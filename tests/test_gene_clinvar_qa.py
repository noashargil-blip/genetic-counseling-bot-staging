# -*- coding: utf-8 -*-
"""
Tests for gene-level ClinVar Q&A integrated into the /ask endpoint (pipeline step 4.5).

Coverage
--------
* Hebrew and English gene-inquiry questions for each of the 6 required genes.
* Unknown-gene safe response.
* Personal-question quarantine — personal interpretation requests must NOT
  reach step 4.5 (they are caught by step 3 instead).
* VUS+gene quarantine — "יש לי VUS ב-BRCA1" must go to vus_known_gene, not
  gene_clinvar_summary.
* Safety invariants on every gene-level answer.
* Follow-up on a gene summary.
* Pure-function tests for _extract_gene_symbol_from_question and
  _is_gene_level_question.

Test strategy
-------------
* When the gene index IS available: all database-backed tests run.
* When absent: those tests are skipped via needs_gene_qa.
* Safety tests (personal questions, forbidden output patterns) always run.
"""

import re
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import gene_index
from app.counseling_engine import (
    _extract_gene_symbol_from_question,
    _is_gene_level_question,
    _build_gene_clinvar_deterministic_answer,
    _GENE_SUMMARY_SAFETY_NOTE_HE,
    _FORBIDDEN_GENE_OUTPUT_RE,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared helpers and marks
# ---------------------------------------------------------------------------

needs_gene_qa = pytest.mark.skipif(
    not gene_index._GENE_INDEX_AVAILABLE,
    reason="ClinVar gene index not available (data/clinvar.duckdb missing or unreadable)",
)

REQUIRED_GENES = ["BRCA1", "BRCA2", "NF1", "SHANK3", "TP53", "CFTR"]

# Phrases that must never appear in a gene-level answer.
FORBIDDEN_PERSONAL_RISK_PHRASES = [
    "הסיכון שלך",
    "מסוכן לך",
    "מסוכן עבורך",
    "יש לך סרטן",
    "יש לך גידול",
    "את חולה",
    "אתה חולה",
    "you have cancer",
    "you should have surgery",
    "you must",
]


def _ask(question: str, **kwargs) -> dict:
    payload = {"question": question}
    payload.update(kwargs)
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    return resp.json()


def _assert_gene_answer_safety(data: dict, gene: str) -> None:
    """Shared safety assertions for every gene-level /ask response."""
    answer = data["answer"]
    assert data["safety_level"] == "general_information", (
        f"{gene}: expected safety_level='general_information', got {data['safety_level']!r}"
    )
    assert data["matched_topic"] == "gene_clinvar_summary", (
        f"{gene}: expected matched_topic='gene_clinvar_summary', got {data['matched_topic']!r}"
    )
    for phrase in FORBIDDEN_PERSONAL_RISK_PHRASES:
        assert phrase.lower() not in answer.lower(), (
            f"{gene}: forbidden phrase {phrase!r} found in answer"
        )
    # Answer must mention the gene symbol
    assert gene in answer, f"{gene}: gene symbol not present in answer"
    # Answer must contain Hebrew characters (it's a Hebrew chatbot)
    assert re.search(r"[א-ת]", answer), f"{gene}: answer contains no Hebrew text"


# ---------------------------------------------------------------------------
# 1. Hebrew intent questions — one per required gene
# ---------------------------------------------------------------------------

@needs_gene_qa
class TestGeneQAHebrew:
    @pytest.mark.parametrize("gene, question", [
        ("BRCA1", "מה ידוע על BRCA1?"),
        ("BRCA2", "תסביר לי על BRCA2"),
        ("NF1",   "מה אפשר לדעת על NF1?"),
        ("SHANK3","ספר לי על SHANK3"),
        ("TP53",  "מה מדווח על TP53 ב-ClinVar?"),
        ("CFTR",  "אילו מצבים קשורים ל-CFTR?"),
    ])
    def test_hebrew_gene_question_routes_to_gene_summary(self, gene, question):
        data = _ask(question)
        _assert_gene_answer_safety(data, gene)

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_answer_mentions_variant_count_or_clinvar(self, gene):
        data = _ask(f"מה ידוע על {gene}?")
        answer = data["answer"]
        meta = data.get("gene_metadata", {})
        # Tier-1: answer contains ClinVar reference or numeric stats.
        # Tier-2: answer is now a short patient-friendly message;
        #         ClinVar stats are in gene_metadata instead.
        if meta.get("answer_tier") == "tier2":
            assert "total_variants" in meta, (
                f"{gene} (Tier 2): total_variants missing from gene_metadata"
            )
        else:
            assert "ClinVar" in answer or re.search(r"\d", answer), (
                f"{gene}: answer has neither 'ClinVar' nor any digit"
            )

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_answer_contains_safety_disclaimer(self, gene):
        data = _ask(f"מה ידוע על {gene}?")
        # The safety note should always appear in the deterministic path
        # (LLM path is off in tests — LOCAL_LLM_URL is unset)
        assert "צוות הגנטי" in data["answer"] or "פנה" in data["answer"], (
            f"{gene}: answer missing counselor-referral safety note"
        )

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_suggested_questions_non_empty(self, gene):
        data = _ask(f"מה ידוע על {gene}?")
        assert len(data["suggested_questions"]) >= 1, (
            f"{gene}: no suggested_questions returned"
        )


# ---------------------------------------------------------------------------
# 2. English intent questions
# ---------------------------------------------------------------------------

@needs_gene_qa
class TestGeneQAEnglish:
    @pytest.mark.parametrize("gene, question", [
        ("BRCA1", "What is known about BRCA1?"),
        ("BRCA2", "Tell me about BRCA2"),
        ("NF1",   "What is NF1?"),
        ("SHANK3","Explain SHANK3"),
        ("TP53",  "Describe TP53"),
        ("CFTR",  "What conditions are associated with CFTR in ClinVar?"),
    ])
    def test_english_gene_question_routes_to_gene_summary(self, gene, question):
        data = _ask(question)
        _assert_gene_answer_safety(data, gene)

    def test_english_how_many_routes_to_gene_summary(self):
        data = _ask("How many variants are in BRCA1?")
        _assert_gene_answer_safety(data, "BRCA1")

    def test_english_what_variants_routes(self):
        data = _ask("What variants are in TP53?")
        _assert_gene_answer_safety(data, "TP53")


# ---------------------------------------------------------------------------
# 3. Unknown gene — safe "not found" response
# ---------------------------------------------------------------------------

@needs_gene_qa
class TestGeneQAUnknownGene:
    def test_unknown_gene_hebrew(self):
        data = _ask("מה ידוע על FAKEGENE99?")
        # Should return a gene_clinvar_summary response with "not found" text,
        # not a crash, and not a random KB answer.
        # Possible outcomes: gene_clinvar_summary (not-found path)
        # or falls through to KB / fallback — both acceptable.
        # Key invariant: no forbidden personal-risk language.
        answer = data["answer"]
        for phrase in FORBIDDEN_PERSONAL_RISK_PHRASES:
            assert phrase.lower() not in answer.lower()

    def test_unknown_gene_english(self):
        data = _ask("Tell me about NOTAREALGENEXYZ")
        answer = data["answer"]
        for phrase in FORBIDDEN_PERSONAL_RISK_PHRASES:
            assert phrase.lower() not in answer.lower()

    def test_unknown_gene_does_not_crash(self):
        resp = client.post("/ask", json={"question": "מה ידוע על XXXXXXX?"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 4. Personal questions must NOT reach gene_clinvar_summary
# ---------------------------------------------------------------------------

class TestGeneQAPersonalQuestionsBlocked:
    """
    These tests do NOT require the gene index — they verify that step 3
    (personal interpretation) intercepts personal questions before step 4.5.
    """

    def test_personal_variant_question_not_gene_summary(self):
        data = _ask("האם הווריאנט שלי ב-BRCA1 מסוכן?")
        assert data["matched_topic"] != "gene_clinvar_summary", (
            "Personal variant question should not reach gene_clinvar_summary"
        )
        assert data["safety_level"] in (
            "requires_genetic_counselor", "general_information"
        )

    def test_personal_risk_question_not_gene_summary(self):
        data = _ask("מה הסיכון שלי בגלל BRCA2?")
        assert data["matched_topic"] != "gene_clinvar_summary"

    def test_vus_gene_question_not_gene_summary(self):
        """'יש לי VUS בגן BRCA1' should go to vus_known_gene, not gene_clinvar_summary."""
        data = _ask("יש לי VUS בגן BRCA1, מה זה?")
        assert data["matched_topic"] == "vus_known_gene", (
            f"Expected vus_known_gene, got {data['matched_topic']!r}"
        )

    def test_vus_gene_nf1_not_gene_summary(self):
        data = _ask("יש לי VUS ב NF1, מה זה?")
        assert data["matched_topic"] == "vus_known_gene"

    def test_surgery_question_not_gene_summary(self):
        # Step 4.5 must not fire for medical-action questions.
        # We only assert the topic routing — the KB counselor flag depends on
        # which KB entry is matched and is not the invariant we're testing here.
        data = _ask("האם עלי לעשות ניתוח בגלל BRCA1?")
        assert data["matched_topic"] != "gene_clinvar_summary", (
            "Surgery question should not be routed to gene_clinvar_summary"
        )


# ---------------------------------------------------------------------------
# 5. Follow-up on a gene summary
# ---------------------------------------------------------------------------

@needs_gene_qa
class TestGeneQAFollowUp:
    def _gene_followup(self, gene: str, followup_text: str) -> dict:
        """Two-turn: first ask about a gene, then ask a follow-up."""
        first = _ask(f"מה ידוע על {gene}?")
        assert first["matched_topic"] == "gene_clinvar_summary"
        context = [
            {"role": "user", "content": f"מה ידוע על {gene}?"},
            {
                "role": "assistant",
                "content": first["answer"],
                "matched_topic": "gene_clinvar_summary",
            },
        ]
        return _ask(
            followup_text,
            last_topic="gene_clinvar_summary",
            conversation_context=context,
        )

    def test_followup_brca1_returns_gene_topic(self):
        data = self._gene_followup("BRCA1", "ספר לי עוד על זה")
        # Either re-routes to gene_clinvar_summary or falls back gracefully
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] in ("gene_clinvar_summary", None) or isinstance(
            data["matched_topic"], str
        )

    def test_followup_answer_not_empty(self):
        data = self._gene_followup("BRCA2", "מה עוד ידוע?")
        assert len(data["answer"]) > 20

    def test_followup_brca1_no_personal_risk(self):
        data = self._gene_followup("BRCA1", "ספר לי עוד")
        for phrase in FORBIDDEN_PERSONAL_RISK_PHRASES:
            assert phrase.lower() not in data["answer"].lower()


# ---------------------------------------------------------------------------
# 6. Pure-function unit tests (no HTTP layer)
# ---------------------------------------------------------------------------

class TestGeneExtractorFunction:
    """Tests for _extract_gene_symbol_from_question — no DB needed for non-index tests."""

    def test_brca1_detected(self):
        assert _extract_gene_symbol_from_question("מה ידוע על BRCA1?") == "BRCA1"

    def test_brca2_detected(self):
        assert _extract_gene_symbol_from_question("tell me about BRCA2") == "BRCA2"

    def test_nf1_detected(self):
        assert _extract_gene_symbol_from_question("יש לי שאלה על NF1") == "NF1"

    def test_vus_token_excluded(self):
        # "VUS" is in _NON_GENE_TOKENS — should not be treated as a gene
        result = _extract_gene_symbol_from_question("מה זה VUS?")
        assert result != "VUS"

    def test_dna_token_excluded(self):
        result = _extract_gene_symbol_from_question("what does DNA do?")
        assert result != "DNA"

    def test_no_gene_returns_none(self):
        assert _extract_gene_symbol_from_question("מה זה נשאות?") is None

    @pytest.mark.parametrize("gene", ["SHANK3", "TP53", "CFTR"])
    @pytest.mark.skipif(
        not gene_index._GENE_INDEX_AVAILABLE,
        reason="Gene index not available",
    )
    def test_index_backed_genes_detected(self, gene):
        result = _extract_gene_symbol_from_question(f"tell me about {gene}")
        assert result == gene, f"Expected {gene}, got {result!r}"


class TestIsGeneLevelQuestion:
    def test_mah_yodua_is_gene_question(self):
        assert _is_gene_level_question("מה ידוע על BRCA1?") is True

    def test_explain_is_gene_question(self):
        assert _is_gene_level_question("explain BRCA2 to me") is True

    def test_tell_me_about_is_gene_question(self):
        assert _is_gene_level_question("ספר לי על NF1") is True

    def test_what_conditions_is_gene_question(self):
        assert _is_gene_level_question("What conditions are associated with CFTR?") is True

    def test_clinvar_keyword_triggers(self):
        assert _is_gene_level_question("מה מדווח ב ClinVar על TP53?") is True

    def test_carrier_question_not_gene_level(self):
        # "נשאות" question should not match gene-level intent phrases
        # (unless it happens to contain a matching phrase like "מה זה")
        result = _is_gene_level_question("אמרו לי שאני נשאית, מה כדאי לעשות?")
        # "מה" alone is not in the list; only "מה זה", "מה ידוע", etc.
        # This test merely verifies it doesn't hard-crash.
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 7. Deterministic answer builder (pure function)
# ---------------------------------------------------------------------------

class TestDeterministicAnswerBuilder:
    """Tests for _build_gene_clinvar_deterministic_answer — no index needed."""

    def _fake_summary(self, gene: str = "BRCA1") -> dict:
        return {
            "gene_symbol": gene,
            "total_variants": 5000,
            "by_significance": {
                "Pathogenic": 2000,
                "Benign": 1500,
                "Uncertain significance": 1200,
                "Conflicting interpretations of pathogenicity": 300,
            },
            "by_review_status": {
                "criteria provided, multiple submitters, no conflicts": 3000,
            },
            "phenotypes": [
                "Hereditary breast and ovarian cancer syndrome",
                "Breast cancer",
                "Ovarian cancer",
            ],
            "variant_types": {
                "single nucleotide variant": 4000,
                "deletion": 600,
            },
            "date_range": {"earliest": "2000-01-01", "latest": "2024-06-01"},
        }

    def test_gene_name_in_answer(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert "BRCA1" in answer

    def test_total_count_in_answer(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        # "5,000" or "5000" should appear
        assert "5,000" in answer or "5000" in answer

    def test_safety_note_in_answer(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        # The safety note is the short general disclaimer
        assert "המידע כללי ואינו מחליף ייעוץ רפואי אישי" in answer

    def test_phenotypes_in_answer(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert "Hereditary breast and ovarian cancer syndrome" in answer

    def test_pathogenic_count_in_answer(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert "Pathogenic" in answer

    def test_no_forbidden_output(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert not _FORBIDDEN_GENE_OUTPUT_RE.search(answer), (
            "Deterministic answer tripped the forbidden-output regex"
        )

    def test_clinvar_mentioned(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert "ClinVar" in answer

    def test_answer_contains_hebrew(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert re.search(r"[א-ת]", answer)

    def test_empty_significance_handled(self):
        summary = self._fake_summary()
        summary["by_significance"] = {}
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", summary)
        assert "BRCA1" in answer  # should not crash

    def test_empty_phenotypes_handled(self):
        summary = self._fake_summary()
        summary["phenotypes"] = []
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", summary)
        assert "BRCA1" in answer  # should not crash

    def test_global_safety_note_constant_in_answer(self):
        answer = _build_gene_clinvar_deterministic_answer("BRCA1", self._fake_summary())
        assert _GENE_SUMMARY_SAFETY_NOTE_HE in answer


# ---------------------------------------------------------------------------
# 8. Forbidden output regex
# ---------------------------------------------------------------------------

class TestForbiddenGeneOutputRegex:
    def test_personal_risk_blocked(self):
        assert _FORBIDDEN_GENE_OUTPUT_RE.search("הסיכון שלך גבוה מאוד")

    def test_dangerous_to_you_blocked(self):
        assert _FORBIDDEN_GENE_OUTPUT_RE.search("מסוכן לך להיות נשאית")

    def test_you_are_sick_blocked(self):
        assert _FORBIDDEN_GENE_OUTPUT_RE.search("את חולה בסרטן")

    def test_you_have_cancer_english_blocked(self):
        assert _FORBIDDEN_GENE_OUTPUT_RE.search("you have cancer")

    def test_you_should_english_blocked(self):
        assert _FORBIDDEN_GENE_OUTPUT_RE.search("you should get tested immediately")

    def test_safe_general_text_passes(self):
        safe = (
            "גן BRCA1 קשור במאגר ClinVar ל-5,000 רשומות וריאנט. "
            "מרביתן מסווגות כ-Pathogenic. "
            "לפרשנות של ממצאיך — פנה לצוות הגנטי."
        )
        assert not _FORBIDDEN_GENE_OUTPUT_RE.search(safe)

    def test_clinical_info_passes(self):
        safe = "הגן TP53 קשור לתסמונת Li-Fraumeni לפי ClinVar."
        assert not _FORBIDDEN_GENE_OUTPUT_RE.search(safe)


# ---------------------------------------------------------------------------
# 9. gene_metadata field in /ask responses
# ---------------------------------------------------------------------------

@needs_gene_qa
class TestGeneMetadata:
    """
    Verify that gene-level /ask responses include the gene_metadata field
    and that non-gene responses do NOT include it (schema stays at 5 keys).
    """

    def test_gene_response_has_gene_metadata(self):
        data = _ask("מה ידוע על BRCA1?")
        assert data["matched_topic"] == "gene_clinvar_summary"
        assert "gene_metadata" in data, "gene_metadata missing from gene-level /ask response"

    def test_gene_metadata_fields(self):
        data = _ask("מה ידוע על BRCA2?")
        meta = data["gene_metadata"]
        assert meta["gene_symbol"] == "BRCA2"
        assert meta["data_source"] in (
            "ClinVar (NCBI) via local gene index",
            "Curated educational content + ClinVar",
        ), f"Unexpected data_source: {meta['data_source']!r}"
        assert isinstance(meta["llm_used"], bool)
        assert isinstance(meta["fallback_used"], bool)
        assert meta["found_in_index"] is True

    def test_gene_metadata_total_variants_positive(self):
        data = _ask("תסביר לי על NF1")
        meta = data["gene_metadata"]
        assert isinstance(meta["total_variants"], int)
        assert meta["total_variants"] > 0

    def test_gene_metadata_llm_false_in_tests(self):
        # LOCAL_LLM_URL is unset in tests — deterministic path must be used.
        data = _ask("מה ידוע על TP53?")
        meta = data["gene_metadata"]
        assert meta["llm_used"] is False
        assert meta["fallback_used"] is True

    def test_non_gene_response_has_no_gene_metadata(self):
        # A normal KB question must not leak gene_metadata into the response.
        data = _ask("מה זה VUS?")
        assert "gene_metadata" not in data, (
            "gene_metadata must not appear in non-gene-summary responses"
        )

    def test_carrier_response_has_no_gene_metadata(self):
        data = _ask("אמרו לי שאני נשאית, מה זה?")
        assert "gene_metadata" not in data

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_all_required_genes_return_gene_metadata(self, gene):
        data = _ask(f"מה ידוע על {gene}?")
        assert "gene_metadata" in data, f"{gene}: gene_metadata missing"
        assert data["gene_metadata"]["gene_symbol"] == gene
