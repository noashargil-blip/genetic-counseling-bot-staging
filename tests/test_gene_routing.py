# -*- coding: utf-8 -*-
"""
tests/test_gene_routing.py

Tests for the gene-level intent routing fix (v2.2.0+).

Problem being tested
--------------------
After a VUS discussion, a follow-on question like "מה המשמעות של הגן APC?"
was misrouted to the VUS follow-up path instead of returning a gene-level
educational answer.  The fix:

1. Broadened _GENE_QUESTION_PHRASES to catch "מה המשמעות של הגן X",
   "מה התפקיד של X", "איזה גן זה X", etc.
2. Extended _GENE_PATTERNS to recognise APC, TP53, SHANK3 without the gene index.
3. Step 4.5 now also fires an educational fallback for pattern-matched genes
   when the gene index is unavailable.
4. Removed "ClinVar" from patient-facing suggested questions (except when the
   user explicitly asked about ClinVar).
"""

import re

import pytest
from fastapi.testclient import TestClient

from app import gene_index as _gene_index_module, gene_cards as _gene_cards_module
from app.counseling_engine import (
    _build_gene_education_fallback,
    _is_gene_level_question,
    _detect_known_gene,
    _GENE_INFO_SUGGESTED_QUESTIONS,
    _FORBIDDEN_GENE_OUTPUT_RE,
)

# Backward-compatible alias so existing test assertions use the same data source.
_GENE_EDUCATION_HE = {g: _gene_cards_module.get_approved_summary(g)
                      for g in _gene_cards_module.list_approved_genes()}
from app.main import app

client = TestClient(app)

FORBIDDEN_PERSONAL_PHRASES = re.compile(
    r"ניתוח|כריתה|קולונוסקופיה|סקירה|מעקב רפואי אישי"
    r"|יש\s+לך\s+(מחלה|סרטן|FAP)"
    r"|את[הן]?\s+צריכ[הי]?\s+(לעשות|לבצע|להיבדק)"
    r"|הסיכון\s+שלך",
    re.IGNORECASE,
)


def _ask(question: str, **extra) -> dict:
    payload = {"question": question}
    payload.update(extra)
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# A. _is_gene_level_question — new phrase coverage
# ---------------------------------------------------------------------------

class TestIsGeneLevelQuestionNewPhrases:
    def test_mashmaut_shel_hagen(self):
        assert _is_gene_level_question("מה המשמעות של הגן APC?") is True

    def test_mashmaut_shel_without_hagen(self):
        assert _is_gene_level_question("מה המשמעות של APC בדוח?") is True

    def test_tafkid(self):
        assert _is_gene_level_question("מה התפקיד של SHANK3?") is True

    def test_tafkid_shel(self):
        assert _is_gene_level_question("תפקיד של TP53") is True

    def test_eizehu_gen(self):
        assert _is_gene_level_question("איזה גן זה BRCA1?") is True

    def test_hagen_haze(self):
        assert _is_gene_level_question("מה הגן הזה עושה?") is True

    def test_existing_mah_yodua_still_works(self):
        assert _is_gene_level_question("מה ידוע על BRCA1?") is True

    def test_existing_saper_still_works(self):
        assert _is_gene_level_question("ספר לי על NF1") is True


# ---------------------------------------------------------------------------
# B. _detect_known_gene — new patterns
# ---------------------------------------------------------------------------

class TestDetectKnownGeneNewPatterns:
    def test_apc_detected(self):
        assert _detect_known_gene("יש לי VUS בגן APC") == "APC"

    def test_apc_lower(self):
        assert _detect_known_gene("מה זה apc?") == "APC"

    def test_tp53_detected(self):
        assert _detect_known_gene("שמעתי על TP53") == "TP53"

    def test_shank3_detected(self):
        assert _detect_known_gene("מה זה SHANK3?") == "SHANK3"

    def test_shank_space_3(self):
        assert _detect_known_gene("גן SHANK 3") == "SHANK3"

    def test_brca1_still_detected(self):
        assert _detect_known_gene("יש לי VUS ב-BRCA1") == "BRCA1"


# ---------------------------------------------------------------------------
# C. _build_gene_education_fallback — unit tests
# ---------------------------------------------------------------------------

class TestBuildGeneEducationFallback:
    def test_apc_uses_curated_content(self):
        result = _build_gene_education_fallback("APC")
        assert "APC" in result["answer"]
        assert "Familial Adenomatous Polyposis" in result["answer"]

    def test_brca1_uses_curated_content(self):
        result = _build_gene_education_fallback("BRCA1")
        assert "BRCA1" in result["answer"]
        assert "DNA" in result["answer"]

    def test_unknown_gene_uses_generic_template(self):
        result = _build_gene_education_fallback("FAKE99")
        assert "FAKE99" in result["answer"]

    def test_safety_level_general_information(self):
        result = _build_gene_education_fallback("APC")
        assert result["safety_level"] == "general_information"
        assert result["needs_genetic_counselor"] is False

    def test_matched_topic_gene_info(self):
        result = _build_gene_education_fallback("APC")
        assert result["matched_topic"] == "gene_info"

    def test_no_personal_recommendations(self):
        for gene in list(_GENE_EDUCATION_HE) + ["UNKNOWN_X"]:
            result = _build_gene_education_fallback(gene)
            assert not FORBIDDEN_PERSONAL_PHRASES.search(result["answer"]), (
                f"Gene {gene!r} education text contains personal recommendation language"
            )

    def test_safety_note_present(self):
        result = _build_gene_education_fallback("APC")
        assert "צוות הגנטי" in result["answer"]

    def test_no_forbidden_gene_output(self):
        for gene in list(_GENE_EDUCATION_HE):
            result = _build_gene_education_fallback(gene)
            assert not _FORBIDDEN_GENE_OUTPUT_RE.search(result["answer"]), (
                f"Gene {gene!r} education text tripped forbidden-output regex"
            )

    def test_suggested_questions_no_clinvar(self):
        result = _build_gene_education_fallback("APC")
        for q in result["suggested_questions"]:
            assert "ClinVar" not in q, f"Suggested question contains ClinVar: {q!r}"

    def test_llm_metadata(self):
        result = _build_gene_education_fallback("APC")
        assert result["llm_used"] is False
        assert result["fallback_used"] is True


# ---------------------------------------------------------------------------
# D. Routing integration: gene-explanation questions go to gene answer
# ---------------------------------------------------------------------------

class TestGeneExplanationRouting:
    """
    These tests run against the live pipeline.  When the gene_index is
    available (the normal test environment), questions about a recognised
    gene return gene_clinvar_summary.  When the index is unavailable
    (monkeypatched), pattern-matched genes return gene_info.
    """

    def test_apc_vus_question_returns_vus_topic(self):
        """'מה זה VUS בגן APC?' must stay on the VUS path, not the gene path."""
        data = _ask("מה זה VUS בגן APC?")
        assert data["matched_topic"] == "vus_known_gene", (
            f"Expected vus_known_gene, got {data['matched_topic']!r}"
        )
        assert "VUS" in data["answer"]
        assert "pathogenic" in data["answer"].lower() or "פתוגני" in data["answer"]

    def test_apc_gene_question_returns_gene_level_answer(self):
        """'מה המשמעות של הגן APC?' must return a gene-level answer, not VUS."""
        data = _ask("מה המשמעות של הגן APC?")
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info"), (
            f"Expected gene-level topic, got {data['matched_topic']!r}"
        )
        assert "APC" in data["answer"]
        assert data["safety_level"] == "general_information"

    def test_apc_gene_question_not_vus_answer(self):
        """The gene-explanation answer for APC must not be the generic VUS template."""
        data = _ask("מה המשמעות של הגן APC?")
        # The VUS template starts with "כאשר מתקבל VUS בגן" — should NOT appear
        assert not data["answer"].startswith("כאשר מתקבל VUS"), (
            "Gene explanation question returned the VUS-known-gene template"
        )

    def test_apc_gene_question_after_vus_context(self):
        """Even with prior VUS context, a gene question about APC returns gene info."""
        vus_ctx = [
            {"role": "user", "content": "הרופא אמר לי שיש לי VUS בגן APC"},
            {
                "role": "assistant",
                "content": "VUS הוא ממצא בעל משמעות לא ידועה...",
                "matched_topic": "vus_known_gene",
            },
        ]
        data = _ask(
            "מה המשמעות של הגן APC?",
            last_topic="vus_known_gene",
            conversation_context=vus_ctx,
        )
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info"), (
            f"With VUS context, gene question returned {data['matched_topic']!r} — "
            "should have been overridden by gene-level intent"
        )

    def test_brca1_gene_question_returns_gene_level_answer(self):
        data = _ask("מה ידוע על BRCA1?")
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info")
        assert data["safety_level"] == "general_information"

    def test_shank3_gene_question_returns_gene_level_answer(self):
        data = _ask("מה התפקיד של SHANK3?")
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info"), (
            f"SHANK3 question returned unexpected topic {data['matched_topic']!r}"
        )
        assert data["safety_level"] == "general_information"

    def test_tp53_tafkid_question(self):
        data = _ask("מה התפקיד של TP53?")
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info")

    def test_eizehu_gen_brca1(self):
        data = _ask("איזה גן זה BRCA1?")
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info")

    def test_mashmaut_shel_nf1(self):
        data = _ask("מה המשמעות של הגן NF1?")
        assert data["matched_topic"] in ("gene_clinvar_summary", "gene_info")


# ---------------------------------------------------------------------------
# E. Gene-index-unavailable fallback path
# ---------------------------------------------------------------------------

class TestGeneEducationFallbackPath:
    """
    Monkeypatches gene_index._GENE_INDEX_AVAILABLE = False to force the
    pattern-based fallback path through _build_gene_education_fallback.
    """

    def test_apc_without_gene_index(self, monkeypatch):
        monkeypatch.setattr(_gene_index_module, "_GENE_INDEX_AVAILABLE", False)
        data = _ask("מה המשמעות של הגן APC?")
        assert data["matched_topic"] == "gene_info", (
            f"Without gene index, APC question should return gene_info, got {data['matched_topic']!r}"
        )
        assert "APC" in data["answer"]
        assert data["safety_level"] == "general_information"

    def test_brca1_without_gene_index(self, monkeypatch):
        monkeypatch.setattr(_gene_index_module, "_GENE_INDEX_AVAILABLE", False)
        data = _ask("מה ידוע על BRCA1?")
        # With index down, "מה ידוע" is a gene-level phrase but BRCA1 is pattern-matched
        assert data["matched_topic"] == "gene_info"
        assert "BRCA1" in data["answer"]

    def test_shank3_without_gene_index(self, monkeypatch):
        monkeypatch.setattr(_gene_index_module, "_GENE_INDEX_AVAILABLE", False)
        data = _ask("מה התפקיד של SHANK3?")
        assert data["matched_topic"] == "gene_info"

    def test_vus_apc_still_goes_to_vus_known_gene_without_index(self, monkeypatch):
        monkeypatch.setattr(_gene_index_module, "_GENE_INDEX_AVAILABLE", False)
        data = _ask("יש לי VUS בגן APC, מה זה?")
        assert data["matched_topic"] == "vus_known_gene"

    def test_gene_info_answer_no_forbidden_phrases(self, monkeypatch):
        monkeypatch.setattr(_gene_index_module, "_GENE_INDEX_AVAILABLE", False)
        data = _ask("מה המשמעות של הגן APC?")
        assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"]), (
            "Gene_info answer contains personal recommendation language"
        )

    def test_gene_info_suggested_questions_no_clinvar(self, monkeypatch):
        monkeypatch.setattr(_gene_index_module, "_GENE_INDEX_AVAILABLE", False)
        data = _ask("מה זה APC?")
        if data["matched_topic"] == "gene_info":
            for q in data["suggested_questions"]:
                assert "ClinVar" not in q, f"Suggested question contains ClinVar: {q!r}"


# ---------------------------------------------------------------------------
# F. Patient-facing suggested questions must not mention ClinVar (unless
#    the user explicitly asked about ClinVar)
# ---------------------------------------------------------------------------

class TestNoClivarInSuggestedQuestions:
    def test_vus_followup_no_clinvar_in_suggestions(self):
        """After a VUS answer, the follow-up suggested questions must not mention ClinVar."""
        vus_ctx = [
            {"role": "user", "content": "יש לי VUS ב-BRCA1"},
            {
                "role": "assistant",
                "content": "...",
                "matched_topic": "vus_known_gene",
            },
        ]
        data = _ask(
            "מה כדאי לעשות?",
            last_topic="vus_known_gene",
            conversation_context=vus_ctx,
        )
        for q in data["suggested_questions"]:
            assert "ClinVar" not in q, (
                f"Suggested question after VUS follow-up contains ClinVar: {q!r}"
            )

    def test_vus_followup_answer_body_no_clinvar_suggested_question(self):
        """The body of a VUS practical follow-up answer must not ask the user
        to check ClinVar as a suggested question."""
        data = _ask("מה ההשלכות?", last_topic="vus_known_gene")
        # The body of _compose_vus_practical_answer is what we check —
        # it must not suggest asking "האם הווריאנט דווח ב-ClinVar"
        assert "דווח בעבר במאגרים כמו ClinVar" not in data["answer"], (
            "VUS follow-up answer still contains the old ClinVar suggested question"
        )

    def test_gene_clinvar_answer_suggested_questions_no_clinvar(self):
        """Gene-level ClinVar stats answer: suggested questions must be patient-friendly."""
        data = _ask("מה ידוע על BRCA1?")
        if data["matched_topic"] == "gene_clinvar_summary":
            for q in data["suggested_questions"]:
                assert "ClinVar" not in q, (
                    f"gene_clinvar_summary suggested question contains ClinVar: {q!r}"
                )

    def test_clinvar_explicit_question_may_mention_clinvar_in_answer(self):
        """When the user explicitly asks about ClinVar, the ANSWER body may mention it."""
        data = _ask("מה ClinVar אומר על APC?")
        # The routing is gene-level — just verify safety
        assert data["safety_level"] == "general_information"
        assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"])


# ---------------------------------------------------------------------------
# G. Safety invariants — gene-level answers must never give personal advice
# ---------------------------------------------------------------------------

class TestGeneLevelAnswerSafetyInvariants:
    GENE_QUESTIONS = [
        "מה המשמעות של הגן APC?",
        "מה ידוע על BRCA1?",
        "מה התפקיד של TP53?",
        "ספר לי על NF1",
        "מה זה APC?",
    ]

    @pytest.mark.parametrize("question", GENE_QUESTIONS)
    def test_no_personal_recommendations(self, question):
        data = _ask(question)
        assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"]), (
            f"Gene-level answer for {question!r} contains personal recommendation language"
        )

    @pytest.mark.parametrize("question", GENE_QUESTIONS)
    def test_safety_level_is_general_information(self, question):
        data = _ask(question)
        assert data["safety_level"] == "general_information", (
            f"Gene-level answer for {question!r} has unexpected safety_level {data['safety_level']!r}"
        )

    @pytest.mark.parametrize("question", GENE_QUESTIONS)
    def test_has_llm_metadata(self, question):
        data = _ask(question)
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_apc_answer_does_not_say_you_have_fap(self):
        data = _ask("מה המשמעות של הגן APC?")
        assert "יש לך FAP" not in data["answer"]
        assert "יש לך Familial Adenomatous Polyposis" not in data["answer"]

    def test_apc_answer_does_not_say_colonoscopy(self):
        data = _ask("מה המשמעות של הגן APC?")
        assert "קולונוסקופיה" not in data["answer"]

    def test_apc_answer_no_surgery_recommendation(self):
        data = _ask("מה המשמעות של הגן APC?")
        assert "ניתוח" not in data["answer"]

    def test_response_schema_complete(self):
        data = _ask("מה המשמעות של הגן APC?")
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions", "llm_used", "fallback_used"}
        assert required.issubset(data.keys()), (
            f"Missing keys: {required - set(data.keys())}"
        )


# ---------------------------------------------------------------------------
# H. Curated education wins over local-index "not found"
# ---------------------------------------------------------------------------

class TestCuratedEducationWinsOverNotFound:
    """
    When gene_index.get_gene_summary() returns None (gene absent from local DB),
    the curated _GENE_EDUCATION_HE content must be returned instead of a generic
    "not found" message.  We monkeypatch get_gene_summary to force this path
    regardless of what the local test database actually contains.
    """

    def _mock_get_gene_summary_none(self, monkeypatch):
        """Force get_gene_summary to return None for every gene."""
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)

    def test_apc_gets_curated_content_when_not_in_index(self, monkeypatch):
        """When APC is absent from the local DB, curated education is returned."""
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)
        data = _ask("מה זה APC?")
        assert data["matched_topic"] == "gene_clinvar_summary"
        assert "לא נמצא במאגר ClinVar" not in data["answer"], (
            "Old 'not found in ClinVar' message served — curated content should have won"
        )
        assert "APC" in data["answer"]
        # Curated APC content mentions cell growth/division
        assert "גדילה" in data["answer"] or "תאים" in data["answer"] or "בקרה" in data["answer"]

    def test_apc_gene_metadata_found_in_index_false_when_not_in_db(self, monkeypatch):
        """When curated content is served for a gene not in the local index,
        gene_metadata.found_in_index is False."""
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)
        data = _ask("מה זה APC?")
        assert "gene_metadata" in data
        assert data["gene_metadata"]["found_in_index"] is False
        assert data["gene_metadata"]["gene_symbol"] == "APC"

    def test_brca1_gets_curated_content_when_not_in_index(self, monkeypatch):
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)
        data = _ask("מה ידוע על BRCA1?")
        assert "לא נמצא במאגר ClinVar" not in data["answer"]
        assert "BRCA1" in data["answer"]

    def test_brca2_gets_curated_content_when_not_in_index(self, monkeypatch):
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)
        data = _ask("ספר לי על BRCA2")
        assert "לא נמצא במאגר ClinVar" not in data["answer"]
        assert "BRCA2" in data["answer"]

    def test_tp53_gets_curated_content_when_not_in_index(self, monkeypatch):
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)
        data = _ask("מה התפקיד של TP53?")
        assert "לא נמצא במאגר ClinVar" not in data["answer"]
        assert "TP53" in data["answer"] or "p53" in data["answer"]

    def test_nf1_in_real_index_has_found_in_index_true(self):
        """NF1 has 5946 records in the local DB — found_in_index must be True."""
        data = _ask("מה ידוע על NF1?")
        assert "gene_metadata" in data
        assert data["gene_metadata"]["found_in_index"] is True, (
            "NF1 is in the local DB so found_in_index should be True"
        )

    def test_curated_gene_has_safety_note_when_not_in_index(self, monkeypatch):
        import app.gene_index as gi
        monkeypatch.setattr(gi, "get_gene_summary", lambda gene: None)
        data = _ask("מה זה APC?")
        assert "כללי בלבד" in data["answer"] or "צוות הגנטי" in data["answer"]

    def test_curated_gene_no_personal_phrases(self):
        """Gene-level answers (regardless of source) must have no personal recommendations."""
        for gene_q in ["מה זה APC?", "ספר לי על BRCA2", "מה התפקיד של TP53?"]:
            data = _ask(gene_q)
            assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"]), (
                f"Gene-level answer for {gene_q!r} contains personal recommendation language"
            )

    def test_apc_any_source_returns_gene_clinvar_summary(self):
        """Regardless of whether the local index has APC, matched_topic is gene_clinvar_summary."""
        data = _ask("מה זה APC?")
        assert data["matched_topic"] == "gene_clinvar_summary"
        assert "APC" in data["answer"]
        assert data["safety_level"] == "general_information"


# ---------------------------------------------------------------------------
# I. Unknown gene fallback — gentle language
# ---------------------------------------------------------------------------

class TestUnknownGeneFallbackLanguage:
    """
    For genes that are not in _GENE_EDUCATION_HE AND not in the local index,
    the response must not imply the gene does not exist globally.
    """

    def test_sox1_not_found_answer_is_gentle(self):
        data = _ask("מה זה SOX1?")
        # Either goes to gene_clinvar_summary (not-found path) or falls through to KB
        if data["matched_topic"] == "gene_clinvar_summary":
            assert "לא קיים" not in data["answer"], "Must not say the gene does not exist"
            assert "אינו ידוע" not in data["answer"], "Must not say the gene is unknown"
            # Should say it's not in the LOCAL dataset, not that the gene doesn't exist
            assert "מאגר" in data["answer"] or "מקומי" in data["answer"] or "צוות" in data["answer"]

    def test_unknown_gene_answer_no_personal_recommendations(self):
        data = _ask("מה זה FAKEGENE999?")
        assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"])

    def test_not_found_message_does_not_say_old_text(self):
        """Old message started with 'הגן X לא נמצא במאגר ClinVar המקומי.' — should be gone."""
        data = _ask("מה ידוע על SOX1?")
        if data["matched_topic"] == "gene_clinvar_summary":
            assert "לא נמצא במאגר ClinVar המקומי" not in data["answer"], (
                "Old 'not found in local ClinVar' phrasing still present"
            )


# ---------------------------------------------------------------------------
# J. VUS + gene includes brief gene description
# ---------------------------------------------------------------------------

class TestVusGeneAnswerIncludesGeneNote:
    """
    "מה זה VUS בגן APC?" should explain what APC is (briefly) AND explain VUS,
    not just VUS alone.
    """

    def test_vus_apc_answer_includes_apc_description(self):
        data = _ask("מה זה VUS בגן APC?")
        assert data["matched_topic"] == "vus_known_gene"
        # Answer should mention APC gene context AND VUS
        assert "APC" in data["answer"]
        assert "VUS" in data["answer"]
        # Gene note should mention cell growth/division (from curated APC content)
        assert "גדילה" in data["answer"] or "תאים" in data["answer"] or "בקרה" in data["answer"], (
            "VUS+APC answer should include a brief note about what APC does"
        )

    def test_vus_brca1_answer_includes_brca1_description(self):
        data = _ask("יש לי VUS ב-BRCA1, מה זה?")
        assert data["matched_topic"] == "vus_known_gene"
        assert "BRCA1" in data["answer"]
        assert "VUS" in data["answer"]
        # Brief gene note should mention DNA repair
        assert "DNA" in data["answer"]

    def test_vus_gene_with_no_curated_content_still_works(self):
        """VUS + a gene symbol that _build_known_gene_answer doesn't have curated content for
        should still return a valid answer.  We test this directly via the function since
        CFTR is not in _GENE_PATTERNS (so step 4 never fires via the API for CFTR).
        For Tier-2 genes (in ClinVar index, no approved card), the answer now includes
        a patient-friendly note instead of a raw ClinVar dump."""
        from app.counseling_engine import _build_known_gene_answer
        result = _build_known_gene_answer("CFTR")
        assert result["matched_topic"] == "vus_known_gene"
        assert "VUS" in result["answer"]
        assert result["safety_level"] == "general_information"
        meta = result.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            # Tier-2: patient-friendly note present; raw ClinVar dump absent
            assert "לגבי הגן CFTR:" in result["answer"], (
                "Tier-2 VUS+gene answer should include patient-friendly gene note"
            )
            assert "נתוני ClinVar" not in result["answer"], (
                "Raw ClinVar header must not appear in Tier-2 VUS+gene answer"
            )
        else:
            # Tier-3 or Tier-1: no gene note; no raw ClinVar dump either
            assert "נתוני ClinVar עבור גן CFTR" not in result["answer"]

    def test_vus_apc_answer_no_personal_recommendations(self):
        data = _ask("מה זה VUS בגן APC?")
        assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"])


# ---------------------------------------------------------------------------
# K. Enriched VUS + gene answer — v2.3 requirements
# ---------------------------------------------------------------------------

class TestEnrichedVusGeneAnswer:
    """
    Tests for the enriched VUS+gene answer that combines:
    - warm VUS opening
    - curated gene education (if available) or generic VUS explanation
    - ClinVar aggregate stats when available
    - safety note

    Also covers:
    - Step 4 gene detection via gene index (HBB, SOX1 — not in _GENE_PATTERNS)
    - Curated education priority over raw ClinVar stats for gene-level questions
    - Condition label filtering ("8 conditions", "not provided", etc.)
    """

    # -- VUS + gene: warm opening ------------------------------------------

    def test_vus_brca1_has_warm_opening(self):
        """Answer must not start with the old robotic template 'כאשר מתקבל VUS'."""
        data = _ask("מה זה VUS בגן BRCA1")
        assert data["matched_topic"] == "vus_known_gene"
        assert not data["answer"].startswith("כאשר מתקבל VUS"), (
            "Old robotic template still in use"
        )
        assert "VUS" in data["answer"]

    def test_vus_brca1_includes_both_vus_and_gene_context(self):
        """VUS+BRCA1 answer must contain both VUS explanation and BRCA1 gene info."""
        data = _ask("מה זה VUS בגן BRCA1")
        assert "VUS" in data["answer"]
        # Curated BRCA1 content mentions DNA
        assert "DNA" in data["answer"], "BRCA1 education (DNA repair) should be in the answer"
        # VUS safety/practical language
        assert "פתוגני" in data["answer"], "Answer should distinguish VUS from pathogenic"

    def test_vus_apc_includes_both_vus_and_gene_context(self):
        """'אמרו לי שיש לי VUS בגן APC מה זה אומר' should include APC gene context."""
        data = _ask("אמרו לי שיש לי VUS בגן APC מה זה אומר?")
        assert data["matched_topic"] == "vus_known_gene"
        assert "VUS" in data["answer"]
        assert "APC" in data["answer"]
        # Curated APC content mentions cell growth / growth control
        assert (
            "גדילה" in data["answer"]
            or "תאים" in data["answer"]
            or "בקרה" in data["answer"]
        ), "APC curated education should appear in VUS+APC answer"

    def test_vus_gene_answer_has_safety_note(self):
        """Every VUS+gene answer must include a safety note about personal interpretation."""
        data = _ask("יש לי VUS ב-BRCA2, מה זה?")
        assert "כללי" in data["answer"] or "צוות הגנטי" in data["answer"], (
            "Safety note about personal interpretation missing"
        )

    def test_vus_gene_answer_no_personal_recommendations(self):
        for question in [
            "מה זה VUS בגן BRCA1",
            "אמרו לי שיש לי VUS בגן APC מה זה אומר?",
            "מה המשמעות של VUS ב-NF1?",
        ]:
            data = _ask(question)
            assert not FORBIDDEN_PERSONAL_PHRASES.search(data["answer"]), (
                f"Personal recommendation in VUS+gene answer: {question!r}"
            )

    # -- Gene detection via gene index (HBB, SOX1) -------------------------

    def test_vus_hbb_detected_via_gene_index(self):
        """'מה המשמעות של VUS ב-HBB' should route to vus_known_gene (gene found via index)."""
        data = _ask("מה המשמעות של VUS ב-HBB?")
        assert data["matched_topic"] == "vus_known_gene", (
            f"Expected vus_known_gene, got {data['matched_topic']!r}"
        )
        assert "VUS" in data["answer"]
        assert data["safety_level"] == "general_information"

    def test_vus_hbb_answer_includes_hbb_context(self):
        """VUS+HBB answer should include HBB gene information."""
        data = _ask("יש לי VUS ב-HBB, מה זה?")
        assert "HBB" in data["answer"], "HBB not mentioned in VUS+HBB answer"
        assert "VUS" in data["answer"]

    def test_vus_sox1_detected_via_gene_index(self):
        """'יש לי VUS בגן SOX1' should also route to vus_known_gene."""
        data = _ask("יש לי VUS בגן SOX1, מה אני צריכה לדעת?")
        assert data["matched_topic"] == "vus_known_gene", (
            f"Expected vus_known_gene, got {data['matched_topic']!r}"
        )
        assert "VUS" in data["answer"]
        assert data["safety_level"] == "general_information"

    def test_vus_gene_no_diagnosis(self):
        """VUS+gene answers for genes found via index must contain no diagnosis."""
        data = _ask("יש לי VUS ב-HBB, מה זה?")
        assert "יש לך" not in data["answer"] or all(
            word not in data["answer"] for word in ["מחלה", "סרטן", "FAP"]
        )

    # -- Curated education priority for gene-level questions ---------------

    def test_apc_gene_level_prefers_curated_over_raw_stats(self):
        """'מה זה APC' should lead with curated education, not raw ClinVar header."""
        data = _ask("מה זה APC")
        assert data["matched_topic"] == "gene_clinvar_summary"
        # Curated APC content is about cell growth control — should appear before raw stats
        assert "APC" in data["answer"]
        # Curated content should be present (cell-growth language)
        assert (
            "גדילה" in data["answer"]
            or "תאים" in data["answer"]
            or "בקרה" in data["answer"]
        ), "Curated APC education should appear in gene-level answer"
        # Raw stats header should NOT be the first line
        assert not data["answer"].strip().startswith("מידע כללי על גן APC ממאגר ClinVar"), (
            "Raw ClinVar stats header should not be the first thing shown"
        )

    def test_brca1_gene_level_prefers_curated(self):
        data = _ask("מה ידוע על BRCA1?")
        assert data["matched_topic"] == "gene_clinvar_summary"
        assert "DNA" in data["answer"], "Curated BRCA1 education should mention DNA repair"

    def test_sox1_gene_level_returns_clinvar_stats(self):
        """SOX1 has no curated education — should return gene index stats."""
        data = _ask("מה זה SOX1?")
        assert data["matched_topic"] == "gene_clinvar_summary"
        assert "SOX1" in data["answer"]
        assert data["safety_level"] == "general_information"
        assert "gene_metadata" in data
        assert data["gene_metadata"]["found_in_index"] is True

    def test_hbb_gene_level_returns_clinvar_stats(self):
        """HBB has no curated education — should return gene index stats."""
        data = _ask("מה זה HBB?")
        assert data["matched_topic"] == "gene_clinvar_summary"
        assert "HBB" in data["answer"]
        assert data["safety_level"] == "general_information"

    # -- Condition label filtering -----------------------------------------

    def test_hbb_answer_does_not_contain_count_condition_labels(self):
        """'8 conditions', '2 conditions', etc. must be filtered from patient-facing answers."""
        data = _ask("מה זה HBB?")
        import re
        count_label_re = re.compile(r'\d+\s+condition', re.IGNORECASE)
        assert not count_label_re.search(data["answer"]), (
            "Non-patient-friendly '8 conditions' style label found in answer"
        )

    def test_gene_answer_does_not_contain_not_provided_label(self):
        """'not provided' condition label must be filtered."""
        data = _ask("מה ידוע על NF1?")
        assert "not provided" not in data["answer"].lower()

    def test_filter_patient_conditions_removes_count_labels(self):
        """Unit test for _filter_patient_conditions."""
        from app.counseling_engine import _filter_patient_conditions
        raw = [
            "Hereditary breast ovarian cancer syndrome",
            "8 conditions",
            "not provided",
            "not specified",
            "See cases",
            "",
            "abc",   # too short
            "Breast-ovarian cancer, familial",
            "42",    # purely numeric
        ]
        clean = _filter_patient_conditions(raw)
        assert "8 conditions" not in clean
        assert "not provided" not in clean
        assert "not specified" not in clean
        assert "See cases" not in clean
        assert "" not in clean
        assert "abc" not in clean
        assert "42" not in clean
        assert "Hereditary breast ovarian cancer syndrome" in clean
        assert "Breast-ovarian cancer, familial" in clean

    # -- Suggested questions must not contain ClinVar jargon ---------------

    def test_vus_gene_suggested_questions_no_clinvar(self):
        """Suggested questions in VUS+gene answers must not mention ClinVar."""
        for question in ["מה זה VUS בגן BRCA1", "יש לי VUS ב-HBB"]:
            data = _ask(question)
            for q in data["suggested_questions"]:
                assert "ClinVar" not in q, (
                    f"Suggested question contains ClinVar for {question!r}: {q!r}"
                )

    def test_gene_level_suggested_questions_no_clinvar(self):
        """Gene-level answers for genes with curated content must have patient-friendly suggestions."""
        data = _ask("מה זה APC?")
        for q in data["suggested_questions"]:
            assert "ClinVar" not in q, f"Suggested question contains ClinVar: {q!r}"

    # -- Response schema ---------------------------------------------------

    def test_vus_gene_response_schema(self):
        """VUS+gene /ask response must have all required fields."""
        data = _ask("מה זה VUS בגן BRCA1")
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions", "llm_used", "fallback_used"}
        assert required.issubset(data.keys())

    def test_gene_level_response_schema(self):
        """Gene-level /ask response must have all required fields + gene_metadata."""
        data = _ask("מה זה APC?")
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions", "llm_used", "fallback_used",
                    "gene_metadata"}
        assert required.issubset(data.keys())
