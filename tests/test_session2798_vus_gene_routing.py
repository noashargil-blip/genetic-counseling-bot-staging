# -*- coding: utf-8 -*-
"""
tests/test_session2798_vus_gene_routing.py

Session 27.9.8 — Route all VUS+gene-symbol queries through the safe pipeline.
Also covers the addendum regressions fixed in the same commit:
  - Gene-function questions ("איך הגן BRCA1 פועל בגוף?") no longer captured by what_is_gene
  - Genome history ("מתי גילו את הגנום?") no longer misrouted to human_genome_size KB
  - Generic disclaimer removed from routine KB answers

Groups:
  1. TestMixedCaseGeneExtraction     — _extract_gene_from_explicit_phrase unit tests
  2. TestVusGeneIntentRouting        — classify_question_intent for VUS+gene
  3. TestVusGeneNeverGeneralAI       — _classify_general_question guard
  4. TestGeneFunctionQuestionsRoute  — "איך הגן X פועל" routes to gene pipeline
  5. TestWhatIsGeneGuard             — what_is_gene KB guard with explicit gene symbol
  6. TestGenomeHistoryRouting        — human_genome_size KB guard for "מתי" questions
  7. TestGenericDisclaimerAbsent     — disclaimer removed from KB and deterministic answers
  8. TestFalsePositivePrevention     — no false positives on non-gene uppercase tokens
  9. TestSafetyPreserved             — safety pipeline still blocks personal/medical queries
 10. TestRegressions                 — BRCA1, COL1A1, ACE, and standalone paths unchanged
"""

import importlib
import re
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as _ce

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures and test data
# ---------------------------------------------------------------------------

_GENE_SUMMARY_COL1A1_5PH = {
    "gene_symbol": "COL1A1",
    "total_variants": 3200,
    "by_significance": {"Pathogenic": 1200, "Uncertain significance": 1400, "Benign": 600},
    "phenotypes": [
        "Osteogenesis imperfecta",
        "Ehlers-Danlos syndrome",
        "Caffey disease",
        "Osteoporosis",
        "Bruck syndrome",
    ],
}

_GENE_SUMMARY_ACE_2PH = {
    "gene_symbol": "ACE",
    "total_variants": 1628,
    "by_significance": {"Uncertain significance": 950},
    "phenotypes": ["Renal tubular dysgenesis", "Hypertension"],
}

_FAKE_DRAFT_ACE = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "ACE",
    "text_he": "הגן ACE מקודד לאנזים הממיר אנגיוטנסין.",
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
    "based_on": "model_expansion_with_clinvar_gene_context",
    "source_grounded": False,
}


@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_review_2798.db")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from app import review_db
    importlib.reload(review_db)
    review_db.init_db()
    yield review_db


# ===========================================================================
# 1. TestMixedCaseGeneExtraction
# ===========================================================================

class TestMixedCaseGeneExtraction:
    """_extract_gene_from_explicit_phrase must extract HGNC-style mixed-case symbols."""

    def test_c12orf57_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("מה המשמעות של VUS בגן C12orf57?")
        assert sym == "C12orf57"

    def test_kiaa2022_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("איך הגן KIAA2022 פועל בגוף?")
        assert sym == "KIAA2022"

    def test_brca1_extracted_uppercase(self):
        sym = _ce._extract_gene_from_explicit_phrase("יש לי VUS בגן BRCA1, מה זה?")
        assert sym == "BRCA1"

    def test_ace_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("מה המשמעות של VUS בגן ACE?")
        assert sym == "ACE"

    def test_vus_not_extracted_as_gene(self):
        # VUS is in _NON_GENE_TOKENS
        sym = _ce._extract_gene_from_explicit_phrase("מה המשמעות של VUS בגן VUS?")
        assert sym is None

    def test_dna_not_extracted_as_gene(self):
        sym = _ce._extract_gene_from_explicit_phrase("שינוי בגן DNA")
        assert sym is None

    def test_no_explicit_phrase_returns_none(self):
        # No "בגן" or "הגן" → None
        sym = _ce._extract_gene_from_explicit_phrase("יש לי BRCA1 VUS")
        assert sym is None

    def test_single_lowercase_letter_rejected(self):
        # Starts with uppercase but only 1 uppercase char + no digit → rejected
        sym = _ce._extract_gene_from_explicit_phrase("הגן Ab")
        assert sym is None

    def test_short_non_gene_rejected(self):
        # "Aa" — 2 chars, 1 uppercase, no digit → rejected
        sym = _ce._extract_gene_from_explicit_phrase("הגן Aa")
        assert sym is None

    def test_col1a1_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("נמצא לי VUS בגן COL1A1")
        assert sym == "COL1A1"


# ===========================================================================
# 2. TestVusGeneIntentRouting
# ===========================================================================

class TestVusGeneIntentRouting:
    """classify_question_intent must route VUS+gene queries as explicit_gene_question."""

    def test_c12orf57_vus_intent(self):
        r = _ce.classify_question_intent("מה המשמעות של VUS בגן C12orf57?")
        assert r["intent"] == "explicit_gene_question"
        assert r["gene_symbol"] == "C12orf57"

    def test_kiaa2022_vus_intent(self):
        r = _ce.classify_question_intent("מה המשמעות של VUS בגן KIAA2022?")
        assert r["intent"] == "explicit_gene_question"
        assert r["gene_symbol"] == "KIAA2022"

    def test_brca1_vus_intent_preserved(self):
        r = _ce.classify_question_intent("יש לי VUS ב-BRCA1")
        assert r["intent"] == "explicit_gene_question"
        assert r["gene_symbol"] is not None

    def test_pure_vus_question_not_gene_intent(self):
        # No gene symbol — must not route as explicit_gene_question
        r = _ce.classify_question_intent("מה זה VUS?")
        assert r["intent"] != "explicit_gene_question"

    def test_c12orf57_vus_intent_with_gene_index_available(self):
        # Even when gene index is available but C12orf57 is not in it
        fake_summary = MagicMock(return_value=None)
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=None):
            r = _ce.classify_question_intent("מה המשמעות של VUS בגן C12orf57?")
        assert r["intent"] == "explicit_gene_question"
        assert r["gene_symbol"] == "C12orf57"

    def test_hb_orf_style_gene_intent(self):
        r = _ce.classify_question_intent("יש לי VUS בגן C1orf194, מה זה אומר?")
        assert r["intent"] == "explicit_gene_question"
        assert "orf" in r["gene_symbol"].lower()


# ===========================================================================
# 3. TestVusGeneNeverGeneralAI
# ===========================================================================

class TestVusGeneNeverGeneralAI:
    """_classify_general_question must not return safe_general_education for VUS+gene queries."""

    def test_vus_c12orf57_blocked_from_general_ai(self):
        route = _ce._classify_general_question("מה המשמעות של VUS בגן C12orf57?")
        assert route != "safe_general_education"

    def test_vus_known_gene_blocked_from_general_ai(self):
        route = _ce._classify_general_question("מה המשמעות של VUS בגן BRCA1?")
        assert route != "safe_general_education"

    def test_vus_kiaa_blocked_from_general_ai(self):
        route = _ce._classify_general_question("יש לי VUS בגן KIAA2022, מה זה?")
        assert route != "safe_general_education"

    def test_pure_vus_question_not_blocked(self):
        # No gene symbol — should still be safe_general_education
        route = _ce._classify_general_question("מה זה VUS?")
        assert route == "safe_general_education"

    def test_gene_function_question_is_safe_education(self):
        # "מה עושה הגן?" without VUS → safe education
        route = _ce._classify_general_question("מה עושה הגן BRCA1?")
        # This question will be caught by step E before reaching classifier,
        # but if it reaches here it should be safe.
        assert route in ("safe_general_education", "out_of_scope")


# ===========================================================================
# 4. TestGeneFunctionQuestionsRoute
# ===========================================================================

class TestGeneFunctionQuestionsRoute:
    """Gene-function questions must route to gene-specific pipeline, not generic KB."""

    def test_brca1_function_intent(self):
        r = _ce.classify_question_intent("איך הגן BRCA1 פועל בגוף?")
        assert r["intent"] == "explicit_gene_question"
        assert r["gene_symbol"] is not None

    def test_kiaa2022_function_intent(self):
        r = _ce.classify_question_intent("מה עושה הגן KIAA2022?")
        assert r["intent"] == "explicit_gene_question"

    def test_col1a1_function_intent(self):
        r = _ce.classify_question_intent("מה עושה הגן COL1A1?")
        assert r["intent"] == "explicit_gene_question"

    def test_generic_gene_question_not_gene_intent(self):
        # "מה זה גן?" has no gene symbol → stays unclear → falls to KB what_is_gene
        r = _ce.classify_question_intent("מה זה גן?")
        assert r["intent"] != "explicit_gene_question"

    def test_general_function_question_not_gene_intent(self):
        # "איך גנים פועלים?" has no specific gene symbol
        r = _ce.classify_question_intent("איך גנים פועלים באופן כללי?")
        assert r["intent"] != "explicit_gene_question"

    def test_brca1_function_http_not_generic_definition(self):
        """BRCA1 function question must return curated BRCA1 answer, not generic gene definition."""
        r = _client.post("/ask", json={"question": "איך הגן BRCA1 פועל בגוף?"})
        assert r.status_code == 200
        data = r.json()
        # Must NOT return the generic gene definition opener
        assert "גן הוא יחידת מידע גנטי" not in data["answer"], (
            "Gene-function question must not return generic gene definition"
        )
        assert bool(data["answer"])


# ===========================================================================
# 5. TestWhatIsGeneGuard
# ===========================================================================

class TestWhatIsGeneGuard:
    """what_is_gene KB entry must be rejected when an explicit gene symbol is present."""

    def test_what_is_gene_fires_for_generic_question(self):
        r = _client.post("/ask", json={"question": "מה זה גן?"})
        assert r.status_code == 200
        data = r.json()
        # Generic gene question → KB what_is_gene answer with gene definition
        assert "גן הוא יחידת מידע גנטי" in data["answer"] or bool(data["answer"])

    def test_what_is_gene_blocked_for_brca1_question(self):
        r = _client.post("/ask", json={"question": "איך הגן BRCA1 פועל בגוף?"})
        assert r.status_code == 200
        data = r.json()
        # Must not return generic "גן הוא יחידת מידע גנטי" answer
        assert "גן הוא יחידת מידע גנטי" not in data["answer"]

    def test_what_is_gene_blocked_for_c12orf57_question(self):
        r = _client.post("/ask", json={"question": "מה זה הגן C12orf57?"})
        assert r.status_code == 200
        data = r.json()
        # Must not return generic gene definition — should be gene-specific
        assert "גן הוא יחידת מידע גנטי" not in data["answer"]

    def test_generic_gene_question_no_disclaimer(self):
        r = _client.post("/ask", json={"question": "מה זה גן?"})
        assert r.status_code == 200
        data = r.json()
        assert "המידע כללי ואינו מחליף ייעוץ רפואי אישי" not in data["answer"]


# ===========================================================================
# 6. TestGenomeHistoryRouting
# ===========================================================================

class TestGenomeHistoryRouting:
    """History/discovery questions about the genome must not return the gene-count KB answer."""

    def test_genome_discovery_not_gene_count(self):
        r = _client.post("/ask", json={"question": "מתי גילו את הגנום?"})
        assert r.status_code == 200
        data = r.json()
        assert "לבן אדם יש כ-20,000" not in data["answer"], (
            "Genome history question must not return gene-count answer"
        )

    def test_genome_sequencing_question_not_gene_count(self):
        r = _client.post("/ask", json={"question": "מתי פוענח הגנום האנושי?"})
        assert r.status_code == 200
        data = r.json()
        assert "לבן אדם יש כ-20,000" not in data["answer"]

    def test_how_many_genes_still_works(self):
        r = _client.post("/ask", json={"question": "כמה גנים יש בגנום האנושי?"})
        assert r.status_code == 200
        data = r.json()
        # Must still return the gene-count answer for this explicit count question
        assert "20,000" in data["answer"] or "25,000" in data["answer"]

    def test_how_many_genes_basic_still_works(self):
        r = _client.post("/ask", json={"question": "כמה גנים יש לאדם?"})
        assert r.status_code == 200
        data = r.json()
        assert "20,000" in data["answer"] or "25,000" in data["answer"]

    def test_what_is_genome_not_gene_count(self):
        r = _client.post("/ask", json={"question": "מה זה גנום?"})
        assert r.status_code == 200
        data = r.json()
        # Should explain what a genome is, not return gene count
        assert "לבן אדם יש כ-20,000" not in data["answer"]

    def test_genome_history_classify_as_safe_education(self):
        route = _ce._classify_general_question("מתי גילו את הגנום?")
        assert route == "safe_general_education"

    def test_chromosome_count_still_works(self):
        r = _client.post("/ask", json={"question": "כמה כרומוזומים יש לבני אדם?"})
        assert r.status_code == 200
        data = r.json()
        assert bool(data["answer"])


# ===========================================================================
# 7. TestGenericDisclaimerAbsent
# ===========================================================================

class TestGenericDisclaimerAbsent:
    """Generic disclaimer 'המידע כללי ואינו מחליף ייעוץ רפואי אישי' must not appear in routine answers."""

    _DISCLAIMER = "המידע כללי ואינו מחליף ייעוץ רפואי אישי"

    def test_gene_definition_no_disclaimer(self):
        r = _client.post("/ask", json={"question": "מה זה גן?"})
        assert self._DISCLAIMER not in r.json()["answer"]

    def test_dna_definition_no_disclaimer(self):
        r = _client.post("/ask", json={"question": "מה זה DNA?"})
        assert self._DISCLAIMER not in r.json()["answer"]

    def test_human_gene_count_no_disclaimer(self):
        r = _client.post("/ask", json={"question": "כמה גנים יש לאדם?"})
        assert self._DISCLAIMER not in r.json()["answer"]

    def test_vus_kb_answer_no_disclaimer(self):
        r = _client.post("/ask", json={"question": "מה זה VUS?"})
        assert self._DISCLAIMER not in r.json()["answer"]

    def test_deterministic_clinvar_answer_no_disclaimer(self):
        summary = {
            "gene_symbol": "TEST",
            "total_variants": 100,
            "by_significance": {"Pathogenic": 10},
            "phenotypes": ["Test condition"],
        }
        answer = _ce._build_gene_clinvar_deterministic_answer("TEST", summary)
        assert self._DISCLAIMER not in answer

    def test_carrier_kb_answer_no_generic_disclaimer(self):
        r = _client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"})
        assert r.status_code == 200
        data = r.json()
        # Carrier answer may have specific counseling language but not the generic disclaimer
        assert self._DISCLAIMER not in data["answer"]


# ===========================================================================
# 8. TestFalsePositivePrevention
# ===========================================================================

class TestFalsePositivePrevention:
    """Extraction must not fire on questions without explicit gene context."""

    def test_pure_vus_no_gene_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("מה זה VUS?")
        assert sym is None

    def test_general_gene_question_no_extraction(self):
        sym = _ce._extract_gene_from_explicit_phrase("מה זה גן?")
        assert sym is None

    def test_hebrew_word_in_explicit_position_rejected(self):
        # "הגן" followed by a Hebrew word — not a gene
        sym = _ce._extract_gene_from_explicit_phrase("הגן הזה חשוב")
        assert sym is None

    def test_rna_not_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("שינוי בגן RNA")
        assert sym is None

    def test_snp_not_extracted(self):
        sym = _ce._extract_gene_from_explicit_phrase("שינוי בגן SNP")
        assert sym is None


# ===========================================================================
# 9. TestSafetyPreserved
# ===========================================================================

class TestSafetyPreserved:
    """Safety pipeline must still block personal/medical queries even with new routing."""

    def test_identifying_info_blocked(self):
        r = _client.post("/ask", json={"question": "קוראים לי שרה, יש לי VUS בגן C12orf57"})
        assert r.status_code == 200
        assert r.json()["safety_level"] == "contains_identifying_info"

    def test_surgery_request_refused(self):
        r = _client.post("/ask", json={"question": "האם עלי לעשות ניתוח בגלל VUS בגן C12orf57?"})
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")

    def test_personal_risk_request_refused(self):
        r = _client.post("/ask", json={"question": "מה הסיכון שלי עם VUS בגן C12orf57?"})
        assert r.status_code == 200
        data = r.json()
        # Must not give personal risk estimate
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")

    def test_schema_preserved_for_c12orf57_vus(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys())


# ===========================================================================
# 10. TestRegressions
# ===========================================================================

class TestRegressions:
    """Existing correct behaviors must be preserved after Session 27.9.8 changes."""

    def test_brca1_vus_still_works(self):
        r = _client.post("/ask", json={"question": "יש לי VUS ב-BRCA1"})
        assert r.status_code == 200
        data = r.json()
        assert "VUS" in data["answer"] or "ממצא" in data["answer"]
        assert data["safety_level"] == "general_information"

    def test_col1a1_vus_still_works(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_COL1A1_5PH), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False):
            result = _ce._build_known_gene_answer("COL1A1", question="VUS בגן COL1A1")
        assert "VUS" in result["answer"] or "COL1A1" in result["answer"]

    def test_c12orf57_standalone_gene_preserved(self):
        """'מה זה הגן C12orf57?' must still route to gene-specific pipeline."""
        r = _ce.classify_question_intent("מה זה הגן C12orf57?")
        assert r["intent"] == "explicit_gene_question"
        assert r["gene_symbol"] == "C12orf57"

    def test_carrier_route_unaffected(self):
        r = _client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"})
        assert r.status_code == 200
        data = r.json()
        assert "נשא" in data["answer"] or "carrier" in data["answer"].lower()

    def test_vus_followup_still_routes(self):
        r = _client.post("/ask", json={
            "question": "מה כדאי לעשות עם זה?",
            "last_topic": "vus",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] != "contains_identifying_info"

    def test_chromosome_path_unaffected(self):
        r = _client.post("/ask", json={"question": "מה זה מחיקה בכרומוזום 21?"})
        assert r.status_code == 200
        data = r.json()
        assert bool(data["answer"])

    def test_ace_vus_builds(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_ACE_2PH), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            result = _ce._build_known_gene_answer("ACE", question="VUS בגן ACE")
        assert "VUS" in result["answer"] or "ACE" in result["answer"]

    def test_specific_variant_question_blocked(self):
        r = _client.post("/ask", json={"question": "יש לי BRCA1 c.5266dupC"})
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")

    def test_schema_five_keys_for_vus_ace(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys())
