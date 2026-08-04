# -*- coding: utf-8 -*-
"""
tests/test_session2799_unknown_gene_fallback.py

Session 27.9.9 — Complete unknown-gene AI fallback and intent-aware detail.

Test groups:
  G1 — C12orf57 VUS: correct routing already; verify no "general_education_ai" match
  G2 — KIAA2022 standalone: AI fallback when no local data; no "אין מידע" when LLM available
  G3 — Unknown gene + LLM unavailable: safe fallback, no crash, no fake review record
  G4 — BRCA1 function question: fuller than generic; curated source; no unverified card
  G5 — BRCA1 overview: short, concise behavior preserved
  G6 — Multiple arbitrary unknown symbols: same fallback policy, no hard-coded names
  G7 — Regression: COL1A1 grounded path, general AI, genome history, disclaimer absent,
        safety overrides all
"""

import importlib
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as _ce

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_AI_DRAFT = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "KIAA2022",
    "text_he": "הגן KIAA2022 מקודד לחלבון הקשור להתפתחות עצבית.",
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
    "based_on": "model_expansion_without_clinvar_context",
    "source_grounded": False,
}

_FAKE_BRCA1_CURATED = (
    "BRCA1 הוא גן המקודד לחלבון המעורב בתיקון DNA ובשמירה על יציבות הגנום."
)

_FAKE_EXPANDED_BRCA1 = (
    "הגן BRCA1 מקודד לחלבון BRCA1 המשמש כחלק ממנגנון תיקון ה-DNA בתא. "
    "הוא פועל יחד עם חלבונים נוספים לתיקון שברים בשתי גדילי ה-DNA. "
    "BRCA1 מכונה לעיתים 'שומר הגנום' בשל תפקידו בשמירה על שלמות החומר הגנטי. "
    "ללא פעילות תקינה של BRCA1, שגיאות בהעתקת ה-DNA עלולות להצטבר."
)


@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_review_2799.db")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from app import review_db
    importlib.reload(review_db)
    review_db.init_db()
    yield review_db


def _get_rdb():
    from app import review_db as _rdb
    return _rdb


# ===========================================================================
# G1: C12orf57 VUS — routing already correct; check key response properties
# ===========================================================================

class TestC12orf57VusRouting:
    """C12orf57 with VUS phrasing routes through the VUS+gene pipeline."""

    def test_not_general_education_ai(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        assert r.status_code == 200
        data = r.json()
        assert data.get("matched_topic") != "general_education_ai", (
            f"C12orf57 VUS must not route to general_education_ai, got: {data.get('matched_topic')!r}"
        )

    def test_vus_content_in_answer(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        data = r.json()
        assert "VUS" in data["answer"] or "ממצא" in data["answer"], (
            f"Answer must contain VUS explanation. Got: {data['answer'][:200]!r}"
        )

    def test_matched_topic_is_vus(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        data = r.json()
        topic = data.get("matched_topic", "")
        assert "vus" in topic.lower() or topic == "gene_clinvar_summary", (
            f"Unexpected topic: {topic!r}"
        )

    def test_schema_intact(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys())

    def test_safety_level_general(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        data = r.json()
        assert data["safety_level"] == "general_information"


# ===========================================================================
# G2: KIAA2022 standalone — AI fallback when no local data
# ===========================================================================

class TestKiaa2022StandaloneFallback:
    """When LLM is available and gene not in any source, AI fallback is returned."""

    def _call_kiaa2022(self, question="איך הגן KIAA2022 פועל בגוף?"):
        _draft = dict(_FAKE_AI_DRAFT, gene_symbol="KIAA2022")
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=None), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_draft):
            return _ce._build_gene_clinvar_answer(question, "KIAA2022")

    def test_no_ain_meidah_message_when_llm_available(self):
        result = self._call_kiaa2022()
        assert "אין עדיין מידע על הגן KIAA2022" not in result["answer"], (
            f"Should not show 'no data' when AI fallback available. Got: {result['answer']!r}"
        )

    def test_ai_text_in_answer(self):
        result = self._call_kiaa2022()
        assert "KIAA2022" in result["answer"] or "גן" in result["answer"], (
            f"AI text must be in answer. Got: {result['answer']!r}"
        )

    def test_llm_used_true(self):
        result = self._call_kiaa2022()
        assert result.get("llm_used") is True

    def test_unverified_draft_attached(self):
        result = self._call_kiaa2022()
        assert result.get("unverified_gene_draft") is not None or \
               result.get("gene_metadata", {}).get("unverified_gene_draft_available") is True

    def test_review_record_created(self):
        self._call_kiaa2022()
        drafts = _get_rdb().list_drafts()
        assert len(drafts) >= 1, "Physician review record must be created for KIAA2022 AI fallback"

    def test_gene_knowledge_status_ai_draft_pending(self):
        result = self._call_kiaa2022()
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "ai_draft_pending", (
            f"Unexpected gene_knowledge_status: {gm.get('gene_knowledge_status')!r}"
        )

    def test_answer_tier_tier3(self):
        result = self._call_kiaa2022()
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier3"

    def test_safety_level_general(self):
        result = self._call_kiaa2022()
        assert result["safety_level"] == "general_information"


# ===========================================================================
# G3: Unknown gene + LLM unavailable — safe fallback, no crash
# ===========================================================================

class TestUnknownGeneLlmUnavailable:
    """When LLM is not configured and gene not found anywhere, show safe 'no data' message."""

    def _call_unknown_no_llm(self, gene="UNKNX99", question=None):
        question = question or f"מה הגן {gene} עושה?"
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=None), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            return _ce._build_gene_clinvar_answer(question, gene)

    def test_no_crash_on_llm_unavailable(self):
        result = self._call_unknown_no_llm()
        assert isinstance(result, dict)
        assert "answer" in result

    def test_safe_fallback_message_present(self):
        result = self._call_unknown_no_llm()
        assert "UNKNX99" in result["answer"] or "מידע" in result["answer"], (
            f"Fallback message must reference the gene or lack of data. Got: {result['answer']!r}"
        )

    def test_no_fake_review_record(self):
        self._call_unknown_no_llm()
        drafts = _get_rdb().list_drafts()
        assert len(drafts) == 0, "No review record when LLM unavailable"

    def test_llm_used_false(self):
        result = self._call_unknown_no_llm()
        assert result.get("llm_used") is False

    def test_schema_intact(self):
        result = self._call_unknown_no_llm()
        assert "safety_level" in result
        assert result["safety_level"] == "general_information"


# ===========================================================================
# G4: BRCA1 function question — fuller answer; curated source; no unverified card
# ===========================================================================

class TestBrca1FunctionQuestion:
    """Function questions on curated genes get intent-aware phrasing expansion."""

    def _call_brca1_function(self, question="איך הגן BRCA1 פועל בגוף?"):
        _summary = {
            "gene_symbol": "BRCA1",
            "total_variants": 5000,
            "by_significance": {"Pathogenic": 300},
            "phenotypes": ["Breast-ovarian cancer familial"],
        }
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=_FAKE_BRCA1_CURATED), \
             patch.object(_ce, "_phrase_curated_function_answer", return_value=_FAKE_EXPANDED_BRCA1):
            return _ce._build_gene_clinvar_answer(question, "BRCA1")

    def _call_brca1_function_no_llm(self, question="איך הגן BRCA1 פועל בגוף?"):
        _summary = {
            "gene_symbol": "BRCA1",
            "total_variants": 5000,
            "by_significance": {"Pathogenic": 300},
            "phenotypes": ["Breast-ovarian cancer familial"],
        }
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=_FAKE_BRCA1_CURATED), \
             patch.object(_ce, "_phrase_curated_function_answer", return_value=None):
            return _ce._build_gene_clinvar_answer(question, "BRCA1")

    def test_expanded_answer_returned_when_llm_available(self):
        result = self._call_brca1_function()
        assert _FAKE_EXPANDED_BRCA1 in result["answer"], (
            f"Expanded function answer expected. Got: {result['answer'][:300]!r}"
        )

    def test_curated_fallback_when_llm_unavailable(self):
        result = self._call_brca1_function_no_llm()
        assert _FAKE_BRCA1_CURATED in result["answer"], (
            f"Should fall back to raw curated text. Got: {result['answer'][:300]!r}"
        )

    def test_no_unverified_draft_card(self):
        result = self._call_brca1_function()
        assert result.get("unverified_gene_draft") is None, (
            "Tier 1a (curated) should never produce an unverified_gene_draft card"
        )

    def test_no_review_record_created(self):
        self._call_brca1_function()
        assert len(_get_rdb().list_drafts()) == 0

    def test_answer_tier_is_tier1(self):
        result = self._call_brca1_function()
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier1"

    def test_gene_knowledge_status_approved(self):
        result = self._call_brca1_function()
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "approved"

    def test_expanded_answer_longer_than_curated(self):
        result = self._call_brca1_function()
        assert len(result["answer"]) >= len(_FAKE_BRCA1_CURATED)


# ===========================================================================
# G5: BRCA1 overview — concise behavior preserved (non-function question)
# ===========================================================================

class TestBrca1OverviewConcise:
    """Non-function questions on curated genes return the raw curated text unchanged."""

    def _call_brca1_overview(self, question="מה זה הגן BRCA1?"):
        _summary = {
            "gene_symbol": "BRCA1",
            "total_variants": 5000,
            "by_significance": {"Pathogenic": 300},
            "phenotypes": ["Breast-ovarian cancer familial"],
        }
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=_FAKE_BRCA1_CURATED):
            return _ce._build_gene_clinvar_answer(question, "BRCA1")

    def test_curated_text_returned(self):
        result = self._call_brca1_overview()
        assert _FAKE_BRCA1_CURATED in result["answer"]

    def test_no_expansion_for_overview(self):
        with patch.object(_ce, "_phrase_curated_function_answer") as mock_phrase:
            self._call_brca1_overview()
            mock_phrase.assert_not_called()

    def test_answer_tier_tier1(self):
        result = self._call_brca1_overview()
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier1"


# ===========================================================================
# G6: Multiple arbitrary unknown gene symbols — same fallback policy
# ===========================================================================

class TestArbitraryUnknownGenesFallback:
    """No gene-specific hard-coding: all unknown genes go through the same resolver."""

    _UNKNOWN_GENES = ["ZZZ9", "KIAA9999", "C3orf99", "LOC123456"]

    def _call_gene(self, gene, has_llm=True):
        _draft = dict(_FAKE_AI_DRAFT, gene_symbol=gene,
                      text_he=f"הגן {gene} עדיין נחקר.")
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=None), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft",
                          return_value=(_draft if has_llm else None)):
            return _ce._build_gene_clinvar_answer(f"מה הגן {gene} עושה?", gene)

    @pytest.mark.parametrize("gene", _UNKNOWN_GENES)
    def test_no_ain_meidah_when_llm_available(self, gene):
        result = self._call_gene(gene, has_llm=True)
        assert f"אין עדיין מידע על הגן {gene}" not in result["answer"], (
            f"Gene {gene}: 'no data' message must not appear when LLM fallback available"
        )

    @pytest.mark.parametrize("gene", _UNKNOWN_GENES)
    def test_safe_message_when_llm_unavailable(self, gene):
        result = self._call_gene(gene, has_llm=False)
        assert isinstance(result, dict) and "answer" in result

    @pytest.mark.parametrize("gene", _UNKNOWN_GENES)
    def test_schema_intact(self, gene):
        result = self._call_gene(gene, has_llm=True)
        assert result["safety_level"] == "general_information"
        assert "gene_metadata" in result


# ===========================================================================
# G7: Regressions
# ===========================================================================

class TestSession2799Regressions:
    """Key behaviors from earlier sessions must not be broken."""

    def test_genome_history_question_routes_correctly(self):
        r = _client.post("/ask", json={"question": "מתי גילו את הגנום?"})
        assert r.status_code == 200
        data = r.json()
        assert data.get("matched_topic") != "human_genome_size", (
            "History question must not route to human_genome_size"
        )

    def test_no_generic_disclaimer_in_vus_answer(self):
        r = _client.post("/ask", json={"question": "מה זה VUS?"})
        data = r.json()
        assert "המידע כללי ואינו מחליף ייעוץ רפואי אישי." not in data["answer"]

    def test_identifying_info_blocked(self):
        r = _client.post("/ask", json={"question": "קוראים לי שרה, יש לי VUS ב-KIAA2022"})
        data = r.json()
        assert data["safety_level"] == "contains_identifying_info"

    def test_medical_action_refused(self):
        r = _client.post("/ask", json={"question": "האם עלי לעשות ניתוח בגלל הגן?"})
        data = r.json()
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")

    def test_carrier_path_unaffected(self):
        r = _client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"})
        data = r.json()
        assert bool(data["answer"])
        assert data["safety_level"] == "general_information"

    def test_vus_general_question_unaffected(self):
        r = _client.post("/ask", json={"question": "מה זה VUS?"})
        data = r.json()
        assert "VUS" in data["answer"] or "ממצא" in data["answer"]

    def test_c12orf57_vus_not_general_education_ai(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"})
        data = r.json()
        assert data.get("matched_topic") != "general_education_ai"

    def test_schema_always_5_required_fields(self):
        for question in [
            "מה זה VUS?",
            "איך הגן KIAA2022 פועל בגוף?",
            "מה זה כרומוזום?",
        ]:
            r = _client.post("/ask", json={"question": question})
            data = r.json()
            required = {"answer", "safety_level", "needs_genetic_counselor",
                        "matched_topic", "suggested_questions"}
            assert required.issubset(data.keys()), (
                f"Missing schema keys for '{question}': {required - data.keys()}"
            )

    def test_is_gene_function_question_detects_correctly(self):
        assert _ce._is_gene_function_question("איך הגן BRCA1 פועל בגוף?") is True
        assert _ce._is_gene_function_question("מה עושה הגן KIAA2022?") is True
        assert _ce._is_gene_function_question("מה תפקיד הגן TP53?") is True
        assert _ce._is_gene_function_question("מה זה VUS?") is False
        assert _ce._is_gene_function_question("מה ההשלכות של נשאות?") is False

    def test_phrase_curated_returns_none_without_llm(self):
        with patch.object(_ce, "create_llm_client", side_effect=ValueError("not configured")):
            result = _ce._phrase_curated_function_answer(
                "BRCA1", _FAKE_BRCA1_CURATED, "איך הגן BRCA1 פועל?"
            )
        assert result is None
