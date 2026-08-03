# -*- coding: utf-8 -*-
"""
tests/test_session2794_clinvar_grounding.py

Session 27.9.4 — Prefer ClinVar grounded summaries over AI drafts.

Decision policy for VUS+gene tier2 path:
  1. Curated gene card (tier1) → curated answer, no note appended, no AI draft
  2. Gene Knowledge Base (tier1b) → GK answer, no note appended, no AI draft
  3. ClinVar index with usable metadata (tier2a) → deterministic ClinVar note,
     NO AI draft, NO physician review record
  4. ClinVar index with no usable metadata (tier2b) → AI biology expansion,
     physician review record created
  5. Not in ClinVar index (tier3) → deterministic VUS text only, no draft

Standalone gene biology questions go through _build_gene_clinvar_answer(),
which has its own grounding/expansion logic and is NOT affected by this change.

All tests use isolated SQLite via REVIEW_DB_SQLITE_PATH.
Gene index and LLM are mocked so tests run offline.
"""

import importlib
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import app

import app.counseling_engine as _ce

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_GENE_SUMMARY_WITH_DATA = {
    "gene_symbol": "ACE",
    "total_variants": 1628,
    "by_significance": {
        "Uncertain significance": 950,
        "Likely benign": 450,
        "Pathogenic": 30,
        "Likely pathogenic": 15,
        "Benign": 183,
    },
    "phenotypes": ["Renal tubular dysgenesis", "Hypertension"],
}

_GENE_SUMMARY_NO_DATA = {
    "gene_symbol": "TESTGENE",
    "total_variants": 0,
    "by_significance": {},
    "phenotypes": [],
}

_GENE_SUMMARY_ONLY_PHENOTYPES = {
    "gene_symbol": "RAREDX",
    "total_variants": 3,
    "by_significance": {},
    "phenotypes": ["Rare inherited neuropathy"],
}

_FAKE_DRAFT_ACE = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "ACE",
    "text_he": (
        "הגן ACE מקודד לאנזים הממיר אנגיוטנסין, המעורב בוויסות לחץ הדם."
    ),
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "based_on": "model_expansion_with_clinvar_gene_context",
    "source_grounded": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
}

_CURATED_ACE_TEXT = "הגן ACE מקודד לאנזים ממיר אנגיוטנסין. (טקסט מאושר)"
_GK_ACE_TEXT = "מידע על הגן ACE ממאגר הידע הגנטי."


@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_review_2794.db")
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
# Group 1: _build_vus_clinvar_gene_note helper — unit tests
# ===========================================================================

class TestBuildVusClinvarNoteHelper:
    """Unit tests for the _build_vus_clinvar_gene_note deterministic helper."""

    def test_returns_none_when_g_summary_is_none(self):
        assert _ce._build_vus_clinvar_gene_note("ACE", None) is None

    def test_returns_none_when_no_variants_and_no_phenotypes(self):
        summary = {"total_variants": 0, "by_significance": {}, "phenotypes": []}
        assert _ce._build_vus_clinvar_gene_note("ACE", summary) is None

    def test_returns_none_when_fewer_than_10_variants_and_no_phenotypes(self):
        summary = {"total_variants": 5, "by_significance": {}, "phenotypes": []}
        assert _ce._build_vus_clinvar_gene_note("ACE", summary) is None

    def test_returns_note_when_total_variants_at_threshold(self):
        summary = {"total_variants": 10, "by_significance": {}, "phenotypes": []}
        note = _ce._build_vus_clinvar_gene_note("ACE", summary)
        assert note is not None

    def test_returns_note_when_phenotypes_present_even_with_low_count(self):
        summary = {"total_variants": 3, "by_significance": {}, "phenotypes": ["Rare neuropathy"]}
        note = _ce._build_vus_clinvar_gene_note("RAREDX", summary)
        assert note is not None

    def test_note_contains_gene_name(self):
        note = _ce._build_vus_clinvar_gene_note("ACE", _GENE_SUMMARY_WITH_DATA)
        assert "ACE" in note

    def test_note_contains_total_variants(self):
        note = _ce._build_vus_clinvar_gene_note("ACE", _GENE_SUMMARY_WITH_DATA)
        assert "1,628" in note or "1628" in note, f"Variant count missing from note: {note!r}"

    def test_note_contains_mandatory_disclaimer(self):
        note = _ce._build_vus_clinvar_gene_note("ACE", _GENE_SUMMARY_WITH_DATA)
        assert "אינו מפרש את הממצא האישי שלך" in note, (
            f"Mandatory disclaimer missing. Note: {note!r}"
        )

    def test_note_phenotypes_limited_to_two(self):
        summary = {
            "total_variants": 100,
            "by_significance": {},
            "phenotypes": ["Disease A", "Disease B", "Disease C", "Disease D"],
        }
        note = _ce._build_vus_clinvar_gene_note("GENE1", summary)
        assert "Disease C" not in note, "Third phenotype must not appear in the note"
        assert "Disease D" not in note

    def test_note_skips_trivial_phenotypes(self):
        summary = {
            "total_variants": 100,
            "by_significance": {},
            "phenotypes": ["not specified", "not provided", "Real disease"],
        }
        note = _ce._build_vus_clinvar_gene_note("GENE2", summary)
        assert "not specified" not in note
        assert "Real disease" in note

    def test_significance_note_includes_vus_and_pathogenic_and_benign(self):
        note = _ce._build_vus_clinvar_gene_note("ACE", _GENE_SUMMARY_WITH_DATA)
        assert "VUS" in note or "pathogenic" in note.lower() or "benign" in note.lower(), (
            f"Significance note missing. Note: {note!r}"
        )

    def test_significance_note_absent_when_no_by_sig(self):
        summary = {"total_variants": 50, "by_significance": {}, "phenotypes": []}
        note = _ce._build_vus_clinvar_gene_note("GENE3", summary)
        assert note is not None
        assert "VUS" not in note
        assert "pathogenic" not in note.lower()

    def test_returns_string_not_bytes(self):
        note = _ce._build_vus_clinvar_gene_note("ACE", _GENE_SUMMARY_WITH_DATA)
        assert isinstance(note, str)

    def test_note_starts_with_clinvar_mention(self):
        note = _ce._build_vus_clinvar_gene_note("ACE", _GENE_SUMMARY_WITH_DATA)
        assert "ClinVar" in note[:50], f"Note should mention ClinVar early: {note[:50]!r}"


# ===========================================================================
# Group 2: VUS + gene with ClinVar data — end-to-end via _build_known_gene_answer
# ===========================================================================

class TestVusAceClinvarGrounding:
    """VUS+gene: resolver generates AI gene explanation (supplemental card), main answer is VUS text."""

    def _call(self, gene_summary=None, question="מה המשמעות של VUS בגן ACE?"):
        if gene_summary is None:
            gene_summary = _GENE_SUMMARY_WITH_DATA
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=gene_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question=question)

    def test_unverified_gene_draft_in_supplemental_card(self):
        """27.9.6: resolver generates AI gene explanation → unverified_gene_draft set."""
        result = self._call()
        assert result.get("unverified_gene_draft") is not None, (
            "unverified_gene_draft must appear in supplemental card when AI gene explanation generated"
        )

    def test_no_clinvar_note_in_default_vus_answer(self):
        """Default VUS+gene main answer must NOT inject raw ClinVar statistics."""
        result = self._call()
        assert "במאגר ClinVar קיימות" not in result["answer"], (
            "ClinVar record count must not appear in default VUS+gene main answer"
        )

    def test_review_draft_id_in_response(self):
        """27.9.6: review_draft_id must be set for AI gene explanation."""
        result = self._call()
        assert result.get("review_draft_id") is not None

    def test_pending_record_in_db(self):
        """27.9.6: AI gene explanation must be persisted to the physician review queue."""
        self._call()
        rdb = _get_rdb()
        assert len(rdb.list_drafts()) == 1

    def test_gene_knowledge_status_ai_draft_pending(self):
        """gene_knowledge_status must be 'ai_draft_pending' when AI gene explanation generated."""
        result = self._call()
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "ai_draft_pending", (
            f"Expected 'ai_draft_pending', got {gm.get('gene_knowledge_status')!r}"
        )

    def test_ai_draft_debug_reason_gene_explanation_ai_generated(self):
        """ai_draft_debug.reason must reflect gene explanation generation."""
        result = self._call()
        debug = result.get("ai_draft_debug", {})
        assert debug.get("reason") == "gene_explanation_ai_generated", (
            f"Expected gene_explanation_ai_generated, got {debug.get('reason')!r}"
        )

    def test_vus_explanation_still_in_answer(self):
        result = self._call()
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_llm_used_false(self):
        """llm_used must remain False — the main answer is deterministic VUS text."""
        result = self._call()
        assert result.get("llm_used") is False

    def test_answer_tier_still_tier2(self):
        result = self._call()
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier2"

    def test_no_disclaimer_in_default_vus_answer(self):
        """The ClinVar DB disclaimer must NOT appear in the default VUS+gene main answer."""
        result = self._call()
        assert "אינו מפרש את הממצא האישי שלך" not in result["answer"]

    def test_no_clinvar_note_for_phenotype_only_summary(self):
        """Gene with few variants (phenotype only) must also not show ClinVar note in main answer."""
        result = self._call(gene_summary=_GENE_SUMMARY_ONLY_PHENOTYPES)
        assert "במאגר ClinVar קיימות" not in result["answer"]


# ===========================================================================
# Group 3: VUS + gene with NO ClinVar data — AI draft path preserved
# ===========================================================================

class TestVusTier3Fallback:
    """Tier2/3 VUS+gene default: no AI draft, no ClinVar note — only deterministic VUS text."""

    def _call_no_data(self):
        """Tier2 with empty ClinVar summary (gene in index but no useful metadata)."""
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_NO_DATA), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def _call_not_in_index(self):
        """Tier3: gene not in ClinVar index at all."""
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=None), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def test_ai_draft_generated_for_tier2_with_empty_data(self):
        """27.9.6: resolver generates AI gene explanation even when ClinVar has empty metadata."""
        result = self._call_no_data()
        assert result.get("unverified_gene_draft") is not None, (
            "AI gene explanation must be generated and set in supplemental card (27.9.6 policy)"
        )

    def test_review_record_created_for_tier2_with_empty_data(self):
        """27.9.6: AI gene explanation is persisted for physician review even with empty ClinVar data."""
        self._call_no_data()
        rdb = _get_rdb()
        assert len(rdb.list_drafts()) == 1, "AI gene explanation must be persisted to review queue"

    def test_no_clinvar_note_when_tier3(self):
        """Tier3 (not in index at all) must not have a ClinVar note in the answer."""
        result = self._call_not_in_index()
        assert result.get("gene_metadata", {}).get("answer_tier") == "tier3"
        assert "במאגר ClinVar קיימות" not in result["answer"], (
            "ClinVar count note must not appear for tier3 gene"
        )

    def test_tier3_vus_explanation_present(self):
        result = self._call_not_in_index()
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]


# ===========================================================================
# Group 4: Curated / GK genes must not have ClinVar note appended
# ===========================================================================

class TestClinvarNoteNotOverridesCurated:
    """Tier1/1b genes use curated answers — ClinVar note must not be injected."""

    def test_curated_gene_no_clinvar_note(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_WITH_DATA), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=_CURATED_ACE_TEXT), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            result = _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier1"
        assert "אינו מפרש את הממצא האישי שלך" not in result["answer"], (
            "ClinVar disclaimer must not appear in tier1 curated answer"
        )

    def test_gk_gene_no_clinvar_note(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_WITH_DATA), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=_GK_ACE_TEXT), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=True), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            result = _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier1b"
        assert "אינו מפרש את הממצא האישי שלך" not in result["answer"]

    def test_curated_gene_answer_contains_curated_text(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_WITH_DATA), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=_CURATED_ACE_TEXT), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            result = _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")
        assert _CURATED_ACE_TEXT in result["answer"]


# ===========================================================================
# Group 5: Standalone gene biology goes through _build_gene_clinvar_answer —
#          ClinVar note helper is NOT called there; behavior unchanged
# ===========================================================================

class TestClinvarNoteDoesNotAffectBiologyPath:
    """_build_gene_clinvar_answer must be unaffected by the new ClinVar note logic."""

    _FAKE_CCR5_SUMMARY = {
        "gene_symbol": "CCR5",
        "total_variants": 500,
        "by_significance": {"Pathogenic": 2, "VUS": 50},
        "phenotypes": ["HIV susceptibility"],
    }
    _FAKE_CCR5_DRAFT = {
        "visible": True,
        "text_he": "הגן CCR5 מקודד לקולטן כמוקין. (CCR5 test draft)",
        "gene_symbol": "CCR5",
        "generated_by_model": "gpt-test",
        "review_status": "unreviewed",
        "approved": False,
        "requires_physician_review": True,
        "ai_content_type": "medical_educational_ai_expansion",
        "based_on": "model_expansion_with_clinvar_gene_context",
        "source_grounded": False,
    }

    def test_biology_path_still_persists_when_no_grounded_context(self):
        """_build_gene_clinvar_answer AI expansion still persists for physician review."""
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=self._FAKE_CCR5_SUMMARY), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=self._FAKE_CCR5_DRAFT):
            _ce._build_gene_clinvar_answer("מה התפקיד הביולוגי של הגן CCR5?", "CCR5")
        rdb = _get_rdb()
        drafts = rdb.list_drafts(gene_symbol="CCR5")
        assert len(drafts) == 1, (
            f"Biology path AI expansion must still persist for physician review. Got {len(drafts)}"
        )

    def test_biology_path_source_grounded_does_not_persist(self):
        """Source-grounded gene biology answer must still NOT create physician review record."""
        grounded_text = "הגן MTHFR קשור לעיבוד חומצה פולית."
        fake_mthfr_summary = {
            "gene_symbol": "MTHFR",
            "total_variants": 200,
            "by_significance": {"Pathogenic": 10},
            "phenotypes": ["Homocystinuria"],
        }
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=fake_mthfr_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=grounded_text), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            _ce._build_gene_clinvar_answer("MTHFR", "MTHFR")
        rdb = _get_rdb()
        assert rdb.list_drafts() == []


# ===========================================================================
# Group 6: Regressions
# ===========================================================================

class TestClinvarGroundingRegressions:
    """Existing behavior must be preserved after the ClinVar grounding change."""

    def test_schema_valid_for_ace_vus_http(self):
        """POST /ask for VUS+ACE must return 200 and all 5 required fields."""
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"

    def test_vus_answer_contains_vus_text_http(self):
        """The VUS+gene answer must still explain VUS (deterministic text not removed)."""
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        data = r.json()
        assert "VUS" in data["answer"] or "ממצא" in data["answer"]

    def test_safety_level_general_information_for_vus(self):
        r = _client.post("/ask", json={"question": "מה זה VUS?"})
        data = r.json()
        assert data["safety_level"] == "general_information"

    def test_chromosome_path_unaffected(self):
        """Chromosome education path must still produce a valid answer."""
        r = _client.post("/ask", json={"question": "מה זה מחיקה בכרומוזום?"})
        assert r.status_code == 200
        data = r.json()
        assert bool(data["answer"])

    def test_general_low_risk_no_clinvar_note(self):
        """General non-genetic question must never contain the ClinVar note disclaimer."""
        r = _client.post("/ask", json={"question": "כמה גנים יש לצב ים?"})
        assert r.status_code == 200
        data = r.json()
        assert "אינו מפרש את הממצא האישי שלך" not in data["answer"]

    def test_brca1_curated_no_clinvar_note(self):
        """BRCA1 is a curated gene — ClinVar note must not be appended to its answer."""
        r = _client.post("/ask", json={"question": "יש לי VUS ב-BRCA1"})
        assert r.status_code == 200
        data = r.json()
        gene_meta = data.get("gene_metadata", {})
        assert gene_meta.get("answer_tier") in ("tier1", "tier1b", "tier2", "tier3", "unknown", None)
        if gene_meta.get("answer_tier") == "tier1":
            assert "אינו מפרש את הממצא האישי שלך" not in data["answer"], (
                "ClinVar note disclaimer must not appear in curated BRCA1 answer"
            )

    def test_format_vus_significance_note_with_all_categories(self):
        """_format_vus_significance_note must list all three categories present."""
        note = _ce._format_vus_significance_note({
            "Uncertain significance": 100,
            "Benign": 50,
            "Pathogenic": 10,
        })
        assert "VUS" in note
        assert "benign" in note.lower()
        assert "pathogenic" in note.lower()

    def test_format_vus_significance_note_empty_returns_empty(self):
        """_format_vus_significance_note with empty input must return empty string."""
        note = _ce._format_vus_significance_note({})
        assert note == ""

    def test_clinvar_note_counts_formatted_with_comma(self):
        """Variant count in ClinVar note must use comma formatting (1,628 not 1628)."""
        summary = {
            "total_variants": 1628,
            "by_significance": {},
            "phenotypes": [],
        }
        note = _ce._build_vus_clinvar_gene_note("ACE", summary)
        assert note is not None
        assert "1,628" in note, f"Expected comma-formatted count. Note: {note!r}"


# ===========================================================================
# Group 7: Explicit ClinVar database queries — statistical note allowed
# ===========================================================================

class TestVusExplicitClinvarQuery:
    """Explicit ClinVar database questions (e.g. 'כמה וריאנטים יש במאגר?') must
    return a cleaned statistical note; the default VUS question must NOT."""

    def _call(self, question):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_WITH_DATA), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            return _ce._build_known_gene_answer("ACE", question=question)

    def test_explicit_query_token_clinvar_detected(self):
        assert _ce._is_explicit_clinvar_db_query("כמה וריאנטים יש בגן ACE?") is True

    def test_explicit_query_token_records_detected(self):
        assert _ce._is_explicit_clinvar_db_query("כמה רשומות יש ב-ACE ב-ClinVar?") is True

    def test_explicit_query_token_classifications_detected(self):
        assert _ce._is_explicit_clinvar_db_query("אילו סיווגים יש לגן ACE?") is True

    def test_default_vus_question_not_detected(self):
        assert _ce._is_explicit_clinvar_db_query("מה המשמעות של VUS בגן ACE?") is False

    def test_followup_question_not_detected(self):
        assert _ce._is_explicit_clinvar_db_query("מה כדאי לעשות עם זה?") is False

    def test_explicit_query_produces_clinvar_note_in_answer(self):
        """An explicit ClinVar database question must include the note."""
        result = self._call("כמה וריאנטים יש בגן ACE במאגר?")
        assert "במאגר ClinVar קיימות" in result["answer"], (
            f"Explicit ClinVar query must show note. Answer: {result['answer'][:300]!r}"
        )

    def test_explicit_query_gene_knowledge_status_clinvar_summarized(self):
        result = self._call("כמה וריאנטים יש בגן ACE במאגר?")
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "clinvar_summarized", (
            f"Expected 'clinvar_summarized', got {gm.get('gene_knowledge_status')!r}"
        )

    def test_explicit_query_no_semicolons_in_answer(self):
        """Phenotype strings with semicolons must be split and cleaned."""
        summary = dict(_GENE_SUMMARY_WITH_DATA)
        summary["phenotypes"] = ["Disease A;Disease B;not specified"]
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            result = _ce._build_known_gene_answer("ACE", question="כמה וריאנטים יש בגן ACE?")
        assert ";" not in result["answer"], (
            f"Semicolons must not appear in the ClinVar note. Answer: {result['answer']!r}"
        )

    def test_default_vus_question_still_no_clinvar_note(self):
        """Default VUS+gene question must still produce no ClinVar note."""
        result = self._call("מה המשמעות של VUS בגן ACE?")
        assert "במאגר ClinVar קיימות" not in result["answer"]

    def test_explicit_query_no_unverified_draft(self):
        """Explicit ClinVar query must not generate an AI draft either."""
        result = self._call("כמה וריאנטים יש בגן ACE במאגר?")
        assert result.get("unverified_gene_draft") is None


# ===========================================================================
# Group 8: _clean_clinvar_phenotypes helper — unit tests
# ===========================================================================

class TestCleanClinvarPhenotypes:
    """Unit tests for the _clean_clinvar_phenotypes deduplication helper."""

    def test_splits_semicolons(self):
        cleaned = _ce._clean_clinvar_phenotypes(["Alpha disease;Beta syndrome;Gamma disorder"])
        assert "Alpha disease" in cleaned
        assert "Beta syndrome" in cleaned
        assert "Gamma disorder" in cleaned

    def test_deduplicates(self):
        cleaned = _ce._clean_clinvar_phenotypes(["Disease X", "Disease X"])
        assert cleaned.count("Disease X") == 1

    def test_filters_trivial_entries(self):
        cleaned = _ce._clean_clinvar_phenotypes(["not specified", "not provided", "Real disease"])
        assert "not specified" not in cleaned
        assert "not provided" not in cleaned
        assert "Real disease" in cleaned

    def test_strips_whitespace(self):
        cleaned = _ce._clean_clinvar_phenotypes(["  Hypertension  "])
        assert "Hypertension" in cleaned

    def test_filters_very_short_entries(self):
        cleaned = _ce._clean_clinvar_phenotypes(["AB", "Real disease"])
        assert "AB" not in cleaned

    def test_empty_input(self):
        assert _ce._clean_clinvar_phenotypes([]) == []

    def test_none_input(self):
        assert _ce._clean_clinvar_phenotypes(None) == []

    def test_deduplicates_case_insensitively(self):
        cleaned = _ce._clean_clinvar_phenotypes(["Hypertension", "hypertension"])
        assert len([x for x in cleaned if x.lower() == "hypertension"]) == 1
