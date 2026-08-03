# -*- coding: utf-8 -*-
"""
tests/test_session2793_persistence.py

Session 27.9.3 — Persist every patient-visible medical AI expansion.

Covers:
  Group 1: VUS + ACE — review record created, metadata in response
  Group 2: VUS + ACE repeated — deduplication (no duplicate pending rows)
  Group 3: Approval lifecycle — approve, edit-approve, approved reuse
  Group 4: Rejection / needs_revision — not reused as approved
  Group 5: Persistence failure — patient answer safe, no fake review_draft_id
  Group 6: Regressions — CCR5/chromosome persist, general/grounded do not

All tests use isolated SQLite via REVIEW_DB_SQLITE_PATH.
Gene index and LLM are mocked so tests run offline.
"""

import importlib
import os
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app.main import app

import app.counseling_engine as _ce

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Draft fixture helpers
# ---------------------------------------------------------------------------

_FAKE_DRAFT_ACE = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "ACE",
    "warning_he": "טיוטה שנוצרה על ידי AI ועדיין לא נבדקה על ידי גנטיקאי/ת.",
    "text_he": (
        "הגן ACE מקודד לאנזים הממיר אנגיוטנסין, המעורב בוויסות לחץ הדם ובמערכת "
        "הרנין-אנגיוטנסין. מוטציות בגן זה קשורות להיבטים שונים של מחלות לב וכלי דם."
    ),
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "based_on": "model_expansion_with_clinvar_gene_context",
    "source_grounded": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
}

_FAKE_DRAFT_CCR5 = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "CCR5",
    "warning_he": "טיוטה שנוצרה על ידי AI ועדיין לא נבדקה על ידי גנטיקאי/ת.",
    "text_he": (
        "הגן CCR5 מקודד לקולטן כמוקין המשמש כ-co-receptor לכניסת HIV לתאי T. "
        "וריאציה מוכרת Delta32 מספקת הגנה חלקית מפני HIV."
    ),
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "based_on": "model_expansion_with_clinvar_gene_context",
    "source_grounded": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
}

_FAKE_GENE_SUMMARY = {
    "gene_symbol": "ACE",
    "total_variants": 1234,
    "by_significance": {"Pathogenic": 5, "VUS": 100},
    "phenotypes": ["Hypertension", "Cardiovascular disease"],
}

_FAKE_GENE_SUMMARY_CCR5 = {
    "gene_symbol": "CCR5",
    "total_variants": 500,
    "by_significance": {"Pathogenic": 2, "VUS": 50},
    "phenotypes": ["HIV susceptibility"],
}

_FAKE_CHR_DRAFT = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "text_he": (
        "מחיקה כרומוזומית היא אובדן של חלק מהחומר הגנטי בכרומוזום. "
        "גודל המחיקה ומיקומה קובעים את ההשפעה הקלינית."
    ),
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
}


# ---------------------------------------------------------------------------
# Fixture: isolated review DB per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    """Each test gets its own SQLite DB; counseling_engine lazy-imports pick it up."""
    db_path = str(tmp_path / "test_review.db")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from app import review_db
    importlib.reload(review_db)
    review_db.init_db()
    yield review_db


def _get_reloaded_rdb():
    """Return the currently active review_db module (post-reload)."""
    from app import review_db as _rdb
    return _rdb


# ===========================================================================
# Group 1: VUS + ACE with ClinVar metadata — deterministic note, no AI draft
# (Session 27.9.4: prefer ClinVar grounded summaries over AI drafts)
# ===========================================================================

class TestVusAcePersistence:
    """VUS+gene with usable ClinVar metadata must use a deterministic note (no AI draft, no review record)."""

    def _call_known_gene(self):
        """Call _build_known_gene_answer with mocked gene_index; ACE has 1234 variants + phenotypes."""
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_FAKE_GENE_SUMMARY), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def test_no_review_draft_id_when_clinvar_note_used(self):
        """When ClinVar note is generated, review_draft_id must NOT appear in response."""
        result = self._call_known_gene()
        assert result.get("review_draft_id") is None, (
            "review_draft_id must not be set when ClinVar structured data is available"
        )

    def test_no_pending_record_in_db_when_clinvar_note_used(self):
        """No physician review record must be created when ClinVar data is used instead of AI draft."""
        self._call_known_gene()
        rdb = _get_reloaded_rdb()
        drafts = rdb.list_drafts(gene_symbol="ACE")
        assert len(drafts) == 0, (
            f"Expected 0 review records for ACE (ClinVar grounded), got {len(drafts)}"
        )

    def test_clinvar_note_appended_to_answer(self):
        """ClinVar note must be appended to the main answer text."""
        result = self._call_known_gene()
        assert "ClinVar" in result["answer"], (
            "ClinVar note must appear in the answer when structured metadata is available"
        )

    def test_clinvar_note_contains_disclaimer(self):
        """ClinVar note must contain the mandatory data-vs-personal interpretation disclaimer."""
        result = self._call_known_gene()
        assert "אינו מפרש את הממצא האישי שלך" in result["answer"], (
            "Disclaimer phrase must appear in the ClinVar note"
        )

    def test_no_review_record_in_list_drafts(self):
        """list_drafts must return empty list after a ClinVar-grounded VUS+ACE call."""
        self._call_known_gene()
        rdb = _get_reloaded_rdb()
        all_drafts = rdb.list_drafts()
        assert all_drafts == [], (
            f"No physician queue entries expected for ClinVar-grounded answer. Got: {all_drafts}"
        )

    def test_gene_knowledge_status_clinvar_summarized(self):
        """gene_knowledge_status in gene_metadata must be 'clinvar_summarized'."""
        result = self._call_known_gene()
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "clinvar_summarized", (
            f"Expected 'clinvar_summarized', got {gm.get('gene_knowledge_status')!r}"
        )

    def test_ai_draft_debug_shows_clinvar_path(self):
        """ai_draft_debug.reason must explain that ClinVar data was used instead of AI."""
        result = self._call_known_gene()
        debug = result.get("ai_draft_debug", {})
        assert debug.get("reason") == "clinvar_structured_data_used_instead", (
            f"Expected clinvar_structured_data_used_instead, got {debug.get('reason')!r}"
        )

    def test_no_unverified_gene_draft_when_clinvar_used(self):
        """unverified_gene_draft must NOT appear in the response when ClinVar note is shown."""
        result = self._call_known_gene()
        assert result.get("unverified_gene_draft") is None, (
            "unverified_gene_draft must not be present when ClinVar grounding is used"
        )

    def test_source_grounded_true_in_metadata(self):
        """gene_metadata.source_grounded must be True when ClinVar structured data is used."""
        result = self._call_known_gene()
        gm = result.get("gene_metadata", {})
        assert gm.get("source_grounded") is True, (
            f"source_grounded must be True for ClinVar-grounded response, got {gm.get('source_grounded')!r}"
        )

    def test_llm_used_false_when_clinvar_summarized(self):
        """llm_used must remain False when ClinVar structured data is used (no AI expansion)."""
        result = self._call_known_gene()
        assert result.get("llm_used") is False, (
            f"llm_used must be False for ClinVar-grounded response, got {result.get('llm_used')!r}"
        )

    def test_patient_answer_still_present(self):
        """The VUS explanation must still be in the main answer (ClinVar note is appended, not replacing)."""
        result = self._call_known_gene()
        assert "VUS" in result["answer"] or "ממצא" in result["answer"], (
            "VUS explanation must still appear in the main answer"
        )

    def test_no_ai_bridging_note_when_clinvar_summarized(self):
        """'מידע כללי נוסף' AI bridging note must NOT appear when ClinVar data is used."""
        result = self._call_known_gene()
        assert "מידע כללי נוסף על הגן" not in result["answer"], (
            "AI bridging note must not appear when ClinVar structured data is used"
        )

    def test_no_internal_clinvar_approval_sentence(self):
        """The internal 'no approved Hebrew summary' sentence must not appear."""
        result = self._call_known_gene()
        assert "אין עדיין סיכום ביולוגי מאושר בעברית" not in result["answer"], (
            "Internal review-architecture sentence must not appear in patient-facing answer"
        )

    def test_no_review_persistence_failed_when_clinvar_used(self):
        """review_persistence_failed must NOT be set when ClinVar data prevents AI draft."""
        result = self._call_known_gene()
        assert not result.get("review_persistence_failed"), (
            "review_persistence_failed must not be set when ClinVar path is taken (no draft to persist)"
        )


# ===========================================================================
# Group 2: VUS + ACE deduplication — same draft reuses existing record
# ===========================================================================

class TestVusAceDeduplication:
    """Second identical VUS+ACE call reuses existing pending record (ClinVar note bypassed to force AI draft path)."""

    def _call(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_FAKE_GENE_SUMMARY), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_build_vus_clinvar_gene_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def test_no_duplicate_pending_rows(self):
        """Two calls with identical draft text must produce exactly one DB record."""
        r1 = self._call()
        r2 = self._call()
        rdb = _get_reloaded_rdb()
        drafts = rdb.list_drafts(gene_symbol="ACE")
        assert len(drafts) == 1, f"Expected 1 draft, got {len(drafts)}"

    def test_same_review_draft_id_returned(self):
        """Both calls must return the same review_draft_id."""
        r1 = self._call()
        r2 = self._call()
        assert r1.get("review_draft_id") == r2.get("review_draft_id"), (
            "Deduplication must return the same review_draft_id for identical text"
        )

    def test_seen_count_incremented(self):
        """The existing record's seen_count must increment on the second call."""
        self._call()
        self._call()
        rdb = _get_reloaded_rdb()
        drafts = rdb.list_drafts(gene_symbol="ACE")
        assert drafts[0]["seen_count"] == 2, (
            f"Expected seen_count=2, got {drafts[0]['seen_count']}"
        )


# ===========================================================================
# Group 3: Approval lifecycle
# ===========================================================================

class TestVusAceApprovalLifecycle:
    """Approve → approved reuse; physician edit takes precedence. ClinVar note bypassed to force AI draft path."""

    def _create_draft_via_engine(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_FAKE_GENE_SUMMARY), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_build_vus_clinvar_gene_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def test_approve_transitions_status(self):
        """update_draft_status('approved') must change status to approved."""
        result = self._create_draft_via_engine()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        updated = rdb.update_draft_status(
            draft_id, new_status="approved", reviewer_identity="physician@test"
        )
        assert updated["review_status"] == "approved"
        assert updated["physician_approved"] is True
        assert updated["physician_reviewed"] is True

    def test_get_approved_draft_returns_approved(self):
        """After approval, get_approved_draft('ACE') must return the record."""
        result = self._create_draft_via_engine()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        rdb.update_draft_status(
            draft_id, new_status="approved", reviewer_identity="physician@test"
        )
        approved = rdb.get_approved_draft("ACE", draft_type="gene_summary")
        assert approved is not None
        assert approved["review_status"] == "approved"
        assert approved["gene_symbol"] == "ACE"

    def test_physician_edited_text_takes_precedence(self):
        """When physician edits text, effective_text must use the edited version."""
        result = self._create_draft_via_engine()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        edited_text = "הגן ACE מקודד לאנזים המעורב בוויסות לחץ הדם. (גרסה ערוכה)"
        rdb.update_draft_status(
            draft_id, new_status="approved", reviewer_identity="physician@test",
            physician_edited_text=edited_text,
        )
        approved = rdb.get_approved_draft("ACE", draft_type="gene_summary")
        assert approved["effective_text"] == edited_text

    def test_approved_text_not_labelled_unreviewed(self):
        """After approval, physician_reviewed must be True, physician_approved True."""
        result = self._create_draft_via_engine()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        rdb.update_draft_status(
            draft_id, new_status="approved", reviewer_identity="physician@test"
        )
        approved = rdb.get_approved_draft("ACE", draft_type="gene_summary")
        assert approved["physician_reviewed"] is True
        assert approved["physician_approved"] is True


# ===========================================================================
# Group 4: Rejection and needs_revision — not reused as approved
# ===========================================================================

class TestVusAceRejectionLifecycle:
    """Rejected / needs_revision drafts must not be reused as approved content. ClinVar note bypassed."""

    def _create_draft(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_FAKE_GENE_SUMMARY), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_build_vus_clinvar_gene_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def test_reject_not_reused_as_approved(self):
        """After rejection, get_approved_draft must return None."""
        result = self._create_draft()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        rdb.update_draft_status(
            draft_id, new_status="rejected", reviewer_identity="physician@test",
            review_comment="Content not accurate enough.",
        )
        approved = rdb.get_approved_draft("ACE", draft_type="gene_summary")
        assert approved is None, "Rejected draft must not appear as approved"

    def test_needs_revision_not_reused_as_approved(self):
        """After needs_revision, get_approved_draft must return None."""
        result = self._create_draft()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        rdb.update_draft_status(
            draft_id, new_status="needs_revision", reviewer_identity="physician@test",
            review_comment="Please clarify the enzyme role.",
        )
        approved = rdb.get_approved_draft("ACE", draft_type="gene_summary")
        assert approved is None, "needs_revision draft must not appear as approved"

    def test_reject_requires_comment(self):
        """Rejected status is still stored correctly (comment is stored)."""
        result = self._create_draft()
        draft_id = result["review_draft_id"]
        rdb = _get_reloaded_rdb()
        updated = rdb.update_draft_status(
            draft_id, new_status="rejected", reviewer_identity="physician@test",
            review_comment="Not accurate.",
        )
        assert updated["review_status"] == "rejected"
        assert updated["review_comment"] == "Not accurate."


# ===========================================================================
# Group 5: Persistence failure — patient answer safe, no fake review_draft_id
# ===========================================================================

class TestPersistenceFailureSafety:
    """DB errors must not reach the patient response. ClinVar note bypassed to force AI draft path."""

    def _call_with_db_failure(self):
        """Call _build_known_gene_answer with mocked gene_index, LLM, and DB failure."""
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_FAKE_GENE_SUMMARY), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_build_vus_clinvar_gene_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_ACE), \
             patch.object(_ce, "_persist_reviewable_ai_expansion", return_value=None):
            return _ce._build_known_gene_answer("ACE", question="מה המשמעות של VUS בגן ACE?")

    def test_patient_answer_still_returned(self):
        """Patient must still receive the VUS answer even when persistence fails."""
        result = self._call_with_db_failure()
        assert result.get("answer"), "Patient answer must not be empty on DB failure"
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_no_fake_review_draft_id(self):
        """review_draft_id must not be set when persistence fails."""
        result = self._call_with_db_failure()
        assert result.get("review_draft_id") is None, (
            f"review_draft_id must not be fabricated on failure: {result.get('review_draft_id')!r}"
        )

    def test_review_persistence_failed_marked(self):
        """review_persistence_failed must be True when persistence returns None."""
        result = self._call_with_db_failure()
        assert result.get("review_persistence_failed") is True, (
            "review_persistence_failed must be set to True when DB persistence fails"
        )

    def test_unverified_draft_still_shown(self):
        """The AI draft card must still appear even when DB persistence fails."""
        result = self._call_with_db_failure()
        assert result.get("unverified_gene_draft") is not None, (
            "Patient-visible draft must still be returned when persistence fails"
        )


# ===========================================================================
# Group 6: Regressions
# ===========================================================================

class TestPersistenceRegressions:
    """Existing paths must still work correctly; new paths must not break them."""

    def test_general_low_risk_ai_not_in_physician_queue(self):
        """GENERAL_LOW_RISK_AI answers (sea turtle) must NOT create physician queue entries."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "לצב ים יש כ-28 זוגות כרומוזומים. מספר הגנים המדויק אינו ידוע עדיין."
        )
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            _ce._build_general_education_answer("כמה גנים יש לצב ים?")
        rdb = _get_reloaded_rdb()
        all_drafts = rdb.list_drafts()
        assert len(all_drafts) == 0, (
            f"General low-risk AI must not create review records. Got: {all_drafts}"
        )

    def test_chromosome_expansion_persists(self):
        """Chromosome education expansion must still create a review record."""
        fake_chr_draft = dict(_FAKE_CHR_DRAFT)
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=fake_chr_draft):
            _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion_general"
            )
        rdb = _get_reloaded_rdb()
        drafts = rdb.list_drafts()
        assert len(drafts) == 1, (
            f"Chromosome expansion must still be persisted. Got {len(drafts)} records."
        )
        assert drafts[0]["draft_type"] == "chromosome_education"

    def test_chromosome_expansion_returns_review_metadata(self):
        """Chromosome answer must now include review_draft_id in the response."""
        fake_chr_draft = dict(_FAKE_CHR_DRAFT)
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=fake_chr_draft):
            result = _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion_general"
            )
        assert "review_draft_id" in result, (
            f"Chromosome expansion must return review_draft_id. Keys: {list(result.keys())}"
        )
        assert result["review_draft_id"] is not None

    def test_ccr5_gene_biology_persists(self):
        """Standalone gene biology expansion (CCR5) must still persist."""
        fake_ccr5_summary = dict(_FAKE_GENE_SUMMARY_CCR5)
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=fake_ccr5_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_CCR5):
            result = _ce._build_gene_clinvar_answer("מה התפקיד הביולוגי של הגן CCR5?", "CCR5")
        rdb = _get_reloaded_rdb()
        drafts = rdb.list_drafts(gene_symbol="CCR5")
        assert len(drafts) == 1, (
            f"Standalone CCR5 expansion must persist. Got {len(drafts)} records."
        )
        assert drafts[0]["review_status"] == "pending"

    def test_ccr5_gene_biology_returns_review_metadata(self):
        """Standalone CCR5 gene biology answer must return review_draft_id."""
        fake_ccr5_summary = dict(_FAKE_GENE_SUMMARY_CCR5)
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=fake_ccr5_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT_CCR5):
            result = _ce._build_gene_clinvar_answer("מה התפקיד הביולוגי של הגן CCR5?", "CCR5")
        assert "review_draft_id" in result, (
            f"CCR5 gene biology answer must return review_draft_id. Keys: {list(result.keys())}"
        )

    def test_source_grounded_mthfr_not_in_physician_queue(self):
        """SOURCE_GROUNDED answers (MTHFR) must NOT enter the physician queue."""
        fake_mthfr_summary = {
            "gene_symbol": "MTHFR",
            "total_variants": 200,
            "by_significance": {"Pathogenic": 10},
            "phenotypes": ["Homocystinuria", "Neural tube defects"],
        }
        grounded_text = "הגן MTHFR קשור לעיבוד חומצה פולית."
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=fake_mthfr_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=grounded_text), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            result = _ce._build_gene_clinvar_answer("MTHFR", "MTHFR")
        rdb = _get_reloaded_rdb()
        all_drafts = rdb.list_drafts()
        assert len(all_drafts) == 0, (
            f"Source-grounded answer must not create physician queue entries. Got: {all_drafts}"
        )

    def test_no_422_regression_on_ask_endpoint(self):
        """POST /ask must return 200 for the VUS+ACE question (no schema errors)."""
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        missing = required - data.keys()
        assert not missing, f"Missing required keys: {missing}"

    def test_vus_ace_answer_still_contains_vus_explanation(self):
        """The VUS+gene deterministic answer must still explain VUS (not replaced by AI)."""
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        assert r.status_code == 200
        data = r.json()
        assert "VUS" in data["answer"] or "ממצא" in data["answer"], (
            f"VUS explanation missing from answer: {data['answer'][:300]!r}"
        )

    def test_persist_helper_does_not_enqueue_short_text(self):
        """_persist_reviewable_ai_expansion must silently skip drafts with text < 10 chars."""
        short_draft = {"text_he": "מה"}
        record = _ce._persist_reviewable_ai_expansion(
            short_draft, "gene_summary", "vus_known_gene", gene_symbol="ACE"
        )
        assert record is None
        rdb = _get_reloaded_rdb()
        assert rdb.list_drafts() == []

    def test_persist_helper_returns_none_on_exception(self):
        """_persist_reviewable_ai_expansion must return None (not raise) on DB error."""
        draft = {"text_he": "הגן ACE מקודד לאנזים המעורב בוויסות לחץ הדם."}
        # Patch create_draft to raise to simulate DB failure
        with patch("app.review_db.create_draft", side_effect=RuntimeError("db_down")):
            record = _ce._persist_reviewable_ai_expansion(
                draft, "gene_summary", "vus_known_gene", gene_symbol="ACE"
            )
        assert record is None
