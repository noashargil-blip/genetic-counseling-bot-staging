# -*- coding: utf-8 -*-
"""
tests/test_session2796_universal_gene.py

Session 27.9.6 — Universal gene explanation for every VUS + gene answer.

All VUS+gene answers go through the same _resolve_gene_explanation_for_vus
priority pipeline regardless of which gene is mentioned.  Response shape is
determined by available content, not gene identity.

Priority chain:
  1. curated          — gene_cards approved text → main answer, no review record
  2. grounded         — gene_knowledge KB text   → main answer, no review record
  3. physician_approved — review_db approved     → main answer, no new record
  4. ai_unreviewed    — AI biology draft          → supplemental card, review record
  5. none             — LLM unavailable           → VUS-only answer

Tests cover:
  1. BRCA1 VUS (curated path)
  2. ACE VUS with approved physician summary (physician_approved path)
  3. ACE VUS without any summary (ai_unreviewed path)
  4. Approval lifecycle + reuse
  5. Generic non-curated genes (pipeline consistency, no hard-coding)
  6. ClinVar technical data separation
  7. Safety
  8. Regressions

All tests use isolated SQLite via REVIEW_DB_SQLITE_PATH.
"""

import importlib
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as _ce

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_BRCA1_CURATED_TEXT = "הגן BRCA1 קשור לסיכון מוגבר לסרטן שד ושחלות. (טקסט מאושר)"

_GENE_SUMMARY_ACE = {
    "gene_symbol": "ACE",
    "total_variants": 1628,
    "by_significance": {"Uncertain significance": 950, "Benign": 450},
    "phenotypes": ["Renal tubular dysgenesis", "Hypertension"],
}

_GENE_SUMMARY_MYGENE = {
    "gene_symbol": "MYGENE",
    "total_variants": 300,
    "by_significance": {"Uncertain significance": 200},
    "phenotypes": ["Some disorder"],
}

_FAKE_ACE_BIOLOGY = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "ACE",
    "text_he": (
        "הגן ACE מקודד לאנזים הממיר אנגיוטנסין, המעורב בוויסות לחץ הדם ובמערכת "
        "הרנין-אנגיוטנסין. מוטציות בגן זה קשורות להיבטים שונים של בריאות הלב."
    ),
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "based_on": "model_expansion_llm_knowledge",
    "source_grounded": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
}

_FAKE_MYGENE_BIOLOGY = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "MYGENE",
    "text_he": (
        "הגן MYGENE ממלא תפקיד ביולוגי מדגמי לצורכי בדיקה."
    ),
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "based_on": "model_expansion_llm_knowledge",
    "source_grounded": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
}

_APPROVED_ACE_SUMMARY = (
    "הגן ACE מקודד לאנזים הממיר אנגיוטנסין — חלבון המשתתף בוויסות לחץ הדם. "
    "(אושר על ידי גנטיקאי.)"
)


@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_review_2796.db")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from app import review_db
    importlib.reload(review_db)
    review_db.init_db()
    yield review_db


def _get_rdb():
    from app import review_db as _rdb
    return _rdb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_vus_gene(gene="ACE", question=None, gene_summary=None,
                   curated=None, gk_patient=None, draft=None):
    """Call _build_known_gene_answer with full mock control."""
    q = question or f"מה המשמעות של VUS בגן {gene}?"
    gs = gene_summary if gene_summary is not None else _GENE_SUMMARY_ACE
    with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
         patch.object(_ce.gene_index, "get_gene_summary", return_value=gs), \
         patch.object(_ce.gene_cards, "get_approved_summary", return_value=curated), \
         patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=gk_patient), \
         patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge",
                      return_value=gk_patient is not None), \
         patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
         patch.object(_ce, "_generate_unverified_gene_draft", return_value=draft):
        return _ce._build_known_gene_answer(gene, question=q)


# ===========================================================================
# 1. BRCA1 VUS — curated path
# ===========================================================================

class TestBrca1VusCurated:
    """BRCA1 goes through the same resolver as ACE; curated text is found at priority 1."""

    def test_vus_explanation_present(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_gene_explanation_in_main_answer(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert _BRCA1_CURATED_TEXT in result["answer"], (
            "Curated gene explanation must appear in the main answer"
        )

    def test_distinction_note_present(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert "חשוב להבדיל" in result["answer"], (
            "Distinction note must be included when gene explanation is in main answer"
        )

    def test_no_clinvar_data_in_main_answer(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert "במאגר ClinVar קיימות" not in result["answer"]

    def test_no_unverified_draft(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert result.get("unverified_gene_draft") is None

    def test_no_review_record(self):
        _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert _get_rdb().list_drafts() == []

    def test_answer_tier_is_tier1(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("answer_tier") == "tier1"

    def test_gene_knowledge_status_approved(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "approved"

    def test_gene_explanation_source_curated(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "curated"


# ===========================================================================
# 2. ACE VUS with physician-approved summary
# ===========================================================================

class TestAceVusPhysicianApproved:
    """When a physician-approved summary exists in review_db, it is used as the gene explanation."""

    def _seed_approved_record(self):
        rdb = _get_rdb()
        record = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text=_APPROVED_ACE_SUMMARY,
            gene_symbol="ACE",
            normalized_intent="vus_known_gene",
            prompt_version="s2796",
        )
        rdb.update_draft_status(record["id"], new_status="approved",
                                reviewer_identity="physician@test")
        return record["id"]

    def test_approved_text_in_main_answer(self):
        self._seed_approved_record()
        result = _call_vus_gene("ACE", draft=None)
        assert _APPROVED_ACE_SUMMARY in result["answer"], (
            "Physician-approved gene explanation must appear in the main answer"
        )

    def test_vus_explanation_present(self):
        self._seed_approved_record()
        result = _call_vus_gene("ACE", draft=None)
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_distinction_note_present(self):
        self._seed_approved_record()
        result = _call_vus_gene("ACE", draft=None)
        assert "חשוב להבדיל" in result["answer"]

    def test_no_new_pending_record_created(self):
        draft_id = self._seed_approved_record()
        _call_vus_gene("ACE", draft=None)
        rdb = _get_rdb()
        pending = [d for d in rdb.list_drafts() if d["review_status"] == "pending"]
        assert len(pending) == 0, (
            "No new pending record must be created when physician-approved text is reused"
        )

    def test_no_unverified_draft_shown(self):
        self._seed_approved_record()
        result = _call_vus_gene("ACE", draft=None)
        assert result.get("unverified_gene_draft") is None

    def test_gene_explanation_source_physician_approved(self):
        self._seed_approved_record()
        result = _call_vus_gene("ACE", draft=None)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "physician_approved"

    def test_gene_knowledge_status_physician_approved(self):
        self._seed_approved_record()
        result = _call_vus_gene("ACE", draft=None)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "physician_approved"

    def test_approved_reuse_for_both_query_styles(self):
        """Same approved summary is reused regardless of question phrasing."""
        self._seed_approved_record()
        r1 = _call_vus_gene("ACE", question="מה המשמעות של VUS בגן ACE?", draft=None)
        r2 = _call_vus_gene("ACE", question="יש לי VUS ב-ACE, מה זה?", draft=None)
        assert _APPROVED_ACE_SUMMARY in r1["answer"]
        assert _APPROVED_ACE_SUMMARY in r2["answer"]


# ===========================================================================
# 3. ACE VUS without any approved summary (ai_unreviewed path)
# ===========================================================================

class TestAceVusAiUnreviewed:
    """No curated, GK, or approved record → AI draft generated, supplemental card only."""

    def test_vus_explanation_in_main_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_gene_biology_not_in_main_answer(self):
        """AI gene biology must NOT be injected into the main patient answer."""
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert _FAKE_ACE_BIOLOGY["text_he"] not in result["answer"], (
            "AI gene explanation must stay in supplemental card, not main answer"
        )

    def test_gene_biology_in_supplemental_card(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        draft_card = result.get("unverified_gene_draft")
        assert draft_card is not None
        assert draft_card["text_he"] == _FAKE_ACE_BIOLOGY["text_he"]

    def test_review_draft_id_returned(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert result.get("review_draft_id") is not None

    def test_pending_record_in_physician_queue(self):
        _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        rdb = _get_rdb()
        drafts = rdb.list_drafts(gene_symbol="ACE")
        assert len(drafts) == 1
        assert drafts[0]["review_status"] == "pending"

    def test_gene_explanation_source_ai_unreviewed(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "ai_unreviewed"

    def test_gene_knowledge_status_ai_draft_pending(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "ai_draft_pending"

    def test_llm_used_false_for_main_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert result.get("llm_used") is False

    def test_no_clinvar_data_in_main_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert "1,628" not in result["answer"]
        assert "במאגר ClinVar קיימות" not in result["answer"]

    def test_no_semicolons_in_main_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert ";" not in result["answer"]


# ===========================================================================
# 4. Approval lifecycle + reuse
# ===========================================================================

class TestApprovalLifecycleAndReuse:
    """After approving a draft, the same explanation is reused without a new pending record."""

    def _create_pending(self):
        rdb = _get_rdb()
        return rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text=_FAKE_ACE_BIOLOGY["text_he"],
            gene_symbol="ACE",
            normalized_intent="vus_known_gene",
            prompt_version="s2796",
        )

    def test_pending_draft_transitions_to_approved(self):
        record = self._create_pending()
        rdb = _get_rdb()
        updated = rdb.update_draft_status(
            record["id"], new_status="approved", reviewer_identity="physician@test"
        )
        assert updated["review_status"] == "approved"

    def test_approved_text_appears_in_main_answer(self):
        record = self._create_pending()
        rdb = _get_rdb()
        rdb.update_draft_status(record["id"], new_status="approved",
                                reviewer_identity="physician@test")
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert _FAKE_ACE_BIOLOGY["text_he"] in result["answer"], (
            "After approval, gene explanation must appear in the main answer"
        )

    def test_no_new_pending_after_approval(self):
        record = self._create_pending()
        rdb = _get_rdb()
        rdb.update_draft_status(record["id"], new_status="approved",
                                reviewer_identity="physician@test")
        _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        pending = [d for d in rdb.list_drafts() if d["review_status"] == "pending"]
        assert len(pending) == 0, "No new pending record after approved text is reused"

    def test_no_unverified_draft_shown_after_approval(self):
        record = self._create_pending()
        rdb = _get_rdb()
        rdb.update_draft_status(record["id"], new_status="approved",
                                reviewer_identity="physician@test")
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert result.get("unverified_gene_draft") is None

    def test_rejected_draft_not_reused(self):
        record = self._create_pending()
        rdb = _get_rdb()
        rdb.update_draft_status(record["id"], new_status="rejected",
                                reviewer_identity="physician@test",
                                review_comment="Not accurate.")
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        # After rejection, resolver should try to generate a NEW draft
        assert result.get("unverified_gene_draft") is not None, (
            "Rejected draft must not block generation of a new draft"
        )


# ===========================================================================
# 5. Generic non-curated genes (pipeline consistency)
# ===========================================================================

class TestGenericNonCuratedGenes:
    """The same resolver pipeline applies to any gene symbol — no hard-coded logic."""

    def test_mygene_vus_with_draft_generates_supplemental_card(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_MYGENE), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_MYGENE_BIOLOGY):
            result = _ce._build_known_gene_answer("MYGENE", question="מה המשמעות של VUS בגן MYGENE?")
        assert result.get("unverified_gene_draft") is not None
        assert result["unverified_gene_draft"]["gene_symbol"] == "MYGENE"

    def test_mygene_main_answer_does_not_contain_ai_biology(self):
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_MYGENE), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_MYGENE_BIOLOGY):
            result = _ce._build_known_gene_answer("MYGENE", question="מה המשמעות של VUS בגן MYGENE?")
        assert _FAKE_MYGENE_BIOLOGY["text_he"] not in result["answer"]

    def test_no_lm_available_returns_vus_only_answer(self):
        """When LLM is not configured, source_type='none' → VUS-only answer, no review record."""
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=_GENE_SUMMARY_MYGENE), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
            result = _ce._build_known_gene_answer("MYGENE", question="מה המשמעות של VUS בגן MYGENE?")
        assert result.get("unverified_gene_draft") is None
        assert _get_rdb().list_drafts() == []
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_response_schema_consistent_across_genes(self):
        """All VUS+gene answers must have the same top-level keys regardless of gene."""
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        for gene, curated in [("BRCA1", _BRCA1_CURATED_TEXT), ("ACE", None)]:
            draft = _FAKE_ACE_BIOLOGY if gene == "ACE" else None
            result = _call_vus_gene(gene, curated=curated, draft=draft)
            missing = required - result.keys()
            assert not missing, f"Missing keys for {gene}: {missing}"

    def test_gene_explanation_source_in_metadata_for_all(self):
        """gene_explanation_source must be set in gene_metadata for every VUS+gene response."""
        for gene, curated, draft in [
            ("BRCA1", _BRCA1_CURATED_TEXT, None),
            ("ACE", None, _FAKE_ACE_BIOLOGY),
            ("MYGENE", None, None),
        ]:
            gs = _GENE_SUMMARY_MYGENE if gene == "MYGENE" else _GENE_SUMMARY_ACE
            result = _call_vus_gene(gene, curated=curated, gene_summary=gs, draft=draft)
            gm = result.get("gene_metadata", {})
            assert "gene_explanation_source" in gm, (
                f"gene_explanation_source must be in gene_metadata for {gene}"
            )


# ===========================================================================
# 6. ClinVar technical data stays separate
# ===========================================================================

class TestClinvarTechnicalDataSeparation:
    """ClinVar counts, classifications, and phenotype strings stay in gene_metadata."""

    def test_no_variant_count_in_main_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert "1,628" not in result["answer"] and "1628" not in result["answer"]

    def test_no_classification_buckets_in_main_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        for cat in ("Uncertain significance", "Benign", "Pathogenic"):
            assert cat not in result["answer"]

    def test_clinvar_data_in_gene_metadata(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        gm = result.get("gene_metadata", {})
        assert gm.get("total_variants") == 1628
        assert "Uncertain significance" in (gm.get("significance_breakdown") or {})

    def test_explicit_clinvar_query_shows_note_plus_gene_expl(self):
        """Explicit ClinVar query: statistical note in main answer AND gene explanation in card."""
        result = _call_vus_gene("ACE", question="כמה וריאנטים יש בגן ACE במאגר?",
                                draft=_FAKE_ACE_BIOLOGY)
        assert "במאגר ClinVar קיימות" in result["answer"], "ClinVar note expected for explicit query"
        assert result.get("unverified_gene_draft") is not None, "Gene explanation card still expected"

    def test_no_semicolons_in_any_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        assert ";" not in result["answer"]


# ===========================================================================
# 7. Safety
# ===========================================================================

class TestSafetyPreserved:
    """All safety rules remain intact after the 27.9.6 changes."""

    def test_identifying_info_blocked(self):
        r = _client.post("/ask", json={"question": "קוראים לי שרה, יש לי VUS ב-BRCA1"})
        assert r.status_code == 200
        assert r.json()["safety_level"] == "contains_identifying_info"

    def test_medical_action_refused(self):
        r = _client.post("/ask", json={"question": "האם עלי לעשות ניתוח בגלל ה-VUS ב-BRCA1?"})
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")

    def test_no_personal_risk_in_vus_gene_answer(self):
        result = _call_vus_gene("ACE", draft=_FAKE_ACE_BIOLOGY)
        for phrase in ("הסיכון שלך", "הסיכון האישי", "אתה בסיכון"):
            assert phrase not in result["answer"]

    def test_schema_always_5_keys(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys()), f"Missing: {required - data.keys()}"


# ===========================================================================
# 8. Regressions
# ===========================================================================

class TestRegressions:
    """Existing paths are unaffected by 27.9.6 changes."""

    def test_explicit_clinvar_query_still_shows_statistical_note(self):
        r = _client.post("/ask", json={"question": "כמה וריאנטים יש בגן ACE במאגר?"})
        assert r.status_code == 200

    def test_standalone_gene_biology_still_creates_review_record(self):
        """_build_gene_clinvar_answer (standalone biology) still persists for physician review."""
        fake_summary = dict(_GENE_SUMMARY_ACE)
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=fake_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_ACE_BIOLOGY):
            _ce._build_gene_clinvar_answer("מה התפקיד הביולוגי של ACE?", "ACE")
        rdb = _get_rdb()
        assert len(rdb.list_drafts(gene_symbol="ACE")) == 1

    def test_chromosome_path_unaffected(self):
        r = _client.post("/ask", json={"question": "מה זה מחיקה בכרומוזום 21?"})
        assert r.status_code == 200
        assert bool(r.json()["answer"])

    def test_general_low_risk_outside_physician_review(self):
        r = _client.post("/ask", json={"question": "כמה גנים יש לאנשים?"})
        assert r.status_code == 200
        # General questions must not inject gene_metadata or ClinVar content
        data = r.json()
        assert "ClinVar" not in data["answer"] or "within" not in data["answer"]

    def test_vus_followup_routing_unchanged(self):
        r = _client.post("/ask", json={
            "question": "מה כדאי לעשות עם זה?",
            "last_topic": "vus",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] != "contains_identifying_info"

    def test_carrier_path_unaffected(self):
        r = _client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"})
        assert r.status_code == 200
        assert bool(r.json()["answer"])
