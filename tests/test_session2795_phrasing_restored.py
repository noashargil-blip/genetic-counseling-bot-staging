# -*- coding: utf-8 -*-
"""
tests/test_session2795_phrasing_restored.py

Session 27.9.5 — Restore patient-facing phrasing; keep ClinVar technical data
out of the main VUS+gene answer.

Tests cover:
  Part A/B — Default VUS+gene: phrased educational answer only, no ClinVar metadata
  Part C   — Explicit ClinVar DB queries: statistical note allowed
  Part D   — Biology path unchanged
  Part E   — Technical card / gene_metadata preserved; phenotype strings cleaned
  Part F   — Regressions: curated genes, VUS safety, chromosome path, schema

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
# Fixtures and shared data
# ---------------------------------------------------------------------------

_GENE_SUMMARY_ACE = {
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

_GENE_SUMMARY_DIRTY_PHENOTYPES = {
    "gene_symbol": "ACE",
    "total_variants": 1628,
    "by_significance": {"Uncertain significance": 950},
    "phenotypes": [
        "Renal tubular dysgenesis;not specified;Hypertension",
        "not provided",
        "Renal tubular dysgenesis",
    ],
}

_FAKE_DRAFT = {
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
    db_path = str(tmp_path / "test_review_2795.db")
    monkeypatch.setenv("REVIEW_DB_SQLITE_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from app import review_db
    importlib.reload(review_db)
    review_db.init_db()
    yield review_db


def _get_rdb():
    from app import review_db as _rdb
    return _rdb


def _vus_ace_call(question="מה המשמעות של VUS בגן ACE?", gene_summary=None):
    gs = gene_summary if gene_summary is not None else _GENE_SUMMARY_ACE
    with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
         patch.object(_ce.gene_index, "get_gene_summary", return_value=gs), \
         patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
         patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
         patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
         patch.object(_ce, "_generate_unverified_gene_draft", return_value=_FAKE_DRAFT):
        return _ce._build_known_gene_answer("ACE", question=question)


# ===========================================================================
# Parts A + B: Default VUS+gene — phrased answer, no ClinVar metadata in text
# ===========================================================================

class TestVusAcePhrasingRestored:
    """Default VUS+gene answer is the deterministic educational VUS text only."""

    def test_vus_explanation_in_answer(self):
        result = _vus_ace_call()
        assert "VUS" in result["answer"] or "ממצא" in result["answer"], (
            "VUS explanation must be present in the patient-facing answer"
        )

    def test_no_variant_count_in_answer(self):
        result = _vus_ace_call()
        assert "1,628" not in result["answer"], "Raw variant count must not appear in VUS answer"
        assert "1628" not in result["answer"]

    def test_no_raw_clinvar_record_string(self):
        result = _vus_ace_call()
        assert "במאגר ClinVar קיימות" not in result["answer"], (
            "ClinVar record-count phrase must not appear in default VUS answer"
        )

    def test_no_classification_categories_in_answer(self):
        result = _vus_ace_call()
        for cat in ("Uncertain significance", "Likely benign", "Likely pathogenic"):
            assert cat not in result["answer"], f"Classification category '{cat}' must not appear"

    def test_no_semicolons_in_answer(self):
        result = _vus_ace_call()
        assert ";" not in result["answer"], "No semicolons must appear in VUS answer"

    def test_no_unverified_draft_in_response(self):
        result = _vus_ace_call()
        assert result.get("unverified_gene_draft") is None

    def test_no_review_db_record_created(self):
        _vus_ace_call()
        assert _get_rdb().list_drafts() == [], "No physician review record must be created"

    def test_gene_metadata_present(self):
        result = _vus_ace_call()
        assert "gene_metadata" in result, "gene_metadata must still be returned"
        gm = result["gene_metadata"]
        assert gm.get("answer_tier") == "tier2"

    def test_gene_knowledge_status_is_clinvar_index_no_expansion(self):
        result = _vus_ace_call()
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "clinvar_index_no_expansion"

    def test_ai_draft_debug_reason_tier2_deterministic(self):
        result = _vus_ace_call()
        debug = result.get("ai_draft_debug", {})
        assert debug.get("reason") == "tier2_deterministic_only"

    def test_source_grounded_not_set_for_default(self):
        result = _vus_ace_call()
        gm = result.get("gene_metadata", {})
        assert gm.get("source_grounded") is not True

    def test_llm_used_false(self):
        result = _vus_ace_call()
        assert result.get("llm_used") is False

    def test_no_clinvar_disclaimer_in_answer(self):
        result = _vus_ace_call()
        assert "אינו מפרש את הממצא האישי שלך" not in result["answer"]


# ===========================================================================
# Part C: Explicit ClinVar database query — statistical note allowed
# ===========================================================================

class TestExplicitClinvarQueryAllowed:
    """When user explicitly asks about the ClinVar database, a cleaned statistical note
    is included. Counts and classifications may appear."""

    def test_explicit_variant_count_question_produces_note(self):
        result = _vus_ace_call(question="כמה וריאנטים יש בגן ACE במאגר?")
        assert "במאגר ClinVar קיימות" in result["answer"], (
            f"Explicit ClinVar query must show note. Answer: {result['answer'][:300]!r}"
        )

    def test_explicit_records_question_produces_note(self):
        result = _vus_ace_call(question="כמה רשומות יש לגן ACE ב-ClinVar?")
        assert "במאגר ClinVar קיימות" in result["answer"]

    def test_explicit_classifications_question_produces_note(self):
        result = _vus_ace_call(question="אילו סיווגים קיימים לגן ACE?")
        assert "במאגר ClinVar קיימות" in result["answer"]

    def test_explicit_query_gene_knowledge_status_clinvar_summarized(self):
        result = _vus_ace_call(question="כמה וריאנטים יש בגן ACE במאגר?")
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "clinvar_summarized"

    def test_explicit_query_no_ai_draft(self):
        result = _vus_ace_call(question="כמה וריאנטים יש בגן ACE במאגר?")
        assert result.get("unverified_gene_draft") is None

    def test_explicit_query_no_review_record(self):
        _vus_ace_call(question="כמה וריאנטים יש בגן ACE במאגר?")
        assert _get_rdb().list_drafts() == []

    def test_explicit_query_variant_count_appears_in_note(self):
        result = _vus_ace_call(question="כמה וריאנטים יש בגן ACE במאגר?")
        assert "1,628" in result["answer"], (
            "Variant count must appear in explicit ClinVar query answer"
        )


# ===========================================================================
# Part E: Phenotype string sanitation — no semicolons, no trivial entries
# ===========================================================================

class TestPhenotypeStringClean:
    """Phenotype strings with semicolons and trivial entries must be cleaned."""

    def test_no_semicolons_in_explicit_query_answer(self):
        result = _vus_ace_call(
            question="כמה וריאנטים יש בגן ACE במאגר?",
            gene_summary=_GENE_SUMMARY_DIRTY_PHENOTYPES,
        )
        assert ";" not in result["answer"], (
            f"Semicolons must be absent from ClinVar note. Answer: {result['answer']!r}"
        )

    def test_trivial_phenotype_not_in_answer(self):
        result = _vus_ace_call(
            question="כמה וריאנטים יש בגן ACE במאגר?",
            gene_summary=_GENE_SUMMARY_DIRTY_PHENOTYPES,
        )
        assert "not specified" not in result["answer"]
        assert "not provided" not in result["answer"]

    def test_real_phenotype_still_in_answer(self):
        result = _vus_ace_call(
            question="כמה וריאנטים יש בגן ACE במאגר?",
            gene_summary=_GENE_SUMMARY_DIRTY_PHENOTYPES,
        )
        assert "Renal tubular dysgenesis" in result["answer"] or \
               "Hypertension" in result["answer"]

    def test_deduplicated_phenotype_appears_once(self):
        result = _vus_ace_call(
            question="כמה וריאנטים יש בגן ACE במאגר?",
            gene_summary=_GENE_SUMMARY_DIRTY_PHENOTYPES,
        )
        count = result["answer"].count("Renal tubular dysgenesis")
        assert count <= 1, f"Phenotype must appear at most once, found {count} times"


# ===========================================================================
# Part D: Biology path unchanged
# ===========================================================================

class TestBiologyPathUnchanged:
    """_build_gene_clinvar_answer still persists for physician review when AI draft
    is returned and there is no grounded context."""

    def test_biology_path_still_creates_review_record(self):
        fake_summary = {
            "gene_symbol": "APOE",
            "total_variants": 200,
            "by_significance": {"Pathogenic": 5},
            "phenotypes": ["Alzheimer disease"],
        }
        fake_draft = {
            "visible": True,
            "text_he": "APOE test draft",
            "gene_symbol": "APOE",
            "generated_by_model": "gpt-test",
            "review_status": "unreviewed",
            "approved": False,
            "requires_physician_review": True,
            "ai_content_type": "medical_educational_ai_expansion",
            "based_on": "model_expansion_with_clinvar_gene_context",
            "source_grounded": False,
        }
        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary", return_value=fake_summary), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=fake_draft):
            _ce._build_gene_clinvar_answer("APOE", "APOE")
        drafts = _get_rdb().list_drafts(gene_symbol="APOE")
        assert len(drafts) == 1, (
            f"Biology path must still persist review record. Got: {len(drafts)}"
        )


# ===========================================================================
# Part F: Regression suite
# ===========================================================================

class TestPhrasingRestoredRegressions:
    """End-to-end regressions: schema, safety, curated genes, chromosome path."""

    def test_schema_vus_ace_http(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys()), f"Missing schema keys: {required - data.keys()}"

    def test_schema_general_vus_http(self):
        r = _client.post("/ask", json={"question": "מה זה VUS?"})
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] == "general_information"

    def test_vus_answer_has_vus_text_http(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        data = r.json()
        assert "VUS" in data["answer"] or "ממצא" in data["answer"]

    def test_no_raw_count_in_http_vus_answer(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן ACE?"})
        data = r.json()
        assert "1,628" not in data["answer"]
        assert "רשומות" not in data["answer"] or "במאגר" not in data["answer"]

    def test_chromosome_path_unaffected(self):
        r = _client.post("/ask", json={"question": "מה זה מחיקה בכרומוזום 21?"})
        assert r.status_code == 200
        data = r.json()
        assert bool(data["answer"])

    def test_carrier_path_unaffected(self):
        r = _client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"})
        assert r.status_code == 200
        data = r.json()
        assert bool(data["answer"])

    def test_identifying_info_blocked(self):
        r = _client.post("/ask", json={"question": "קוראים לי שרה, יש לי VUS ב-ACE"})
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] == "contains_identifying_info"

    def test_medical_action_refused(self):
        r = _client.post("/ask", json={"question": "האם עלי לעשות ניתוח בגלל ה-VUS?"})
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")

    def test_brca1_curated_not_contaminated_with_clinvar_note(self):
        r = _client.post("/ask", json={"question": "יש לי VUS ב-BRCA1"})
        assert r.status_code == 200
        data = r.json()
        gm = data.get("gene_metadata", {})
        if gm.get("answer_tier") == "tier1":
            assert "במאגר ClinVar קיימות" not in data["answer"]

    def test_vus_followup_stays_on_topic(self):
        r = _client.post("/ask", json={
            "question": "מה כדאי לעשות עם זה?",
            "last_topic": "vus",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["safety_level"] != "contains_identifying_info"

    def test_general_question_no_clinvar_note(self):
        r = _client.post("/ask", json={"question": "כמה גנים יש לאנשים?"})
        assert r.status_code == 200
        data = r.json()
        assert "אינו מפרש את הממצא האישי שלך" not in data["answer"]
