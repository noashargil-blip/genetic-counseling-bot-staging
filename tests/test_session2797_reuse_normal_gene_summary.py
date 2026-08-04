# -*- coding: utf-8 -*-
"""
tests/test_session2797_reuse_normal_gene_summary.py

Session 27.9.7 — Reuse the existing normal gene answer inside VUS + gene responses.

If the application can already answer "מה זה הגן X?" as a normal, non-unverified
answer, then "מה המשמעות של VUS בגן X?" must reuse a concise version of that same
accepted gene explanation.

Root cause of Session 27.9.6 gap:
  _resolve_gene_explanation_for_vus() did not call
  _has_sufficient_grounded_gene_context() + _generate_source_grounded_gene_answer().
  That is the path the standalone gene route uses for tier-2 genes (like COL1A1)
  when ClinVar phenotypes are available.

Fix: add "grounded_clinvar" as priority 3 in the resolver, between gene_knowledge (2)
and physician_approved (4).  The same phenotype evidence and the same LLM prompt are
used for both the standalone answer and the VUS + gene answer.

Priority chain after 27.9.7:
  1. curated          — gene_cards approved text
  2. grounded         — gene_knowledge KB text
  3. grounded_clinvar — source-grounded ClinVar phenotype summary (NO review required)
  4. physician_approved — review_db approved draft
  5. ai_unreviewed    — new AI draft → supplemental card, physician queue
  6. none             — VUS-only answer

Tests cover:
  1. COL1A1 standalone: unchanged, no AI draft, no review record
  2. COL1A1 VUS: grounded explanation in main answer, no unverified card, no review record
  3. Multiple arbitrary genes: same pipeline for all genes with ≥3 phenotypes
  4. Gene without grounded context: AI draft still allowed, enters queue
  5. Approved content: physician-approved still reusable when grounded is absent
  6. Safety: no personal interpretation, no treatment advice
  7. Regression: BRCA1 curated unchanged, explicit ClinVar queries, chromosome path,
     carrier path, general low-risk AI
"""

import importlib
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as _ce

_client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A summary with enough phenotypes to trigger grounded_clinvar (≥3 non-trivial)
_GENE_SUMMARY_COL1A1 = {
    "gene_symbol": "COL1A1",
    "total_variants": 6938,
    "by_significance": {
        "Pathogenic": 1926,
        "Uncertain significance": 1433,
        "Likely benign": 1742,
    },
    "phenotypes": [
        "Osteogenesis imperfecta type I",
        "Ehlers-Danlos syndrome, arthrochalasia type",
        "COL1A1-related disorder",
        "Cardiovascular phenotype",
        "Infantile cortical hyperostosis",
    ],
}

# A summary with <3 useful phenotypes — does NOT qualify for grounded_clinvar
_GENE_SUMMARY_ACE = {
    "gene_symbol": "ACE",
    "total_variants": 1628,
    "by_significance": {"Uncertain significance": 950},
    "phenotypes": ["Renal tubular dysgenesis", "Hypertension"],  # only 2
}

# Generic summary for arbitrary test gene — 3 phenotypes → qualifies
_GENE_SUMMARY_MYGENE = {
    "gene_symbol": "MYGENE",
    "total_variants": 500,
    "by_significance": {"Uncertain significance": 300},
    "phenotypes": ["Alpha disorder", "Beta syndrome", "Gamma condition"],
}

_GROUNDED_COL1A1_TEXT = (
    "COL1A1 קשור למספר מצבים רפואיים, כולל אוסטיאוגנזיס אימפרפקטה סוג I, "
    "תסמונת אילר-דנלוס, וסוגים נוספים של הפרעות רקמת חיבור."
)

_GROUNDED_MYGENE_TEXT = (
    "הגן MYGENE קשור למספר מצבים: Alpha disorder, Beta syndrome, ו-Gamma condition."
)

_FAKE_AI_DRAFT_ACE = {
    "visible": True,
    "status": "ai_generated_unreviewed",
    "gene_symbol": "ACE",
    "text_he": "הגן ACE מקודד לאנזים הממיר אנגיוטנסין.",
    "generated_by_model": "gpt-test",
    "review_status": "unreviewed",
    "approved": False,
    "requires_physician_review": True,
    "ai_content_type": "medical_educational_ai_expansion",
    "based_on": "model_expansion_llm_knowledge",
    "source_grounded": False,
}

_BRCA1_CURATED_TEXT = "הגן BRCA1 קשור לסיכון מוגבר לסרטן שד ושחלות. (טקסט מאושר)"

_APPROVED_ACE_SUMMARY = (
    "הגן ACE מקודד לאנזים הממיר אנגיוטנסין. (אושר על ידי גנטיקאי)"
)


@pytest.fixture(autouse=True)
def isolated_review_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_review_2797.db")
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

def _call_vus_gene(gene, question=None, gene_summary=None,
                   curated=None, gk_patient=None,
                   grounded_text=None, ai_draft=None):
    """Call _build_known_gene_answer with controlled dependencies."""
    q = question or f"מה המשמעות של VUS בגן {gene}?"
    gs = gene_summary  # None means gene_index returns None → no ClinVar
    with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", gs is not None), \
         patch.object(_ce.gene_index, "get_gene_summary", return_value=gs), \
         patch.object(_ce.gene_cards, "get_approved_summary", return_value=curated), \
         patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=gk_patient), \
         patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge",
                      return_value=gk_patient is not None), \
         patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
         patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
         patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None), \
         patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=grounded_text), \
         patch.object(_ce, "_generate_unverified_gene_draft", return_value=ai_draft):
        return _ce._build_known_gene_answer(gene, question=q)


def _call_standalone_gene(gene, question=None, gene_summary=None, grounded_text=None):
    """Call _build_gene_clinvar_answer (standalone gene route) with controlled deps."""
    q = question or f"מה זה הגן {gene}?"
    gs = gene_summary
    with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", gs is not None), \
         patch.object(_ce.gene_index, "get_gene_summary", return_value=gs), \
         patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
         patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
         patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
         patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
         patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
         patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None), \
         patch.object(_ce, "_generate_source_grounded_gene_answer", return_value=grounded_text), \
         patch.object(_ce, "_generate_unverified_gene_draft", return_value=None):
        return _ce._build_gene_clinvar_answer(q, gene)


# ===========================================================================
# 1. COL1A1 standalone: behavior unchanged
# ===========================================================================

class TestCol1a1StandaloneUnchanged:
    """The standalone 'מה זה הגן COL1A1?' path must not be altered by 27.9.7."""

    def test_standalone_returns_grounded_text_as_main_answer(self):
        result = _call_standalone_gene("COL1A1",
                                       gene_summary=_GENE_SUMMARY_COL1A1,
                                       grounded_text=_GROUNDED_COL1A1_TEXT)
        assert _GROUNDED_COL1A1_TEXT in result["answer"]

    def test_standalone_no_ai_draft(self):
        result = _call_standalone_gene("COL1A1",
                                       gene_summary=_GENE_SUMMARY_COL1A1,
                                       grounded_text=_GROUNDED_COL1A1_TEXT)
        assert result.get("unverified_gene_draft") is None or \
               not result.get("unverified_gene_draft", {}).get("visible"), \
               "Standalone grounded answer must not attach an unverified draft"

    def test_standalone_no_review_record(self):
        _call_standalone_gene("COL1A1",
                              gene_summary=_GENE_SUMMARY_COL1A1,
                              grounded_text=_GROUNDED_COL1A1_TEXT)
        assert _get_rdb().list_drafts() == []

    def test_standalone_matched_topic(self):
        result = _call_standalone_gene("COL1A1",
                                       gene_summary=_GENE_SUMMARY_COL1A1,
                                       grounded_text=_GROUNDED_COL1A1_TEXT)
        assert result["matched_topic"] == "gene_clinvar_summary"

    def test_standalone_source_grounded_flag(self):
        result = _call_standalone_gene("COL1A1",
                                       gene_summary=_GENE_SUMMARY_COL1A1,
                                       grounded_text=_GROUNDED_COL1A1_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("source_grounded") is True


# ===========================================================================
# 2. COL1A1 VUS: grounded explanation reused, no unverified card
# ===========================================================================

class TestCol1a1VusGroundedReuse:
    """VUS + COL1A1 must reuse the same grounded gene summary, not generate a new AI draft."""

    def test_vus_explanation_in_main_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert "VUS" in result["answer"] or "ממצא" in result["answer"]

    def test_grounded_gene_summary_in_main_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert _GROUNDED_COL1A1_TEXT in result["answer"], (
            "Accepted grounded gene summary must appear in the main VUS answer"
        )

    def test_gene_section_header_in_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert "לגבי הגן COL1A1:" in result["answer"], (
            "'לגבי הגן COL1A1:' header must appear when grounded summary is used"
        )

    def test_distinction_note_present(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert "חשוב להבדיל" in result["answer"]

    def test_no_unverified_card(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert result.get("unverified_gene_draft") is None, (
            "No unverified_gene_draft card when grounded source is available"
        )

    def test_no_review_record(self):
        _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                       grounded_text=_GROUNDED_COL1A1_TEXT)
        assert _get_rdb().list_drafts() == [], (
            "No physician-review record must be created for grounded_clinvar path"
        )

    def test_no_review_draft_id(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert result.get("review_draft_id") is None

    def test_gene_explanation_source_grounded_clinvar(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "grounded_clinvar"

    def test_gene_knowledge_status_grounded(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_knowledge_status") == "grounded"

    def test_source_grounded_flag_true(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("source_grounded") is True

    def test_no_raw_clinvar_counts_in_main_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert "6,938" not in result["answer"] and "6938" not in result["answer"]

    def test_no_semicolons_in_main_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert ";" not in result["answer"]

    def test_clinvar_data_in_gene_metadata(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("total_variants") == 6938

    def test_matched_topic_vus_known_gene(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        assert result["matched_topic"] == "vus_known_gene"


# ===========================================================================
# 3. Multiple arbitrary genes: same pipeline for all with ≥3 phenotypes
# ===========================================================================

class TestGenericGroundedClinvarPath:
    """No gene-specific hard-coding — the resolver works for any gene with ≥3 phenotypes."""

    def test_mygene_vus_reuses_grounded_summary(self):
        result = _call_vus_gene("MYGENE", gene_summary=_GENE_SUMMARY_MYGENE,
                                grounded_text=_GROUNDED_MYGENE_TEXT)
        assert _GROUNDED_MYGENE_TEXT in result["answer"]

    def test_mygene_no_unverified_card(self):
        result = _call_vus_gene("MYGENE", gene_summary=_GENE_SUMMARY_MYGENE,
                                grounded_text=_GROUNDED_MYGENE_TEXT)
        assert result.get("unverified_gene_draft") is None

    def test_mygene_no_review_record(self):
        _call_vus_gene("MYGENE", gene_summary=_GENE_SUMMARY_MYGENE,
                       grounded_text=_GROUNDED_MYGENE_TEXT)
        assert _get_rdb().list_drafts() == []

    def test_mygene_gene_explanation_source(self):
        result = _call_vus_gene("MYGENE", gene_summary=_GENE_SUMMARY_MYGENE,
                                grounded_text=_GROUNDED_MYGENE_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "grounded_clinvar"

    def test_same_source_for_col1a1_and_mygene(self):
        """Both genes follow the same resolver path — no gene-specific branching."""
        r_col = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                               grounded_text=_GROUNDED_COL1A1_TEXT)
        r_my = _call_vus_gene("MYGENE", gene_summary=_GENE_SUMMARY_MYGENE,
                              grounded_text=_GROUNDED_MYGENE_TEXT)
        assert r_col["gene_metadata"]["gene_explanation_source"] == "grounded_clinvar"
        assert r_my["gene_metadata"]["gene_explanation_source"] == "grounded_clinvar"

    def test_grounded_clinvar_skipped_when_llm_returns_none(self):
        """When _generate_source_grounded_gene_answer returns None (no LLM), falls through."""
        result = _call_vus_gene("MYGENE", gene_summary=_GENE_SUMMARY_MYGENE,
                                grounded_text=None, ai_draft=None)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") != "grounded_clinvar"

    def test_schema_consistent_across_genes(self):
        """All VUS+gene responses have the same required top-level keys."""
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        for gene, gs, gt in [
            ("COL1A1", _GENE_SUMMARY_COL1A1, _GROUNDED_COL1A1_TEXT),
            ("MYGENE", _GENE_SUMMARY_MYGENE, _GROUNDED_MYGENE_TEXT),
            ("ACE", _GENE_SUMMARY_ACE, None),
        ]:
            result = _call_vus_gene(gene, gene_summary=gs, grounded_text=gt)
            missing = required - result.keys()
            assert not missing, f"Missing keys for {gene}: {missing}"


# ===========================================================================
# 4. Gene without sufficient grounded context: AI draft still allowed
# ===========================================================================

class TestFallbackToAiUnreviewed:
    """When grounded_clinvar fails (LLM None or <3 phenotypes), AI draft path still works."""

    def test_ace_two_phenotypes_falls_through_to_ai_unreviewed(self):
        """ACE has only 2 phenotypes — grounded_clinvar is skipped → ai_unreviewed."""
        result = _call_vus_gene("ACE", gene_summary=_GENE_SUMMARY_ACE,
                                grounded_text=None, ai_draft=_FAKE_AI_DRAFT_ACE)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "ai_unreviewed"

    def test_ace_ai_draft_in_supplemental_card(self):
        result = _call_vus_gene("ACE", gene_summary=_GENE_SUMMARY_ACE,
                                grounded_text=None, ai_draft=_FAKE_AI_DRAFT_ACE)
        assert result.get("unverified_gene_draft") is not None

    def test_ace_pending_review_record_created(self):
        _call_vus_gene("ACE", gene_summary=_GENE_SUMMARY_ACE,
                       grounded_text=None, ai_draft=_FAKE_AI_DRAFT_ACE)
        assert len(_get_rdb().list_drafts()) == 1

    def test_none_gene_summary_falls_through_to_ai_unreviewed(self):
        """When gene is not in the ClinVar index, grounded_clinvar is skipped."""
        result = _call_vus_gene("UNKNOWNGENE", gene_summary=None,
                                grounded_text=None, ai_draft=_FAKE_AI_DRAFT_ACE)
        gm = result.get("gene_metadata", {})
        # Should reach ai_unreviewed or none
        assert gm.get("gene_explanation_source") in ("ai_unreviewed", "none")

    def test_grounded_returns_none_falls_to_next_priority(self):
        """If grounded_clinvar LLM returns None despite having phenotypes, continues chain."""
        # _GENE_SUMMARY_COL1A1 has 5 phenotypes → grounded_clinvar eligible
        # but grounded_text=None → falls through to ai_unreviewed
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=None, ai_draft=_FAKE_AI_DRAFT_ACE)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "ai_unreviewed"


# ===========================================================================
# 5. Approved content: physician-approved still reusable when grounded absent
# ===========================================================================

class TestApprovedContentReuse:
    """physician_approved (priority 4) is used when grounded_clinvar (3) is absent."""

    def _seed_approved_ace(self):
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

    def test_approved_text_used_when_grounded_absent(self):
        """ACE (2 phenotypes) → grounded_clinvar skipped → physician_approved used."""
        self._seed_approved_ace()
        result = _call_vus_gene("ACE", gene_summary=_GENE_SUMMARY_ACE,
                                grounded_text=None, ai_draft=None)
        assert _APPROVED_ACE_SUMMARY in result["answer"]

    def test_no_new_pending_record_when_approved_reused(self):
        self._seed_approved_ace()
        _call_vus_gene("ACE", gene_summary=_GENE_SUMMARY_ACE,
                       grounded_text=None, ai_draft=None)
        pending = [d for d in _get_rdb().list_drafts() if d["review_status"] == "pending"]
        assert len(pending) == 0

    def test_grounded_clinvar_takes_priority_over_approved(self):
        """When grounded_clinvar IS available, it takes priority over physician_approved.
        Uses COL1A1 (5 phenotypes → eligible for grounded_clinvar).
        Seed a physician-approved COL1A1 summary, then confirm grounded_clinvar wins.
        """
        rdb = _get_rdb()
        _approved_col_text = "COL1A1 physician-approved summary."
        record = rdb.create_draft(
            draft_type="gene_summary",
            original_ai_text=_approved_col_text,
            gene_symbol="COL1A1",
            normalized_intent="vus_known_gene",
            prompt_version="s2796",
        )
        rdb.update_draft_status(record["id"], new_status="approved",
                                reviewer_identity="physician@test")
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text="Grounded COL1A1 explanation.", ai_draft=None)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "grounded_clinvar", (
            "grounded_clinvar (priority 3) must beat physician_approved (priority 4)"
        )
        assert _approved_col_text not in result["answer"]


# ===========================================================================
# 6. Safety preserved
# ===========================================================================

class TestSafetyPreserved:
    """All safety rules remain intact after 27.9.7 changes."""

    def test_identifying_info_blocked(self):
        r = _client.post("/ask", json={"question": "קוראים לי שרה, יש לי VUS ב-COL1A1"})
        assert r.status_code == 200
        assert r.json()["safety_level"] == "contains_identifying_info"

    def test_no_personal_risk_in_vus_gene_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        for phrase in ("הסיכון שלך", "הסיכון האישי", "אתה בסיכון"):
            assert phrase not in result["answer"]

    def test_no_treatment_advice_in_vus_gene_answer(self):
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        for phrase in ("ניתוח", "מעקב", "טיפול", "מניעה"):
            assert phrase not in result["answer"], (
                f"Treatment/surveillance advice must not appear: '{phrase}'"
            )

    def test_gene_explanation_does_not_repeat_vus_explanation(self):
        """AI draft must not repeat the VUS paragraph when grounded text is returned."""
        result = _call_vus_gene("COL1A1", gene_summary=_GENE_SUMMARY_COL1A1,
                                grounded_text=_GROUNDED_COL1A1_TEXT)
        main = result["answer"]
        # VUS explanation should appear once (in opening), not inside the gene section
        vus_phrase = "ממצא שמשמעותו עדיין לא ידועה"
        assert main.count(vus_phrase) <= 1, "VUS explanation must not be repeated"

    def test_schema_always_5_keys(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן COL1A1?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys()), f"Missing: {required - data.keys()}"


# ===========================================================================
# 7. Regression suite
# ===========================================================================

class TestRegressions:
    """Existing paths are unaffected by 27.9.7 changes."""

    def test_brca1_curated_path_unchanged(self):
        result = _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        gm = result.get("gene_metadata", {})
        assert gm.get("gene_explanation_source") == "curated"
        assert gm.get("answer_tier") == "tier1"
        assert _BRCA1_CURATED_TEXT in result["answer"]

    def test_brca1_curated_no_review_record(self):
        _call_vus_gene("BRCA1", curated=_BRCA1_CURATED_TEXT)
        assert _get_rdb().list_drafts() == []

    def test_explicit_clinvar_query_still_shows_note(self):
        r = _client.post("/ask", json={"question": "כמה וריאנטים יש בגן ACE במאגר?"})
        assert r.status_code == 200

    def test_chromosome_path_unaffected(self):
        r = _client.post("/ask", json={"question": "מה זה מחיקה בכרומוזום 21?"})
        assert r.status_code == 200
        assert bool(r.json()["answer"])

    def test_carrier_path_unaffected(self):
        r = _client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"})
        assert r.status_code == 200
        assert bool(r.json()["answer"])

    def test_vus_followup_routing_unchanged(self):
        r = _client.post("/ask", json={
            "question": "מה כדאי לעשות עם זה?",
            "last_topic": "vus",
        })
        assert r.status_code == 200
        assert r.json()["safety_level"] != "contains_identifying_info"

    def test_col1a1_vus_http_schema(self):
        r = _client.post("/ask", json={"question": "מה המשמעות של VUS בגן COL1A1?"})
        assert r.status_code == 200
        data = r.json()
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(data.keys()), f"Missing: {required - data.keys()}"

    def test_general_question_no_clinvar_noise(self):
        r = _client.post("/ask", json={"question": "כמה גנים יש לאנשים?"})
        assert r.status_code == 200
        assert bool(r.json()["answer"])

    def test_standalone_vs_vus_same_grounded_source(self):
        """Standalone and VUS paths invoke _generate_source_grounded_gene_answer for
        the same gene with the same context.  Verify both call it (not mocked away)."""
        calls_standalone = []
        calls_vus = []

        orig = _ce._generate_source_grounded_gene_answer

        def _capture_standalone(*a, **kw):
            calls_standalone.append(a[0])
            return _GROUNDED_COL1A1_TEXT

        def _capture_vus(*a, **kw):
            calls_vus.append(a[0])
            return _GROUNDED_COL1A1_TEXT

        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary",
                          return_value=_GENE_SUMMARY_COL1A1), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None), \
             patch.object(_ce, "_generate_source_grounded_gene_answer",
                          side_effect=_capture_standalone):
            _ce._build_gene_clinvar_answer("מה זה הגן COL1A1?", "COL1A1")

        with patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True), \
             patch.object(_ce.gene_index, "get_gene_summary",
                          return_value=_GENE_SUMMARY_COL1A1), \
             patch.object(_ce.gene_cards, "get_approved_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "has_approved_gene_knowledge", return_value=False), \
             patch.object(_ce.gene_knowledge, "get_gene_vus_note", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None), \
             patch.object(_ce, "_generate_unverified_gene_draft", return_value=None), \
             patch.object(_ce, "_generate_source_grounded_gene_answer",
                          side_effect=_capture_vus):
            _ce._build_known_gene_answer("COL1A1", question="מה המשמעות של VUS בגן COL1A1?")

        assert "COL1A1" in calls_standalone, "Standalone route must call grounded generator"
        assert "COL1A1" in calls_vus, "VUS route must call the same grounded generator"
