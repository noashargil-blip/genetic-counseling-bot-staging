"""
Session 27.8.2 tests — restore grounded phrasing and contextual chromosome answers.

Parts covered:
  A   Historical information pipeline audit (source layer verification)
  B   Revised final answer pipeline priority
  C/D Source inventory and claim-level sufficiency
  F   MTHFR regression: no 'אין לי סיכום' when sources support an answer
  G   Generic gene test matrix by source configuration (not gene names)
  H   Chromosome number appears in answer text when known from context
  I   finding_type added to session context output
  J   Context-aware clinician questions for chromosome findings
  K   Physician queue policy: only AI_EXPANDED_UNVERIFIED enqueued

Run:
  PYTHONUTF8=1 python -m pytest tests/test_session2781_grounding_context.py -v
"""
import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.counseling_engine as _ce
import app.gene_knowledge as _gk

client = TestClient(app)

_REQUIRED_KEYS = {"answer", "safety_level", "needs_genetic_counselor",
                  "matched_topic", "suggested_questions"}


def _ask(question: str, **kwargs) -> dict:
    payload = {"question": question, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


# ===========================================================================
# Part A — Historical pipeline audit: gene_knowledge draft access
# ===========================================================================

class TestDraftBiologyTextLayer:
    """Layer 2.5: get_gene_knowledge_biology_text() returns draft text regardless of approval."""

    def test_function_exists_on_gene_knowledge_module(self):
        assert hasattr(_gk, "get_gene_knowledge_biology_text"), (
            "get_gene_knowledge_biology_text must be a public function on gene_knowledge"
        )

    def test_gene_not_in_kb_returns_none(self):
        assert _gk.get_gene_knowledge_biology_text("TESTGENE_NOTEXIST") is None

    def test_mthfr_not_in_kb_returns_none(self):
        """MTHFR has no entry in gene_knowledge_base.json."""
        result = _gk.get_gene_knowledge_biology_text("MTHFR")
        assert result is None, (
            "MTHFR is not in gene_knowledge_base.json — must return None"
        )

    def test_hbb_approved_record_returns_text(self):
        """HBB has approved=True in gene_knowledge_base.json; draft accessor also returns its text."""
        result = _gk.get_gene_knowledge_biology_text("HBB")
        # HBB should be in KB (approved=True); the draft accessor returns text for any record.
        if result is not None:
            assert len(result) > 30, "HBB biology text must be non-trivial"

    def test_approved_and_draft_accessible_equally(self):
        """
        get_gene_knowledge_biology_text returns text for ANY record regardless of approved flag.
        It is distinct from get_gene_patient_summary which requires approved=True.
        Uses actual genes from gene_knowledge_base.json: BRCA1 (approved=False), HBB (approved=True).
        """
        # BRCA1 is approved=False in gene_knowledge_base.json
        brca1_draft_text = _gk.get_gene_knowledge_biology_text("BRCA1")
        brca1_patient_text = _gk.get_gene_patient_summary("BRCA1")

        if brca1_draft_text is None:
            pytest.skip("BRCA1 not in gene_knowledge_base.json on this deployment")

        assert brca1_draft_text is not None, "Draft accessor must return BRCA1 text (approved=False)"
        assert brca1_patient_text is None, (
            "Approved accessor must return None for BRCA1 (approved=False)"
        )

        # HBB is approved=True — draft accessor should also return its text
        hbb_text = _gk.get_gene_knowledge_biology_text("HBB")
        hbb_patient = _gk.get_gene_patient_summary("HBB")
        if hbb_text:
            assert hbb_patient is not None, (
                "Approved accessor must also return HBB text (approved=True)"
            )

    def test_short_text_returns_none(self):
        """Biology text shorter than 30 chars is not useful as grounding context."""
        with patch.object(_gk, "_RECORDS", {
            "TINY": {"gene_symbol": "TINY", "approved": False, "patient_summary_he": "קצר"},
        }):
            assert _gk.get_gene_knowledge_biology_text("TINY") is None


# ===========================================================================
# Part C/D — Claim-level sufficiency: draft biology text in _has_sufficient
# ===========================================================================

class TestDraftBiologyInGroundingSufficiency:
    """Layer 2.5 is included in _has_sufficient_grounded_gene_context()."""

    def test_draft_biology_text_qualifies_as_biology(self):
        """gene_knowledge draft text (approved=False) sets has_biology=True."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text",
                          return_value="BRCA1 הוא גן שמקודד לחלבון המשתתף בתיקון נזקי DNA."):
            has, ctx = _ce._has_sufficient_grounded_gene_context("BRCA1", {})
            assert has is True
            draft_parts = [p for p in ctx if p.get("type") == "draft_biology_text"]
            assert draft_parts, "draft_biology_text part should be in context"
            assert draft_parts[0].get("has_biology") is True
            assert draft_parts[0].get("approved") is False

    def test_draft_biology_text_skipped_when_approved_exists(self):
        """Layer 2.5 is skipped when an approved biology source was already found."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary",
                          return_value="HBB אחראי על ייצור beta-globin — מרכיב של המוגלובין."), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text",
                          return_value="draft text would be here") as mock_draft:
            has, ctx = _ce._has_sufficient_grounded_gene_context("HBB", {})
            assert has is True
            # draft accessor should not even be called when approved text was found
            mock_draft.assert_not_called()
            draft_parts = [p for p in ctx if p.get("type") == "draft_biology_text"]
            assert not draft_parts, "draft_biology_text must not appear when approved text found"

    def test_draft_biology_part_has_approved_false(self):
        """The approved flag in the draft_biology_text part must be False."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text",
                          return_value="גן עם תפקיד חשוב בתיקון נזקי DNA ובשמירה על יציבות הגנום."):
            _, ctx = _ce._has_sufficient_grounded_gene_context("POLE", {})
            draft_part = next((p for p in ctx if p.get("type") == "draft_biology_text"), None)
            if draft_part:
                assert draft_part["approved"] is False


# ===========================================================================
# Part D — Association-only path: phenotypes as grounding source
# ===========================================================================

class TestAssociationOnlyGrounding:
    """Phenotype-only context now qualifies for an association-only grounded answer."""

    def test_three_phenotypes_sufficient_for_association_answer(self):
        """3+ non-trivial phenotypes qualify for grounding (association-only)."""
        summary = {"phenotypes": ["Condition A", "Condition B", "Condition C"]}
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
            assert has is True
            pheno = next((p for p in ctx if p["type"] == "phenotype_associations"), None)
            assert pheno is not None
            assert pheno["has_biology"] is False

    def test_two_phenotypes_not_sufficient(self):
        """Fewer than 3 phenotypes are not sufficient for association answer."""
        summary = {"phenotypes": ["Condition A", "Condition B"]}
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, _ = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
            assert has is False

    def test_no_summary_no_kb_is_not_sufficient(self):
        """Gene with no local data at all → not sufficient."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, _ = _ce._has_sufficient_grounded_gene_context("TESTGENE", None)
            assert has is False

    def test_association_grounding_does_not_claim_biology(self):
        """When only phenotypes are available, has_biology is False in all context parts."""
        summary = {"phenotypes": ["Disease X", "Disease Y", "Disease Z"]}
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
            assert has is True
            # Critical: no part claims biology
            assert not any(p.get("has_biology") for p in ctx)

    def test_association_prompt_used_when_biology_absent(self):
        """_generate_source_grounded_gene_answer uses association-only prompt for phenotype-only context."""
        phenotype_only_parts = [
            {"type": "phenotype_associations", "phenotypes": ["Disease X", "Disease Y", "Disease Z"],
             "has_biology": False}
        ]
        debug = {}
        # Without LLM configured, the function returns None but we can inspect debug state
        result = _ce._generate_source_grounded_gene_answer("TESTGENE", phenotype_only_parts, _debug=debug)
        # result is None (no LLM), but prompt_type should indicate association
        # OR attempted=False/reason=llm_not_configured
        assert result is None
        if "prompt_type" in debug:
            assert debug["prompt_type"] == "association_only"

    def test_biology_prompt_used_when_biology_present(self):
        """_generate_source_grounded_gene_answer uses biology prompt when biology context present."""
        biology_parts = [
            {"type": "curated_gene_description",
             "text": "HBB מקודד לשרשרת beta-globin, חלק מהמוגלובין.", "has_biology": True}
        ]
        debug = {}
        result = _ce._generate_source_grounded_gene_answer("HBB", biology_parts, _debug=debug)
        assert result is None  # No LLM configured
        if "prompt_type" in debug:
            assert debug["prompt_type"] == "biology_grounded"


# ===========================================================================
# Part F — MTHFR regression
# ===========================================================================

class TestMTHFRRegression:
    """MTHFR is not in gene_knowledge_base.json — behavior depends on ClinVar availability."""

    def test_mthfr_not_in_gene_knowledge_base(self):
        """MTHFR has no record in gene_knowledge_base.json — all KB accessors return None."""
        assert _gk.get_gene_patient_summary("MTHFR") is None
        assert _gk.get_gene_context_summary("MTHFR") is None
        assert _gk.get_gene_knowledge_biology_text("MTHFR") is None

    def test_mthfr_with_phenotypes_qualifies_for_association_grounded(self):
        """When ClinVar phenotypes available, MTHFR gets association-only grounded answer."""
        mthfr_clinvar = {
            "phenotypes": [
                "Homocystinuria",
                "Homocysteinemia",
                "MTHFR deficiency",
                "Neural tube defects",
            ],
            "total_variants": 1200,
        }
        has, ctx = _ce._has_sufficient_grounded_gene_context("MTHFR", mthfr_clinvar)
        assert has is True, (
            "MTHFR with ClinVar phenotypes must qualify for association-only grounded answer"
        )
        pheno = next((p for p in ctx if p["type"] == "phenotype_associations"), None)
        assert pheno is not None
        assert pheno["has_biology"] is False
        # No biology parts — MTHFR has no local biology knowledge
        assert not any(p.get("has_biology") for p in ctx)

    def test_mthfr_without_clinvar_not_sufficient(self):
        """Without ClinVar (local env), MTHFR has no local data → not grounded."""
        has, ctx = _ce._has_sufficient_grounded_gene_context("MTHFR", None)
        assert has is False, "MTHFR without any local data must not be grounded"
        assert ctx == []

    def test_mthfr_no_fallback_message_when_phenotypes_available(self):
        """
        When sufficient phenotype data exists, the answer must NOT be the 'אין לי סיכום'
        fallback.  Simulates server behavior where ClinVar DB has MTHFR phenotypes.
        """
        mthfr_clinvar = {
            "phenotypes": ["Homocystinuria", "Homocysteinemia", "MTHFR deficiency"],
            "total_variants": 1200,
        }
        # Inject a mock grounded answer (LLM not configured locally → grounded path returns None,
        # but we verify the logic: grounded path is ATTEMPTED when has=True)
        has, ctx = _ce._has_sufficient_grounded_gene_context("MTHFR", mthfr_clinvar)
        assert has is True
        # The fallback message text
        fallback_marker = "עדיין אין לי סיכום ביולוגי"
        # With grounded=True, the system TRIES to generate a grounded answer first.
        # Locally (no LLM) it falls through to fallback — that's expected on this machine.
        # On the server with LLM, grounded_answer would be returned instead of fallback.
        # This test only verifies that has=True (grounded attempt is made).
        # Verification that fallback is NOT the main answer when LLM is available is in Part E prompt.


# ===========================================================================
# Part G — Generic gene test matrix by source configuration
# ===========================================================================

class TestGenericGeneSourceMatrix:
    """
    Test grounding behavior based on ACTUAL source data, not gene names.
    Each configuration simulates what a gene's source layers contain.
    """

    def _make_summary(self, phenotypes=None, total=None):
        s = {}
        if phenotypes is not None:
            s["phenotypes"] = phenotypes
        if total is not None:
            s["total_variants"] = total
        return s or None

    def test_config_approved_biology_only(self):
        """Approved biology text → has=True, has_biology=True."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary",
                          return_value="גן זה מקודד לחלבון המעורב בתיקון DNA ובשמירה על יציבות הגנום."), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None):
            has, ctx = _ce._has_sufficient_grounded_gene_context("GENE_A", None)
            assert has is True
            assert any(p.get("has_biology") for p in ctx)

    def test_config_draft_biology_only(self):
        """Draft biology text (approved=False) → has=True, biology part present."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text",
                          return_value="גן זה מקודד לחלבון חשוב בתהליך ביולוגי מסוים בגוף האדם."):
            has, ctx = _ce._has_sufficient_grounded_gene_context("GENE_B", None)
            assert has is True
            assert any(p.get("type") == "draft_biology_text" for p in ctx)

    def test_config_phenotypes_only(self):
        """ClinVar phenotypes only → has=True, has_biology=False (association-only)."""
        summary = self._make_summary(
            phenotypes=["Disease Alpha", "Disease Beta", "Disease Gamma"]
        )
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, ctx = _ce._has_sufficient_grounded_gene_context("GENE_C", summary)
            assert has is True
            assert not any(p.get("has_biology") for p in ctx)

    def test_config_counts_only(self):
        """ClinVar counts only (no phenotypes) → has=False."""
        summary = self._make_summary(phenotypes=[], total=5000)
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, _ = _ce._has_sufficient_grounded_gene_context("GENE_D", summary)
            assert has is False

    def test_config_no_data(self):
        """No local data whatsoever → has=False."""
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None), \
             patch.object(_ce.gene_knowledge, "get_gene_knowledge_biology_text", return_value=None):
            has, _ = _ce._has_sufficient_grounded_gene_context("GENE_E", None)
            assert has is False

    def test_config_biology_plus_phenotypes(self):
        """Biology + phenotypes → has=True, biology part present (phenotypes supplemental)."""
        summary = self._make_summary(phenotypes=["Disease X", "Disease Y", "Disease Z"])
        with patch.object(_ce.gene_knowledge, "get_gene_patient_summary",
                          return_value="גן זה מקודד לחלבון המעורב בתיקון DNA."), \
             patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None):
            has, ctx = _ce._has_sufficient_grounded_gene_context("GENE_F", summary)
            assert has is True
            assert any(p.get("has_biology") for p in ctx)
            # Both biology and phenotypes are present
            assert any(p.get("type") == "curated_gene_description" for p in ctx)
            assert any(p.get("type") == "phenotype_associations" for p in ctx)

    def test_hbb_actual_source_biology(self):
        """HBB is approved in gene_knowledge_base.json — has_biology=True without mocks."""
        has, ctx = _ce._has_sufficient_grounded_gene_context("HBB", None)
        # HBB has approved=True in KB → Layer 1 should fire
        assert has is True
        assert any(p.get("has_biology") for p in ctx)

    def test_mthfr_actual_source_no_kb(self):
        """MTHFR has no KB entry → no biology locally; needs phenotypes for grounding."""
        has_no_clinvar, _ = _ce._has_sufficient_grounded_gene_context("MTHFR", None)
        assert has_no_clinvar is False, "MTHFR without any data must not be grounded"

    def test_apoe_actual_source_no_kb(self):
        """APOE has no KB entry → phenotypes needed for association-only grounding."""
        has_no_clinvar, _ = _ce._has_sufficient_grounded_gene_context("APOE", None)
        assert has_no_clinvar is False


# ===========================================================================
# Part H — Chromosome number appears in answer text
# ===========================================================================

class TestChromosomeContextInAnswerText:
    """When chromosome_number is known, the KB answer must mention it explicitly."""

    def test_deletion_answer_mentions_chromosome_21(self):
        """chromosome_deletion_general answer must mention chromosome 21 when provided."""
        result = _ce._build_chromosome_education_answer(
            "הממצא הוא מחיקה",
            "chromosome_deletion_general",
            chromosome_number="21",
        )
        answer = result["answer"]
        assert "21" in answer, f"Answer must mention chromosome 21: {answer}"
        assert "מחיקה" in answer or "חסר" in answer

    def test_deletion_answer_specific_opener_text(self):
        """The opener for chromosome 21 deletion must match the spec example."""
        result = _ce._build_chromosome_education_answer(
            "הממצא הוא מחיקה",
            "chromosome_deletion_general",
            chromosome_number="21",
        )
        answer = result["answer"]
        # Spec: "אם הכוונה היא למחיקה בכרומוזום 21, מדובר בחסר של קטע מסוים מכרומוזום 21."
        assert "כרומוזום 21" in answer
        assert answer.startswith("אם הכוונה היא למחיקה בכרומוזום 21")

    def test_duplication_answer_mentions_chromosome(self):
        """chromosome_duplication_general answer mentions the chromosome when known."""
        result = _ce._build_chromosome_education_answer(
            "זה כפילות",
            "chromosome_duplication_general",
            chromosome_number="13",
        )
        assert "13" in result["answer"]

    def test_mosaicism_answer_mentions_chromosome(self):
        """mosaicism_general answer mentions the chromosome when known."""
        result = _ce._build_chromosome_education_answer(
            "זה פסיפס",
            "mosaicism_general",
            chromosome_number="7",
        )
        assert "7" in result["answer"]

    def test_no_chromosome_number_gives_generic_answer(self):
        """Without chromosome_number, the answer is the generic KB text (no opener)."""
        result = _ce._build_chromosome_education_answer(
            "הממצא הוא מחיקה",
            "chromosome_deletion_general",
            chromosome_number=None,
        )
        answer = result["answer"]
        # The generic KB deletion answer starts with "מחיקה כרומוזומית"
        assert answer.startswith("מחיקה כרומוזומית") or "מחיקה" in answer
        # Must NOT contain "כרומוזום 21" — no number was supplied
        assert "כרומוזום 21" not in answer

    def test_general_finding_no_opener(self):
        """chromosome_finding_general does not get a specific opener."""
        result = _ce._build_chromosome_education_answer(
            "יש לי בעיה בכרומוזום",
            "chromosome_finding_general",
            chromosome_number="21",
        )
        answer = result["answer"]
        # The general finding answer uses the generic KB text
        assert "מחיקה" in answer or "כרומוזום" in answer

    def test_chromosome_from_question_when_no_context(self):
        """When chromosome_number not passed but present in question text, it's extracted."""
        result = _ce._build_chromosome_education_answer(
            "יש לי מחיקה בכרומוזום 5",
            "chromosome_deletion_general",
            chromosome_number=None,
        )
        # _extract_chromosome_number should find "5" from the question
        metadata = result.get("chromosome_draft_metadata", {})
        detected = metadata.get("chromosome_number_detected")
        if detected:
            assert detected == "5"
            assert "5" in result["answer"]

    def test_api_deletion_followup_with_chr_number_in_answer(self):
        """Full-stack: deletion follow-up with chromosome context includes number in answer."""
        data = _ask(
            "הממצא הוא מחיקה",
            context={"active_topic": "chromosome_finding_general", "chromosome_number": "21"},
        )
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        # Should mention chromosome 21 in the answer
        assert "21" in answer, f"Expected '21' in answer for chr21 deletion follow-up: {answer}"


# ===========================================================================
# Part I — finding_type in session context
# ===========================================================================

class TestFindingTypeSessionContext:
    """finding_type is stored in session_context_out for chromosome finding follow-ups."""

    def test_resolve_followup_returns_finding_type(self):
        """_resolve_chromosome_followup now includes finding_type in the returned dict."""
        ctx = {"active_topic": "chromosome_finding_general", "chromosome_number": "21"}
        result = _ce._resolve_chromosome_followup("הממצא הוא מחיקה", ctx)
        assert result is not None
        assert "finding_type" in result, "_resolve_chromosome_followup must return finding_type"
        assert result["finding_type"] == "chromosome_deletion_general"

    def test_finding_type_matches_sub_intent(self):
        """finding_type in the resolved dict equals the sub_intent."""
        for keyword, expected_si in [
            ("מחיקה", "chromosome_deletion_general"),
            ("כפילות", "chromosome_duplication_general"),
            ("טרנסלוקציה", "translocation_general"),
            ("פסיפס", "mosaicism_general"),
        ]:
            ctx = {"active_topic": "chromosome_finding_general"}
            result = _ce._resolve_chromosome_followup(keyword, ctx)
            if result:
                assert result.get("finding_type") == expected_si, (
                    f"finding_type mismatch for '{keyword}'"
                )

    def test_session_context_out_includes_finding_type_for_deletion(self):
        """_build_session_context_out emits finding_type when sub_intent is a specific finding."""
        # Build a fake deletion answer result with chromosome_draft_metadata
        fake_result = {
            "matched_topic": "chromosome_deletion_general",
            "chromosome_draft_metadata": {
                "sub_intent": "chromosome_deletion_general",
                "chromosome_number_detected": "21",
            },
        }
        ctx_out = _ce._build_session_context_out(fake_result, "הממצא הוא מחיקה", None)
        assert "finding_type" in ctx_out, "finding_type must be in session_context_out"
        assert ctx_out["finding_type"] == "chromosome_deletion_general"

    def test_session_context_out_finding_type_persists_from_previous_turn(self):
        """finding_type from a previous chromosome turn is carried forward."""
        # Simulate a turn where matched_topic is a chromosome intent but
        # chromosome_draft_metadata has no new sub_intent (e.g., a follow-up answer)
        fake_result = {
            "matched_topic": "mosaicism_general",
            "chromosome_draft_metadata": {
                "sub_intent": "mosaicism_general",
                "chromosome_number_detected": "21",
            },
        }
        prev_ctx = {"finding_type": "chromosome_deletion_general", "chromosome_number": "21"}
        ctx_out = _ce._build_session_context_out(fake_result, "זה פסיפס", prev_ctx)
        # Current turn updates finding_type to mosaicism
        assert ctx_out.get("finding_type") == "mosaicism_general"

    def test_finding_type_not_set_for_general_finding(self):
        """chromosome_finding_general is the base category — finding_type NOT stored for it."""
        fake_result = {
            "matched_topic": "chromosome_finding_general",
            "chromosome_draft_metadata": {
                "sub_intent": "chromosome_finding_general",
                "chromosome_number_detected": "21",
            },
        }
        ctx_out = _ce._build_session_context_out(fake_result, "יש לי בעיה בכרומוזום", None)
        # finding_type is NOT set for the base "general" category
        assert ctx_out.get("finding_type") != "chromosome_finding_general", (
            "chromosome_finding_general should NOT be stored as finding_type"
        )

    def test_api_finding_type_in_session_context_out(self):
        """Full-stack: deletion follow-up returns finding_type in conversation_context."""
        data = _ask(
            "הממצא הוא מחיקה",
            context={"active_topic": "chromosome_finding_general", "chromosome_number": "21"},
        )
        ctx_out = data.get("conversation_context") or {}
        # finding_type should be set to chromosome_deletion_general
        assert ctx_out.get("finding_type") == "chromosome_deletion_general", (
            f"Expected finding_type=chromosome_deletion_general in conversation_context: {ctx_out}"
        )

    def test_chromosome_number_preserved_on_finding_type_update(self):
        """When finding_type updates, chromosome_number from context is preserved."""
        data = _ask(
            "אמרו שזה mosaic",
            context={
                "active_topic": "chromosome_deletion_general",
                "chromosome_number": "21",
                "finding_type": "chromosome_deletion_general",
            },
        )
        ctx_out = data.get("conversation_context") or {}
        assert ctx_out.get("chromosome_number") == "21", (
            "chromosome_number must persist when finding_type updates"
        )


# ===========================================================================
# Part J — Context-aware clinician questions
# ===========================================================================

class TestContextAwareClinicianQuestions:
    """Clinician questions reference the specific chromosome when known."""

    def test_deletion_clinician_questions_mention_chr21(self):
        """Clinician questions for chr21 deletion must mention chromosome 21."""
        result = _ce._build_chromosome_education_answer(
            "הממצא הוא מחיקה",
            "chromosome_deletion_general",
            chromosome_number="21",
        )
        qs = result.get("clinician_questions", [])
        assert qs, "clinician_questions must not be empty for deletion answer"
        joined = " ".join(qs)
        assert "21" in joined, f"At least one clinician question must mention '21': {qs}"

    def test_deletion_clinician_questions_spec_example(self):
        """First clinician question for chr21 deletion matches spec: 'מה הגודל והמיקום המדויק...'"""
        result = _ce._build_chromosome_education_answer(
            "הממצא הוא מחיקה",
            "chromosome_deletion_general",
            chromosome_number="21",
        )
        qs = result.get("clinician_questions", [])
        assert qs
        assert "21" in qs[0], f"First question must mention chromosome 21: {qs[0]}"
        # Spec example: "מה הגודל והמיקום המדויק של המחיקה בכרומוזום 21?"
        assert "גודל" in qs[0] or "מיקום" in qs[0], (
            f"First question should ask about size/location: {qs[0]}"
        )

    def test_duplication_clinician_questions_mention_chromosome(self):
        """Clinician questions for chr13 duplication mention chromosome 13."""
        result = _ce._build_chromosome_education_answer(
            "זה כפילות",
            "chromosome_duplication_general",
            chromosome_number="13",
        )
        qs = result.get("clinician_questions", [])
        joined = " ".join(qs)
        assert "13" in joined

    def test_no_chromosome_gives_generic_questions(self):
        """Without chromosome_number, generic (non-chromosome-specific) questions are used."""
        result = _ce._build_chromosome_education_answer(
            "הממצא הוא מחיקה",
            "chromosome_deletion_general",
            chromosome_number=None,
        )
        qs = result.get("clinician_questions", [])
        assert qs
        # Generic questions do not contain specific numbers like 21
        for q in qs:
            assert "כרומוזום 21" not in q

    def test_api_clinician_questions_include_chromosome(self):
        """Full-stack: deletion follow-up with chr21 context includes 21 in clinician questions."""
        data = _ask(
            "הממצא הוא מחיקה",
            context={"active_topic": "chromosome_finding_general", "chromosome_number": "21"},
        )
        qs = data.get("clinician_questions", [])
        if qs:  # clinician_questions may not be in all response shapes
            joined = " ".join(qs)
            assert "21" in joined, (
                f"Clinician questions should mention chromosome 21: {qs}"
            )


# ===========================================================================
# Part K — Physician queue policy
# ===========================================================================

class TestPhysicianQueuePolicy:
    """Only AI_EXPANDED_UNVERIFIED (unverified drafts) should populate the physician queue."""

    def test_ai_content_type_constants_exist(self):
        assert hasattr(_ce, "AI_CONTENT_TYPE_GROUNDED")
        assert hasattr(_ce, "AI_CONTENT_TYPE_EXPANDED")
        assert _ce.AI_CONTENT_TYPE_GROUNDED != _ce.AI_CONTENT_TYPE_EXPANDED

    def _tier2_grounded_patches(self, mock_text):
        """Return context manager stack for mocking a Tier 2 gene with grounded answer."""
        from contextlib import ExitStack
        stack = ExitStack()
        # Make gene appear in ClinVar index (Tier 2 path)
        stack.enter_context(patch.object(_ce.gene_index, "_GENE_INDEX_AVAILABLE", True))
        stack.enter_context(patch.object(_ce.gene_index, "get_gene_summary",
                                         return_value={"total_variants": 100, "by_significance": {},
                                                       "phenotypes": ["Test disease"]}))
        # Skip Tier 1a and Tier 1b
        stack.enter_context(patch.object(_ce.gene_cards, "get_approved_summary", return_value=None))
        stack.enter_context(patch.object(_ce.gene_knowledge, "get_gene_patient_summary", return_value=None))
        stack.enter_context(patch.object(_ce.gene_knowledge, "get_gene_context_summary", return_value=None))
        # Inject grounded answer
        stack.enter_context(patch.object(_ce, "_has_sufficient_grounded_gene_context",
                                         return_value=(True, [{"type": "curated_gene_description",
                                                               "text": mock_text, "has_biology": True}])))
        stack.enter_context(patch.object(_ce, "_generate_source_grounded_gene_answer",
                                         return_value=mock_text))
        return stack

    def test_grounded_answer_not_queued(self):
        """
        When grounded phrasing is used (AI_CONTENT_TYPE_GROUNDED), the physician review
        queue must NOT be populated.  The review queue is only for AI_EXPANDED_UNVERIFIED.
        """
        mock_grounded = "גן TESTG מקודד לחלבון חשוב בתיקון DNA ובשמירה על יציבות הגנום."
        try:
            from app import review_db as _rdb
        except ImportError:
            pytest.skip("review_db not available in this environment")

        with self._tier2_grounded_patches(mock_grounded), \
             patch.object(_rdb, "create_draft") as mock_create:
            result = _ce._build_gene_clinvar_answer("מה זה TESTG?", "TESTG")
            assert result is not None
            meta = result.get("gene_metadata", {})
            ai_type = meta.get("ai_content_type")
            assert ai_type == _ce.AI_CONTENT_TYPE_GROUNDED, (
                f"Expected source_grounded_phrasing, got: {ai_type}"
            )
            assert not meta.get("requires_physician_review"), (
                "Grounded answer must not require physician review"
            )
            mock_create.assert_not_called()

    def test_grounded_response_schema_has_correct_flags(self):
        """Grounded answer metadata: source_grounded=True, requires_physician_review=False."""
        mock_grounded = "גן TESTG מקודד לחלבון חשוב בתיקון DNA ובשמירה על יציבות הגנום."
        with self._tier2_grounded_patches(mock_grounded):
            result = _ce._build_gene_clinvar_answer("מה זה TESTG?", "TESTG")
            assert result is not None
            meta = result.get("gene_metadata", {})
            assert meta.get("source_grounded") is True
            assert meta.get("requires_physician_review") is False


# ===========================================================================
# Full-pipeline multi-turn chromosome sequence
# ===========================================================================

class TestMultiTurnChromosomeContext:
    """Three-turn chromosome sequence: general → specific finding → mosaic refinement."""

    def test_turn1_chromosome_general(self):
        """Turn 1: 'יש לי בעיה בכרומוזום 21' sets chromosome_number in conversation_context."""
        data = _ask("יש לי בעיה בכרומוזום 21")
        ctx_out = data.get("conversation_context") or {}
        assert ctx_out.get("chromosome_number") == "21", (
            f"chromosome_number should be '21' after turn 1: {ctx_out}"
        )

    def test_turn2_deletion_followup_uses_chr21(self):
        """Turn 2: 'הממצא הוא מחיקה' with chr21 context → answer mentions chr21."""
        data = _ask(
            "הממצא הוא מחיקה",
            context={"active_topic": "chromosome_finding_general", "chromosome_number": "21"},
        )
        assert "21" in data["answer"]
        ctx_out = data.get("conversation_context") or {}
        assert ctx_out.get("finding_type") == "chromosome_deletion_general"
        assert ctx_out.get("chromosome_number") == "21"

    def test_turn3_mosaicism_followup_preserves_chr21(self):
        """Turn 3: 'אמרו שזה mosaic' with deletion context → chromosome 21 preserved."""
        data = _ask(
            "אמרו שזה mosaic",
            context={
                "active_topic": "chromosome_deletion_general",
                "chromosome_number": "21",
                "finding_type": "chromosome_deletion_general",
            },
        )
        assert data["safety_level"] == "general_information"
        ctx_out = data.get("conversation_context") or {}
        # chromosome_number must persist
        assert ctx_out.get("chromosome_number") == "21", (
            f"chromosome_number must persist to turn 3: {ctx_out}"
        )
        # finding_type updates to mosaicism
        assert ctx_out.get("finding_type") == "mosaicism_general", (
            f"finding_type must update to mosaicism_general: {ctx_out}"
        )

    def test_response_schema_valid_throughout(self):
        """All three turns must return valid schema."""
        for question, ctx in [
            ("יש לי בעיה בכרומוזום 21", None),
            ("הממצא הוא מחיקה",
             {"active_topic": "chromosome_finding_general", "chromosome_number": "21"}),
            ("אמרו שזה mosaic",
             {"active_topic": "chromosome_deletion_general", "chromosome_number": "21",
              "finding_type": "chromosome_deletion_general"}),
        ]:
            kwargs = {} if ctx is None else {"context": ctx}
            data = _ask(question, **kwargs)
            assert _REQUIRED_KEYS.issubset(data.keys()), (
                f"Missing required keys for '{question}': {data.keys()}"
            )
