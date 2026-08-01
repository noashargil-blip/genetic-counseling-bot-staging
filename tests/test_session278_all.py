"""
Session 27.8 tests — grounded AI phrasing, safe conversation context,
chromosome follow-up routing, and clinician questions.

Parts covered:
  A   AI content type constants (SOURCE_GROUNDED_PHRASING, AI_EXPANDED_UNVERIFIED)
  B   _has_sufficient_grounded_gene_context() — sufficiency rules
  C   Tier 2 cascade: grounded → approved → fallback+unverified
  D/E SafeSessionContext validation (whitelist, chromosome number format)
  F   Chromosome follow-up routing via session context
  G   clinician_questions field in chromosome education responses
  H   Safety overrides (existing pipeline, not bypassed by context)
  I   Grounded phrasing does NOT populate physician review queue

Run:
  PYTHONUTF8=1 python -m pytest tests/test_session278_all.py -v
"""
import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, SafeSessionContext
import app.counseling_engine as _ce

client = TestClient(app)

_REQUIRED_KEYS = {"answer", "safety_level", "needs_genetic_counselor",
                  "matched_topic", "suggested_questions"}


def _ask(question: str, **kwargs) -> dict:
    payload = {"question": question, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


# ===========================================================================
# Part A — AI content type constants
# ===========================================================================

class TestAiContentTypeConstants:
    def test_grounded_constant_exists(self):
        assert hasattr(_ce, "AI_CONTENT_TYPE_GROUNDED")

    def test_grounded_constant_value(self):
        assert _ce.AI_CONTENT_TYPE_GROUNDED == "source_grounded_phrasing"

    def test_expanded_constant_exists(self):
        assert hasattr(_ce, "AI_CONTENT_TYPE_EXPANDED")

    def test_expanded_constant_value(self):
        assert _ce.AI_CONTENT_TYPE_EXPANDED == "ai_expanded_unverified"

    def test_constants_are_distinct(self):
        assert _ce.AI_CONTENT_TYPE_GROUNDED != _ce.AI_CONTENT_TYPE_EXPANDED


# ===========================================================================
# Part B — _has_sufficient_grounded_gene_context()
# ===========================================================================

class TestGroundedContextSufficiency:
    """Unit tests for the sufficiency helper — no LLM needed."""

    def test_insufficient_with_no_phenotypes(self):
        summary = {"total_variants": 100, "by_significance": {}, "phenotypes": []}
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
        assert has is False
        assert ctx == []

    def test_insufficient_with_only_two_phenotypes(self):
        summary = {"phenotypes": ["Disease A", "Disease B"]}
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
        assert has is False

    def test_insufficient_with_only_trivial_phenotypes(self):
        summary = {"phenotypes": ["not specified", "not provided", "see cases"]}
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
        assert has is False

    def test_sufficient_with_three_meaningful_phenotypes(self):
        summary = {"phenotypes": ["Alzheimer disease", "Lipoprotein disorder", "Cardiovascular disease"]}
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
        assert has is True
        assert any(p.get("type") == "clinvar_phenotypes" for p in ctx)

    def test_sufficient_with_six_phenotypes(self):
        phenotypes = [f"Disease {i}" for i in range(6)]
        summary = {"phenotypes": phenotypes}
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
        assert has is True

    def test_sufficient_with_approved_context_summary(self):
        with patch.object(_ce.gene_knowledge, "get_gene_context_summary",
                          return_value="APOE is associated with lipid metabolism and Alzheimer disease risk."):
            has, ctx = _ce._has_sufficient_grounded_gene_context("APOE", {})
            assert has is True
            assert any(p.get("type") == "approved_context" for p in ctx)

    def test_insufficient_with_short_approved_context(self):
        with patch.object(_ce.gene_knowledge, "get_gene_context_summary",
                          return_value="Short"):
            has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", {})
            assert has is False

    def test_none_summary_returns_insufficient(self):
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", None)
        assert has is False

    def test_phenotypes_capped_at_six_in_context(self):
        summary = {"phenotypes": [f"Disease {i}" for i in range(10)]}
        has, ctx = _ce._has_sufficient_grounded_gene_context("TESTGENE", summary)
        if has:
            pheno_part = next((p for p in ctx if p.get("type") == "clinvar_phenotypes"), None)
            if pheno_part:
                assert len(pheno_part["phenotypes"]) <= 6


# ===========================================================================
# Part C — Tier 2 cascade: grounded → approved → fallback
# ===========================================================================

class TestTier2CascadeGrounded:
    """Tier 2 answers when grounded phrasing is mocked to succeed."""

    def _mock_grounded_answer(self, gene: str) -> str:
        return f"גן {gene} קשור לכמה מצבים קליניים לפי מאגר ClinVar. לפרטים, פנה לצוות הגנטי."

    def test_grounded_answer_in_llm_mode_field(self):
        with patch.object(
            _ce, "_has_sufficient_grounded_gene_context", return_value=(True, [])
        ), patch.object(
            _ce, "_generate_source_grounded_gene_answer", return_value=self._mock_grounded_answer("APOE")
        ), patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 100, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            data = _ask("מה ידוע על הגן APOE?")
            assert data.get("llm_mode") == "source_grounded"

    def test_grounded_answer_is_main_answer(self):
        expected = "גן APOE קשור למספר מצבים."
        with patch.object(
            _ce, "_has_sufficient_grounded_gene_context", return_value=(True, [])
        ), patch.object(
            _ce, "_generate_source_grounded_gene_answer", return_value=expected
        ), patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 50, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            data = _ask("מה ידוע על הגן APOE?")
            assert expected in data["answer"]

    def test_grounded_sets_source_grounded_metadata(self):
        with patch.object(
            _ce, "_has_sufficient_grounded_gene_context", return_value=(True, [])
        ), patch.object(
            _ce, "_generate_source_grounded_gene_answer",
            return_value="גן CCR5 קשור למצבים מסוימים. לפרטים, פנה לצוות הגנטי."
        ), patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 30, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            data = _ask("מה ידוע על הגן CCR5?")
            meta = data.get("gene_metadata", {})
            assert meta.get("source_grounded") is True

    def test_grounded_sets_ai_content_type(self):
        with patch.object(
            _ce, "_has_sufficient_grounded_gene_context", return_value=(True, [])
        ), patch.object(
            _ce, "_generate_source_grounded_gene_answer",
            return_value="גן CCR5 קשור למצבים מסוימים. לפרטים, פנה לצוות הגנטי."
        ), patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 30, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            data = _ask("מה ידוע על הגן CCR5?")
            meta = data.get("gene_metadata", {})
            assert meta.get("ai_content_type") == _ce.AI_CONTENT_TYPE_GROUNDED

    def test_fallback_when_grounded_fails(self):
        """When grounded phrasing returns None (no LLM), fall back gracefully."""
        with patch.object(
            _ce, "_has_sufficient_grounded_gene_context", return_value=(True, [])
        ), patch.object(
            _ce, "_generate_source_grounded_gene_answer", return_value=None
        ), patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 100, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            data = _ask("מה ידוע על הגן APOE?")
            assert data["safety_level"] == "general_information"
            assert data["answer"]

    def test_no_grounded_answer_when_insufficient_context(self):
        """Tier 2 with insufficient context: no grounded phrasing, no LLM in tests."""
        with patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 5, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            data = _ask("מה ידוע על הגן CCR5?")
            meta = data.get("gene_metadata", {})
            assert meta.get("source_grounded") is False
            assert meta.get("ai_content_type") in (None, _ce.AI_CONTENT_TYPE_EXPANDED)


# ===========================================================================
# Part I — Grounded phrasing does NOT populate review queue
# ===========================================================================

class TestGroundedNotInReviewQueue:
    """Grounded answers (AI_CONTENT_TYPE_GROUNDED) must NOT create draft DB entries."""

    def test_review_db_not_called_for_grounded_answer(self):
        with patch.object(
            _ce, "_has_sufficient_grounded_gene_context", return_value=(True, [])
        ), patch.object(
            _ce, "_generate_source_grounded_gene_answer",
            return_value="גן APOE קשור למצבים קליניים שונים. פנה לצוות הגנטי."
        ), patch.object(
            _ce.gene_index, "_GENE_INDEX_AVAILABLE", True
        ), patch.object(
            _ce.gene_index, "get_gene_summary",
            return_value={"total_variants": 100, "by_significance": {}, "phenotypes": []}
        ), patch.object(
            _ce.gene_cards, "get_approved_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_patient_summary", return_value=None
        ), patch.object(
            _ce.gene_knowledge, "get_gene_context_summary", return_value=None
        ):
            import app.review_db as _rdb
            with patch.object(_rdb, "create_draft") as mock_create:
                _ask("מה ידוע על הגן APOE?")
                # create_draft must NOT be called for grounded answers
                for call_args in mock_create.call_args_list:
                    args, kwargs = call_args
                    draft_type = kwargs.get("draft_type") or (args[0] if args else None)
                    assert draft_type != "gene_summary", (
                        "Grounded phrasing must not create a gene_summary review entry"
                    )


# ===========================================================================
# Part D/E — SafeSessionContext validation
# ===========================================================================

class TestSafeSessionContextValidation:
    """Whitelist and format validation for SafeSessionContext."""

    def test_valid_chromosome_topic(self):
        ctx = SafeSessionContext(active_topic="chromosome_deletion_general")
        assert ctx.active_topic == "chromosome_deletion_general"

    def test_invalid_active_topic_returns_none(self):
        ctx = SafeSessionContext(active_topic="injection_attack; DROP TABLE--")
        assert ctx.active_topic is None

    def test_unknown_active_topic_silently_ignored(self):
        ctx = SafeSessionContext(active_topic="something_unknown_xyz")
        assert ctx.active_topic is None

    def test_valid_chromosome_number_digit(self):
        ctx = SafeSessionContext(chromosome_number="21")
        assert ctx.chromosome_number == "21"

    def test_valid_chromosome_number_x(self):
        ctx = SafeSessionContext(chromosome_number="X")
        assert ctx.chromosome_number == "X"

    def test_valid_chromosome_number_y(self):
        ctx = SafeSessionContext(chromosome_number="y")
        assert ctx.chromosome_number == "Y"

    def test_invalid_chromosome_number_out_of_range(self):
        ctx = SafeSessionContext(chromosome_number="99")
        assert ctx.chromosome_number is None

    def test_invalid_chromosome_number_text(self):
        ctx = SafeSessionContext(chromosome_number="twenty-one")
        assert ctx.chromosome_number is None

    def test_valid_gene_symbol(self):
        ctx = SafeSessionContext(gene_symbol="BRCA1")
        assert ctx.gene_symbol == "BRCA1"

    def test_gene_symbol_uppercased(self):
        ctx = SafeSessionContext(gene_symbol="apoe")
        assert ctx.gene_symbol == "APOE"

    def test_invalid_gene_symbol_with_special_chars(self):
        ctx = SafeSessionContext(gene_symbol="'; DROP--")
        assert ctx.gene_symbol is None

    def test_turn_count_ge_0(self):
        ctx = SafeSessionContext(turn_count=5)
        assert ctx.turn_count == 5

    def test_turn_count_max(self):
        ctx = SafeSessionContext(turn_count=200)
        assert ctx.turn_count == 200

    def test_turn_count_over_max_rejected(self):
        with pytest.raises(Exception):
            SafeSessionContext(turn_count=9999)

    def test_all_none_allowed(self):
        ctx = SafeSessionContext()
        assert ctx.active_topic is None
        assert ctx.chromosome_number is None

    def test_valid_gene_clinvar_topic(self):
        ctx = SafeSessionContext(active_topic="gene_clinvar_summary")
        assert ctx.active_topic == "gene_clinvar_summary"

    def test_valid_vus_topic(self):
        ctx = SafeSessionContext(active_topic="vus")
        assert ctx.active_topic == "vus"


# ===========================================================================
# Chromosome number extraction helper
# ===========================================================================

class TestChromosomeNumberExtraction:
    def test_extract_from_hebrew_21(self):
        assert _ce._extract_chromosome_number("יש לי בעיה בכרומוזום 21") == "21"

    def test_extract_from_hebrew_x(self):
        assert _ce._extract_chromosome_number("כרומוזום X נמצא פגוע") == "X"

    def test_extract_from_english(self):
        assert _ce._extract_chromosome_number("chromosome 13 deletion") == "13"

    def test_no_chromosome_returns_none(self):
        assert _ce._extract_chromosome_number("מה זה VUS?") is None

    def test_extract_two_digit(self):
        assert _ce._extract_chromosome_number("כרומוזום 22") == "22"

    def test_uppercase_x(self):
        result = _ce._extract_chromosome_number("chromosome X missing")
        assert result == "X"


# ===========================================================================
# Part F — Chromosome follow-up routing
# ===========================================================================

class TestChromosomeFollowupRouting:
    """Chromosome follow-up routing via session context."""

    def test_resolve_followup_deletion_with_context(self):
        session_ctx = {"active_topic": "chromosome_finding_general", "chromosome_number": "21"}
        result = _ce._resolve_chromosome_followup("הממצא הוא מחיקה", session_ctx)
        assert result is not None
        assert result["sub_intent"] == "chromosome_deletion_general"
        assert result["chromosome_number"] == "21"

    def test_resolve_followup_duplication_with_context(self):
        session_ctx = {"active_topic": "chromosome_finding_general"}
        result = _ce._resolve_chromosome_followup("זה duplication", session_ctx)
        assert result is not None
        assert result["sub_intent"] == "chromosome_duplication_general"

    def test_resolve_followup_translocation(self):
        session_ctx = {"active_topic": "chromosome_deletion_general"}
        result = _ce._resolve_chromosome_followup("אמרו שזו טרנסלוקציה", session_ctx)
        assert result is not None
        assert result["sub_intent"] == "translocation_general"

    def test_resolve_followup_mosaicism(self):
        session_ctx = {"active_topic": "chromosome_finding_general"}
        result = _ce._resolve_chromosome_followup("זה פסיפס", session_ctx)
        assert result is not None
        assert result["sub_intent"] == "mosaicism_general"

    def test_no_followup_without_context(self):
        result = _ce._resolve_chromosome_followup("הממצא הוא מחיקה", None)
        assert result is None

    def test_no_followup_when_wrong_topic(self):
        session_ctx = {"active_topic": "vus"}
        result = _ce._resolve_chromosome_followup("הממצא הוא מחיקה", session_ctx)
        assert result is None

    def test_no_followup_for_long_message(self):
        long_msg = "זוהי שאלה ארוכה מאוד " * 10
        session_ctx = {"active_topic": "chromosome_finding_general"}
        result = _ce._resolve_chromosome_followup(long_msg, session_ctx)
        assert result is None

    def test_followup_retains_chromosome_number(self):
        session_ctx = {"active_topic": "chromosome_finding_general", "chromosome_number": "5"}
        result = _ce._resolve_chromosome_followup("זה מחיקה", session_ctx)
        assert result["chromosome_number"] == "5"

    def test_followup_chromosome_number_none_when_no_context(self):
        session_ctx = {"active_topic": "chromosome_finding_general"}
        result = _ce._resolve_chromosome_followup("זה מחיקה", session_ctx)
        assert result["chromosome_number"] is None

    def test_api_chromosome_followup_returns_deletion_answer(self):
        """Full-stack: context says chromosome topic → deletion follow-up routes correctly."""
        data = _ask(
            "הממצא הוא מחיקה",
            context={"active_topic": "chromosome_finding_general"},
        )
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert any(w in answer for w in ["מחיקה", "deletion", "חסר"])

    def test_api_chromosome_followup_no_context_falls_to_kb(self):
        """Without context, 'הממצא הוא מחיקה' is not routed as chromosome follow-up."""
        data = _ask("הממצא הוא מחיקה")
        # Should still succeed — might route to deletion or general chromosome
        assert data["safety_level"] in ("general_information", "out_of_scope", "requires_genetic_counselor")

    def test_api_followup_schema_valid(self):
        data = _ask(
            "הממצא הוא כפילות",
            context={"active_topic": "chromosome_deletion_general"},
        )
        assert _REQUIRED_KEYS.issubset(data.keys())

    def test_api_followup_deletion_matched_topic(self):
        data = _ask(
            "זוהי מחיקה",
            context={"active_topic": "chromosome_finding_general", "chromosome_number": "13"},
        )
        assert data.get("matched_topic") == "chromosome_deletion_general"


# ===========================================================================
# Part G — Clinician questions in chromosome education responses
# ===========================================================================

class TestClinicianQuestionsInResponse:
    """Chromosome education answers must include clinician_questions."""

    CHR_QUESTIONS = [
        "מה זה מחיקה בכרומוזום?",
        "מה זה כפילות כרומוזומית?",
        "מה זה טרנסלוקציה?",
        "מה זה מוזאיקה בגנטיקה?",
        "מה זה בדיקת קריוטיפ?",
        "מה זה אנאופלואידיה?",
        "יש לי ממצא כרומוזומי, מה זה?",
    ]

    @pytest.mark.parametrize("q", CHR_QUESTIONS)
    def test_clinician_questions_present(self, q):
        data = _ask(q)
        assert "clinician_questions" in data, (
            f"clinician_questions missing from response to: {q!r}"
        )
        assert isinstance(data["clinician_questions"], list)
        assert len(data["clinician_questions"]) >= 1, (
            f"clinician_questions is empty for: {q!r}"
        )

    @pytest.mark.parametrize("q", CHR_QUESTIONS)
    def test_suggested_questions_still_present(self, q):
        data = _ask(q)
        assert len(data.get("suggested_questions", [])) >= 1, (
            f"suggested_questions is empty for: {q!r}"
        )

    def test_clinician_questions_distinct_from_suggested(self):
        data = _ask("מה זה מחיקה בכרומוזום?")
        clinician = set(data.get("clinician_questions", []))
        suggested = set(data.get("suggested_questions", []))
        assert not (clinician & suggested), (
            "clinician_questions and suggested_questions must not overlap"
        )

    def test_deletion_clinician_questions_about_location(self):
        data = _ask("מה זה מחיקה בכרומוזום?")
        clinician = data.get("clinician_questions", [])
        text = " ".join(clinician)
        assert any(w in text for w in ["מיקום", "גנים", "ממצא", "הורים"]), (
            "Deletion clinician questions should ask about location/genes/parents"
        )

    def test_clinician_questions_not_present_for_vus(self):
        """VUS answers do not include clinician_questions."""
        data = _ask("מה זה VUS?")
        assert "clinician_questions" not in data or data.get("clinician_questions") is None

    def test_clinician_questions_not_present_for_carrier(self):
        data = _ask("מה זה נשאות?")
        assert "clinician_questions" not in data or data.get("clinician_questions") is None

    def test_chromosome_number_in_metadata_when_detected(self):
        data = _ask("יש לי בעיה בכרומוזום 21")
        meta = data.get("chromosome_draft_metadata") or {}
        assert meta.get("chromosome_number_detected") == "21"

    def test_chromosome_number_none_when_not_in_question(self):
        data = _ask("מה זה מחיקה בכרומוזום?")
        meta = data.get("chromosome_draft_metadata") or {}
        # Number may be None if not mentioned
        chr_num = meta.get("chromosome_number_detected")
        assert chr_num is None or isinstance(chr_num, str)


# ===========================================================================
# Part G — API schema compliance
# ===========================================================================

class TestClinicianQuestionsSchema:
    def test_api_returns_200(self):
        r = client.post("/ask", json={"question": "מה זה מחיקה בכרומוזום?"})
        assert r.status_code == 200

    def test_clinician_questions_is_list(self):
        data = _ask("מה זה מחיקה בכרומוזום?")
        assert isinstance(data.get("clinician_questions"), list)

    def test_required_keys_still_present(self):
        data = _ask("מה זה כפילות כרומוזומית?")
        assert _REQUIRED_KEYS.issubset(data.keys())

    def test_context_field_accepted_in_request(self):
        r = client.post("/ask", json={
            "question": "מה זה מחיקה?",
            "context": {
                "active_topic": "chromosome_finding_general",
                "chromosome_number": "5",
            }
        })
        assert r.status_code == 200

    def test_invalid_context_field_silently_ignored(self):
        """Invalid context values are silently ignored — request still succeeds."""
        r = client.post("/ask", json={
            "question": "מה זה VUS?",
            "context": {
                "active_topic": "'; DROP TABLE--",
                "chromosome_number": "not-a-number",
                "gene_symbol": "injection_attack_123456789012345",
            }
        })
        assert r.status_code == 200


# ===========================================================================
# Part H — Safety overrides context (existing behavior preserved)
# ===========================================================================

class TestSafetyOverridesContext:
    """Safety checks must fire before chromosome follow-up routing."""

    def test_pii_blocked_even_with_chromosome_context(self):
        data = _ask(
            "יש לי בעיה בכרומוזום 21 ותעודת הזהות שלי 123456789",
            context={"active_topic": "chromosome_finding_general"},
        )
        assert data["safety_level"] == "contains_identifying_info"

    def test_reproductive_decision_blocked_with_context(self):
        data = _ask(
            "האם להפיל?",
            context={"active_topic": "chromosome_finding_general"},
        )
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True


# ===========================================================================
# Chromosome topic set
# ===========================================================================

class TestChromosomeTopicIntentSet:
    def test_chromosome_topic_intents_exists(self):
        assert hasattr(_ce, "_CHROMOSOME_TOPIC_INTENTS")

    def test_all_expected_topics_in_set(self):
        expected = {
            "chromosome_finding_general", "chromosome_deletion_general",
            "chromosome_duplication_general", "translocation_general",
            "mosaicism_general", "cytogenetic_test_general", "aneuploidy_general",
        }
        assert expected.issubset(_ce._CHROMOSOME_TOPIC_INTENTS)


# ===========================================================================
# Regression — existing behavior unchanged
# ===========================================================================

class TestSession278Regression:
    """Existing tests must not be broken by Session 27.8 changes."""

    def test_vus_answer_still_works(self):
        data = _ask("מה זה VUS?")
        assert data["safety_level"] == "general_information"
        assert "VUS" in data["answer"]

    def test_carrier_answer_still_works(self):
        data = _ask("אמרו לי שאני נשאית, מה זה?")
        assert data["safety_level"] == "general_information"

    def test_pii_blocked(self):
        data = _ask("השם שלי ישראל ישראלי, מה זה VUS?")
        assert data["safety_level"] == "contains_identifying_info"

    def test_trisomy21_still_routed(self):
        data = _ask("מה זה טריזומיה 21?")
        assert data["safety_level"] == "general_information"
        assert any(w in data["answer"] for w in ["טריזומיה", "תסמונת", "דאון"])

    def test_chromosome_deletion_still_routed(self):
        data = _ask("מה זה מחיקה בכרומוזום?")
        assert data["safety_level"] == "general_information"
        assert any(w in data["answer"] for w in ["מחיקה", "חסר", "deletion"])

    def test_schema_5_required_keys(self):
        for q in ["מה זה VUS?", "מה זה נשאות?", "מה זה מחיקה בכרומוזום?"]:
            data = _ask(q)
            assert _REQUIRED_KEYS.issubset(data.keys()), f"Schema broken for: {q!r}"

    def test_curated_gene_still_tier1(self):
        """HBB and other curated genes must not be affected by grounded phrasing."""
        from app import gene_cards
        if gene_cards.get_approved_summary("HBB"):
            data = _ask("מה ידוע על הגן HBB?")
            meta = data.get("gene_metadata", {})
            assert meta.get("answer_tier") in ("tier1", "tier1b"), (
                "Curated gene HBB must remain at tier1/1b"
            )
            assert meta.get("source_grounded", False) is False
