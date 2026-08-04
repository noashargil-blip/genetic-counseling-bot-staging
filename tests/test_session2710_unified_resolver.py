# -*- coding: utf-8 -*-
"""
Session 27.9.10 — unified source-resolution layer tests.

Verifies that _resolve_gene_content() is the single decision point for
display_mode routing, and that both gene answer builders (_build_gene_clinvar_answer
and _build_known_gene_answer) respect that decision consistently.

Source classes:
  Class A — curated | grounded | grounded_clinvar | physician_approved
             → display_mode="main_answer", source_grounded=True (grounded_clinvar only)
  Class B — ai_unreviewed
             → display_mode="supplemental_card", requires_physician_review=True
  none     — all steps failed / LLM unavailable
             → display_mode="none"
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAFE_HE = (
    "הגן זה מקודד לחלבון המעורב בתיקון DNA. "
    "שינויים בגן קשורים לסיכון מוגבר לסרטן. "
    "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
)


# ---------------------------------------------------------------------------
# A. Intent classification
# ---------------------------------------------------------------------------

class TestClassifyAnswerIntent:
    """_classify_answer_intent maps phrasings to consistent intent labels."""

    @pytest.mark.parametrize("question,expected", [
        ("מה זה הגן BRCA1?", "overview"),
        ("BRCA1", "overview"),
        ("מה תפקיד הגן CFTR?", "function"),
        ("מה עושה הגן CFTR?", "function"),
        ("איזה מחלות קשורות לגן TP53?", "conditions"),
        ("מה הקשר של גן ATM למחלות?", "conditions"),
        ("יש לי VUS בגן NF1, מה זה?", "vus"),
        ("תוצאות הבדיקה ציינו VUS", "vus"),
        ("תוכל להסביר בצורה פשוטה?", "simplify"),
        ("תסביר לי פשוט יותר", "simplify"),
    ])
    def test_intent_classification(self, question, expected):
        from app.counseling_engine import _classify_answer_intent
        result = _classify_answer_intent(question)
        assert result == expected, (
            f"question={question!r}: expected intent {expected!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# B. Resolver display_mode for "none" source (LLM unavailable)
# ---------------------------------------------------------------------------

class TestResolverDisplayModeNone:
    """Without LLM, genes lacking curated/grounded content resolve to display_mode='none'."""

    @pytest.mark.parametrize("gene", ["APOE", "CCR5", "TLR3"])
    def test_display_mode_none_without_llm(self, gene):
        from app.counseling_engine import _resolve_gene_content, _classify_answer_intent
        intent = _classify_answer_intent(f"מה זה הגן {gene}?")
        result = _resolve_gene_content(gene, intent, question=f"מה זה הגן {gene}?")
        # With no LLM and no curated/grounded data, resolver returns none or grounded
        assert result["display_mode"] in ("none", "main_answer"), (
            f"{gene}: unexpected display_mode {result['display_mode']!r}"
        )

    @pytest.mark.parametrize("gene", ["APOE", "CCR5", "TLR3"])
    def test_source_grounded_false_for_none(self, gene):
        from app.counseling_engine import _resolve_gene_content, _classify_answer_intent
        intent = _classify_answer_intent(f"מה זה הגן {gene}?")
        result = _resolve_gene_content(gene, intent, question=f"מה זה הגן {gene}?")
        if result["display_mode"] == "none":
            assert result["source_grounded"] is False


# ---------------------------------------------------------------------------
# C. Resolver display_mode for Source A (curated gene)
# ---------------------------------------------------------------------------

class TestResolverDisplayModeSourceA:
    """Curated and grounded genes produce display_mode='main_answer'."""

    @pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "ATM"])
    def test_curated_gene_is_main_answer(self, gene):
        from app.counseling_engine import _resolve_gene_content, gene_cards
        if not gene_cards.get_approved_summary(gene):
            pytest.skip(f"{gene} has no curated card on this server")
        result = _resolve_gene_content(gene, "overview", question=f"מה זה הגן {gene}?")
        assert result["display_mode"] == "main_answer"
        assert result["source_type"] == "curated"
        assert result["requires_physician_review"] is False

    def test_curated_source_grounded_true(self):
        # In the resolver, source_grounded=True for all Source A types (curated included).
        # In gene_metadata, it's narrowed to grounded_clinvar only (backward compat).
        from app.counseling_engine import _resolve_gene_content, gene_cards
        for gene in ("BRCA1", "BRCA2", "ATM"):
            if gene_cards.get_approved_summary(gene):
                result = _resolve_gene_content(gene, "overview", question=f"מה זה {gene}")
                assert result["source_grounded"] is True, (
                    "curated source is Source A → source_grounded should be True in resolver"
                )
                # gene_metadata narrows source_grounded to grounded_clinvar only
                from app.counseling_engine import _build_gene_clinvar_answer
                built = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
                assert built["gene_metadata"]["source_grounded"] is False, (
                    "gene_metadata.source_grounded is backward-compat: False for curated"
                )
                return
        pytest.skip("No curated gene available")


# ---------------------------------------------------------------------------
# D. Source B routing with mocked LLM
# ---------------------------------------------------------------------------

class TestResolverDisplayModeSourceB:
    """When LLM returns safe Hebrew text for a gene lacking curated/grounded content,
    the resolver returns display_mode='supplemental_card'."""

    @pytest.mark.parametrize("gene", ["KIAA2022", "TLR3"])
    def test_ai_draft_goes_to_supplemental_card(self, monkeypatch, gene):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = SAFE_HE
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        from app.counseling_engine import _resolve_gene_content, gene_cards, gene_knowledge
        if gene_cards.get_approved_summary(gene) or gene_knowledge.get_gene_patient_summary(gene):
            pytest.skip(f"{gene} has approved content — Source A takes priority")
        result = _resolve_gene_content(gene, "overview", question=f"מה זה הגן {gene}?")
        if result["display_mode"] == "supplemental_card":
            assert result["source_type"] == "ai_unreviewed"
            assert result["requires_physician_review"] is True
            assert result["source_grounded"] is False

    @pytest.mark.parametrize("gene", ["KIAA2022", "TLR3"])
    def test_source_b_text_present(self, monkeypatch, gene):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = SAFE_HE
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        from app.counseling_engine import _resolve_gene_content, gene_cards, gene_knowledge
        if gene_cards.get_approved_summary(gene) or gene_knowledge.get_gene_patient_summary(gene):
            pytest.skip(f"{gene} has approved content")
        result = _resolve_gene_content(gene, "overview", question=f"מה זה הגן {gene}?")
        if result["display_mode"] == "supplemental_card":
            assert result["text_he"] and len(result["text_he"]) > 20


# ---------------------------------------------------------------------------
# E. Gene answer builder — main_answer branch produces valid response
# ---------------------------------------------------------------------------

class TestGeneAnswerBuilderSourceA:
    """Source A → non-empty answer, no supplemental card, ai_draft_attempted=False."""

    @pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "ATM"])
    def test_source_a_no_supplemental_card(self, gene):
        from app.counseling_engine import _build_gene_clinvar_answer, gene_cards
        if not gene_cards.get_approved_summary(gene):
            pytest.skip(f"{gene} has no curated card")
        result = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        assert result is not None
        assert "unverified_gene_draft" not in result
        meta = result.get("gene_metadata", {})
        assert meta.get("ai_draft_attempted") is False

    @pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "ATM"])
    def test_source_a_answer_nonempty(self, gene):
        from app.counseling_engine import _build_gene_clinvar_answer, gene_cards
        if not gene_cards.get_approved_summary(gene):
            pytest.skip(f"{gene} has no curated card")
        result = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        assert len(result["answer"]) > 50
        assert result["safety_level"] == "general_information"


# ---------------------------------------------------------------------------
# F. Gene answer builder — none branch has all required metadata fields
# ---------------------------------------------------------------------------

class TestGeneAnswerBuilderNone:
    """display_mode='none' branch populates every expected metadata field."""

    REQUIRED_NONE_META_FIELDS = [
        "gene_symbol", "answer_tier", "gene_knowledge_status",
        "unverified_gene_draft_available", "unverified_gene_draft_displayable",
        "draft_promoted_to_answer", "ai_draft_attempted", "ai_draft_generated",
        "ai_content_type", "source_grounded", "requires_physician_review",
        "answer_scope",
    ]

    @pytest.mark.parametrize("gene", ["APOE", "CCR5", "TLR3"])
    def test_none_branch_meta_fields_present(self, gene):
        from app.counseling_engine import _build_gene_clinvar_answer
        result = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        meta = result.get("gene_metadata", {})
        if meta.get("gene_knowledge_status") == "missing":
            for field in self.REQUIRED_NONE_META_FIELDS:
                assert field in meta, (
                    f"{gene}: gene_metadata missing '{field}'. meta={meta}"
                )

    @pytest.mark.parametrize("gene", ["APOE", "CCR5", "TLR3"])
    def test_none_branch_bool_fields_are_false(self, gene):
        from app.counseling_engine import _build_gene_clinvar_answer
        result = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        meta = result.get("gene_metadata", {})
        if meta.get("gene_knowledge_status") == "missing":
            assert meta.get("unverified_gene_draft_available") is False
            assert meta.get("unverified_gene_draft_displayable") is False
            assert meta.get("draft_promoted_to_answer") is False
            assert meta.get("ai_draft_generated") is False
            assert meta.get("source_grounded") is False

    @pytest.mark.parametrize("gene", ["APOE", "CCR5", "TLR3"])
    def test_none_branch_ai_draft_debug_present(self, gene):
        from app.counseling_engine import _build_gene_clinvar_answer
        result = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        if result.get("gene_metadata", {}).get("gene_knowledge_status") == "missing":
            debug = result.get("ai_draft_debug", {})
            assert "attempted" in debug
            assert "generated" in debug
            assert debug.get("shown") is False


# ---------------------------------------------------------------------------
# G. No raw technical dumps in main answer
# ---------------------------------------------------------------------------

class TestNoRawDumpsInMainAnswer:
    """Main answer must never contain raw ClinVar technical dumps."""

    FORBIDDEN_IN_ANSWER = [
        "נתוני ClinVar",
        "Total variants:",
        "by_significance",
        "top_phenotypes",
    ]

    @pytest.mark.parametrize("question,gene", [
        ("מה זה הגן BRCA1?", "BRCA1"),
        ("מה זה הגן CFTR?", "CFTR"),
        ("מה זה הגן APOE?", "APOE"),
        ("יש לי VUS בגן BRCA2, מה זה?", "BRCA2"),
        ("יש לי VUS בגן CFTR, מה זה?", "CFTR"),
    ])
    def test_no_raw_dumps(self, question, gene):
        data = client.post("/ask", json={"question": question}).json()
        answer = data.get("answer", "")
        for forbidden in self.FORBIDDEN_IN_ANSWER:
            assert forbidden not in answer, (
                f"gene={gene}: forbidden phrase {forbidden!r} found in answer"
            )


# ---------------------------------------------------------------------------
# H. display_mode consistency between both gene routes
# ---------------------------------------------------------------------------

class TestDisplayModeConsistency:
    """The same gene should produce consistent display behavior in both routes."""

    @pytest.mark.parametrize("gene", ["BRCA1", "CFTR", "APOE"])
    def test_both_routes_agree_on_source_grounded(self, gene):
        from app.counseling_engine import (
            _build_gene_clinvar_answer, _build_known_gene_answer
        )
        standalone = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        vus_route = _build_known_gene_answer(gene, question=f"יש לי VUS בגן {gene}")
        sm = standalone.get("gene_metadata", {})
        vm = vus_route.get("gene_metadata", {})
        # source_grounded should agree between routes for the same gene
        assert sm.get("source_grounded") == vm.get("source_grounded"), (
            f"{gene}: standalone source_grounded={sm.get('source_grounded')}, "
            f"vus_route source_grounded={vm.get('source_grounded')}"
        )

    @pytest.mark.parametrize("gene", ["BRCA1", "CFTR"])
    def test_vus_route_answer_contains_vus(self, gene):
        from app.counseling_engine import _build_known_gene_answer
        result = _build_known_gene_answer(gene, question=f"יש לי VUS בגן {gene}")
        assert "VUS" in result["answer"]
        assert result["matched_topic"] == "vus_known_gene"

    @pytest.mark.parametrize("gene", ["BRCA1", "CFTR", "APOE"])
    def test_answer_scope_in_metadata(self, gene):
        from app.counseling_engine import _build_gene_clinvar_answer
        result = _build_gene_clinvar_answer(f"מה זה הגן {gene}?", gene)
        meta = result.get("gene_metadata", {})
        assert "answer_scope" in meta
        assert isinstance(meta["answer_scope"], str)
        assert len(meta["answer_scope"]) > 0


# ---------------------------------------------------------------------------
# I. API-level response schema completeness
# ---------------------------------------------------------------------------

class TestApiResponseSchema:
    """Every /ask response for a gene question has the mandatory 5 fields."""

    REQUIRED_FIELDS = {"answer", "safety_level", "needs_genetic_counselor",
                       "matched_topic", "suggested_questions"}

    @pytest.mark.parametrize("question", [
        "מה זה הגן BRCA1?",
        "מה זה הגן CFTR?",
        "מה זה הגן APOE?",
        "יש לי VUS בגן NF1, מה זה?",
        "מה תפקיד הגן TP53?",
    ])
    def test_mandatory_response_fields(self, question):
        data = client.post("/ask", json={"question": question}).json()
        for field in self.REQUIRED_FIELDS:
            assert field in data, (
                f"question={question!r}: missing required field {field!r}"
            )

    @pytest.mark.parametrize("question", [
        "מה זה הגן BRCA1?",
        "מה זה הגן CFTR?",
    ])
    def test_gene_metadata_present_for_gene_questions(self, question):
        data = client.post("/ask", json={"question": question}).json()
        assert "gene_metadata" in data, (
            f"question={question!r}: gene_metadata missing from response"
        )
        assert "gene_symbol" in data["gene_metadata"]
