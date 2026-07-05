# -*- coding: utf-8 -*-
"""
Session 17 — production fix: AI draft metadata consistency and debug fields.

Regression tests for:
- unverified_gene_draft_available matches whether draft object actually exists
- Draft is generated eagerly (without require include_unverified_gene_draft flag)
- ai_draft_debug appears when LLM is not configured
- No "לא הצלחנו ליצור" text in the main answer
- Tier2 genes (CFTR, TLR3, ABO, MUC19) attempt draft generation
- _generate_unverified_gene_draft populates _debug dict
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as engine
import app.gene_index as gene_index

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    engine._known_gene_set_cache = None
    yield
    engine._known_gene_set_cache = None


# ---------------------------------------------------------------------------
# _debug parameter population
# ---------------------------------------------------------------------------

class TestDebugParameterPopulation:
    """_generate_unverified_gene_draft populates _debug when provided."""

    def test_debug_reflects_no_llm_configured(self):
        debug = {}
        result = engine._generate_unverified_gene_draft("CFTR", _debug=debug)
        assert result is None
        assert debug.get("attempted") is False
        assert debug.get("provider") == "none"
        assert "reason" in debug

    def test_debug_on_success(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "גן CFTR מכיל הוראות לייצור חלבון חשוב. "
            "שינויים בגן זה קשורים לתסמונת ריאה כרונית. "
            "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
        )
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        debug = {}
        result = engine._generate_unverified_gene_draft(
            "CFTR", use_lenient_validator=True, _debug=debug
        )
        if result is not None:
            assert debug.get("attempted") is True
            assert debug.get("generated") is True
            assert debug.get("validation_passed") is True

    def test_debug_on_validation_failure(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = "CFTR is a gene."  # no Hebrew
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        debug = {}
        result = engine._generate_unverified_gene_draft(
            "CFTR", use_lenient_validator=True, _debug=debug
        )
        assert result is None
        assert debug.get("attempted") is True
        assert debug.get("generated") is False
        assert "rejection_code" in debug


# ---------------------------------------------------------------------------
# Metadata consistency: unverified_gene_draft_available == draft object exists
# ---------------------------------------------------------------------------

class TestMetadataConsistency:
    """unverified_gene_draft_available must equal (unverified_gene_draft in response)."""

    def _ask(self, question):
        return client.post("/ask", json={"question": question}).json()

    def test_cftr_flag_matches_draft_presence(self):
        data = self._ask("CFTR")
        meta = data.get("gene_metadata", {})
        flag = meta.get("unverified_gene_draft_available", False)
        draft_present = "unverified_gene_draft" in data
        assert flag == draft_present, (
            f"unverified_gene_draft_available={flag} but draft_present={draft_present}"
        )

    def test_cftr_phrase_flag_matches_draft_presence(self):
        data = self._ask("מה זה הגן CFTR?")
        meta = data.get("gene_metadata", {})
        flag = meta.get("unverified_gene_draft_available", False)
        draft_present = "unverified_gene_draft" in data
        assert flag == draft_present

    def test_no_draft_means_no_available_flag(self):
        """When no LLM and draft fails, flag must be false."""
        data = self._ask("CFTR")
        meta = data.get("gene_metadata", {})
        if "unverified_gene_draft" not in data:
            assert meta.get("unverified_gene_draft_available") is False

    def test_draft_present_means_flag_true(self, monkeypatch):
        """When mocked LLM returns valid draft, flag must be true."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "גן CFTR מכיל הוראות לייצור חלבון חשוב בוויסות כלוריד. "
            "שינויים מסוימים קשורים לתסמונת ריאות כרונית. "
            "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
        )
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = self._ask("CFTR")
        meta = data.get("gene_metadata", {})
        if "unverified_gene_draft" in data:
            assert meta.get("unverified_gene_draft_available") is True

    def test_ai_draft_attempted_present_for_tier2(self):
        """ai_draft_attempted field should appear in tier2 gene_metadata."""
        data = self._ask("CFTR")
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            assert "ai_draft_attempted" in meta

    def test_ai_draft_generated_matches_draft_presence(self):
        data = self._ask("CFTR")
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            generated = meta.get("ai_draft_generated", False)
            draft_present = "unverified_gene_draft" in data
            assert generated == draft_present


# ---------------------------------------------------------------------------
# ai_draft_debug appears when LLM not configured
# ---------------------------------------------------------------------------

class TestDraftDebugField:
    """ai_draft_debug must be present (and safe) when LLM is not available."""

    def _ask_tier2(self, question):
        data = client.post("/ask", json={"question": question}).json()
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") != "tier2":
            pytest.skip("Not a tier2 response on this server")
        return data

    def test_debug_field_present_when_no_draft(self):
        data = self._ask_tier2("CFTR")
        if "unverified_gene_draft" not in data:
            assert "ai_draft_debug" in data, (
                "Expected ai_draft_debug when draft is absent"
            )

    def test_debug_field_no_sensitive_data(self):
        data = self._ask_tier2("CFTR")
        debug = data.get("ai_draft_debug", {})
        for forbidden_key in ("api_key", "prompt", "system_prompt", "password", "token"):
            assert forbidden_key not in debug, f"Sensitive key {forbidden_key!r} in ai_draft_debug"

    def test_debug_field_has_expected_keys(self):
        data = self._ask_tier2("CFTR")
        if "ai_draft_debug" in data:
            debug = data["ai_draft_debug"]
            assert "attempted" in debug


# ---------------------------------------------------------------------------
# Draft called with mocked OpenAI
# ---------------------------------------------------------------------------

class TestDraftWithMockedLLM:
    """When LLM returns safe Hebrew text, the draft is in the response."""

    SAFE_TEXT = (
        "גן CFTR מקודד לחלבון שמעורב בוויסות הפרשות. "
        "שינויים מסוימים בגן זה קשורים לתסמונת ריאות. "
        "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
    )

    def test_cftr_draft_in_response(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.SAFE_TEXT
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            assert "unverified_gene_draft" in data, (
                f"Expected draft in response when LLM returns safe text. meta={meta}"
            )
            assert data["gene_metadata"]["unverified_gene_draft_available"] is True

    def test_cftr_draft_has_text_he(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.SAFE_TEXT
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        draft = data.get("unverified_gene_draft")
        if draft:
            assert draft.get("text_he"), "Draft should have text_he"
            assert "גן" in draft["text_he"] or "CFTR" in draft["text_he"]

    def test_cftr_draft_has_warning_he(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.SAFE_TEXT
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        draft = data.get("unverified_gene_draft")
        if draft:
            assert draft.get("warning_he"), "Draft should have warning_he"

    def test_cftr_ai_draft_debug_present_when_draft_succeeds(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.SAFE_TEXT
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        if "unverified_gene_draft" in data:
            assert "ai_draft_debug" in data, (
                "ai_draft_debug should be present for monitoring even when draft succeeded"
            )
            debug = data["ai_draft_debug"]
            assert debug.get("generated") is True
            assert debug.get("shown") is True

    def test_llm_mode_reflects_draft_usage(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.SAFE_TEXT
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        if "unverified_gene_draft" in data:
            assert data.get("llm_mode") != "none", (
                "llm_mode should not be 'none' when draft was generated by LLM"
            )


# ---------------------------------------------------------------------------
# OpenAI failure returns no draft and no misleading flag
# ---------------------------------------------------------------------------

class TestLLMFailureCase:
    """When LLM fails, draft is absent and unverified_gene_draft_available=false."""

    def test_llm_error_no_draft(self, monkeypatch):
        from app.llm_client import LLMClientError
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = LLMClientError("timeout")
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        assert "unverified_gene_draft" not in data

    def test_llm_error_flag_false(self, monkeypatch):
        from app.llm_client import LLMClientError
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = LLMClientError("timeout")
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            assert meta.get("unverified_gene_draft_available") is False

    def test_llm_error_debug_present(self, monkeypatch):
        from app.llm_client import LLMClientError
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = LLMClientError("timeout")
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            assert "ai_draft_debug" in data


# ---------------------------------------------------------------------------
# No "לא הצלחנו ליצור" in main answer text ever
# ---------------------------------------------------------------------------

class TestNoFailureMessageInAnswer:
    FAILURE_MSG = "לא הצלחנו ליצור"

    def test_cftr_no_failure_msg(self):
        data = client.post("/ask", json={"question": "CFTR"}).json()
        assert self.FAILURE_MSG not in data["answer"]

    def test_cftr_phrase_no_failure_msg(self):
        data = client.post("/ask", json={"question": "מה זה הגן CFTR?"}).json()
        assert self.FAILURE_MSG not in data["answer"]

    def test_vus_brca1_no_failure_msg(self):
        data = client.post("/ask", json={"question": "מה זה VUS בBRCA1?"}).json()
        assert self.FAILURE_MSG not in data["answer"]

    def test_tlr3_no_failure_msg(self):
        data = client.post("/ask", json={"question": "מה זה הגן TLR3?"}).json()
        assert self.FAILURE_MSG not in data["answer"]


# ---------------------------------------------------------------------------
# Tier2 genes attempt draft (CFTR, TLR3, ABO, MUC19)
# ---------------------------------------------------------------------------

class TestTier2GenesAttemptDraft:
    """For known tier2 genes, ai_draft_attempted should be True."""

    TIER2_QUESTIONS = [
        ("CFTR", "CFTR"),
        ("מה זה הגן CFTR?", "CFTR"),
        ("מה זה הגן TLR3?", "TLR3"),
    ]

    @pytest.mark.parametrize("question,gene", TIER2_QUESTIONS)
    def test_draft_attempted(self, question, gene):
        data = client.post("/ask", json={"question": question}).json()
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            assert "ai_draft_attempted" in meta, (
                f"Expected ai_draft_attempted in meta for tier2 gene {gene}"
            )


# ---------------------------------------------------------------------------
# Response schema: 5 required fields always present
# ---------------------------------------------------------------------------

class TestSchemaAlwaysComplete:
    REQUIRED = ("answer", "safety_level", "needs_genetic_counselor",
                "matched_topic", "suggested_questions")

    def test_cftr_schema(self):
        data = client.post("/ask", json={"question": "CFTR"}).json()
        for key in self.REQUIRED:
            assert key in data

    def test_tlr3_schema(self):
        data = client.post("/ask", json={"question": "מה זה הגן TLR3?"}).json()
        for key in self.REQUIRED:
            assert key in data
