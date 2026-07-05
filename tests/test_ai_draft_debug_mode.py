# -*- coding: utf-8 -*-
"""
Tests for the staging-only AI_DRAFT_DEBUG_SHOW_REJECTED feature.
IMPORTANT: temporary — must NOT be enabled in customer-facing deployment.
"""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Mostly English — fails validation (no Hebrew majority)
REJECTED = (
    "APOE gene is associated with lipid metabolism and cardiovascular risk. "
    "This text is mostly English and should fail validation."
)
# Hebrew-dominant — passes validation
PASSING = (
    "הגן APOE קשור לעיבוד שומנים. "
    "מידע זה כללי בלבד."
)
Q_APOE = "מה זה הגן APOE?"


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    for var in ("LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "APP_ENV", "AI_DRAFT_DEBUG_SHOW_REJECTED"):
        monkeypatch.delenv(var, raising=False)
    import app.counseling_engine as eng
    eng._known_gene_set_cache = None
    yield


class TestDebugModeActiveStaging:
    def test_rejected_in_debug_meta(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        debug = data.get("ai_draft_debug", {})
        assert "unverified_gene_draft" not in data
        assert "raw_rejected_text_he" in debug

    def test_rejected_marked_debug_only(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        debug = data.get("ai_draft_debug", {})
        if "raw_rejected_text_he" in debug:
            assert debug.get("raw_rejected_status") == "ai_generated_rejected_debug_only"
            assert "DEBUG ONLY" in debug.get("raw_rejected_warning", "")

    def test_rejected_truncated_500(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = "APOE gene test. " * 50
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        raw = data.get("ai_draft_debug", {}).get("raw_rejected_text_he", "")
        if raw:
            assert len(raw) <= 500

    def test_passing_not_in_raw_rejected(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = PASSING
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "raw_rejected_text_he" not in data.get("ai_draft_debug", {})


class TestDebugModeInactiveProduction:
    def test_not_exposed_in_production(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "raw_rejected_text_he" not in data.get("ai_draft_debug", {})

    def test_not_exposed_without_env(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "raw_rejected_text_he" not in data.get("ai_draft_debug", {})

    def test_flag_off_in_staging(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "false")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "raw_rejected_text_he" not in data.get("ai_draft_debug", {})


class TestDebugModeHighStakesBlock:
    HIGH_STAKES = [
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "יש לי מוטציה בAPOE מה הסיכון שלי?",
        "יש לי שינוי בBRCA1 האם לעשות ניתוח?",
    ]

    @pytest.mark.parametrize("question", HIGH_STAKES)
    def test_no_raw_for_high_stakes(self, monkeypatch, question):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": question}).json()
        assert "raw_rejected_text_he" not in data.get("ai_draft_debug", {})
        assert "unverified_gene_draft" not in data


class TestDebugAllowlist:
    def test_non_allowlist_gene_not_exposed(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "מה זה הגן EGFR?"}).json()
        assert "raw_rejected_text_he" not in data.get("ai_draft_debug", {})


class TestDebugNoSecrets:
    FORBIDDEN = ("api_key", "prompt", "system_prompt", "password", "token", "secret")

    def test_no_secrets_in_debug(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        debug = data.get("ai_draft_debug", {})
        for key in debug:
            assert key.lower() not in [f.lower() for f in self.FORBIDDEN]

    def test_no_api_key_in_raw(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_DRAFT_DEBUG_SHOW_REJECTED", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-99")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = REJECTED
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "sk-test-secret-99" not in str(data.get("ai_draft_debug", {}))
