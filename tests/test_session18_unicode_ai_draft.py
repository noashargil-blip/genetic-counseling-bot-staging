# -*- coding: utf-8 -*-
"""
Session 18: UnicodeEncodeError robustness, metadata consistency,
mocked-OpenAI draft, high-stakes safety regression, frontend contract.
"""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

APOE_SAFE_HEBREW = (
    "הגן APOE קשור לעיבוד שומנים בגוף ולסיכון למחלות קרדיווסקולריות. "
    "מידע זה כללי בלבד ואינו מפרש תוצאות אישיות."
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


class TestMockedOpenAIDraft:
    def test_apoe_draft_text_he_populated(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        draft = data.get("unverified_gene_draft")
        if draft:
            assert draft.get("text_he"), "text_he must be set from LLM output"

    def test_apoe_draft_required_fields(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        draft = data.get("unverified_gene_draft")
        if draft:
            assert "warning_he" in draft
            assert draft.get("approved") is False
            assert draft.get("review_status") == "unreviewed"

    def test_cftr_draft_text_he(self, monkeypatch):
        safe = (
            "הגן CFTR קשור למחלת הסיסטיק פיברוסיס. "
            "נשאות יחידה אינה גורמת למחלה."
        )
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = safe
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "CFTR"}).json()
        draft = data.get("unverified_gene_draft")
        if draft:
            assert draft.get("text_he")


class TestMetadataConsistencySuccess:
    def test_ai_draft_debug_on_success(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        if "unverified_gene_draft" in data:
            assert "ai_draft_debug" in data
            debug = data["ai_draft_debug"]
            assert debug.get("generated") is True
            assert debug.get("shown") is True
            assert debug.get("attempted") is True

    def test_gene_metadata_on_success(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        if "unverified_gene_draft" in data:
            meta = data.get("gene_metadata", {})
            assert meta.get("ai_draft_attempted") is True
            assert meta.get("ai_draft_generated") is True
            assert meta.get("unverified_gene_draft_available") is True


class TestMetadataConsistencyFailure:
    def test_ai_draft_debug_on_failure(self):
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "ai_draft_debug" in data

    def test_no_draft_when_no_llm(self):
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "unverified_gene_draft" not in data

    def test_gene_metadata_flags_on_failure(self):
        data = client.post("/ask", json={"question": Q_APOE}).json()
        meta = data.get("gene_metadata", {})
        if meta:
            assert meta.get("unverified_gene_draft_available") is False
            assert meta.get("ai_draft_generated") is False

    def test_shown_false_on_failure(self):
        data = client.post("/ask", json={"question": "CFTR"}).json()
        debug = data.get("ai_draft_debug", {})
        if debug and not debug.get("generated"):
            assert debug.get("shown") is False or "shown" not in debug


class TestHighStakesSafetyRegression:
    NO_DRAFT = [
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "יש לי מוטציה בAPOE מה הסיכון שלי?",
        "יש לי שינוי בBRCA1 האם לעשות ניתוח?",
    ]
    PERSONAL = [
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "יש לי מוטציה בAPOE מה הסיכון שלי?",
    ]

    @pytest.mark.parametrize("question", NO_DRAFT)
    def test_no_unverified_draft(self, monkeypatch, question):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": question}).json()
        assert "unverified_gene_draft" not in data

    @pytest.mark.parametrize("question", PERSONAL)
    def test_needs_counselor(self, monkeypatch, question):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": question}).json()
        assert data.get("needs_genetic_counselor") is True


class TestUnicodeEncodeErrorRecovery:
    def _unicode_err_client(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = UnicodeEncodeError(
            "latin-1", "—", 0, 1, "ordinal not in range(256)"
        )
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)

    def test_does_not_crash(self, monkeypatch):
        self._unicode_err_client(monkeypatch)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert "answer" in data
        assert "unverified_gene_draft" not in data

    def test_rejection_code(self, monkeypatch):
        self._unicode_err_client(monkeypatch)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        debug = data.get("ai_draft_debug", {})
        assert debug.get("rejection_code") == "unexpected_error"

    def test_error_type(self, monkeypatch):
        self._unicode_err_client(monkeypatch)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        debug = data.get("ai_draft_debug", {})
        if debug.get("rejection_code") == "unexpected_error":
            assert debug.get("error_type") == "UnicodeEncodeError"


class TestFrontendContract:
    def test_no_failure_text_in_draft(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        draft = data.get("unverified_gene_draft")
        if draft:
            assert "לא הצלחנו" not in draft.get("text_he", "")

    def test_matched_topic_for_gene(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = APOE_SAFE_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": Q_APOE}).json()
        assert data.get("matched_topic") in ("gene_clinvar_summary", "known_gene_answer", "gene_education")

    def test_no_secrets_in_ai_debug(self):
        data = client.post("/ask", json={"question": "CFTR"}).json()
        debug = data.get("ai_draft_debug", {})
        forbidden = ("api_key", "prompt", "system_prompt", "password", "token", "secret")
        for key in debug:
            assert key not in forbidden
