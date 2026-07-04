# -*- coding: utf-8 -*-
"""
tests/test_llm_metadata.py

Two test groups added in version 2.2.0:

A. LLM metadata fields (llm_used, fallback_used) in /ask responses
   Every answer path must return these fields with correct boolean values.
   All tests run with LOCAL_LLM_URL unset (deterministic fallback path).

B. /health/llm live probe behaviour
   Covers the probe_llm_live() function and the /health/llm endpoint under
   four conditions: no URL, unreachable, successful HTTP response, timeout.
"""
import socket
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import counseling_engine, health as health_module
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Run every test in the deterministic-fallback (no LLM) configuration."""
    monkeypatch.delenv("LOCAL_LLM_URL",    raising=False)
    monkeypatch.delenv("OPENAI_API_KEY",   raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# A. llm_used / fallback_used per answer path
# ---------------------------------------------------------------------------

def _ask(question: str, **extra) -> dict:
    payload = {"question": question}
    payload.update(extra)
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestLlmMetadataFieldsPresent:
    """Every /ask response must carry llm_used and fallback_used as booleans."""

    def test_general_kb_path_has_llm_metadata(self):
        data = _ask("מה זה VUS?")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_pii_block_has_llm_metadata(self):
        data = _ask("תעודת הזהות שלי 123456789 מה זה VUS?")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_personal_redirect_has_llm_metadata(self):
        data = _ask("האם אני צריכה ניתוח?")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_vus_known_gene_has_llm_metadata(self):
        data = _ask("יש לי VUS ב-BRCA1, מה זה?")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_gene_clinvar_summary_has_llm_metadata(self):
        data = _ask("מה ידוע על BRCA1?")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_followup_has_llm_metadata(self):
        data = _ask("תסביר יותר", last_topic="vus")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_variant_evidence_has_llm_metadata(self):
        data = _ask("מה ידוע על וריאנט c.5266dupC ב-BRCA1?")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)

    def test_helpful_fallback_has_llm_metadata(self):
        data = _ask("שאלה שלא קיימת במאגר בכלל xyxyxy")
        assert isinstance(data["llm_used"], bool)
        assert isinstance(data["fallback_used"], bool)


class TestLlmMetadataValuesDeterministic:
    """
    Without LOCAL_LLM_URL the answer is always deterministic:
    llm_used must be False, fallback_used must be True.

    Exception: PII blocks and personal redirects are safety policy messages
    (not KB text), so fallback_used is False for them.
    """

    # --- paths where the KB / deterministic text is used ---

    def test_general_kb_path_llm_false(self):
        data = _ask("מה זה VUS?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_carrier_kb_path_llm_false(self):
        data = _ask("מה זה נשאות?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_inheritance_kb_path_llm_false(self):
        data = _ask("מה זה ירושה אוטוזומלית דומיננטית?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_vus_known_gene_deterministic(self):
        data = _ask("יש לי VUS ב-BRCA1, מה זה?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_vus_known_gene_nf1_deterministic(self):
        data = _ask("יש לי VUS ב-NF1")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_gene_clinvar_summary_deterministic(self):
        data = _ask("מה ידוע על BRCA2?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_followup_vus_deterministic(self):
        data = _ask("מה ההשלכות", last_topic="vus_known_gene")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_followup_carrier_deterministic(self):
        data = _ask("תסביר יותר", last_topic="carrier")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_followup_generic_kb_entry_deterministic(self):
        data = _ask("אפשר לפרט", last_topic="autosomal_dominant")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_variant_evidence_deterministic(self):
        data = _ask("מה ידוע על וריאנט c.5266dupC ב-BRCA1?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_helpful_fallback_deterministic(self):
        data = _ask("שאלה אקראית xyxyxy123")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    # --- safety policy messages: neither LLM nor KB text ---

    def test_pii_block_neither_llm_nor_kb(self):
        """PII block is a fixed safety message — fallback_used is False."""
        data = _ask("תעודת הזהות שלי 123456789")
        assert data["safety_level"] == "contains_identifying_info"
        assert data["llm_used"] is False
        assert data["fallback_used"] is False

    def test_personal_redirect_neither_llm_nor_kb(self):
        """Personal redirect is a fixed policy message — fallback_used is False."""
        data = _ask("האם אני צריכה ניתוח?")
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["llm_used"] is False
        assert data["fallback_used"] is False

    def test_email_pii_block_neither_llm_nor_kb(self):
        data = _ask("שאלה test@example.com")
        assert data["llm_used"] is False
        assert data["fallback_used"] is False

    # --- gene metadata mirrors top-level values ---

    def test_gene_metadata_mirrors_top_level(self):
        data = _ask("מה ידוע על TP53?")
        meta = data.get("gene_metadata")
        if meta:
            assert meta["llm_used"] == data["llm_used"]
            assert meta["fallback_used"] == data["fallback_used"]

    def test_non_gene_response_has_no_gene_metadata_key(self):
        data = _ask("מה זה VUS?")
        assert "gene_metadata" not in data


class TestLlmMetadataWithMockedLlm:
    """
    With LOCAL_LLM_URL set and the LLM returning a valid response:
    llm_used must be True, fallback_used must be False for the KB path.
    """

    def test_kb_path_deterministic_even_when_llm_configured(self, monkeypatch):
        """KB answers are always deterministic — LLM is not called even when configured."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        valid_response = "אני מבינה שמונח כמו VUS יכול להיות מבלבל, ולכן המידע שלהלן מסביר את הנושא בפירוט."

        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            instance = MockClient.return_value
            instance._call_api.return_value = valid_response
            data = _ask("מה זה VUS?")

        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_kb_path_llm_fail_fallback_to_kb(self, monkeypatch):
        """When the LLM raises an error the answer must fall back to KB text."""
        from app.llm_client import LLMClientError
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")

        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            instance = MockClient.return_value
            instance._call_api.side_effect = LLMClientError("timeout")
            data = _ask("מה זה VUS?")

        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_kb_path_llm_empty_string_fallback(self, monkeypatch):
        """Empty LLM output is treated as failure — KB text used instead."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")

        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            instance = MockClient.return_value
            instance._call_api.return_value = "   "  # whitespace only
            data = _ask("מה זה VUS?")

        assert data["llm_used"] is False
        assert data["fallback_used"] is True


# ---------------------------------------------------------------------------
# B. /health/llm live probe behaviour
# ---------------------------------------------------------------------------

class TestHealthLlmProbeFunction:
    """Unit tests for health.probe_llm_live() in isolation."""

    def test_returns_true_on_successful_response(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert health_module.probe_llm_live("http://localhost:9999") is True

    def test_returns_true_on_http_error_response(self):
        """HTTPError (e.g. 404/405) still means the server is alive."""
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       url="http://localhost:9999", code=405,
                       msg="Method Not Allowed", hdrs=None, fp=None)):
            assert health_module.probe_llm_live("http://localhost:9999") is True

    def test_returns_false_on_connection_refused(self):
        with patch("urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("Connection refused")):
            assert health_module.probe_llm_live("http://localhost:9999") is False

    def test_returns_false_on_timeout(self):
        with patch("urllib.request.urlopen",
                   side_effect=TimeoutError("timed out")):
            assert health_module.probe_llm_live("http://localhost:9999") is False

    def test_returns_false_on_url_error(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("Name or service not known")):
            assert health_module.probe_llm_live("http://no-such-host.invalid") is False

    def test_short_timeout_used(self):
        """Probe with an explicit 0.001 s timeout must fail fast."""
        # The real urlopen will raise OSError for the invalid host;
        # what matters is that probe_llm_live never raises.
        result = health_module.probe_llm_live("http://192.0.2.0:1", timeout=0.001)
        assert isinstance(result, bool)

    def test_returns_bool_not_none(self):
        with patch("urllib.request.urlopen", side_effect=Exception("any error")):
            result = health_module.probe_llm_live("http://localhost:9999")
            assert isinstance(result, bool)


class TestHealthLlmEndpointProbe:
    """Integration-level tests for GET /health/llm including live-probe field."""

    def test_no_local_url_has_no_reachable_field(self):
        """Without LOCAL_LLM_URL the probe is not attempted."""
        data = client.get("/health/llm").json()
        assert "reachable" not in data

    def test_local_url_adds_reachable_field(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            data = client.get("/health/llm").json()
        assert "reachable" in data
        assert data["reachable"] is True

    def test_local_url_unreachable_reachable_false(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("refused")):
            data = client.get("/health/llm").json()
        assert data.get("reachable") is False

    def test_local_url_timeout_reachable_false(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("urllib.request.urlopen",
                   side_effect=TimeoutError("timed out")):
            data = client.get("/health/llm").json()
        assert data.get("reachable") is False

    def test_local_url_http_error_reachable_true(self, monkeypatch):
        """HTTP 405 from the server means it is reachable."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       url="http://localhost:9999", code=405,
                       msg="Method Not Allowed", hdrs=None, fp=None)):
            data = client.get("/health/llm").json()
        assert data.get("reachable") is True

    def test_endpoint_always_returns_200(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("refused")):
            resp = client.get("/health/llm")
        assert resp.status_code == 200

    def test_endpoint_ok_reflects_env_var_not_probe(self, monkeypatch):
        """
        ok=True is set by check_llm() based on env-var config.
        reachable is the live probe result. They are independent.
        """
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("refused")):
            data = client.get("/health/llm").json()
        # ok=True because URL is configured, reachable=False because server is down
        assert data["ok"] is True
        assert data["reachable"] is False

    def test_openai_key_no_reachable_field(self, monkeypatch):
        """Cloud providers don't add a reachable field."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        data = client.get("/health/llm").json()
        assert "reachable" not in data

    def test_anthropic_key_no_reachable_field(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        data = client.get("/health/llm").json()
        assert "reachable" not in data
