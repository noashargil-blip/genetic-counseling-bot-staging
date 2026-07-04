# -*- coding: utf-8 -*-
"""
tests/test_staging_launch.py

Staging / beta launch readiness tests.

Covers:
  - LLM provider abstraction (LLM_PROVIDER=none/local/openai)
  - OpenAI provider does not expose API key in health or errors
  - Provider failure falls back safely
  - /health/llm in all provider modes
  - No LLM call for curated FAQ answers when LOCAL_LLM_URL is unset
  - No LLM call for reproductive/abortion question (safety redirect)
  - App runs without Slurm/GPU/university paths
  - /app static page served
  - / redirects to /app
  - Viewport meta tag present in index.html
  - Mobile-responsive CSS (@media) present in styles.css
  - Privacy notice present in index.html
  - DISABLE_UPLOADS=true gates upload endpoints
  - No PII forwarded past safety layer
"""

import json
import os
import pathlib
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "app" / "static"


# ---------------------------------------------------------------------------
# Client fixture — always starts clean (no env vars set)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    """FastAPI TestClient with all LLM env vars cleared."""
    for var in (
        "LLM_PROVIDER", "LOCAL_LLM_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
        "ANTHROPIC_API_KEY", "HF_ENDPOINT_URL", "HF_TOKEN",
        "LLM_TIMEOUT_SECONDS", "LLM_MAX_TOKENS", "LLM_TEMPERATURE",
        "DISABLE_UPLOADS",
    ):
        monkeypatch.delenv(var, raising=False)
    from app.main import app
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# LLM provider factory — create_llm_client()
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    """create_llm_client() selects the correct backend from env vars."""

    def test_provider_none_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "none")
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError, match="explicitly disabled"):
            create_llm_client()

    def test_provider_local_without_url_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError, match="LOCAL_LLM_URL"):
            create_llm_client()

    def test_provider_local_with_url_returns_local_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LOCAL_LLM_URL", "http://fake-server:8765/generate")
        from app.llm_client import create_llm_client, LocalLLMClient
        client = create_llm_client()
        assert isinstance(client, LocalLLMClient)

    def test_provider_openai_without_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_llm_client()

    def test_provider_openai_with_key_returns_openai_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-staging")
        from app.llm_client import create_llm_client, OpenAILLMClient
        client = create_llm_client()
        assert isinstance(client, OpenAILLMClient)

    def test_provider_openai_uses_openai_model_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        from app.llm_client import create_llm_client, OpenAILLMClient
        client = create_llm_client()
        assert isinstance(client, OpenAILLMClient)
        assert client._model == "gpt-4o"

    def test_provider_openai_default_model_is_mini(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        from app.llm_client import create_llm_client, OpenAILLMClient
        client = create_llm_client()
        assert isinstance(client, OpenAILLMClient)
        assert client._model == "gpt-4o-mini"

    def test_provider_huggingface_without_url_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "huggingface_endpoint")
        monkeypatch.delenv("HF_ENDPOINT_URL", raising=False)
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError, match="HF_ENDPOINT_URL"):
            create_llm_client()

    def test_provider_huggingface_with_url_returns_hf_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "huggingface_endpoint")
        monkeypatch.setenv("HF_ENDPOINT_URL", "https://api-inference.huggingface.co/models/test")
        from app.llm_client import create_llm_client, HuggingFaceEndpointLLMClient
        client = create_llm_client()
        assert isinstance(client, HuggingFaceEndpointLLMClient)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "foobar_provider")
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            create_llm_client()

    def test_no_provider_no_keys_raises(self, monkeypatch):
        for var in ("LLM_PROVIDER", "LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError):
            create_llm_client()

    def test_legacy_auto_detect_local_url(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LOCAL_LLM_URL", "http://compute:8765/generate")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.llm_client import create_llm_client, LocalLLMClient
        assert isinstance(create_llm_client(), LocalLLMClient)

    def test_legacy_auto_detect_openai(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-autodetect")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from app.llm_client import create_llm_client, OpenAILLMClient
        assert isinstance(create_llm_client(), OpenAILLMClient)


# ---------------------------------------------------------------------------
# OpenAI provider — no key exposure
# ---------------------------------------------------------------------------

class TestOpenAINoKeyExposure:
    """OPENAI_API_KEY must never appear in error messages or health responses."""

    def test_error_does_not_contain_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-staging-key")
        from app.llm_client import create_llm_client
        client = create_llm_client()
        # Simulate API failure by making _call_api raise
        import urllib.error
        with patch.object(client._openai, "Client", side_effect=Exception("connection error")):
            try:
                client._call_api("test", "system")
            except Exception as exc:
                assert "sk-super-secret-staging-key" not in str(exc)

    def test_health_llm_openai_does_not_expose_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-private-key-do-not-expose")
        from app.main import app
        with TestClient(app) as c:
            resp = c.get("/health/llm")
        body = resp.text
        assert "sk-private-key-do-not-expose" not in body
        data = resp.json()
        assert data["provider"] == "openai"
        assert data["configured"] is True


# ---------------------------------------------------------------------------
# /health/llm in all provider modes
# ---------------------------------------------------------------------------

class TestHealthLlmProviderModes:

    def test_health_llm_provider_none(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "none")
        from app.main import app
        with TestClient(app) as c:
            data = c.get("/health/llm").json()
        assert data["provider"] == "none"
        assert data["ok"] is False
        assert data["deterministic_fallback_available"] is True

    def test_health_llm_provider_openai_configured(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from app.main import app
        with TestClient(app) as c:
            data = c.get("/health/llm").json()
        assert data["provider"] == "openai"
        assert data["ok"] is True
        assert data["configured"] is True
        assert data["deterministic_fallback_available"] is True

    def test_health_llm_provider_openai_not_configured(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.main import app
        with TestClient(app) as c:
            data = c.get("/health/llm").json()
        assert data["provider"] == "openai"
        assert data["ok"] is False
        assert "reason" in data

    def test_health_llm_no_provider_no_keys_returns_not_ok(self, monkeypatch):
        for var in ("LLM_PROVIDER", "LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        from app.main import app
        with TestClient(app) as c:
            data = c.get("/health/llm").json()
        assert data["ok"] is False
        assert data["deterministic_fallback_available"] is True

    def test_health_llm_local_provider_no_reachability_check_for_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from app.main import app
        with TestClient(app) as c:
            data = c.get("/health/llm").json()
        # OpenAI health should NOT include "reachable" key (no expensive probe)
        assert "reachable" not in data

    def test_health_llm_returns_200_always(self, monkeypatch):
        for var in ("LLM_PROVIDER", "LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        from app.main import app
        with TestClient(app) as c:
            resp = c.get("/health/llm")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /app and / routing
# ---------------------------------------------------------------------------

class TestRoutingAndStaticPages:

    def test_app_endpoint_returns_200(self, client):
        resp = client.get("/app")
        assert resp.status_code == 200

    def test_app_endpoint_returns_html(self, client):
        resp = client.get("/app")
        ct = resp.headers.get("content-type", "")
        assert "text/html" in ct

    def test_root_redirects_to_app(self, client):
        # follow_redirects=False to see the redirect itself
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/app" in resp.headers.get("location", "")

    def test_root_following_redirect_reaches_html(self, client):
        resp = client.get("/", follow_redirects=True)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Frontend HTML — viewport, privacy notice
# ---------------------------------------------------------------------------

class TestFrontendHTML:

    @pytest.fixture()
    def index_html(self):
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @pytest.fixture()
    def styles_css(self):
        return (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    def test_viewport_meta_tag_present(self, index_html):
        assert 'name="viewport"' in index_html
        assert "width=device-width" in index_html

    def test_privacy_notice_element_present(self, index_html):
        assert 'id="privacy-notice"' in index_html

    def test_privacy_notice_has_hebrew_text(self, index_html):
        # Check for the required Hebrew notice text
        assert "אין להזין" in index_html
        assert "ייעוץ רפואי" in index_html

    def test_privacy_notice_has_confirm_button(self, index_html):
        assert 'id="privacy-ok"' in index_html

    def test_mobile_responsive_css_present(self, styles_css):
        assert "@media" in styles_css
        assert "max-width" in styles_css

    def test_mobile_css_covers_narrow_screens(self, styles_css):
        # Should have a breakpoint at 600px or smaller
        assert "600px" in styles_css or "480px" in styles_css

    def test_privacy_card_css_present(self, styles_css):
        assert "#privacy-notice" in styles_css
        assert ".privacy-card" in styles_css
        assert ".privacy-btn" in styles_css

    def test_no_horizontal_overflow_rule_implied(self, styles_css):
        # overflow-x: auto on containers (prevents horizontal scroll)
        assert "overflow-x" in styles_css


# ---------------------------------------------------------------------------
# DISABLE_UPLOADS gate
# ---------------------------------------------------------------------------

class TestDisableUploads:

    def test_upload_disabled_returns_503(self, monkeypatch):
        import app.main as main_mod
        monkeypatch.setattr(main_mod, "_UPLOADS_DISABLED", True)
        from app.main import app
        with TestClient(app) as c:
            resp = c.post("/upload", files={"file": ("test.csv", b"gene,rsid\nBRCA1,rs123", "text/csv")})
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()

    def test_analyze_upload_disabled_returns_503(self, monkeypatch):
        import app.main as main_mod
        monkeypatch.setattr(main_mod, "_UPLOADS_DISABLED", True)
        from app.main import app
        with TestClient(app) as c:
            resp = c.post(
                "/analyze-upload",
                files={"file": ("test.csv", b"gene,rsid\nBRCA1,rs123", "text/csv")},
            )
        assert resp.status_code == 503

    def test_upload_enabled_by_default(self, monkeypatch):
        import app.main as main_mod
        monkeypatch.setattr(main_mod, "_UPLOADS_DISABLED", False)
        from app.main import app
        with TestClient(app) as c:
            resp = c.post("/upload", files={"file": ("test.csv", b"gene,rsid\nBRCA1,rs123", "text/csv")})
        # Not 503 — upload was accepted (may fail for other reasons, that's ok)
        assert resp.status_code != 503


# ---------------------------------------------------------------------------
# /ask — deterministic for curated FAQ (no LOCAL_LLM_URL → no LLM)
# ---------------------------------------------------------------------------

class TestAskDeterministicMode:
    """When LOCAL_LLM_URL is unset, /ask is fully deterministic."""

    @pytest.fixture(autouse=True)
    def _clear_llm_url(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    def test_vus_question_returns_answer(self, client):
        resp = client.post("/ask", json={"question": "מה זה VUS?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert data["safety_level"] == "general_information"
        assert data["needs_genetic_counselor"] is False

    def test_vus_answer_llm_not_used(self, client):
        resp = client.post("/ask", json={"question": "מה זה VUS?"})
        data = resp.json()
        # Without LOCAL_LLM_URL, llm_used must be False
        assert data["llm_used"] is False

    def test_faq_answer_fallback_used(self, client):
        resp = client.post("/ask", json={"question": "מה זה נשאות?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert data["llm_used"] is False

    def test_response_schema_has_5_required_fields(self, client):
        resp = client.post("/ask", json={"question": "מה זה גן?"})
        data = resp.json()
        for field in ("answer", "safety_level", "needs_genetic_counselor",
                      "matched_topic", "suggested_questions"):
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# /ask — safety for high-stakes / restricted topics
# ---------------------------------------------------------------------------

class TestAskSafetyRequirements:
    """Safety-critical questions must never be answered by free-form LLM."""

    @pytest.fixture(autouse=True)
    def _clear_llm_url(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    def test_abortion_question_redirected_to_counselor(self, client):
        resp = client.post(
            "/ask",
            json={"question": "יש לי VUS ב-BRCA2 — כדאי להפסיק את ההריון?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True
        assert data["llm_used"] is False

    def test_personal_variant_risk_redirected(self, client):
        # "האם הווריאנט שלי מסוכן?" → personal interpretation → requires counselor
        resp = client.post("/ask", json={"question": "האם הווריאנט שלי מסוכן?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True

    def test_personal_risk_question_redirected(self, client):
        resp = client.post("/ask", json={"question": "מה הסיכון שלי לחלות?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["llm_used"] is False

    def test_identifying_info_blocked(self, client):
        resp = client.post(
            "/ask",
            json={"question": "קוראים לי שרה, יש לי מוטציה ב BRCA1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # PII blocks the question; no counselor redirect needed (different safety path)
        assert data["safety_level"] == "contains_identifying_info"
        assert data["llm_used"] is False


# ---------------------------------------------------------------------------
# PII — no personal data forwarded past the safety layer
# ---------------------------------------------------------------------------

class TestPIISafety:
    """PII detection fires before any LLM call or KB lookup."""

    @pytest.fixture(autouse=True)
    def _clear_llm_url(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    def test_israeli_id_blocked(self, client):
        resp = client.post("/ask", json={"question": "מספר תעודת זהות שלי 123456782"})
        assert resp.status_code == 200
        assert resp.json()["safety_level"] == "contains_identifying_info"

    def test_email_blocked(self, client):
        resp = client.post("/ask", json={"question": "שלחו לי תשובה ל user@example.com"})
        assert resp.status_code == 200
        assert resp.json()["safety_level"] == "contains_identifying_info"

    def test_name_phrase_blocked(self, client):
        resp = client.post("/ask", json={"question": "קוראים לי נועה"})
        assert resp.status_code == 200
        assert resp.json()["safety_level"] == "contains_identifying_info"


# ---------------------------------------------------------------------------
# App runs without university-specific paths
# ---------------------------------------------------------------------------

class TestNoUniversityDependencies:
    """App must start and answer questions without Slurm/GPU/compute-node paths."""

    @pytest.fixture(autouse=True)
    def _clear_university_env(self, monkeypatch):
        for var in ("LOCAL_LLM_URL", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising=False)

    def test_app_imports_without_crash(self):
        from app.main import app
        assert app is not None

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_ask_works_without_local_llm(self, client):
        resp = client.post("/ask", json={"question": "מה זה pathogenic?"})
        assert resp.status_code == 200
        assert resp.json()["answer"]

    def test_app_page_served(self, client):
        resp = client.get("/app")
        assert resp.status_code == 200

    def test_topics_endpoint_works(self, client):
        resp = client.get("/topics")
        assert resp.status_code == 200
        assert "topics" in resp.json()


# ---------------------------------------------------------------------------
# LLM_PROVIDER=local — still works with mocked endpoint
# ---------------------------------------------------------------------------

class TestProviderLocalMocked:
    """LLM_PROVIDER=local with a mocked local HTTP server."""

    def test_local_provider_client_created(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LOCAL_LLM_URL", "http://fake-slurm:8765/generate")
        from app.llm_client import create_llm_client, LocalLLMClient
        c = create_llm_client()
        assert isinstance(c, LocalLLMClient)
        assert c._url == "http://fake-slurm:8765/generate"

    def test_local_provider_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LOCAL_LLM_URL", "http://fake:8765/generate")
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
        from app.llm_client import create_llm_client, LocalLLMClient
        c = create_llm_client()
        assert isinstance(c, LocalLLMClient)
        assert c.TIMEOUT == 120

    def test_local_provider_connection_error_raises_llm_client_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LOCAL_LLM_URL", "http://nowhere-server:9999/generate")
        from app.llm_client import create_llm_client, LLMClientError
        c = create_llm_client()
        with pytest.raises(LLMClientError):
            c._call_api("test question", "system prompt")


# ---------------------------------------------------------------------------
# OpenAI provider — configurable params
# ---------------------------------------------------------------------------

class TestOpenAIConfigurableParams:

    def test_max_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MAX_TOKENS", "512")
        from app.llm_client import create_llm_client, OpenAILLMClient
        c = create_llm_client()
        assert isinstance(c, OpenAILLMClient)
        assert c._max_tokens == 512

    def test_temperature_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
        from app.llm_client import create_llm_client, OpenAILLMClient
        c = create_llm_client()
        assert isinstance(c, OpenAILLMClient)
        assert abs(c._temperature - 0.1) < 0.001

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45")
        from app.llm_client import create_llm_client, OpenAILLMClient
        c = create_llm_client()
        assert isinstance(c, OpenAILLMClient)
        assert c._timeout == 45
