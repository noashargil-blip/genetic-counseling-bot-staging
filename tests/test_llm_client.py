"""Tests for app/llm_client.py — all Anthropic API calls are mocked."""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.llm_client import LLMClient, LLMClientError


VALID_LLM_RESPONSE = {
    "answer": "BRCA1 c.5266dupC is a well-established pathogenic variant.",
    "evidence": [
        "Variant: NM_007294.4(BRCA1):c.5266dupC | significance: Pathogenic",
    ],
    "limitations": [
        "Evidence based on ClinVar submissions only.",
    ],
    "safety_disclaimer": (
        "This information is for educational purposes only and does not constitute "
        "medical advice. Please consult a certified genetics professional or genetic "
        "counselor for clinical interpretation and personalized recommendations."
    ),
}


def _mock_client(response_text: str) -> LLMClient:
    """Return an LLMClient whose underlying Anthropic client is fully mocked."""
    client = LLMClient.__new__(LLMClient)
    mock_anthropic = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    mock_anthropic.messages.create.return_value = MagicMock(content=[content_block])
    client._client = mock_anthropic
    return client


class TestLLMClientAsk:
    def test_valid_response_parsed(self):
        client = _mock_client(json.dumps(VALID_LLM_RESPONSE))
        result = client.ask("What is BRCA1 c.5266dupC?", [])
        assert result["answer"] == VALID_LLM_RESPONSE["answer"]
        assert isinstance(result["evidence"], list)
        assert isinstance(result["limitations"], list)
        assert "safety_disclaimer" in result

    def test_markdown_fences_stripped(self):
        wrapped = f"```json\n{json.dumps(VALID_LLM_RESPONSE)}\n```"
        client = _mock_client(wrapped)
        result = client.ask("question", [])
        assert result["answer"] == VALID_LLM_RESPONSE["answer"]

    def test_invalid_json_raises_error(self):
        client = _mock_client("this is not json")
        with pytest.raises(LLMClientError, match="not valid JSON"):
            client.ask("question", [])

    def test_missing_fields_raises_error(self):
        partial = {"answer": "something"}  # missing evidence, limitations, safety_disclaimer
        client = _mock_client(json.dumps(partial))
        with pytest.raises(LLMClientError, match="missing fields"):
            client.ask("question", [])

    def test_api_status_error_raises_llm_client_error(self):
        import anthropic
        client = LLMClient.__new__(LLMClient)
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = anthropic.APIStatusError(
            message="rate limit",
            response=MagicMock(status_code=429),
            body={},
        )
        client._client = mock_anthropic
        with pytest.raises(LLMClientError):
            client.ask("question", [])

    def test_api_connection_error_raises_llm_client_error(self):
        import anthropic
        client = LLMClient.__new__(LLMClient)
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )
        client._client = mock_anthropic
        with pytest.raises(LLMClientError, match="reach the Anthropic API"):
            client.ask("question", [])


class TestLLMClientInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            LLMClient()

    def test_initialises_with_explicit_key(self):
        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="sk-test-key")
            assert client is not None


# ---------------------------------------------------------------------------
# _strip_fences (the regex-based fence stripper that replaced removeprefix)
# ---------------------------------------------------------------------------

class TestStripFences:
    """Directly test the module-level fence-stripping helper."""

    def setup_method(self):
        from app.llm_client import _strip_fences
        self._strip = _strip_fences

    def test_plain_json_unchanged(self):
        payload = '{"answer": "x"}'
        assert self._strip(payload) == payload

    def test_json_fence_removed(self):
        payload = '{"answer": "x"}'
        assert self._strip(f"```json\n{payload}\n```") == payload

    def test_plain_fence_removed(self):
        payload = '{"answer": "x"}'
        assert self._strip(f"```\n{payload}\n```") == payload

    def test_trailing_newline_before_fence(self):
        """removeprefix/removesuffix would fail here; regex must not."""
        payload = '{"answer": "x"}'
        assert self._strip(f"```json\n{payload}\n\n```") == payload

    def test_no_fence_content_whitespace_stripped(self):
        assert self._strip("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# LLMRateLimitError is a subclass of LLMClientError
# ---------------------------------------------------------------------------

class TestRateLimitError:
    def test_rate_limit_is_llm_client_error(self):
        from app.llm_client import LLMRateLimitError, LLMClientError
        err = LLMRateLimitError("too fast")
        assert isinstance(err, LLMClientError)

    def test_anthropic_rate_limit_raises_llm_rate_limit_error(self):
        import anthropic
        from app.llm_client import AnthropicLLMClient, LLMRateLimitError
        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        mock_api = MagicMock()
        mock_api.messages.create.side_effect = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body={},
        )
        client._client = mock_api
        with pytest.raises(LLMRateLimitError):
            client.ask("question", [])

    def test_rate_limit_caught_separately_from_generic_error(self):
        """Verify callers can distinguish rate-limit from other failures."""
        from app.llm_client import LLMRateLimitError, LLMClientError
        caught_rate_limit = False
        caught_generic = False
        try:
            raise LLMRateLimitError("429")
        except LLMRateLimitError:
            caught_rate_limit = True
        except LLMClientError:
            caught_generic = True
        assert caught_rate_limit is True
        assert caught_generic is False


# ---------------------------------------------------------------------------
# OpenAILLMClient
# ---------------------------------------------------------------------------

class TestOpenAILLMClient:
    """Tests for the OpenAI backend. The openai package is mocked throughout."""

    def _build_client(self, response_text: str):
        """Return an OpenAILLMClient with the openai package fully mocked."""
        from app.llm_client import OpenAILLMClient

        mock_openai = MagicMock()
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.APIStatusError = type("APIStatusError", (Exception,), {"status_code": 500})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})

        choice = MagicMock()
        choice.message.content = response_text
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
            choices=[choice]
        )

        client = OpenAILLMClient.__new__(OpenAILLMClient)
        client._openai = mock_openai
        client._client = mock_openai.OpenAI.return_value
        return client

    def test_valid_response_parsed(self):
        client = self._build_client(json.dumps(VALID_LLM_RESPONSE))
        result = client.ask("What is BRCA1 c.5266dupC?", [])
        assert result["answer"] == VALID_LLM_RESPONSE["answer"]
        assert isinstance(result["evidence"], list)

    def test_invalid_json_raises_error(self):
        from app.llm_client import LLMClientError
        client = self._build_client("not json at all")
        with pytest.raises(LLMClientError, match="not valid JSON"):
            client.ask("question", [])

    def test_rate_limit_raises_llm_rate_limit_error(self):
        from app.llm_client import OpenAILLMClient, LLMRateLimitError

        client = OpenAILLMClient.__new__(OpenAILLMClient)
        mock_openai = MagicMock()
        RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.RateLimitError = RateLimitError
        mock_openai.APIStatusError = type("APIStatusError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = RateLimitError("429")
        client._openai = mock_openai
        client._client = mock_openai.OpenAI.return_value

        with pytest.raises(LLMRateLimitError):
            client.ask("question", [])

    def test_missing_openai_package_raises_import_error(self, monkeypatch):
        """If openai isn't installed, __init__ must raise ImportError clearly."""
        from app.llm_client import OpenAILLMClient
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(ImportError, match="openai"):
            OpenAILLMClient()


# ---------------------------------------------------------------------------
# create_llm_client factory
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    def test_prefers_anthropic_when_both_keys_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        with patch("anthropic.Anthropic"):
            from app.llm_client import create_llm_client, AnthropicLLMClient
            client = create_llm_client()
        assert isinstance(client, AnthropicLLMClient)

    def test_falls_back_to_openai_when_only_openai_key_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from importlib import reload
            import app.llm_client as llm_mod
            reload(llm_mod)
            client = llm_mod.create_llm_client()
        assert type(client).__name__ == "OpenAILLMClient"

    def test_raises_when_no_keys_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.llm_client import create_llm_client
        with pytest.raises(ValueError, match="No LLM configured"):
            create_llm_client()
