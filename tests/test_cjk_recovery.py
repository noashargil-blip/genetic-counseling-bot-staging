"""
tests/test_cjk_recovery.py

Tests for the CJK artifact cleaning and CJK-specific retry mechanism.

Covers:
  - _strip_tiny_cjk_artifacts: cleans ≤3 chars, refuses ≥4 chars, handles edge cases
  - _attempt_cjk_recovery: cleaning path, retry path, both-fail path
  - _apply_llm_layer: CJK rejection triggers recovery; non-CJK rejections do not
  - At most one retry is ever made
  - Cleaned output still fails if medical advice present
  - Debug metadata: llm_retry_used, llm_repaired, llm_repair_reason
  - Existing safety invariants unchanged
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from fastapi.testclient import TestClient
from app.main import app
from app.counseling_engine import (
    _strip_tiny_cjk_artifacts,
    _apply_llm_layer,
    _validate_controlled_framing,
    _validate_intro_with_reason,
    LLMLayerResult,
)

client = TestClient(app)

DET = "התשובה הדטרמיניסטית הקבועה שאינה משתנה."


def _unset_llm_env():
    for var in ("LOCAL_LLM_URL", "LLM_MODE", "LLM_DEBUG"):
        os.environ.pop(var, None)


def ask(question: str, **kw) -> dict:
    payload = {"question": question}
    payload.update(kw)
    resp = client.post("/ask", json=payload)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# _strip_tiny_cjk_artifacts
# ---------------------------------------------------------------------------

class TestStripTinyCjkArtifacts:

    def test_no_cjk_unchanged(self):
        text = "שלום, זהו טקסט עברי רגיל."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert cleaned == text
        assert was_cleaned is False

    def test_single_cjk_char_removed(self):
        text = "שלום日 זהו טקסט."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is True
        assert "日" not in cleaned
        assert "שלום" in cleaned

    def test_two_cjk_chars_removed(self):
        text = "שלום日本 מידע."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is True
        assert not any(c in cleaned for c in "日本")

    def test_three_cjk_chars_removed(self):
        text = "שלום日本語 מידע גנטי."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is True
        assert not any(c in cleaned for c in "日本語")

    def test_four_cjk_chars_not_removed(self):
        text = "שלום日本語テ מידע גנטי."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert cleaned == text
        assert was_cleaned is False

    def test_five_cjk_chars_not_removed(self):
        text = "日本語テスト שלום"
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert cleaned == text
        assert was_cleaned is False

    def test_many_cjk_chars_not_removed(self):
        text = "日" * 20 + " שלום"
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert cleaned == text
        assert was_cleaned is False

    def test_empty_string_unchanged(self):
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts("")
        assert cleaned == ""
        assert was_cleaned is False

    def test_hebrew_content_preserved_after_cleaning(self):
        text = "מידע על BRCA1日 נמצא כאן."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is True
        assert "BRCA1" in cleaned
        assert "מידע" in cleaned
        assert "נמצא" in cleaned

    def test_exactly_three_cjk_chars_at_threshold(self):
        text = "שלום日本語 גנטי."
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is True

    def test_exactly_four_at_threshold_not_cleaned(self):
        text = "שלום日本語テ גנטי."
        _, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is False

    def test_katakana_counted_as_cjk(self):
        text = "שלום テスト גנטי."
        # テスト = 3 katakana chars → should be cleaned
        cleaned, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is True

    def test_hiragana_counted_as_cjk(self):
        text = "שלום ひらがな גנטי."
        # ひらがな = 4 hiragana chars → should NOT be cleaned
        _, was_cleaned = _strip_tiny_cjk_artifacts(text)
        assert was_cleaned is False


# ---------------------------------------------------------------------------
# _apply_llm_layer — CJK rejection triggers recovery
# ---------------------------------------------------------------------------

class TestCjkRecoveryInApplyLlmLayer:

    def setup_method(self):
        _unset_llm_env()

    def test_many_cjk_triggers_retry_which_also_fails_returns_deterministic(self):
        """Many CJK chars (>3): cleaning skipped, retry with same bad output → deterministic."""
        cjk_heavy = "日本語テスト שלום"
        mock_client = MagicMock()
        mock_client._call_api.return_value = cjk_heavy

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is False
        assert result.answer == DET
        assert result.attempted is True
        assert result.rejection_reason is not None
        # Retry was attempted (call_api called twice: initial + retry)
        assert mock_client._call_api.call_count == 2
        assert result.retry_used is True

    def test_tiny_cjk_artifact_cleaned_returns_valid(self):
        """1 CJK char in otherwise valid Hebrew: cleaned and accepted."""
        text_with_artifact = "ברוך הבא לשאלה שלך על גנטיקה日."
        mock_client = MagicMock()
        mock_client._call_api.return_value = text_with_artifact

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is True
        assert result.repaired is True
        assert result.repair_reason == "tiny_cjk_artifacts_removed"
        assert result.retry_used is False
        # Original _call_api called only once (no retry needed)
        assert mock_client._call_api.call_count == 1
        assert DET in result.answer
        assert "日" not in result.answer

    def test_tiny_cjk_with_medical_content_rejected_after_cleaning(self):
        """1 CJK char cleaned, but cleaned text contains medical advice → rejected."""
        text_with_artifact = "יש לך צורך בניתוח日."  # ניתוח = surgery
        mock_client = MagicMock()
        # First call returns cleaned-but-invalid text; retry returns deterministic-safe text
        valid_retry = "ברוך הבא לשאלה שלך על גנטיקה."
        mock_client._call_api.side_effect = [text_with_artifact, valid_retry]

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        # Cleaning produced text with surgery → failed validation
        # But then retry succeeded with valid_retry
        assert result.llm_used is True
        assert result.retry_used is True
        assert DET in result.answer

    def test_at_most_one_retry_when_cjk_heavy(self):
        """_call_api called exactly twice: initial + one retry. Never more."""
        cjk_heavy = "日本語テスト שלום"  # 5 CJK chars
        mock_client = MagicMock()
        mock_client._call_api.return_value = cjk_heavy

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert mock_client._call_api.call_count == 2

    def test_at_most_one_retry_for_controlled_framing(self):
        """controlled_framing CJK rejection: initial + one retry. Never falls through to intro_only."""
        cjk_heavy = "日本語テスト שלום"
        mock_client = MagicMock()
        mock_client._call_api.return_value = cjk_heavy

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="controlled_framing")

        # Should be exactly 2 calls: initial controlled_framing + CJK retry
        # NOT 3 (which would be initial + CJK retry + intro_only fallback)
        assert mock_client._call_api.call_count == 2
        assert result.llm_used is False
        assert result.retry_used is True

    def test_retry_success_returns_llm_used_true(self):
        """After CJK rejection, retry returns valid Hebrew → llm_used=True, retry_used=True."""
        cjk_heavy = "日本語テスト שלום"
        valid_retry = "ברוך הבא לשאלה שלך על גנטיקה."
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [cjk_heavy, valid_retry]

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is True
        assert result.retry_used is True
        assert result.repaired is False
        assert result.rejection_reason is None
        assert valid_retry in result.answer
        assert DET in result.answer

    def test_retry_failure_returns_deterministic_fallback(self):
        """CJK on initial, CJK on retry → deterministic fallback, llm_used=False."""
        cjk_heavy = "日本語テスト שלום"
        mock_client = MagicMock()
        mock_client._call_api.return_value = cjk_heavy  # both calls return bad output

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is False
        assert result.answer == DET
        assert result.retry_used is True

    def test_non_cjk_rejection_does_not_trigger_retry_intro_mode(self):
        """Medical advice rejection (non-CJK) does NOT trigger CJK recovery."""
        surgery_text = "יש לך צורך בניתוח דחוף."
        mock_client = MagicMock()
        mock_client._call_api.return_value = surgery_text

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        # No retry: only one call for non-CJK rejections
        assert mock_client._call_api.call_count == 1
        assert result.llm_used is False
        assert result.retry_used is False

    def test_controlled_framing_non_cjk_still_falls_to_intro_only(self):
        """Non-CJK rejection in controlled_framing still falls back to intro_only (2 calls total)."""
        too_long = "א" * 601
        valid_intro = "ברוך הבא לשאלה שלך."
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [too_long, valid_intro]

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="controlled_framing")

        assert mock_client._call_api.call_count == 2
        assert result.llm_used is True
        assert result.retry_used is False  # intro_only fallback, not CJK retry


# ---------------------------------------------------------------------------
# Debug metadata — llm_retry_used, llm_repaired, llm_repair_reason
# ---------------------------------------------------------------------------

class TestCjkDebugMetadata:

    def setup_method(self):
        _unset_llm_env()

    def test_kb_path_no_llm_retry_debug_fields(self):
        """KB path is now deterministic — LLM not called, no retry debug fields."""
        cjk_heavy = "日本語テスト שלום"
        valid_retry = "ברוך הבא לשאלה שלך."
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [cjk_heavy, valid_retry]

        with patch.dict(os.environ,
                        {"LOCAL_LLM_URL": "http://localhost:11434", "LLM_DEBUG": "1"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            data = ask("מה זה VUS?")

        # KB path doesn't call LLM → no retry/repair debug fields
        assert data.get("llm_used") is False
        assert "llm_retry_used" not in data

    def test_kb_path_no_llm_repair_debug_fields(self):
        """KB path is now deterministic — LLM not called, no repair debug fields."""
        artifact_text = "ברוך הבא לשאלה שלך על גנטיקה日."
        mock_client = MagicMock()
        mock_client._call_api.return_value = artifact_text

        with patch.dict(os.environ,
                        {"LOCAL_LLM_URL": "http://localhost:11434", "LLM_DEBUG": "1"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            data = ask("מה זה VUS?")

        # KB path doesn't call LLM → no repair debug fields
        assert data.get("llm_used") is False
        assert "llm_repaired" not in data

    def test_no_cjk_debug_fields_when_llm_debug_off(self):
        cjk_heavy = "日本語テスト שלום"
        valid_retry = "ברוך הבא לשאלה שלך."
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [cjk_heavy, valid_retry]

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            data = ask("מה זה VUS?")

        assert "llm_retry_used" not in data
        assert "llm_repaired" not in data
        assert "llm_repair_reason" not in data

    def test_kb_path_no_llm_rejected_reason_field(self):
        """KB path is now deterministic — LLM not called, llm_rejected_reason absent."""
        cjk_heavy = "日本語テスト שלום"
        valid_retry = "ברוך הבא לשאלה שלך."
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [cjk_heavy, valid_retry]

        with patch.dict(os.environ,
                        {"LOCAL_LLM_URL": "http://localhost:11434", "LLM_DEBUG": "1"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            data = ask("מה זה VUS?")

        # KB path doesn't call LLM → no rejected_reason, llm_used=False
        assert "llm_rejected_reason" not in data
        assert data.get("llm_used") is False


# ---------------------------------------------------------------------------
# Safety invariants unchanged
# ---------------------------------------------------------------------------

class TestSafetyInvariantsPreserved:
    """Verify that CJK recovery never bypasses safety checks."""

    def setup_method(self):
        _unset_llm_env()

    def test_medical_advice_still_blocked_even_with_cjk_retry(self):
        """Retry output containing medical advice is still rejected.
        Uses 4 CJK chars (above cleaning threshold) so the retry path is exercised."""
        cjk_output = "日本語テ שלום"  # 4 CJK chars — above threshold, no cleaning
        medical_retry = "יש לך סרטן ועליך לפנות לניתוח."
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [cjk_output, medical_retry]

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is False
        assert result.answer == DET

    def test_personal_risk_in_retry_still_blocked(self):
        """Personal risk statement in retry output is rejected.
        Uses 4 CJK chars (above cleaning threshold) so the retry path is exercised."""
        cjk_output = "テストです שלום"  # 4 katakana chars — above threshold, no cleaning
        risky_retry = "הסיכון האישי שלך גבוה."  # "personal risk" → rejected
        mock_client = MagicMock()
        mock_client._call_api.side_effect = [cjk_output, risky_retry]

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is False
        assert result.answer == DET

    def test_deterministic_content_always_present(self):
        """Regardless of LLM path, deterministic content is in the answer."""
        artifact_text = "ברוך הבא לשאלה שלך日."
        mock_client = MagicMock()
        mock_client._call_api.return_value = artifact_text

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET)

        assert DET in result.answer

    def test_cjk_rejection_still_works_when_cleaning_skipped(self):
        """4+ CJK chars → no cleaning, only retry; retry with CJK → deterministic."""
        four_cjk = "שלום日本語テ מידע."  # exactly 4 CJK
        mock_client = MagicMock()
        mock_client._call_api.return_value = four_cjk

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", DET, mode="intro_only")

        assert result.llm_used is False
        assert result.repaired is False  # no cleaning done
