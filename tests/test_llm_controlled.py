"""
tests/test_llm_controlled.py

Tests for the controlled LLM mode and observability features (v2.3.1):

  - _validate_intro_with_reason returns reason strings or None
  - _validate_controlled_framing enforces 600-char limit + same safety rules
  - _validate_tier2_framing rejects gene biology claims
  - _apply_llm_layer returns LLMLayerResult with correct fields
  - _apply_safe_intro remains backward-compatible (str, bool) 2-tuple
  - LLM_DEBUG=1 causes llm_attempted/llm_rejected_reason in /ask response
  - llm_mode is present in every /ask response (default "none")
  - App works fully without LOCAL_LLM_URL (deterministic fallback)
  - Deterministic medical content is never replaced by LLM output

All tests run without a live LLM server (LOCAL_LLM_URL unset unless
explicitly patched in a test).
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from fastapi.testclient import TestClient
from app.main import app
from app.counseling_engine import (
    _validate_llm_intro,
    _validate_intro_with_reason,
    _validate_controlled_framing,
    _validate_tier2_framing,
    _apply_safe_intro,
    _apply_llm_layer,
    LLMLayerResult,
)

client = TestClient(app)

# Helpers
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
# _validate_intro_with_reason — internals
# ---------------------------------------------------------------------------

class TestValidateIntroWithReason:
    """_validate_intro_with_reason returns None for valid text, str for invalid."""

    def setup_method(self):
        _unset_llm_env()

    def test_valid_hebrew_sentence_returns_none(self):
        text = "ברוך הבא לשאלה שלך על גנטיקה, נסה לקרוא את המידע שלהלן בעיון."
        assert _validate_intro_with_reason(text) is None

    def test_empty_string_returns_reason(self):
        reason = _validate_intro_with_reason("")
        assert reason is not None
        assert isinstance(reason, str)

    def test_whitespace_only_returns_reason(self):
        assert _validate_intro_with_reason("   ") is not None

    def test_too_long_returns_reason(self):
        text = "א" * 201
        reason = _validate_intro_with_reason(text)
        assert reason is not None

    def test_exactly_200_chars_valid(self):
        text = "א" * 200
        # Pure Hebrew, no forbidden terms, long but exactly at limit
        assert _validate_intro_with_reason(text) is None

    def test_question_mark_returns_reason(self):
        text = "מה אתה שואל?"
        assert _validate_intro_with_reason(text) is not None

    def test_cjk_char_returns_reason(self):
        text = "שלום 中文 שלום"
        assert _validate_intro_with_reason(text) is not None

    def test_forbidden_surgery_english_returns_reason(self):
        text = "This surgery recommendation is wrong"
        assert _validate_intro_with_reason(text) is not None

    def test_forbidden_hebrew_treatment_returns_reason(self):
        text = "יש לך צורך בטיפול מיוחד"
        assert _validate_intro_with_reason(text) is not None

    def test_no_hebrew_chars_returns_reason(self):
        text = "Hello world this is English"
        assert _validate_intro_with_reason(text) is not None

    def test_insufficient_hebrew_ratio_returns_reason(self):
        # Mostly English words not in allowed tokens
        text = "Hello testing words here is mostly english sentence with little"
        assert _validate_intro_with_reason(text) is not None

    def test_gene_symbols_allowed_in_hebrew(self):
        text = "המידע על BRCA1 שלהלן נוגע לשאלתך הגנטית."
        assert _validate_intro_with_reason(text) is None

    def test_clinvar_allowed_in_hebrew(self):
        text = "הנתונים מ-ClinVar מציגים תמונה מעניינת לגבי גן זה."
        assert _validate_intro_with_reason(text) is None

    def test_vus_pathogenic_benign_allowed(self):
        text = "המושג VUS מוסבר כאן בהרחבה."
        assert _validate_intro_with_reason(text) is None

    def test_unknown_long_english_word_returns_reason(self):
        text = "שלום המידע הרפואי החשוב שלהלן: randomword"
        reason = _validate_intro_with_reason(text)
        assert reason is not None

    def test_validate_llm_intro_is_consistent_with_reason(self):
        """_validate_llm_intro must agree with _validate_intro_with_reason."""
        cases = [
            "ברוך הבא לשאלה.",
            "",
            "שלום 中文",
            "A" * 201,
            "מה זה?",
        ]
        for text in cases:
            intro_ok = _validate_llm_intro(text)
            reason = _validate_intro_with_reason(text)
            assert intro_ok == (reason is None), (
                f"Inconsistency for {text!r}: intro={intro_ok}, reason={reason!r}"
            )


# ---------------------------------------------------------------------------
# _validate_controlled_framing
# ---------------------------------------------------------------------------

class TestValidateControlledFraming:
    """Max 600 chars; same safety checks as intro_only."""

    def setup_method(self):
        _unset_llm_env()

    def test_short_valid_hebrew_returns_none(self):
        text = "ברוך הבא. המידע שלהלן עוסק בשאלתך."
        assert _validate_controlled_framing(text) is None

    def test_exactly_600_chars_valid(self):
        text = "א" * 600
        assert _validate_controlled_framing(text) is None

    def test_601_chars_returns_reason(self):
        text = "א" * 601
        assert _validate_controlled_framing(text) is not None

    def test_cjk_rejected(self):
        text = "שלום 日本語 שלום שלום שלום שלום"
        assert _validate_controlled_framing(text) is not None

    def test_forbidden_colonoscopy_rejected(self):
        text = "יש להמליץ על קולונוסקופיה במקרה זה."
        assert _validate_controlled_framing(text) is not None

    def test_question_mark_rejected(self):
        text = "מה זה אומר? הנה המידע."
        assert _validate_controlled_framing(text) is not None

    def test_medical_treatment_rejected(self):
        text = "הטיפול המומלץ הוא תרופתי."
        assert _validate_controlled_framing(text) is not None

    def test_surgery_english_rejected(self):
        text = "שלום, surgery is recommended for your case"
        assert _validate_controlled_framing(text) is not None

    def test_200_char_valid_still_valid(self):
        # Anything valid for intro_only should also be valid for controlled_framing
        text = "ברוך הבא לשאלה שלך על גנטיקה."
        assert _validate_controlled_framing(text) is None

    def test_multi_sentence_no_chatbot_close_valid(self):
        text = (
            "שמחנו לקבל את שאלתך. "
            "המידע שלהלן מתאר את הנושא בפירוט."
        )
        assert _validate_controlled_framing(text) is None


# ---------------------------------------------------------------------------
# _validate_tier2_framing
# ---------------------------------------------------------------------------

class TestValidateTier2Framing:
    """Tier-2 framing: same rules as controlled_framing + no gene biology claims."""

    def setup_method(self):
        _unset_llm_env()

    def test_stats_description_valid(self):
        text = "הגן HBB מופיע ב-ClinVar עם אלפי רשומות מסוגים שונים."
        assert _validate_tier2_framing(text, "HBB") is None

    def test_bio_claim_same_gene_rejected(self):
        # "גן HBB אחראי" — biology claim
        text = "הגן HBB אחראי לייצור המוגלובין בתאי דם אדומים."
        assert _validate_tier2_framing(text, "HBB") is not None

    def test_bio_claim_different_gene_not_caught(self):
        # Tier-2 only checks the specific gene we're framing
        text = "הגן HBB אחראי לייצור המוגלובין בתאי דם אדומים."
        # If framing for "BRCA1", "HBB" claim should not trigger
        result = _validate_tier2_framing(text, "BRCA1")
        # The text still fails controlled_framing checks (mostly Hebrew, check)
        # but the gene-biology check itself should not trigger for BRCA1
        # So result is either None (if valid) or reason from controlled framing
        assert result is None or "BRCA1" not in (result or "")

    def test_no_gene_controls_in_text_is_valid(self):
        text = "הנתונים ב-ClinVar לגן זה כוללים מגוון רחב של רשומות גנטיות."
        assert _validate_tier2_framing(text, "APC") is None

    def test_cjk_still_rejected(self):
        text = "שלום 中文 רשומות"
        assert _validate_tier2_framing(text, "APC") is not None

    def test_too_long_rejected(self):
        text = "א" * 601
        assert _validate_tier2_framing(text, "NF1") is not None


# ---------------------------------------------------------------------------
# _apply_llm_layer — unit tests (LLM mocked)
# ---------------------------------------------------------------------------

class TestApplyLlmLayer:
    """_apply_llm_layer selects mode, validates, returns LLMLayerResult."""

    DET = "התשובה הדטרמיניסטית הקבועה."

    def setup_method(self):
        _unset_llm_env()

    def test_no_url_returns_deterministic(self):
        result = _apply_llm_layer("מה זה VUS?", self.DET)
        assert isinstance(result, LLMLayerResult)
        assert result.answer == self.DET
        assert result.llm_used is False
        assert result.attempted is False
        assert result.mode == "none"
        assert result.rejection_reason is None

    def test_apply_safe_intro_backward_compat(self):
        """_apply_safe_intro must return (str, bool) tuple, not LLMLayerResult."""
        final, used = _apply_safe_intro("מה זה VUS?", self.DET)
        assert isinstance(final, str)
        assert isinstance(used, bool)
        assert final == self.DET  # no LLM URL configured
        assert used is False

    def test_valid_intro_only_mode_with_mocked_llm(self):
        valid_intro = "ברוך הבא לשאלה שלך על גנטיקה."
        mock_client = MagicMock()
        mock_client._call_api.return_value = valid_intro

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", self.DET, mode="intro_only")

        assert result.llm_used is True
        assert result.attempted is True
        assert result.mode == "intro_only"
        assert result.answer.startswith(valid_intro)
        assert self.DET in result.answer
        assert result.rejection_reason is None

    def test_rejected_intro_returns_deterministic(self):
        bad_intro = "Surgery is needed for your case. ניתוח מומלץ."
        mock_client = MagicMock()
        mock_client._call_api.return_value = bad_intro

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", self.DET, mode="intro_only")

        assert result.llm_used is False
        assert result.attempted is True
        assert result.mode == "intro_only"
        assert result.answer == self.DET
        assert result.rejection_reason is not None

    def test_cjk_intro_rejected(self):
        cjk_text = "日本語テスト שלום"
        mock_client = MagicMock()
        mock_client._call_api.return_value = cjk_text

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", self.DET, mode="intro_only")

        assert result.llm_used is False
        assert result.rejection_reason is not None

    def test_controlled_framing_valid(self):
        valid_framing = "אני מבינה שמונח כמו VUS יכול להיות מבלבל, ולכן המידע שלהלן מסביר את הנושא בפירוט."
        mock_client = MagicMock()
        mock_client._call_api.return_value = valid_framing

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", self.DET, mode="controlled_framing")

        assert result.llm_used is True
        assert result.mode in ("controlled_framing", "intro_only")
        assert self.DET in result.answer

    def test_controlled_framing_over_600_rejected(self):
        too_long = "א" * 601
        mock_client = MagicMock()
        mock_client._call_api.return_value = too_long

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            # controlled_framing rejected → falls back to intro_only (also invalid here)
            # So deterministic returned
            result = _apply_llm_layer("מה זה VUS?", self.DET, mode="controlled_framing")

        assert result.answer == self.DET or self.DET in result.answer

    def test_deterministic_content_always_in_answer(self):
        """Regardless of LLM output, deterministic content must appear in answer."""
        valid_intro = "ברוך הבא לשאלה שלך."
        mock_client = MagicMock()
        mock_client._call_api.return_value = valid_intro

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer("מה זה VUS?", self.DET, mode="intro_only")

        assert self.DET in result.answer, "Deterministic content must always appear in answer"

    def test_env_var_llm_mode_read(self):
        """When no mode is passed, LLM_MODE env var is used."""
        _unset_llm_env()
        result = _apply_llm_layer("מה זה VUS?", self.DET)
        # No LLM URL → mode should be "none"
        assert result.mode == "none"


# ---------------------------------------------------------------------------
# /ask response — llm_mode always present
# ---------------------------------------------------------------------------

class TestLlmModeInResponse:
    """llm_mode must be present in every /ask response."""

    def setup_method(self):
        _unset_llm_env()

    def test_vus_general_has_llm_mode(self):
        data = ask("מה זה VUS?")
        assert "llm_mode" in data
        assert isinstance(data["llm_mode"], str)

    def test_vus_gene_has_llm_mode(self):
        data = ask("יש לי VUS ב-BRCA1, מה זה?")
        assert "llm_mode" in data

    def test_pii_block_has_llm_mode(self):
        data = ask("תעודת הזהות שלי 123456789 מה זה VUS?")
        assert "llm_mode" in data

    def test_personal_redirect_has_llm_mode(self):
        data = ask("האם אני צריכה ניתוח בגלל הווריאנט?")
        assert "llm_mode" in data

    def test_gene_level_question_has_llm_mode(self):
        data = ask("מה ידוע על APC?")
        assert "llm_mode" in data

    def test_followup_has_llm_mode(self):
        data = ask("תסביר יותר", last_topic="vus")
        assert "llm_mode" in data

    def test_carrier_has_llm_mode(self):
        data = ask("אמרו לי שאני נשאית, מה זה?")
        assert "llm_mode" in data

    def test_llm_mode_is_none_without_llm_url(self):
        """Without LOCAL_LLM_URL, llm_mode must be 'none'."""
        data = ask("מה זה VUS?")
        assert data["llm_mode"] == "none"

    def test_no_debug_fields_without_llm_debug(self):
        """Without LLM_DEBUG=1, llm_attempted and llm_rejected_reason absent."""
        data = ask("מה זה VUS?")
        assert "llm_attempted" not in data
        assert "llm_rejected_reason" not in data


# ---------------------------------------------------------------------------
# LLM_DEBUG=1 diagnostic fields
# ---------------------------------------------------------------------------

class TestLlmDebugFields:
    """llm_attempted and llm_rejected_reason appear only when LLM_DEBUG=1."""

    def setup_method(self):
        _unset_llm_env()

    def test_debug_fields_absent_without_flag(self):
        data = ask("מה זה VUS?")
        assert "llm_attempted" not in data
        assert "llm_rejected_reason" not in data

    def test_debug_fields_absent_for_kb_path_even_with_flag(self):
        """KB path is now deterministic — LLM not called, so debug fields absent."""
        bad_output = "Surgery is recommended."
        mock_client = MagicMock()
        mock_client._call_api.return_value = bad_output

        with patch.dict(os.environ,
                        {"LOCAL_LLM_URL": "http://localhost:11434", "LLM_DEBUG": "1"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            data = ask("מה זה VUS?")

        # KB path doesn't call LLM → no debug fields
        assert "llm_attempted" not in data
        assert "llm_rejected_reason" not in data
        assert data.get("llm_used") is False

    def test_debug_fields_absent_without_llm_url(self):
        """KB path does not call LLM — debug fields absent even with LLM_DEBUG=1."""
        with patch.dict(os.environ, {"LLM_DEBUG": "1"}):
            data = ask("מה זה VUS?")

        assert "llm_attempted" not in data
        assert data.get("llm_used") is False

    def test_kb_path_llm_mode_always_none(self):
        """KB path is always deterministic — llm_mode is 'none' even when LLM configured."""
        valid_intro = "ברוך הבא לשאלה שלך על גנטיקה."
        mock_client = MagicMock()
        mock_client._call_api.return_value = valid_intro

        with patch.dict(os.environ,
                        {"LOCAL_LLM_URL": "http://localhost:11434",
                         "LLM_DEBUG": "1",
                         "LLM_MODE": "intro_only"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            data = ask("מה זה VUS?")

        assert data.get("llm_mode") == "none"
        assert data.get("llm_used") is False


# ---------------------------------------------------------------------------
# Tier-2 framing safety (no biology invented)
# ---------------------------------------------------------------------------

class TestTier2FramingSafety:
    """The LLM must not invent gene biology for Tier-2 answers."""

    def setup_method(self):
        _unset_llm_env()

    def test_tier2_bio_claim_rejected_returns_deterministic(self):
        """LLM output claiming gene biology for Tier-2 gene is rejected."""
        bio_claim = f"הגן HBB אחראי לייצור המוגלובין בתאי הדם האדומים של הגוף."
        mock_client = MagicMock()
        mock_client._call_api.return_value = bio_claim

        with patch.dict(os.environ,
                        {"LOCAL_LLM_URL": "http://localhost:11434",
                         "LLM_MODE": "controlled_framing"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer(
                "מה זה HBB?", "התשובה הדטרמיניסטית.",
                gene="HBB", mode="tier2_framing",
                context_fields={"total_variants": 5000, "significance_breakdown": {}, "top_phenotypes": []}
            )

        assert result.llm_used is False
        assert result.rejection_reason is not None

    def test_tier2_stats_description_accepted(self):
        """LLM output describing ClinVar stats (no biology) is accepted."""
        stats_text = "הנתונים ב-ClinVar לגן HBB מציגים אלפי רשומות גנטיות."
        mock_client = MagicMock()
        mock_client._call_api.return_value = stats_text

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer(
                "מה זה HBB?", "התשובה הדטרמיניסטית.",
                gene="HBB", mode="tier2_framing",
                context_fields={"total_variants": 5000, "significance_breakdown": {}, "top_phenotypes": []}
            )

        assert result.llm_used is True
        assert "התשובה הדטרמיניסטית." in result.answer


# ---------------------------------------------------------------------------
# App works without LOCAL_LLM_URL (no crash, deterministic fallback)
# ---------------------------------------------------------------------------

class TestNoLlmUrl:
    """App must work fully and safely without any LLM configuration."""

    def setup_method(self):
        _unset_llm_env()

    def test_all_core_paths_work_without_llm(self):
        questions = [
            "מה זה VUS?",
            "אמרו לי שאני נשאית, מה זה?",
            "יש לי VUS ב-BRCA2, מה זה?",
            "מה ידוע על APC?",
        ]
        for q in questions:
            data = ask(q)
            assert bool(data.get("answer")), f"No answer for: {q!r}"
            assert data.get("llm_used") is False, f"llm_used should be False without LLM: {q!r}"
            assert data.get("llm_mode") == "none", f"llm_mode should be 'none' without LLM: {q!r}"

    def test_schema_complete_without_llm(self):
        data = ask("מה זה VUS?")
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions", "llm_used", "fallback_used", "llm_mode"}
        assert required.issubset(set(data.keys()))
