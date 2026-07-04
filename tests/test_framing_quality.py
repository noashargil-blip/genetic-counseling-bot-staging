"""
tests/test_framing_quality.py — Controlled-framing quality-gate tests.

Verifies that _validate_controlled_framing correctly rejects malformed Hebrew,
chatbot closing phrases, and VUS-defining sentences, while accepting clean
one-sentence professional framing as produced by the 7B model.

These tests complement test_llm_controlled.py (which covers general structural
validation) by focusing specifically on the _FRAMING_QUALITY_RE rules.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.counseling_engine import (
    _apply_llm_layer,
    _validate_controlled_framing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rejected(text: str) -> bool:
    return _validate_controlled_framing(text) is not None


def _rejection_reason(text: str) -> str:
    return _validate_controlled_framing(text) or ""


# ---------------------------------------------------------------------------
# Malformed / invented Hebrew words
# ---------------------------------------------------------------------------

class TestMalformedHebrew:
    def test_ממצאון_rejected(self):
        text = "הממצאון שנמצא בגן שלך אינו מאפיין מחלה."
        reason = _rejection_reason(text)
        assert reason, "Expected rejection"
        assert "quality-rejected" in reason

    def test_ממצאון_without_prefix_rejected(self):
        text = "ממצאון כזה נדיר מאוד בגנטיקה."
        assert _is_rejected(text)

    def test_bad_example_from_real_llm_rejected(self):
        # Actual output from 7B that triggered this task
        text = (
            "הVUS בגן APC מציין שהממצאון הזה לא מאפיין בהכרח "
            "את הסוגיה שלך כמחלה ספציפית."
        )
        assert _is_rejected(text), "Real bad LLM output must be rejected"


# ---------------------------------------------------------------------------
# Chatbot closing phrases
# ---------------------------------------------------------------------------

class TestChatbotPhrases:
    def test_אני_כאן_לעזור_rejected(self):
        text = "שמחנו לקבל את שאלתך. אני כאן לעזור."
        reason = _rejection_reason(text)
        assert reason
        assert "quality-rejected" in reason

    def test_שאלות_נוספות_rejected(self):
        text = "המידע שלהלן יסביר את הנושא. אם יש לך שאלות נוספות נשמח לענות."
        assert _is_rejected(text)

    def test_אם_יש_לך_שאלות_rejected(self):
        text = "נשמח לסייע. אם יש לך שאלות, פנה אלינו."
        assert _is_rejected(text)

    def test_full_chatbot_close_rejected(self):
        # Exact phrase from bad LLM output
        text = "אם יש לך שאלות נוספות אני כאן לעזור."
        assert _is_rejected(text)


# ---------------------------------------------------------------------------
# Unprofessional phrasing
# ---------------------------------------------------------------------------

class TestUnprofessionalPhrasing:
    def test_סוגיה_שלך_rejected(self):
        text = "הממצאון הזה קשור לסוגיה שלך בלבד."
        assert _is_rejected(text)

    def test_מאפיין_בהכרח_rejected(self):
        text = "ממצא זה לא מאפיין בהכרח מחלה."
        assert _is_rejected(text)

    def test_מחלה_ספציפית_rejected(self):
        text = "VUS אינו מלמד על מחלה ספציפית."
        assert _is_rejected(text)


# ---------------------------------------------------------------------------
# VUS-defining phrases (framing must NOT explain VUS)
# ---------------------------------------------------------------------------

class TestVusDefinitionInFraming:
    def test_vus_הוא_rejected(self):
        text = "VUS הוא וריאנט שמשמעותו אינה ברורה."
        reason = _rejection_reason(text)
        assert reason
        assert "quality-rejected" in reason

    def test_vus_מציין_rejected(self):
        text = "VUS מציין ממצא שטרם סווג."
        assert _is_rejected(text)

    def test_vus_פירושו_rejected(self):
        text = "VUS פירושו וריאנט בעל משמעות לא ידועה."
        assert _is_rejected(text)

    def test_vus_מייצג_rejected(self):
        text = "VUS מייצג מצב ביניים בסיווג גנטי."
        assert _is_rejected(text)

    def test_vus_מגדיר_rejected(self):
        text = "VUS מגדיר ממצא שדורש מחקר נוסף."
        assert _is_rejected(text)

    def test_vus_מסמן_rejected(self):
        text = "VUS מסמן שינוי גנטי שמשמעותו לא ברורה."
        assert _is_rejected(text)

    def test_פירוש_vus_rejected(self):
        text = "פירוש VUS הוא וריאנט בעל משמעות לא ידועה."
        assert _is_rejected(text)


# ---------------------------------------------------------------------------
# Acceptable clean framing sentences (from task description examples)
# ---------------------------------------------------------------------------

class TestAcceptableFraming:
    def test_vus_gene_style_accepted(self):
        # Style from task: VUS+gene — APC example
        text = (
            "אני מבינה שתוצאה עם מונח כמו VUS יכולה להיות מבלבלת, "
            "במיוחד כשמדובר בגן מוכר כמו APC."
        )
        assert not _is_rejected(text), _rejection_reason(text)

    def test_general_vus_style_accepted(self):
        # Style from task: general VUS
        text = (
            "אני מבינה שמונח כמו VUS יכול להיות מבלבל, "
            "ולכן חשוב להפריד בין מידע כללי לבין המשמעות האישית של התוצאה."
        )
        assert not _is_rejected(text), _rejection_reason(text)

    def test_short_empathetic_sentence_accepted(self):
        text = "אני מבינה שתוצאה כזו יכולה להיות מבלבלת, והמידע שלהלן מסביר את הנושא בצורה כללית."
        assert not _is_rejected(text), _rejection_reason(text)

    def test_rely_on_genetics_team_accepted(self):
        text = "מידע גנטי יכול להיות מורכב, ולכן חשוב לדון עם הצוות הגנטי שלך לגבי המשמעות האישית."
        assert not _is_rejected(text), _rejection_reason(text)

    def test_brca2_gene_mention_accepted(self):
        text = "אני מבינה שתוצאה עם מונח כמו VUS יכולה להיות מבלבלת, במיוחד כשמדובר בגן כמו BRCA2."
        assert not _is_rejected(text), _rejection_reason(text)

    def test_nf1_gene_mention_accepted(self):
        text = "אני מבינה שתוצאה עם מונח כמו VUS יכולה להיות מבלבלת, במיוחד כשמדובר בגן כמו NF1."
        assert not _is_rejected(text), _rejection_reason(text)


# ---------------------------------------------------------------------------
# Integration: deterministic answer is preserved when framing is accepted
# ---------------------------------------------------------------------------

class TestFramingPreservesDeterministicAnswer:
    """
    When a clean framing sentence passes validation, the deterministic answer
    must appear verbatim in the final output.
    """
    DET = "VUS הוא וריאנט בעל משמעות לא ודאית. הצוות הגנטי ידון עמך על משמעות הממצא."

    def test_clean_framing_prepended_to_deterministic(self):
        from unittest.mock import MagicMock, patch

        clean_framing = (
            "אני מבינה שמונח כמו VUS יכול להיות מבלבל, "
            "ולכן המידע שלהלן מסביר את הנושא בפירוט."
        )
        mock_client = MagicMock()
        mock_client._call_api.return_value = clean_framing

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer(
                "מה זה VUS?", self.DET, mode="controlled_framing"
            )

        assert result.llm_used is True, f"Expected llm_used=True, got {result.llm_used}; reason: {result.rejection_reason}"
        assert self.DET in result.answer, "Deterministic answer must be verbatim in output"
        assert clean_framing in result.answer, "Framing sentence must be in output"

    def test_bad_framing_falls_back_to_deterministic(self):
        from unittest.mock import MagicMock, patch

        bad_framing = "הממצאון הזה לא מאפיין בהכרח את הסוגיה שלך. אני כאן לעזור."
        mock_client = MagicMock()
        mock_client._call_api.return_value = bad_framing

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer(
                "מה זה VUS?", self.DET, mode="controlled_framing"
            )

        # Deterministic answer must always be present regardless of LLM result
        assert self.DET in result.answer, "Deterministic answer must survive bad framing"
        # Bad framing text must not appear verbatim
        assert "הממצאון" not in result.answer, "Malformed word must not reach user"

    def test_vus_definition_framing_rejected_deterministic_preserved(self):
        from unittest.mock import MagicMock, patch

        # Framing that tries to define VUS — must be rejected
        vus_def_framing = "VUS הוא וריאנט שמשמעותו אינה ברורה לחלוטין בשלב זה."
        mock_client = MagicMock()
        mock_client._call_api.return_value = vus_def_framing

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            result = _apply_llm_layer(
                "מה זה VUS?", self.DET, mode="controlled_framing"
            )

        assert self.DET in result.answer, "Deterministic answer must survive VUS-defining framing"
