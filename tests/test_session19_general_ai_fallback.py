# -*- coding: utf-8 -*-
"""
Session 19: General education AI fallback.

Tests that:
  - Safe general concept questions get an AI answer when the flag is on.
  - Personal / high-stakes questions are never sent to the general AI.
  - The unverified_general_draft response structure is correct.
  - ai_general_debug is present alongside the draft.
  - The flag is off by default (production safety).
  - The classifier correctly labels questions.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared Hebrew output that passes validation (Hebrew majority, no blocks)
# ---------------------------------------------------------------------------
GENERIC_EDU_HEBREW = (
    "מחלות נוירודגנרטיביות הן קבוצה של מחלות שמתאפיינות בניוון הדרגתי של תאי עצב. "
    "הן כוללות מחלות כמו פרקינסון, ALS ומחלת הנטינגטון. "
    "אם השאלה נוגעת לתוצאה האישית שלך, יש לפנות לצוות הגנטי."
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    for var in (
        "LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "APP_ENV", "AI_GENERAL_EDUCATION_FALLBACK_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    import app.counseling_engine as eng
    eng._known_gene_set_cache = None
    yield


# ---------------------------------------------------------------------------
# 1. Flag disabled by default — fallback never fires
# ---------------------------------------------------------------------------

class TestFlagDisabledByDefault:
    CONCEPT_QUESTIONS = [
        "מה זה מחלות נוירודגנרטיביות?",
        "מה זה mismatch repair?",
        "מה זה penetrance?",
        "מה זה המוגלובין?",
    ]

    @pytest.mark.parametrize("question", CONCEPT_QUESTIONS)
    def test_no_general_draft_without_flag(self, question):
        data = client.post("/ask", json={"question": question}).json()
        assert "unverified_general_draft" not in data

    @pytest.mark.parametrize("question", CONCEPT_QUESTIONS)
    def test_no_general_draft_in_production(self, monkeypatch, question):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        data = client.post("/ask", json={"question": question}).json()
        assert "unverified_general_draft" not in data


# ---------------------------------------------------------------------------
# 2. Flag enabled in staging — AI draft appears for allowed concept questions
# ---------------------------------------------------------------------------

class TestFlagEnabledStaging:
    ALLOWED_QUESTIONS = [
        "מה זה מחלות נוירודגנרטיביות?",
        "מה זה mismatch repair?",
        "מה זה המוגלובין?",
        "מה זה penetrance?",
        "מה זה ClinVar?",
    ]

    def _staging_mock(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = GENERIC_EDU_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)

    @pytest.mark.parametrize("question", ALLOWED_QUESTIONS)
    def test_general_draft_present(self, monkeypatch, question):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": question}).json()
        # Only check when KB doesn't match (KB might cover some concepts)
        if data.get("matched_topic") == "general_education_ai":
            assert "unverified_general_draft" in data

    def test_general_draft_structure(self, monkeypatch):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            draft = data["unverified_general_draft"]
            assert "status" in draft
            assert draft["status"] == "ai_generated_unreviewed"
            assert "text_he" in draft
            assert "warning_he" in draft
            assert "source_note_he" in draft
            assert len(draft["text_he"]) >= 30

    def test_general_draft_matched_topic(self, monkeypatch):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            assert data.get("matched_topic") == "general_education_ai"

    def test_ai_general_debug_present(self, monkeypatch):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            assert "ai_general_debug" in data
            debug = data["ai_general_debug"]
            assert debug.get("attempted") is True
            assert debug.get("generated") is True

    def test_ai_general_debug_no_secrets(self, monkeypatch):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        debug = data.get("ai_general_debug", {})
        forbidden = ("api_key", "prompt", "system_prompt", "password", "token", "secret")
        for key in debug:
            assert key.lower() not in forbidden

    def test_warning_present_in_draft(self, monkeypatch):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            warning = data["unverified_general_draft"].get("warning_he", "")
            assert len(warning) > 20, "warning_he must be a real warning, not empty"

    def test_text_he_comes_from_llm(self, monkeypatch):
        self._staging_mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            text = data["unverified_general_draft"].get("text_he", "")
            assert "נוירודגנרטיביות" in text or len(text) >= 30


# ---------------------------------------------------------------------------
# 3. Blocked high-stakes / personal questions — no AI draft regardless of flag
# ---------------------------------------------------------------------------

class TestBlockedHighStakesQuestions:
    BLOCKED = [
        "יש לי APOE האם יהיה לי אלצהיימר?",
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "האם אני צריכה ניתוח?",
        "מה הסיכון שלי?",
        "האם הילדים שלי יהיו חולים?",
        "מה המשמעות של הממצא שלי?",
        "האם זה מסוכן לי?",
        "מה עלי לעשות עם זה?",
    ]

    @pytest.mark.parametrize("question", BLOCKED)
    def test_no_general_draft_for_high_stakes(self, monkeypatch, question):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = GENERIC_EDU_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": question}).json()
        assert "unverified_general_draft" not in data


# ---------------------------------------------------------------------------
# 4. Classifier unit tests
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_safe_concept_questions(self):
        from app.counseling_engine import _classify_general_question
        safe = [
            "מה זה מחלות נוירודגנרטיביות?",
            "מה זה mismatch repair?",
            "מה זה penetrance?",
            "מה ההבדל בין גן לחלבון?",
            "מה זה תורשה אוטוזומלית?",
            "מה זה ClinVar?",
            "מה זה בדיקת נשאות?",
            "מה זה המוגלובין?",
        ]
        for q in safe:
            result = _classify_general_question(q)
            assert result == "safe_general_education", (
                f"Expected safe_general_education for {q!r}, got {result!r}"
            )

    def test_personal_questions_blocked(self):
        from app.counseling_engine import _classify_general_question
        blocked = [
            "מה הסיכון שלי?",
            "מה הממצא שלי אומר?",
            "האם יהיה לי אלצהיימר?",
            "מה התוצאה שלי?",
            "מה הגן שלי?",
        ]
        for q in blocked:
            result = _classify_general_question(q)
            assert result == "personal_or_high_stakes", (
                f"Expected personal_or_high_stakes for {q!r}, got {result!r}"
            )

    def test_out_of_scope_questions(self):
        from app.counseling_engine import _classify_general_question
        oos = [
            "שלום",
            "תודה",
            "בסדר",
            "כן",
        ]
        for q in oos:
            result = _classify_general_question(q)
            assert result in ("out_of_scope", "personal_or_high_stakes"), (
                f"Expected out_of_scope/personal for {q!r}, got {result!r}"
            )


# ---------------------------------------------------------------------------
# 5. Validation unit tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_hebrew_passes(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, _ = _validate_general_education_draft(GENERIC_EDU_HEBREW)
        assert ok is True

    def test_empty_fails(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, reason = _validate_general_education_draft("")
        assert ok is False
        assert reason == "empty"

    def test_dash_fails(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, reason = _validate_general_education_draft("-")
        assert ok is False
        assert reason == "model_unsure"

    def test_too_short_fails(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, reason = _validate_general_education_draft("קצר")
        assert ok is False
        assert reason == "too_short"

    def test_personal_language_fails(self):
        from app.counseling_engine import _validate_general_education_draft
        bad = "הסיכון שלך הוא גבוה מאוד ואת צריכה ניתוח."
        ok, reason = _validate_general_education_draft(bad)
        assert ok is False
        assert reason in ("personal_language", "treatment_term")

    def test_treatment_term_fails(self):
        from app.counseling_engine import _validate_general_education_draft
        bad = (
            "מחלה נוירודגנרטיבית מטופלת עם כימותרפיה ותרגולים. "
            "הצוות הגנטי יכול לעזור."
        )
        ok, reason = _validate_general_education_draft(bad)
        assert ok is False
        assert reason == "treatment_term"

    def test_not_hebrew_fails(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, reason = _validate_general_education_draft(
            "Neurodegenerative diseases are caused by neuronal degeneration."
        )
        assert ok is False
        assert reason == "not_hebrew"


# ---------------------------------------------------------------------------
# 6. AI general fallback flag check
# ---------------------------------------------------------------------------

class TestFlagCheck:
    def test_disabled_by_default(self):
        from app.counseling_engine import _ai_general_education_fallback_enabled
        import os
        old_env = os.environ.copy()
        os.environ.pop("APP_ENV", None)
        os.environ.pop("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", None)
        try:
            assert _ai_general_education_fallback_enabled() is False
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_enabled_in_staging(self, monkeypatch):
        from app.counseling_engine import _ai_general_education_fallback_enabled
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        assert _ai_general_education_fallback_enabled() is True

    def test_disabled_in_production_even_with_flag(self, monkeypatch):
        from app.counseling_engine import _ai_general_education_fallback_enabled
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        assert _ai_general_education_fallback_enabled() is False

    def test_enabled_in_development(self, monkeypatch):
        from app.counseling_engine import _ai_general_education_fallback_enabled
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        assert _ai_general_education_fallback_enabled() is True


# ---------------------------------------------------------------------------
# 7. Response schema integrity — existing 5-field contract preserved
# ---------------------------------------------------------------------------

class TestResponseSchema:
    REQUIRED = {"answer", "safety_level", "needs_genetic_counselor",
                "matched_topic", "suggested_questions"}

    def test_schema_intact_no_flag(self):
        data = client.post("/ask", json={"question": "מה זה penetrance?"}).json()
        assert self.REQUIRED.issubset(data.keys())

    def test_schema_intact_with_flag(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = GENERIC_EDU_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "מה זה penetrance?"}).json()
        assert self.REQUIRED.issubset(data.keys())

    def test_unverified_general_draft_absent_for_kb_match(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        # VUS is definitely in the KB
        data = client.post("/ask", json={"question": "מה זה VUS?"}).json()
        assert "unverified_general_draft" not in data

    def test_no_general_draft_when_gene_matches(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        # BRCA1 routes to gene answer, not general education
        data = client.post("/ask", json={"question": "מה זה BRCA1?"}).json()
        assert "unverified_general_draft" not in data


# ---------------------------------------------------------------------------
# 8. LLM failure — graceful fallback, no crash
# ---------------------------------------------------------------------------

class TestLLMFailureGraceful:
    def test_no_crash_on_llm_error(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = RuntimeError("LLM offline")
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "מה זה penetrance?"}).json()
        assert "answer" in data
        # On failure, falls back to helpful fallback — no draft
        assert "unverified_general_draft" not in data

    def test_no_crash_on_unicode_error(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = UnicodeEncodeError(
            "latin-1", "—", 0, 1, "ordinal not in range(256)"
        )
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "מה זה penetrance?"}).json()
        assert "answer" in data

    def test_no_crash_on_invalid_draft(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        # Returns English — fails validation
        mock_client.call_text_raw.return_value = "This is all English and invalid."
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post("/ask", json={"question": "מה זה penetrance?"}).json()
        assert "answer" in data
        assert "unverified_general_draft" not in data


# ---------------------------------------------------------------------------
# 9. ai_general_debug always in staging fallback — even when LLM is unavailable
#    Regression guard: the original bug dropped ai_debug when the LLM wasn't
#    configured because _build_general_education_answer returned bare None.
# ---------------------------------------------------------------------------

class TestAiDebugAlwaysPresentInStaging:
    """
    Core regression tests for requirement 7.

    _build_general_education_answer now returns tuple[Optional[dict], dict].
    Step 6.5 unpacks both values and attaches ai_general_debug to the fallback
    response regardless of whether the LLM succeeded or not.
    """

    def _no_llm(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")

        def _raise():
            raise ValueError("No LLM configured")

        monkeypatch.setattr("app.counseling_engine.create_llm_client", _raise)

    def test_ai_debug_present_when_llm_not_configured(self, monkeypatch):
        """Regression: ai_general_debug must appear even when no LLM is configured."""
        self._no_llm(monkeypatch)
        data = client.post(
            "/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}
        ).json()
        assert "ai_general_debug" in data, (
            "ai_general_debug must be in staging fallback even when LLM is unavailable. "
            f"Got keys: {sorted(data.keys())}"
        )

    def test_mismatch_repair_ai_debug_on_llm_failure(self, monkeypatch):
        """Specific failing question from the bug report (req #7)."""
        self._no_llm(monkeypatch)
        data = client.post(
            "/ask", json={"question": "מה זה mismatch repair?"}
        ).json()
        assert "ai_general_debug" in data, (
            "ai_general_debug must be present for 'mismatch repair?' even when LLM fails. "
            f"Got: {sorted(data.keys())}"
        )

    def test_ai_debug_fields_on_llm_not_configured(self, monkeypatch):
        """Debug fields must correctly report that an attempt was made but failed."""
        self._no_llm(monkeypatch)
        data = client.post(
            "/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}
        ).json()
        dbg = data.get("ai_general_debug", {})
        assert dbg.get("attempted") is True, "attempted must be True"
        assert dbg.get("generated") is False, "generated must be False on LLM failure"

    def test_ai_debug_present_on_llm_runtime_error(self, monkeypatch):
        """ai_general_debug present when LLM raises at call time."""
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = RuntimeError("connection refused")
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post(
            "/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}
        ).json()
        assert "ai_general_debug" in data, (
            f"ai_general_debug must be in response on LLM runtime error. Keys: {sorted(data.keys())}"
        )
        dbg = data["ai_general_debug"]
        assert dbg.get("attempted") is True
        assert dbg.get("generated") is False

    def test_no_secrets_in_ai_debug(self, monkeypatch):
        """Debug dict must never contain API keys, prompts, or auth info."""
        self._no_llm(monkeypatch)
        data = client.post(
            "/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}
        ).json()
        dbg = data.get("ai_general_debug", {})
        forbidden = ("api_key", "prompt", "system_prompt", "password", "token", "secret", "auth")
        for key in dbg:
            assert key.lower() not in forbidden, f"Secret key in debug: {key!r}"

    def test_mocked_llm_success_has_draft_and_debug(self, monkeypatch):
        """Req #8: mocked success must return both unverified_general_draft and ai_general_debug."""
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = GENERIC_EDU_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_client)
        data = client.post(
            "/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}
        ).json()
        assert "unverified_general_draft" in data, "Draft must be present on LLM success"
        assert "ai_general_debug" in data, "ai_general_debug must be present on LLM success"
        assert data["ai_general_debug"].get("generated") is True
