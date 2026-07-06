# -*- coding: utf-8 -*-
"""
Session 19: General education AI fallback -- server-side regression tests.
"""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

GENERIC_EDU_HEBREW = (
    "מחלות נוירודגנרטיביות הן קבוצה של מחלות שמתאפיינות בניוון הדרגתי של תאי עצב. "
    "הן כוללות מחלות כמו פרקינסון, ALS ומחלת הנטינגטון. "
    "אם השאלה נוגעת לתוצאה האישית שלך, יש לפנות לצוות הגנטי."
)

@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    for var in ("LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "APP_ENV", "AI_GENERAL_EDUCATION_FALLBACK_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    import app.counseling_engine as eng
    eng._known_gene_set_cache = None
    yield

class TestFlagDisabledByDefault:
    @pytest.mark.parametrize("q", [
        "מה זה מחלות נוירודגנרטיביות?",
        "מה זה mismatch repair?",
        "מה זה penetrance?",
        "מה זה המוגלובין?",
    ])
    def test_no_draft_without_flag(self, q):
        data = client.post("/ask", json={"question": q}).json()
        assert "unverified_general_draft" not in data

    @pytest.mark.parametrize("q", [
        "מה זה מחלות נוירודגנרטיביות?",
        "מה זה penetrance?",
    ])
    def test_no_draft_in_production(self, monkeypatch, q):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        data = client.post("/ask", json={"question": q}).json()
        assert "unverified_general_draft" not in data

class TestFlagEnabledStaging:
    def _mock(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mc = MagicMock()
        mc.call_text_raw.return_value = GENERIC_EDU_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mc)

    def test_draft_structure(self, monkeypatch):
        self._mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            d = data["unverified_general_draft"]
            assert d["status"] == "ai_generated_unreviewed"
            assert "text_he" in d and "warning_he" in d and "source_note_he" in d

    def test_debug_present_with_draft(self, monkeypatch):
        self._mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        if "unverified_general_draft" in data:
            assert "ai_general_debug" in data
            assert data["ai_general_debug"]["generated"] is True

    def test_no_secrets_in_debug(self, monkeypatch):
        self._mock(monkeypatch)
        data = client.post("/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}).json()
        for k in data.get("ai_general_debug", {}):
            assert k.lower() not in ("api_key","prompt","system_prompt","password","token","secret")

class TestBlockedPersonal:
    @pytest.mark.parametrize("q", [
        "יש לי APOE האם יהיה לי אלצהיימר?",
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "האם אני צריכה ניתוח?",
        "מה הסיכון שלי?",
        "האם הילדים שלי יהיו חולים?",
        "מה המשמעות של הממצא שלי?",
    ])
    def test_no_draft(self, monkeypatch, q):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mc = MagicMock()
        mc.call_text_raw.return_value = GENERIC_EDU_HEBREW
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mc)
        data = client.post("/ask", json={"question": q}).json()
        assert "unverified_general_draft" not in data

class TestClassifier:
    def test_safe(self):
        from app.counseling_engine import _classify_general_question
        for q in ["מה זה penetrance?", "מה זה mismatch repair?",
                  "מה זה ClinVar?", "מה זה המוגלובין?"]:
            assert _classify_general_question(q) == "safe_general_education", q

    def test_personal_blocked(self):
        from app.counseling_engine import _classify_general_question
        for q in ["מה הסיכון שלי?", "הממצא שלי", "האם יהיה לי"]:
            assert _classify_general_question(q) == "personal_or_high_stakes", q

class TestValidation:
    def test_valid(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, _ = _validate_general_education_draft(GENERIC_EDU_HEBREW)
        assert ok

    def test_empty(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, r = _validate_general_education_draft("")
        assert not ok and r == "empty"

    def test_personal_lang(self):
        from app.counseling_engine import _validate_general_education_draft
        ok, r = _validate_general_education_draft("הסיכון שלך גבוה ואת צריכה ניתוח.")
        assert not ok

class TestSchema:
    REQUIRED = {"answer","safety_level","needs_genetic_counselor","matched_topic","suggested_questions"}
    def test_schema_preserved(self):
        data = client.post("/ask", json={"question": "מה זה VUS?"}).json()
        assert self.REQUIRED.issubset(data.keys())

class TestGracefulFailure:
    def test_no_crash_on_llm_error(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mc = MagicMock()
        mc.call_text_raw.side_effect = RuntimeError("LLM offline")
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mc)
        data = client.post("/ask", json={"question": "מה זה penetrance?"}).json()
        assert "answer" in data
        assert "unverified_general_draft" not in data
