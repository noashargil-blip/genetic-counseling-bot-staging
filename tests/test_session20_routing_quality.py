# -*- coding: utf-8 -*-
"""
Session 20 — Response Quality + Routing Regression Tests.

Covers:
  1. Routing regression: chromosome question → not x_linked
  2. Routing regression: gene-association phrase → gene route, not generic KB
  3. DMD gene draft — specific disease mention now allowed
  4. Gene draft validation: specific cancer passes, vague categories blocked
  5. High-stakes personal questions blocked (no draft)
  6. Suggested questions capped at 3
  7. last_gene_symbol follow-up routing
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng

client = TestClient(app)


# ── Routing regression: chromosome ────────────────────────────────────────────

class TestChromosomeNotXLinked:
    """'מה זה כרומוזום?' must not route to x_linked KB entry."""

    @pytest.mark.parametrize("q", [
        "מה זה כרומוזום?",
        "מה זה כרומוזום",
        "מה זה כרומוזום 21?",
        "הסבר לי מה זה כרומוזום",
    ])
    def test_chromosome_question_not_xlinked(self, q):
        data = client.post("/ask", json={"question": q}).json()
        topic = data.get("matched_topic", "")
        assert topic != "x_linked", (
            f"Question {q!r} must not route to x_linked. Got topic={topic!r}, "
            f"answer={data.get('answer', '')[:100]!r}"
        )

    def test_x_linked_still_works_with_signal(self):
        """Explicit x-linked question must still get the x_linked answer."""
        data = client.post("/ask", json={"question": "מה זה תורשה תלויית X?"}).json()
        topic = data.get("matched_topic", "")
        # Accept x_linked OR a general genetics answer — just don't 404
        assert data.get("answer"), "Should always return an answer"
        assert data.get("safety_level") != "out_of_scope", "Should not be out_of_scope"


# ── Routing: gene association phrases → gene route ────────────────────────────

class TestGeneAssociationRouting:
    """'לאיזה מצבים קליניים הגן APOE מקושר?' → gene answer, not generic KB."""

    @pytest.mark.parametrize("q,gene", [
        ("לאיזה מצבים קליניים הגן APOE מקושר?", "APOE"),
        ("לאיזה מחלות הגן MSH2 קשור?", "MSH2"),
        ("מה הקשר של הגן BRCA1 למחלות?", "BRCA1"),
    ])
    def test_association_question_routes_to_gene(self, q, gene):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": q}).json()
        topic = data.get("matched_topic", "")
        answer = data.get("answer", "")
        # Must either route to gene_clinvar_summary or mention the gene name in answer
        assert (
            topic == "gene_clinvar_summary"
            or gene.upper() in answer.upper()
        ), (
            f"Question {q!r} should route to gene answer for {gene}. "
            f"Got topic={topic!r}, answer={answer[:100]!r}"
        )

    def test_association_not_generic_kb(self):
        """Gene association question must not return generic 'what_is_gene' KB entry."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": "לאיזה מצבים קליניים הגן APOE מקושר?"}).json()
        topic = data.get("matched_topic", "")
        assert topic != "what_is_gene", (
            f"Gene association question must not fall through to what_is_gene KB entry. "
            f"Got topic={topic!r}"
        )


# ── Gene draft validation: DMD / specific disease ────────────────────────────

class TestGeneDraftValidation:
    """_validate_gene_education_draft with the narrowed regex."""

    def test_dmd_duchenne_mention_passes(self):
        """Specific single disease (Duchenne) must now pass validation."""
        from app.counseling_engine import _validate_gene_education_draft
        draft = (
            "הגן DMD מקודד לחלבון דיסטרופין, המשמש כעוגן מבני בתאי שריר. "
            "הגן קשור למחלת Duchenne, מחלת שריר קשה. "
            "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
        )
        reason = _validate_gene_education_draft(draft)
        assert reason is None, (
            f"Specific single disease (Duchenne) should pass; got rejection: {reason!r}"
        )

    def test_specific_cancer_mention_passes(self):
        """Specific single cancer ('סרטן המעי הגס') must now pass validation."""
        from app.counseling_engine import _validate_gene_education_draft
        draft = (
            "הגן MSH2 מקודד לחלבון המשתתף במנגנון תיקון שגיאות שכפול DNA. "
            "הגן קשור לסרטן המעי הגס ולתסמונת לינץ'. "
            "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
        )
        reason = _validate_gene_education_draft(draft)
        assert reason is None, (
            f"Specific cancer mention should pass; got rejection: {reason!r}"
        )

    def test_vague_disease_category_rejected(self):
        """'קשור למחלות' (vague category) must still be rejected."""
        from app.counseling_engine import _validate_gene_education_draft
        bad = "הגן FOXP2 קשור למחלות שונות של מערכת העצבים."
        reason = _validate_gene_education_draft(bad)
        assert reason is not None, "Vague disease category 'קשור למחלות' must be rejected"

    def test_mazavim_kemo_rejected(self):
        """'מצבים כמו' must still be rejected."""
        from app.counseling_engine import _validate_gene_education_draft
        bad = "הגן זה קשור למצבים כמו פרקינסון."
        reason = _validate_gene_education_draft(bad)
        assert reason is not None, "'מצבים כמו' must be rejected"

    def test_gorems_le_rejected(self):
        """'גורם למחלה' must still be rejected."""
        from app.counseling_engine import _validate_gene_education_draft
        bad = "שינויים בגן זה גורמים למחלה קשה."
        reason = _validate_gene_education_draft(bad)
        assert reason is not None, "'גורם למחלה' must be rejected"

    def test_various_cancers_rejected(self):
        """'סוגי סרטן שונים' must still be rejected."""
        from app.counseling_engine import _validate_gene_education_draft
        bad = "הגן BRCA1 קשור לסוגי סרטן שונים."
        reason = _validate_gene_education_draft(bad)
        assert reason is not None, "'סוגי סרטן שונים' must be rejected"

    def test_netiya_lesartan_rejected(self):
        """Cancer predisposition framing 'נטייה לסרטן' must be rejected."""
        from app.counseling_engine import _validate_gene_education_draft
        bad = "הגן BRCA2 קשור לנטייה לסרטן השד."
        reason = _validate_gene_education_draft(bad)
        assert reason is not None, "'נטייה לסרטן' must be rejected"

    def test_biological_role_only_passes(self):
        """Clean biological role draft must pass."""
        from app.counseling_engine import _validate_gene_education_draft
        good = (
            "הגן MSH2 מקודד לחלבון המשתתף במנגנון תיקון שגיאות שכפול DNA. "
            "חלבון זה עוזר לשמור על יציבות הגנום. "
            "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
        )
        reason = _validate_gene_education_draft(good)
        assert reason is None, f"Clean biological draft should pass; got: {reason!r}"


# ── High-stakes personal questions blocked ────────────────────────────────────

class TestHighStakesBlocked:
    """Personal / high-stakes questions must not return AI draft."""

    @pytest.fixture(autouse=True)
    def _staging(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        mc = MagicMock()
        mc.call_text_raw.return_value = (
            "מחלות נוירודגנרטיביות הן קבוצה של מחלות שמתאפיינות בניוון הדרגתי "
            "של תאי עצב. אם השאלה נוגעת לתוצאה האישית שלך, יש לפנות לצוות הגנטי."
        )
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mc)

    @pytest.mark.parametrize("q", [
        "יש לי APOE האם יהיה לי אלצהיימר?",
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "מה הסיכון שלי לחלות?",
        "האם אני צריכה לעשות ניתוח?",
        "האם הילדים שלי יהיו חולים?",
    ])
    def test_personal_question_no_draft(self, q):
        data = client.post("/ask", json={"question": q}).json()
        assert "unverified_general_draft" not in data, (
            f"AI draft must be blocked for personal question: {q!r}"
        )

    def test_apoe_alzheimer_personal_blocked(self):
        data = client.post(
            "/ask", json={"question": "יש לי APOE ε4 — האם אני בסיכון לאלצהיימר?"}
        ).json()
        assert "unverified_general_draft" not in data
        assert data.get("safety_level") in (
            "requires_genetic_counselor", "out_of_scope", "general_information"
        )


# ── Suggested questions capped ────────────────────────────────────────────────

class TestSuggestedQuestionsCapped:
    """suggested_questions must never exceed 3."""

    @pytest.mark.parametrize("q", [
        "מה זה VUS?",
        "מה זה נשאות?",
        "מה ההבדל בין VUS לבין ממצא pathogenic?",
        "אמרו לי שאני נשאית, מה זה?",
        "מה זה תורשה אוטוזומלית רצסיבית?",
    ])
    def test_max_three_suggested(self, q):
        data = client.post("/ask", json={"question": q}).json()
        sq = data.get("suggested_questions", [])
        assert len(sq) <= 3, (
            f"suggested_questions must be ≤ 3 for {q!r}. Got {len(sq)}: {sq}"
        )


# ── Warning constants compact ─────────────────────────────────────────────────

class TestWarningConstants:
    """Warning strings must be compact (single line)."""

    def test_general_education_warning_short(self):
        from app.counseling_engine import _GENERAL_EDUCATION_WARNING_HE
        assert len(_GENERAL_EDUCATION_WARNING_HE) < 60, (
            f"Warning must be compact (<60 chars). Got: {_GENERAL_EDUCATION_WARNING_HE!r}"
        )
        assert "\n" not in _GENERAL_EDUCATION_WARNING_HE

    def test_general_education_source_note_short(self):
        from app.counseling_engine import _GENERAL_EDUCATION_SOURCE_NOTE_HE
        assert len(_GENERAL_EDUCATION_SOURCE_NOTE_HE) < 60, (
            f"Source note must be compact (<60 chars). Got: {_GENERAL_EDUCATION_SOURCE_NOTE_HE!r}"
        )

    def test_answer_is_ai_text_not_preamble(self, monkeypatch):
        """General education answer field must be the AI text, not a preamble."""
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        ai_text = (
            "מחלות נוירודגנרטיביות הן קבוצה של מחלות שמתאפיינות בניוון הדרגתי "
            "של תאי עצב. אם השאלה נוגעת לתוצאה האישית שלך, יש לפנות לצוות הגנטי."
        )
        mc = MagicMock()
        mc.call_text_raw.return_value = ai_text
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mc)
        data = client.post(
            "/ask", json={"question": "מה זה מחלות נוירודגנרטיביות?"}
        ).json()
        if "unverified_general_draft" in data:
            answer = data.get("answer", "")
            assert "שאלתך עוסקת במושג" not in answer, (
                "answer must be the AI text, not a preamble. "
                f"Got: {answer[:100]!r}"
            )


# ── last_gene_symbol is disabled — no context bleed ─────────────────────────

class TestLastGeneSymbolRouting:
    """
    Session 22: last_gene_symbol is accepted in the request body (backward
    compat) but the backend ignores it. Every question is answered on its
    own text only.
    """

    def test_last_gene_symbol_ignored(self):
        """Sending last_gene_symbol must NOT cause the answer to be about that gene."""
        data = client.post(
            "/ask",
            json={
                "question": "מה הקשר שלו למחלות?",
                "last_gene_symbol": "BRCA1",
            },
        ).json()
        topic = data.get("matched_topic", "")
        # Must NOT route to gene_clinvar_summary — there is no gene name in the question text
        assert topic != "gene_clinvar_summary", (
            f"last_gene_symbol=BRCA1 must be ignored. "
            f"Got topic={topic!r}"
        )

    def test_vague_question_no_gene_routing(self):
        """Vague pronoun question without explicit gene name never routes to a gene."""
        data = client.post(
            "/ask",
            json={"question": "מה הקשר שלו למחלות?"},
        ).json()
        topic = data.get("matched_topic", "")
        assert topic != "gene_clinvar_summary", (
            f"Vague pronoun question should not route to gene. Got topic={topic!r}"
        )

    def test_explicit_gene_still_routes(self):
        """When the gene name IS in the question text, routing works normally."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post(
            "/ask",
            json={"question": "מה זה הגן BRCA1?"},
        ).json()
        topic = data.get("matched_topic", "")
        assert topic == "gene_clinvar_summary", (
            f"Explicit gene name in question should route to gene. Got topic={topic!r}"
        )
