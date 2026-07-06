# -*- coding: utf-8 -*-
"""
Session 23 — Patient value: educational personal context, trisomy21, VUS options.

Covers:
  1. Trisomy 21 / Down syndrome questions → educational answer, not blocked
  2. VUS options questions → practical options answer
  3. Educational personal context → routed to education, not blocked
  4. Personal decision questions still blocked
  5. Fallback is helpful, not restrictive
  6. Gene AI summary with mocked index + OpenAI
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng

client = TestClient(app)


# ── 1. Trisomy 21 / Down syndrome educational ────────────────────────────────

class TestTrisomy21Educational:
    """Questions about trisomy 21 must return an educational answer, never a refusal."""

    @pytest.mark.parametrize("q", [
        "מה זה טריזומיה 21?",
        "מה זה כרומוזום 21 עודף?",
        "הרופא אמר שיש לתינוק שלי כרומוזום 21 עודף, מה זה אומר?",
        "מה זה כרומוזום 21 נוסף?",
        "מה זה תסמונת דאון?",
    ])
    def test_trisomy21_returns_educational_answer(self, q):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        answer = data.get("answer", "")
        assert safety == "general_information", (
            f"Trisomy21 question must not be blocked. Got safety={safety!r} for {q!r}"
        )
        assert len(answer) > 50, f"Answer too short for {q!r}: {answer!r}"
        # Must mention trisomy21 or Down syndrome
        mention = (
            "דאון" in answer
            or "down" in answer.lower()
            or "טריזומי" in answer
            or "trisomy" in answer.lower()
            or "כרומוזום 21" in answer
        )
        assert mention, (
            f"Answer for {q!r} must mention trisomy21/Down syndrome:\n{answer[:300]!r}"
        )

    def test_trisomy21_matched_topic(self):
        data = client.post("/ask", json={"question": "מה זה טריזומיה 21?"}).json()
        topic = data.get("matched_topic", "")
        assert topic == "trisomy21", f"Expected topic=trisomy21, got {topic!r}"

    def test_trisomy21_has_suggested_questions(self):
        data = client.post("/ask", json={"question": "מה זה כרומוזום 21 עודף?"}).json()
        sq = data.get("suggested_questions", [])
        assert len(sq) > 0, "Trisomy21 answer must have suggested questions"
        assert len(sq) <= 3, "Suggested questions must be ≤ 3"

    def test_trisomy21_abortion_still_blocked(self):
        """'האם לעשות הפלה בגלל כרומוזום 21?' must still be blocked."""
        data = client.post(
            "/ask", json={"question": "האם לעשות הפלה בגלל כרומוזום 21?"}
        ).json()
        safety = data.get("safety_level", "")
        assert safety == "requires_genetic_counselor", (
            f"Abortion question must be blocked. Got safety={safety!r}"
        )

    def test_trisomy21_not_xlinked(self):
        data = client.post("/ask", json={"question": "מה זה כרומוזום 21 עודף?"}).json()
        assert data.get("matched_topic") != "x_linked"


# ── 2. VUS options answer ────────────────────────────────────────────────────

class TestVusOptionsAnswer:
    """VUS + options/steps questions must return a practical answer, not a refusal."""

    @pytest.mark.parametrize("q,gene", [
        ("אמרו לי שיש לי VUS בגן BRCA1, מהן האפשרויות העומדות מולי?", "BRCA1"),
        ("יש לי VUS בBRCA2, מה לעשות?", "BRCA2"),
        ("יש לי VUS ב-NF1, מה האפשרויות?", "NF1"),
    ])
    def test_vus_options_returns_educational_answer(self, q, gene):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        answer = data.get("answer", "")
        assert safety == "general_information", (
            f"VUS options question must not be blocked. Got safety={safety!r} for {q!r}"
        )
        assert len(answer) > 60, f"VUS options answer too short for {q!r}: {answer!r}"

    def test_vus_options_content(self):
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS בגן BRCA1, מהן האפשרויות העומדות מולי?"},
        ).json()
        answer = data.get("answer", "")
        # Must contain practical info about VUS, not just "VUS is uncertain"
        has_practical = (
            "לא מקבלים החלטות" in answer
            or "תיעוד" in answer
            or "עתיד" in answer
            or "אפשרויות" in answer
            or "אפשר" in answer
            or "צוות" in answer
        )
        assert has_practical, (
            f"VUS options answer must contain practical information:\n{answer[:400]!r}"
        )

    def test_vus_options_topic(self):
        data = client.post(
            "/ask",
            json={"question": "יש לי VUS ב-NF1, מהן האפשרויות?"},
        ).json()
        assert data.get("matched_topic") == "vus_known_gene", (
            f"VUS options must route to vus_known_gene topic. Got {data.get('matched_topic')!r}"
        )

    def test_detect_vus_options_request_direct(self):
        """_is_vus_options_request detects options-seeking phrases."""
        assert ceng._is_vus_options_request("מהן האפשרויות העומדות מולי?")
        assert ceng._is_vus_options_request("מה לעשות עם זה?")
        assert ceng._is_vus_options_request("what are my options")
        assert not ceng._is_vus_options_request("מה זה VUS?")

    def test_build_vus_options_answer_direct(self):
        """_build_vus_options_answer returns a clean dict with practical info."""
        result = ceng._build_vus_options_answer("BRCA1")
        assert result["safety_level"] == "general_information"
        assert "BRCA1" in result["answer"]
        assert result["needs_genetic_counselor"] is False
        assert len(result["answer"]) > 50


# ── 3. Educational personal context ─────────────────────────────────────────

class TestEducationalPersonalContext:
    """Personal phrasing that seeks education must route to education, not be blocked."""

    def test_amru_li_carrier(self):
        """'אמרו לי שאני נשאית, מה זה?' must not be blocked."""
        data = client.post("/ask", json={"question": "אמרו לי שאני נשאית, מה זה?"}).json()
        safety = data.get("safety_level", "")
        assert safety != "requires_genetic_counselor", (
            f"'אמרו לי שאני נשאית' must not be blocked. Got safety={safety!r}"
        )

    def test_rofeh_amar_vus(self):
        """'הרופא אמר שיש לי VUS ב-MSH2, מה זה אומר?' → educational."""
        data = client.post(
            "/ask", json={"question": "הרופא אמר שיש לי VUS ב-MSH2, מה זה אומר?"}
        ).json()
        safety = data.get("safety_level", "")
        assert safety != "requires_genetic_counselor", (
            f"Educational VUS question must not be blocked. Got safety={safety!r}"
        )

    def test_is_educational_personal_context_direct(self):
        """_is_educational_personal_context correctly classifies."""
        # Should be educational
        assert ceng._is_educational_personal_context("אמרו לי שיש לי VUS, מה זה?")
        assert ceng._is_educational_personal_context("הרופא אמר שיש לתינוק שלי כרומוזום 21 עודף")
        assert ceng._is_educational_personal_context("נמצא אצלי ממצא, מה זה אומר?")
        # Should NOT be educational (has decision block terms)
        assert not ceng._is_educational_personal_context("אמרו לי שיש לי VUS, האם לעשות ניתוח?")
        assert not ceng._is_educational_personal_context("הרופא אמר שמה הסיכון שלי?")
        # No personal phrase → not educational personal context
        assert not ceng._is_educational_personal_context("מה זה VUS?")

    def test_detect_trisomy21_direct(self):
        """_detect_trisomy21 correctly identifies trisomy 21 signals."""
        assert ceng._detect_trisomy21("כרומוזום 21 עודף")
        assert ceng._detect_trisomy21("טריזומיה 21")
        assert ceng._detect_trisomy21("trisomy 21")
        assert ceng._detect_trisomy21("תסמונת דאון")
        assert ceng._detect_trisomy21("Down syndrome")
        assert not ceng._detect_trisomy21("מה זה BRCA1?")
        assert not ceng._detect_trisomy21("כרומוזום X")


# ── 4. Personal decisions still blocked ─────────────────────────────────────

class TestPersonalDecisionsStillBlocked:
    """Medical decision requests must still be blocked."""

    @pytest.mark.parametrize("q", [
        "האם אני צריכה ניתוח בגלל BRCA1?",
        "מה הסיכון שלי לחלות?",
        "האם לעשות הפלה?",
        "אני צריכה כריתה, מה לעשות?",
    ])
    def test_personal_decision_blocked(self, q):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        assert safety == "requires_genetic_counselor", (
            f"Personal decision must be blocked. Got safety={safety!r} for {q!r}"
        )

    def test_personal_with_educational_phrasing_but_surgery_blocked(self):
        """'אמרו לי שיש לי VUS, לעשות ניתוח?' must still be blocked."""
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS בBRCA1, לעשות ניתוח?"},
        ).json()
        safety = data.get("safety_level", "")
        assert safety == "requires_genetic_counselor", (
            f"Surgery request must be blocked even with personal prefix. Got {safety!r}"
        )


# ── 5. Helpful fallback ──────────────────────────────────────────────────────

class TestHelpfulFallback:
    """Fallback message must be helpful, not a restrictive scope statement."""

    RESTRICTIVE_PHRASES = [
        "אני יכול/ה לענות רק על בסיס מידע כללי שאושר מראש",
        "לא ניתן לקבוע משמעות אישית",
    ]

    def test_fallback_not_restrictive(self):
        """Out-of-scope fallback must not open with a restrictive 'I can only...' sentence."""
        data = client.post("/ask", json={"question": "מה שעות הפתיחה של הקליניקה?"}).json()
        answer = data.get("answer", "")
        for phrase in self.RESTRICTIVE_PHRASES:
            assert phrase not in answer, (
                f"Restrictive phrase found in fallback: {phrase!r}\nAnswer: {answer[:200]!r}"
            )


# ── 6. Gene AI summary with mocked index + OpenAI ───────────────────────────

class TestGeneAiSummaryMocked:
    """Gene AI summary must function for unapproved genes when OpenAI + index are mocked."""

    @pytest.fixture
    def mock_foxp2_environment(self, monkeypatch):
        """Mock gene index + OpenAI so FOXP2 gets an AI summary in Tier 2."""
        import app.gene_index as gene_index
        import app.gene_cards as gene_cards
        import app.gene_knowledge as gene_knowledge

        # Make gene index appear available with a ClinVar record for FOXP2
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)

        foxp2_summary = {
            "gene_symbol": "FOXP2",
            "total_variants": 120,
            "by_significance": {"pathogenic": 3, "vus": 80, "benign": 37},
            "phenotypes": ["speech disorder", "language delay"],
        }
        monkeypatch.setattr(gene_index, "get_gene_summary",
                            lambda g: foxp2_summary if g == "FOXP2" else None)

        # No approved card or knowledge entry for FOXP2 → forces Tier 2
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)

        # Mock _extract_gene_with_correction to find FOXP2 in the question text
        monkeypatch.setattr(
            ceng, "_extract_gene_with_correction",
            lambda text: ("FOXP2", None) if "FOXP2" in text.upper() else (None, None),
        )

        # Mock OpenAI to return a valid Hebrew educational text
        ai_text = (
            "הגן FOXP2 מקודד לגורם שעתוק המעורב בהתפתחות יכולות דיבור ושפה. "
            "הוא ממוקם על כרומוזום 7 ומשחק תפקיד בהתפתחות מוחית."
        )
        mc = MagicMock()
        mc.call_text_raw.return_value = ai_text
        monkeypatch.setattr(ceng, "create_llm_client", lambda: mc)

        return ai_text

    def test_foxp2_gets_ai_summary(self, mock_foxp2_environment):
        """FOXP2 must return an AI summary when gene index and OpenAI are mocked."""
        data = client.post("/ask", json={"question": "מה זה הגן FOXP2?"}).json()
        answer = data.get("answer", "")
        topic = data.get("matched_topic", "")
        assert topic == "gene_clinvar_summary", (
            f"FOXP2 should route to gene_clinvar_summary. Got {topic!r}"
        )
        bland_phrases = ["אין עדיין סיכום", "אין עדיין מידע", "לא נמצא"]
        for phrase in bland_phrases:
            assert phrase not in answer, (
                f"FOXP2 with mocked AI must not return bland fallback: {phrase!r}\nAnswer: {answer!r}"
            )
