# -*- coding: utf-8 -*-
"""
Session 24 — AI-first educational behavior + remove referral overuse.

Covers:
  1. No referral overuse in ordinary educational answers
  2. Gene AI summary (FOXP2/MTHFR) with mocked OpenAI
  3. Extra chromosome (XXY/XXX/XYY/כרומוזום X עודף) → educational, not x_linked
  4. Educational personal context for extra chromosome
  5. General medical education questions → educational answer (mocked AI)
  6. Out-of-domain questions → short out-of-scope response
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng

client = TestClient(app)

# Boilerplate phrases injected by AI prompts that must NOT appear in answers.
# (Does NOT ban "לפנות לצוות הגנטי" which may legitimately appear in KB content.)
_BANNED_REFERRAL_PHRASES = [
    "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי",
    "המשמעות האישית נקבעת על ידי הצוות הגנטי",
    "לפרשנות אישית יש לפנות לצוות הגנטי",
    "לתשובה אישית לפי תוצאות הבדיקה שלך",
]


# ── 1. No referral overuse ───────────────────────────────────────────────────

class TestNoReferralOveruse:
    """Ordinary educational answers must not contain boilerplate referral sentences."""

    @pytest.mark.parametrize("q", [
        "מה זה VUS?",
        "מה זה נשאות?",
        "מה זה תורשה אוטוזומלית רצסיבית?",
        "מה זה מוטציה?",
        "מה זה כרומוזום?",
    ])
    def test_no_referral_in_ordinary_answers(self, q):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        if safety == "requires_genetic_counselor":
            return  # blocked answers may have referral text
        answer = data.get("answer", "")
        for phrase in _BANNED_REFERRAL_PHRASES:
            assert phrase not in answer, (
                f"Banned referral phrase in answer for {q!r}:\n"
                f"  Phrase: {phrase!r}\n"
                f"  Answer: {answer[:300]!r}"
            )


# ── 2. Gene AI summary with mocked OpenAI ───────────────────────────────────

class TestGeneAiSummaryMocked:
    """Gene questions for unrecognized genes get AI summary when index + LLM are mocked."""

    @pytest.fixture
    def mock_foxp2_environment(self, monkeypatch):
        import app.gene_index as gene_index
        import app.gene_cards as gene_cards
        import app.gene_knowledge as gene_knowledge

        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        foxp2_summary = {
            "gene_symbol": "FOXP2",
            "total_variants": 120,
            "by_significance": {"pathogenic": 3, "vus": 80, "benign": 37},
            "phenotypes": ["speech disorder", "language delay"],
        }
        monkeypatch.setattr(gene_index, "get_gene_summary",
                            lambda g: foxp2_summary if g == "FOXP2" else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            ceng, "_extract_gene_with_correction",
            lambda text: ("FOXP2", None) if "FOXP2" in text.upper() else (None, None),
        )
        ai_text = (
            "הגן FOXP2 מקודד לגורם שעתוק המעורב בהתפתחות יכולות דיבור ושפה. "
            "הוא ממוקם על כרומוזום 7 ומשחק תפקיד בהתפתחות מוחית."
        )
        mc = MagicMock()
        mc.call_text_raw.return_value = ai_text
        monkeypatch.setattr(ceng, "create_llm_client", lambda: mc)
        return ai_text

    def test_foxp2_gets_ai_summary(self, mock_foxp2_environment):
        data = client.post("/ask", json={"question": "מה זה הגן FOXP2?"}).json()
        topic = data.get("matched_topic", "")
        answer = data.get("answer", "")
        assert topic == "gene_clinvar_summary", (
            f"FOXP2 must route to gene_clinvar_summary. Got {topic!r}"
        )
        bland_phrases = ["אין עדיין סיכום", "אין עדיין מידע", "לא נמצא"]
        for phrase in bland_phrases:
            assert phrase not in answer, (
                f"FOXP2 with mocked AI must not return bland fallback: {phrase!r}\nAnswer: {answer!r}"
            )

    def test_foxp2_no_referral_in_summary(self, mock_foxp2_environment):
        """AI-generated gene summary must not contain banned referral phrases."""
        data = client.post("/ask", json={"question": "מה זה הגן FOXP2?"}).json()
        answer = data.get("answer", "")
        for phrase in _BANNED_REFERRAL_PHRASES:
            assert phrase not in answer, (
                f"Banned referral phrase in gene AI summary:\n  {phrase!r}\n  Answer: {answer[:300]!r}"
            )


# ── 3. Extra chromosome findings ─────────────────────────────────────────────

class TestChromosomeFindings:
    """XXY/XXX/XYY and 'כרומוזום X עודף' must return chromosomal_finding educational answer."""

    @pytest.mark.parametrize("q", [
        "מה זה XXY?",
        "מה זה תסמונת קליינפלטר?",
        "מה זה Klinefelter?",
        "מה זה XXX?",
        "מה זה XYY?",
        "יש לי כרומוזום X עודף, מה זה?",
        "כרומוזום X נוסף, מה המשמעות?",
    ])
    def test_extra_chromosome_educational_answer(self, q):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        answer = data.get("answer", "")
        assert safety == "general_information", (
            f"Extra chromosome question must not be blocked. Got safety={safety!r} for {q!r}"
        )
        assert len(answer) > 50, f"Answer too short for {q!r}: {answer!r}"

    def test_extra_chromosome_matched_topic(self):
        data = client.post("/ask", json={"question": "מה זה XXY?"}).json()
        topic = data.get("matched_topic", "")
        assert topic == "chromosomal_finding", (
            f"XXY must route to chromosomal_finding. Got {topic!r}"
        )

    def test_extra_chromosome_not_xlinked(self):
        """Extra chromosome questions must NOT route to the x_linked KB entry."""
        for q in ["מה זה XXY?", "כרומוזום X עודף, מה זה?"]:
            data = client.post("/ask", json={"question": q}).json()
            assert data.get("matched_topic") != "x_linked", (
                f"Extra chromosome question must not route to x_linked: {q!r}"
            )

    def test_detect_extra_chromosome_direct(self):
        """_detect_extra_chromosome correctly identifies extra sex chromosome signals."""
        assert ceng._detect_extra_chromosome("מה זה XXY?")
        assert ceng._detect_extra_chromosome("מה זה Klinefelter?")
        assert ceng._detect_extra_chromosome("כרומוזום X עודף")
        assert ceng._detect_extra_chromosome("XYY")
        assert ceng._detect_extra_chromosome("XXX")
        # Trisomy 21 must NOT be captured by extra_chromosome (has its own handler)
        assert not ceng._detect_extra_chromosome("כרומוזום 21 עודף")
        assert not ceng._detect_extra_chromosome("טריזומיה 21")

    def test_extra_chromosome_suggested_questions(self):
        data = client.post("/ask", json={"question": "מה זה XXY?"}).json()
        sq = data.get("suggested_questions", [])
        assert len(sq) > 0, "Extra chromosome answer must have suggested questions"
        assert len(sq) <= 3, "Suggested questions must be ≤ 3"


# ── 4. Educational personal context — extra chromosome ───────────────────────

class TestEducationalPersonalContextS24:
    """Personal phrasing about extra chromosomes must route to education."""

    def test_fetus_extra_x_educational(self):
        """'הרופא אמר שיש לעובר כרומוזום X עודף' → educational answer."""
        data = client.post(
            "/ask",
            json={"question": "הרופא אמר שיש לעובר כרומוזום X עודף, מה זה?"},
        ).json()
        safety = data.get("safety_level", "")
        assert safety == "general_information", (
            f"'כרומוזום X עודף' educational question must not be blocked. Got {safety!r}"
        )

    def test_abortion_because_extra_x_blocked(self):
        """'האם לעשות הפלה בגלל כרומוזום X עודף?' must be blocked."""
        data = client.post(
            "/ask",
            json={"question": "האם לעשות הפלה בגלל כרומוזום X עודף?"},
        ).json()
        safety = data.get("safety_level", "")
        assert safety == "requires_genetic_counselor", (
            f"Abortion request must be blocked. Got {safety!r}"
        )

    def test_xxy_question_not_blocked(self):
        """'אמרו לי שלתינוק שלי יש XXY, מה זה אומר?' must not be blocked."""
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שלתינוק שלי יש XXY, מה זה אומר?"},
        ).json()
        safety = data.get("safety_level", "")
        assert safety == "general_information", (
            f"XXY educational question must not be blocked. Got {safety!r}"
        )


# ── 5. General medical education ─────────────────────────────────────────────

class TestGeneralMedicalEducation:
    """Medical education questions must get informative answers, not refusals."""

    @pytest.mark.parametrize("q", [
        "מה זה שבץ מוחי?",
        "מה זה אלצהיימר?",
        "מה זה מחלה נוירודגנרטיבית?",
    ])
    def test_medical_question_not_blocked(self, q):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        assert safety != "requires_genetic_counselor", (
            f"General medical question must not be blocked. Got {safety!r} for {q!r}"
        )
        answer = data.get("answer", "")
        assert len(answer) > 20, f"Answer too short for {q!r}: {answer!r}"

    def test_alzheimer_death_question(self):
        """'מתים מאלצהיימר?' — medical education question, must not be blocked."""
        data = client.post("/ask", json={"question": "מתים מאלצהיימר?"}).json()
        safety = data.get("safety_level", "")
        assert safety != "requires_genetic_counselor", (
            f"'מתים מאלצהיימר?' must not be blocked. Got {safety!r}"
        )

    def test_classify_general_medical_education(self):
        """_classify_general_question classifies medical questions as safe."""
        result = ceng._classify_general_question("מה זה שבץ מוחי?")
        assert result == "safe_general_education", (
            f"'מה זה שבץ מוחי?' must be safe_general_education. Got {result!r}"
        )
        result2 = ceng._classify_general_question("מתים מאלצהיימר?")
        assert result2 == "safe_general_education", (
            f"'מתים מאלצהיימר?' must be safe_general_education. Got {result2!r}"
        )


# ── 6. Out-of-domain ─────────────────────────────────────────────────────────

class TestOutOfDomain:
    """Obviously non-genetics questions must get a short out-of-scope response."""

    @pytest.mark.parametrize("q", [
        "מה זה מגדל אייפל?",
        "מי ראש ממשלת ישראל?",
        "מה מחיר הדירה בתל אביב?",
    ])
    def test_out_of_domain_short_response(self, q):
        data = client.post("/ask", json={"question": q}).json()
        answer = data.get("answer", "")
        safety = data.get("safety_level", "")
        # Must be a short "not my domain" message (not the long helpful fallback)
        assert len(answer) < 200, (
            f"Out-of-domain answer must be short. Got {len(answer)} chars for {q!r}:\n{answer!r}"
        )
        assert safety == "out_of_scope", (
            f"Out-of-domain must return safety=out_of_scope. Got {safety!r} for {q!r}"
        )

    def test_detect_out_of_domain_direct(self):
        """_detect_out_of_domain correctly identifies non-genetics questions."""
        assert ceng._detect_out_of_domain("מה זה מגדל אייפל?")
        assert ceng._detect_out_of_domain("מי ראש ממשלת ישראל?")
        assert not ceng._detect_out_of_domain("מה זה VUS?")
        assert not ceng._detect_out_of_domain("מה זה אלצהיימר?")

    def test_classify_out_of_domain(self):
        """_classify_general_question returns 'out_of_scope' for non-genetics questions."""
        assert ceng._classify_general_question("מה זה מגדל אייפל?") == "out_of_scope"
        assert ceng._classify_general_question("מי ראש ממשלת ישראל?") == "out_of_scope"
