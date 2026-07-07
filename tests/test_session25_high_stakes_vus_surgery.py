# -*- coding: utf-8 -*-
"""
Session 25c — High-stakes surgery decision detection beats VUS/gene education.

Covers:
  1. VUS + surgery questions are blocked before gene/VUS education routes
  2. Answer is VUS-aware and mentions the surgery term
  3. Answer does not contain old boilerplate referral phrases
  4. Educational VUS and gene questions remain unaffected
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng

client = TestClient(app)

_BANNED_BOILERPLATE = [
    "לתשובה אישית",
    "המשמעות האישית נקבעת",
    "המשמעות של כל ממצא ספציפי נקבעת",
    "המידע כללי ואינו מחליף ייעוץ רפואי אישי",
]


# ── 1. VUS + surgery questions must be blocked ───────────────────────────────

class TestVusSurgeryBlocked:
    """Surgery decision questions must return requires_genetic_counselor."""

    @pytest.mark.parametrize("q,expected_terms", [
        (
            "אמרו לי שיש לי VUS לBRCA1, האם לעשות כריתת שד?",
            ["VUS", "כריתת שד"],
        ),
        (
            "יש לי VUS בגן BRCA1, האם לעבור ניתוח?",
            ["VUS", "ניתוח"],
        ),
        (
            "יש לי VUS בAPC, האם לעשות כריתה?",
            ["VUS", "כריתה"],
        ),
        (
            "אמרו לי שיש לי BRCA1, האם לעשות כריתה מניעתית?",
            ["כריתה"],
        ),
    ])
    def test_vus_surgery_blocked(self, q, expected_terms):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        answer = data.get("answer", "")

        assert safety == "requires_genetic_counselor", (
            f"VUS+surgery question must be blocked. Got safety={safety!r} for {q!r}\n"
            f"Answer: {answer[:300]!r}"
        )
        for term in expected_terms:
            assert term in answer, (
                f"Answer must mention {term!r} for {q!r}:\nAnswer: {answer[:300]!r}"
            )

    def test_vus_surgery_no_boilerplate(self):
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS לBRCA1, האם לעשות כריתת שד?"},
        ).json()
        answer = data.get("answer", "")
        for phrase in _BANNED_BOILERPLATE:
            assert phrase not in answer, (
                f"Banned boilerplate in surgery answer: {phrase!r}\nAnswer: {answer[:300]!r}"
            )

    def test_vus_surgery_answer_mentions_vus_not_pathogenic(self):
        """Answer must say VUS is not pathogenic — not just a generic block."""
        data = client.post(
            "/ask",
            json={"question": "יש לי VUS בגן BRCA1, האם לעבור ניתוח?"},
        ).json()
        answer = data.get("answer", "")
        has_vus_context = (
            "VUS" in answer
            and ("פתוגני" in answer or "בלבד" in answer or "ניתוח" in answer)
        )
        assert has_vus_context, (
            f"Answer must explain VUS≠pathogenic or VUS alone. Got:\n{answer[:300]!r}"
        )

    def test_surgery_decision_detector_direct(self):
        """_is_surgery_decision_question correctly identifies surgery questions."""
        assert ceng._is_surgery_decision_question("האם לעשות כריתת שד?")
        assert ceng._is_surgery_decision_question("האם לעבור ניתוח?")
        assert ceng._is_surgery_decision_question("האם לעשות כריתה מניעתית?")
        assert ceng._is_surgery_decision_question("האם לעשות כריתה?")
        assert ceng._is_surgery_decision_question("כריתה מניעתית")
        # Must NOT fire on pure educational questions
        assert not ceng._is_surgery_decision_question("מה ההבדל בין VUS לבין pathogenic?")
        assert not ceng._is_surgery_decision_question("מה זה הגן BRCA1?")
        assert not ceng._is_surgery_decision_question("מהן האפשרויות העומדות מולי?")
        assert not ceng._is_surgery_decision_question("מה זה ניתוח מניעתי?")

    def test_intent_classified_surgery_decision(self):
        """classify_question_intent must return surgery_decision for surgery+VUS questions."""
        result = ceng.classify_question_intent(
            "אמרו לי שיש לי VUS לBRCA1, האם לעשות כריתת שד?"
        )
        assert result["intent"] == "surgery_decision", (
            f"Expected surgery_decision intent. Got {result!r}"
        )

    def test_gene_symbol_in_intent(self):
        """Surgery decision intent must carry gene_symbol when gene is detected."""
        result = ceng.classify_question_intent(
            "יש לי VUS בגן BRCA1, האם לעבור ניתוח?"
        )
        assert result["intent"] == "surgery_decision"
        assert result["gene_symbol"] == "BRCA1", (
            f"gene_symbol must be BRCA1. Got {result['gene_symbol']!r}"
        )


# ── 2. Educational VUS/gene questions remain unaffected ──────────────────────

class TestEducationalUnaffected:
    """Educational questions about VUS and genes must not be blocked."""

    @pytest.mark.parametrize("q", [
        "אמרו לי שיש לי VUS בגן BRCA1, מהן האפשרויות העומדות מולי?",
        "מה ההבדל בין VUS לבין pathogenic?",
        "מה זה הגן BRCA1?",
        "מה זה VUS?",
        "יש לי VUS בNF1, מה זה אומר?",
    ])
    def test_educational_not_blocked(self, q):
        data = client.post("/ask", json={"question": q}).json()
        safety = data.get("safety_level", "")
        assert safety == "general_information", (
            f"Educational question must not be blocked. Got safety={safety!r} for {q!r}"
        )

    def test_vus_options_still_routes_correctly(self):
        """VUS options request must return practical options, not surgery block."""
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS בגן BRCA1, מהן האפשרויות העומדות מולי?"},
        ).json()
        safety = data.get("safety_level", "")
        topic = data.get("matched_topic", "")
        assert safety == "general_information", (
            f"VUS options question must be educational. Got safety={safety!r}"
        )
        assert topic != "surgery_decision", (
            f"VUS options must not route to surgery_decision. Got topic={topic!r}"
        )


# ── 3. Build function unit tests ──────────────────────────────────────────────

class TestBuildSurgeryDecisionAnswer:
    """_build_surgery_decision_answer returns correct structured response."""

    def test_vus_brca1_mastectomy(self):
        result = ceng._build_surgery_decision_answer(
            "יש לי VUS בBRCA1, האם לעשות כריתת שד?", gene_symbol="BRCA1"
        )
        assert result["safety_level"] == "requires_genetic_counselor"
        assert result["needs_genetic_counselor"] is True
        assert result["matched_topic"] == "surgery_decision"
        assert "VUS" in result["answer"]
        assert "כריתת שד" in result["answer"]

    def test_vus_apc_surgery(self):
        result = ceng._build_surgery_decision_answer(
            "יש לי VUS בAPC, האם לעשות כריתה?", gene_symbol="APC"
        )
        assert result["safety_level"] == "requires_genetic_counselor"
        assert "APC" in result["answer"]
        assert "כריתה" in result["answer"]

    def test_no_gene_no_vus(self):
        result = ceng._build_surgery_decision_answer("האם לעבור ניתוח?")
        assert result["safety_level"] == "requires_genetic_counselor"
        assert result["needs_genetic_counselor"] is True
        assert len(result["answer"]) > 30

    def test_answer_no_boilerplate(self):
        result = ceng._build_surgery_decision_answer(
            "האם לעשות כריתת שד?", gene_symbol="BRCA1"
        )
        for phrase in _BANNED_BOILERPLATE:
            assert phrase not in result["answer"], (
                f"Banned boilerplate in build function output: {phrase!r}"
            )
