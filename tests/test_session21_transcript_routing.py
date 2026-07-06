# -*- coding: utf-8 -*-
"""
Session 21 — Transcript-level integration tests.

Simulates real multi-turn conversations via /ask and verifies:
  - Standalone Q2 is answered about Q2's own topic, NOT the Q1 gene.
  - Gene follow-up with pronouns ("שלו", "הוא") DOES use prior gene context.
  - Standalone concept questions never bleed into prior gene context.
  - classify_question_intent() returns the correct intent structure.
  - _has_gene_followup_signal() is conservative (pronouns only).
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng

client = TestClient(app)


# ── _has_gene_followup_signal unit tests ─────────────────────────────────────

class TestFollowupPronounSignals:
    """_has_gene_followup_signal() must be conservative — pronouns only."""

    @pytest.mark.parametrize("text, expected", [
        # Pronouns / possessives → True
        ("מה התפקיד שלו?", True),
        ("לאיזה מצבים הוא קשור?", True),
        ("מה קורה בו?", True),
        ("ומה לגביו?", True),
        ("מה אפשר לדעת עליו?", True),
        ("מה הוא עושה?", True),
        ("מה הוא מייצר?", True),
        # Standalone concepts — no pronoun → False
        ("מה זה המוגלובין?", False),
        ("מה זה כרומוזום?", False),
        ("מה זה אלצהיימר?", False),
        ("מה זה שבץ מוחי?", False),
        ("מה זה מחלה נוירודגנרטיבית?", False),
        ("מה זה הגן APOE?", False),
        ("לאיזה מצבים קליניים הגן APOE מקושר?", False),
        ("כרומוזום 21", False),
    ])
    def test_pronoun_detection(self, text, expected):
        result = ceng._has_gene_followup_signal(text)
        assert result == expected, (
            f"_has_gene_followup_signal({text!r}) → expected {expected}, got {result}"
        )


# ── classify_question_intent unit tests ──────────────────────────────────────

class TestClassifyQuestionIntent:
    """Unit tests for classify_question_intent() routing function."""

    def test_privacy_blocked(self):
        info = ceng.classify_question_intent("תעודת זהות 123456789", last_gene_symbol="BRCA1")
        assert info["intent"] == "privacy_identifier"

    def test_personal_high_stakes_risk(self):
        """Questions caught by safety.is_personal_interpretation_request."""
        info = ceng.classify_question_intent("מה הסיכון שלי?", last_gene_symbol="BRCA1")
        assert info["intent"] == "personal_high_stakes"

    def test_personal_high_stakes_surgery(self):
        info = ceng.classify_question_intent("האם אני צריכה ניתוח?")
        assert info["intent"] == "personal_high_stakes"

    def test_personal_high_stakes_treatment(self):
        info = ceng.classify_question_intent("איזה טיפול לקחת?")
        assert info["intent"] == "personal_high_stakes"

    def test_specific_variant(self):
        info = ceng.classify_question_intent("יש לי BRCA1 c.5266dupC")
        assert info["intent"] == "specific_variant"

    def test_explicit_gene_no_context(self):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        info = ceng.classify_question_intent("מה זה הגן APOE?")
        assert info["intent"] == "explicit_gene_question"
        assert info["gene_symbol"] == "APOE"

    def test_gene_followup_classify_direct_call(self):
        """classify_question_intent() supports gene_followup when called directly with last_gene_symbol.
        Note: answer_question always passes None, so /ask never uses this path."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        info = ceng.classify_question_intent("מה התפקיד שלו?", last_gene_symbol="APOE")
        assert info["intent"] == "gene_followup"
        assert info["gene_symbol"] == "APOE"

    def test_gene_followup_pronoun_hu_direct_call(self):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        info = ceng.classify_question_intent("לאיזה מצבים הוא קשור?", last_gene_symbol="DMD")
        assert info["intent"] == "gene_followup"
        assert info["gene_symbol"] == "DMD"

    def test_standalone_hemoglobin_not_gene_followup(self):
        """'מה זה המוגלובין?' must NOT be gene_followup even with prior HBB context."""
        info = ceng.classify_question_intent("מה זה המוגלובין?", last_gene_symbol="HBB")
        assert info["intent"] not in ("gene_followup", "explicit_gene_question"), (
            f"Standalone concept must not route to prior gene HBB. Got {info['intent']!r}"
        )

    def test_standalone_chromosome_not_gene_followup(self):
        info = ceng.classify_question_intent("מה זה כרומוזום?", last_gene_symbol="DMD")
        assert info["intent"] not in ("gene_followup", "explicit_gene_question")

    def test_standalone_stroke_not_gene_followup(self):
        info = ceng.classify_question_intent("מה זה שבץ מוחי?", last_gene_symbol="HBB")
        assert info["intent"] not in ("gene_followup", "explicit_gene_question")

    def test_standalone_alzheimer_not_gene_followup(self):
        info = ceng.classify_question_intent("מה זה אלצהיימר?", last_gene_symbol="APOE")
        assert info["intent"] not in ("gene_followup", "explicit_gene_question")

    @pytest.mark.parametrize("pronoun_q", [
        "מה התפקיד שלו?",
        "לאיזה מצבים הוא קשור?",
        "מה קורה בו?",
        "מה עוד ידוע עליו?",
    ])
    def test_pronoun_triggers_gene_followup(self, pronoun_q):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        info = ceng.classify_question_intent(pronoun_q, last_gene_symbol="APOE")
        assert info["intent"] == "gene_followup", (
            f"Pronoun question {pronoun_q!r} should trigger gene_followup. "
            f"Got {info['intent']!r}"
        )


# ── Conversation 1: HBB → hemoglobin (standalone concept) ────────────────────

class TestConversation1HBBThenHemoglobin:
    """
    Q1: מה זה הגן HBB?   → gene answer about HBB
    Q2: מה זה המוגלובין? → must NOT be answered as HBB
    """

    def test_q1_hbb_is_gene_answer(self):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": "מה זה הגן HBB?"}).json()
        assert data.get("matched_topic") == "gene_clinvar_summary"
        assert data.get("gene_metadata", {}).get("gene_symbol") == "HBB"

    def test_q2_hemoglobin_not_hbb(self):
        """Q2 standalone concept must not be intercepted by HBB gene context."""
        data = client.post(
            "/ask", json={"question": "מה זה המוגלובין?", "last_gene_symbol": "HBB"}
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        # Critical: must not answer as HBB gene
        assert gene_sym != "HBB", (
            f"Standalone hemoglobin question must NOT route to HBB gene. "
            f"Got gene_symbol={gene_sym!r}"
        )
        answer = data.get("answer", "")
        # Answer should not LEAD with HBB gene metadata text
        assert "gene_clinvar_summary" != data.get("matched_topic") or gene_sym == "", (
            "If topic=gene_clinvar_summary, gene_symbol must not be HBB"
        )


# ── Conversation 2: HBB → stroke ─────────────────────────────────────────────

class TestConversation2HBBThenStroke:
    """
    Q1: מה זה הגן HBB?   → gene answer
    Q2: מה זה שבץ מוחי?  → general concept, NOT HBB
    """

    def test_q2_stroke_not_hbb(self):
        data = client.post(
            "/ask", json={"question": "מה זה שבץ מוחי?", "last_gene_symbol": "HBB"}
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "HBB", (
            f"Stroke question must not route to HBB gene. gene_symbol={gene_sym!r}"
        )


# ── Conversation 3: DMD → chromosome ─────────────────────────────────────────

class TestConversation3DMDThenChromosome:
    """
    Q1: מה זה הגן DMD?  → gene answer
    Q2: מה זה כרומוזום? → must NOT be DMD, must NOT be x_linked
    """

    def test_q2_chromosome_not_dmd(self):
        data = client.post(
            "/ask", json={"question": "מה זה כרומוזום?", "last_gene_symbol": "DMD"}
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "DMD", (
            f"Chromosome question must not route to DMD. gene_symbol={gene_sym!r}"
        )
        assert data.get("matched_topic") != "x_linked", (
            "Chromosome question must not return x_linked answer"
        )

    def test_q2_chromosome21_not_dmd(self):
        data = client.post(
            "/ask", json={"question": "מה זה כרומוזום 21?", "last_gene_symbol": "DMD"}
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "DMD"
        assert data.get("matched_topic") != "x_linked"


# ── Conversation 4: APOE → pronoun question (no context routing in Session 22) ─

class TestConversation4APOEFollowup:
    """
    Session 22: last_gene_symbol is ignored by the backend.
    Pronoun questions ("לאיזה מצבים הוא קשור?") without an explicit gene name
    in the text are answered on their own — they do NOT route to the prior gene.
    The key invariant: no context bleed (APOE must not appear via last_gene_symbol).
    """

    def test_q2_pronoun_no_context_bleed(self):
        """Pronoun question must NOT route to APOE even when last_gene_symbol=APOE is sent."""
        data = client.post(
            "/ask",
            json={"question": "לאיזה מצבים הוא קשור?", "last_gene_symbol": "APOE"},
        ).json()
        # Gene metadata must not be APOE — last_gene_symbol is ignored
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "APOE", (
            f"last_gene_symbol=APOE must be ignored. "
            f"Got gene_symbol={gene_sym!r}, topic={data.get('matched_topic')!r}"
        )
        assert data.get("matched_topic") != "gene_clinvar_summary" or gene_sym == "", (
            "If topic=gene_clinvar_summary, gene must not be APOE"
        )

    def test_q2_possessive_no_context_bleed(self):
        """Possessive question must NOT route to APOE even with last_gene_symbol."""
        data = client.post(
            "/ask",
            json={"question": "מה התפקיד שלו?", "last_gene_symbol": "APOE"},
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "APOE", (
            f"last_gene_symbol=APOE must be ignored. Got gene_symbol={gene_sym!r}"
        )


# ── Standalone concept: no prior gene contamination ───────────────────────────

class TestStandaloneConceptQuestions:
    """Standalone concepts must never be answered using a prior gene."""

    @pytest.mark.parametrize("q", [
        "מה זה כרומוזום?",
        "מה זה כרומוזום 21?",
        "הסבר לי מה זה כרומוזום",
    ])
    def test_chromosome_not_xlinked(self, q):
        data = client.post("/ask", json={"question": q}).json()
        assert data.get("matched_topic") != "x_linked", (
            f"Chromosome question {q!r} must not return x_linked. "
            f"Got topic={data.get('matched_topic')!r}"
        )

    @pytest.mark.parametrize("q,prior_gene", [
        ("מה זה המוגלובין?", "HBB"),
        ("מה זה שבץ מוחי?", "HBB"),
        ("מה זה מחלה נוירודגנרטיבית?", "APOE"),
        ("מה זה אלצהיימר?", "APOE"),
        ("מה זה כרומוזום?", "DMD"),
        ("מה זה כרומוזום 21?", "BRCA1"),
    ])
    def test_standalone_ignores_prior_gene(self, q, prior_gene):
        """The CRITICAL invariant: standalone question must NEVER route to prior gene."""
        data = client.post(
            "/ask", json={"question": q, "last_gene_symbol": prior_gene}
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        # Must not route to prior gene — even if answer is out_of_scope (no AI in prod)
        assert gene_sym != prior_gene, (
            f"Standalone {q!r} must not route to prior gene {prior_gene}. "
            f"Got gene_symbol={gene_sym!r}, topic={data.get('matched_topic')!r}"
        )

    def test_hemoglobin_no_prior_gene(self):
        """Without prior gene context, hemoglobin must not return gene_clinvar_summary."""
        data = client.post("/ask", json={"question": "מה זה המוגלובין?"}).json()
        assert data.get("gene_metadata", {}).get("gene_symbol", "") == "", (
            "Hemoglobin question without prior gene must not have gene_metadata"
        )

    def test_chromosome_no_prior_gene(self):
        data = client.post("/ask", json={"question": "מה זה כרומוזום?"}).json()
        assert data.get("matched_topic") != "x_linked"
        assert data.get("gene_metadata", {}).get("gene_symbol", "") == ""


# ── High-stakes personal questions ───────────────────────────────────────────

class TestHighStakesActuallyBlocked:
    """
    These questions ARE caught by safety.is_personal_interpretation_request
    and must return requires_genetic_counselor.
    """

    @pytest.mark.parametrize("q", [
        "מה הסיכון שלי?",
        "האם אני צריכה ניתוח?",
        "האם הילדים שלי יהיו חולים?",
        "יש לי שינוי בMSH2 האם יש לי סרטן?",
        "איזה טיפול לקחת?",
    ])
    def test_personal_question_blocked(self, q):
        data = client.post("/ask", json={"question": q}).json()
        assert data.get("safety_level") == "requires_genetic_counselor", (
            f"Personal question {q!r} must be blocked. "
            f"Got safety_level={data.get('safety_level')!r}"
        )
        assert "unverified_general_draft" not in data, (
            f"Blocked question must not get AI draft: {q!r}"
        )

    def test_personal_question_with_prior_gene_blocked(self):
        data = client.post(
            "/ask",
            json={"question": "מה הסיכון שלי?", "last_gene_symbol": "BRCA1"},
        ).json()
        assert data.get("safety_level") == "requires_genetic_counselor"

    def test_surgery_blocked(self):
        data = client.post("/ask", json={"question": "האם אני צריכה ניתוח?"}).json()
        assert data.get("needs_genetic_counselor") is True
        assert data.get("safety_level") == "requires_genetic_counselor"


# ── No AI draft for personal/high-stakes ─────────────────────────────────────

class TestNoAIDraftForSensitiveQuestions:
    """
    Questions that are NOT blocked by safety classifier but should not get
    free-form AI draft (they route to KB or fallback instead).
    """

    @pytest.mark.parametrize("q", [
        "יש לי APOE האם יהיה לי אלצהיימר?",
        "האם אני בסיכון לסרטן?",
    ])
    def test_no_ai_draft(self, q, monkeypatch):
        """These may return KB or fallback, but must not get AI draft."""
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "true")
        from unittest.mock import MagicMock
        mc = MagicMock()
        mc.call_text_raw.return_value = "מידע כללי על גנטיקה."
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mc)
        data = client.post("/ask", json={"question": q}).json()
        assert "unverified_general_draft" not in data, (
            f"Sensitive question {q!r} must not get free AI draft"
        )


# ── Gene answer must have content ────────────────────────────────────────────

class TestGeneAnswerContent:
    """Gene answers should be informative and non-empty."""

    def test_gene_answer_informative(self):
        """Approved-card genes (BRCA1) must return a substantive answer."""
        import app.gene_index as gene_index
        import app.gene_cards as gc
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        # Only test genes known to have an approved curated card
        for gene in ["BRCA1"]:
            if not gc.get_approved_summary(gene):
                pytest.skip(f"{gene} has no approved card in this build")
            data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
            answer = data.get("answer", "")
            assert len(answer) > 50, f"Gene answer for {gene} too short: {answer!r}"
            assert data.get("matched_topic") == "gene_clinvar_summary"

    def test_gene_metadata_present(self):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": "מה זה הגן BRCA1?"}).json()
        assert data.get("gene_metadata", {}).get("gene_symbol") == "BRCA1"
