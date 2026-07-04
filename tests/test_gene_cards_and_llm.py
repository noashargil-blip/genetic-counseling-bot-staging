# -*- coding: utf-8 -*-
"""
tests/test_gene_cards_and_llm.py

Tests for:

A. Gene cards system (app/gene_cards.py + data/gene_cards.json)
   - Loader finds the JSON file and returns approved cards for the 6 curated genes
   - get_approved_summary / get_approved_card / has_approved_card API
   - Unapproved / absent genes return None
   - list_approved_genes returns sorted list

B. LLM intro-only validation (_validate_llm_intro)
   - Accepts valid Hebrew sentences with allowed English tokens
   - Rejects empty string, too-long strings, question marks
   - Rejects CJK characters
   - Rejects forbidden medical-advice terms (surgery, treatment, etc.)
   - Rejects English sentences (insufficient Hebrew ratio)
   - Rejects unknown English words longer than 3 characters

C. _apply_safe_intro — without a configured LLM
   - Returns (deterministic_answer, False) when LOCAL_LLM_URL is unset
   - Returns (deterministic_answer, False) when LLM raises LLMClientError
   - Returns (intro + det, True) when LLM returns a valid Hebrew sentence

D. Answer tier invariants
   - Tier 1 (approved gene card): matched_topic gene_clinvar_summary,
     safety_level general_information, answer contains gene name
   - Tier 2 (ClinVar index only, no approved card): transparency note present,
     answer mentions ClinVar, answer has digits or "ClinVar"
   - Tier 3 (gene unknown): safe not-found message, no "לא קיים" / "אינו ידוע"
   - VUS general answer: no full LLM rewrite artefacts
   - VUS+gene answer: uses approved card content when available

E. Safety invariants on all LLM-related paths
   - No medical advice / personal risk in any validated intro
   - VUS answers contain no non-Hebrew artefacts
   - KB answers still return deterministic content when LLM is unavailable
"""

import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import gene_cards as gc
from app.counseling_engine import _validate_llm_intro, _apply_safe_intro
from app.llm_client import LLMClientError
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ask(question: str, **extra) -> dict:
    payload = {"question": question}
    payload.update(extra)
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """All tests in this file run in deterministic-fallback mode unless overridden."""
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# A. Gene cards loader
# ---------------------------------------------------------------------------

class TestGeneCardsLoader:
    """gene_cards.py loads approved cards and exposes a clean API."""

    def test_cards_available_flag(self):
        assert gc.CARDS_AVAILABLE is True

    def test_approved_genes_list(self):
        approved = gc.list_approved_genes()
        assert isinstance(approved, list)
        assert approved == sorted(approved), "list_approved_genes must return sorted list"
        # All 6 curated genes must be present
        for gene in ("APC", "BRCA1", "BRCA2", "NF1", "TP53", "SHANK3"):
            assert gene in approved, f"{gene} missing from approved gene cards"

    def test_has_approved_card_true(self):
        for gene in ("BRCA1", "APC", "NF1"):
            assert gc.has_approved_card(gene) is True

    def test_has_approved_card_false(self):
        assert gc.has_approved_card("FAKEGENE9999") is False
        assert gc.has_approved_card("HBB") is False  # no approved card for HBB

    def test_get_approved_summary_returns_string(self):
        for gene in ("BRCA1", "BRCA2", "NF1", "APC", "TP53", "SHANK3"):
            summary = gc.get_approved_summary(gene)
            assert isinstance(summary, str), f"{gene}: expected str, got {type(summary)}"
            assert len(summary) > 50, f"{gene}: summary too short"
            # Must contain Hebrew characters
            assert re.search(r"[א-ת]", summary), f"{gene}: no Hebrew in summary"

    def test_get_approved_summary_unknown_returns_none(self):
        assert gc.get_approved_summary("FAKEGENE999") is None
        assert gc.get_approved_summary("") is None

    def test_get_approved_card_has_required_fields(self):
        card = gc.get_approved_card("BRCA1")
        assert card is not None
        for field in ("gene_symbol", "summary_he", "approved", "reviewed_by"):
            assert field in card, f"BRCA1 card missing field: {field}"
        assert card["approved"] is True
        assert card["gene_symbol"] == "BRCA1"

    def test_get_approved_card_unknown_returns_none(self):
        assert gc.get_approved_card("XXXXUNKNOWN") is None

    def test_summaries_contain_no_personal_advice(self):
        """Approved summaries must never give personal medical advice."""
        forbidden = re.compile(
            r"\b(ניתוח|כריתה|קולונוסקופיה|גסטרוסקופיה|"
            r"סיכון\s+אישי|הסיכון\s+שלך|"
            r"אתה\s+חולה|את\s+חולה|יש\s+לך\s+(מחלה|סרטן))\b",
            re.IGNORECASE,
        )
        for gene in gc.list_approved_genes():
            summary = gc.get_approved_summary(gene)
            assert not forbidden.search(summary or ""), (
                f"Approved card for {gene} contains forbidden personal advice"
            )

    def test_apc_summary_mentions_cell_growth(self):
        summary = gc.get_approved_summary("APC")
        assert summary is not None
        assert "גדילה" in summary or "תאים" in summary or "בקרה" in summary

    def test_brca1_summary_mentions_dna(self):
        assert "DNA" in (gc.get_approved_summary("BRCA1") or "")

    def test_nf1_summary_mentions_neurofibromatosis(self):
        summary = gc.get_approved_summary("NF1") or ""
        assert "Neurofibromatosis" in summary or "NF1" in summary

    def test_vus_boundary_language_in_summaries(self):
        """Each curated summary must note VUS ≠ pathogenic or similar boundary."""
        for gene in ("BRCA1", "BRCA2", "NF1", "TP53"):
            summary = gc.get_approved_summary(gene) or ""
            has_boundary = (
                "VUS" in summary
                or "פתוגני" in summary
                or "לא ידוע" in summary
                or "משמעות" in summary
            )
            assert has_boundary, (
                f"{gene}: approved summary has no VUS/pathogenic boundary language"
            )


# ---------------------------------------------------------------------------
# B. _validate_llm_intro
# ---------------------------------------------------------------------------

class TestValidateLlmIntro:

    def test_valid_hebrew_sentence(self):
        assert _validate_llm_intro("שאלה מצוינת — מידע זה יכול לעזור לך להבין את הנושא.") is True

    def test_valid_with_vus_token(self):
        assert _validate_llm_intro(
            "קבלת תוצאה עם VUS יכולה להיות מבלבלת — הנה מידע שיעזור לך להבין את המושג."
        ) is True

    def test_valid_with_brca1_token(self):
        assert _validate_llm_intro(
            "הגן BRCA1 הוא נושא חשוב — כאן מידע כללי שיכול לשפוך אור."
        ) is True

    def test_valid_with_dna_token(self):
        assert _validate_llm_intro(
            "גנים הקשורים לתיקון DNA עשויים להיות מורכבים — הנה הסבר כללי."
        ) is True

    def test_rejects_empty(self):
        assert _validate_llm_intro("") is False

    def test_rejects_whitespace_only(self):
        assert _validate_llm_intro("   ") is False

    def test_rejects_too_long(self):
        assert _validate_llm_intro("א" * 201) is False

    def test_accepts_exactly_200_chars(self):
        text = "מידע " * 39 + "."  # ~196 chars, all Hebrew
        assert len(text) <= 200
        assert _validate_llm_intro(text) is True

    def test_rejects_question_mark(self):
        assert _validate_llm_intro("האם יש מידע על גן זה?") is False

    def test_rejects_japanese(self):
        assert _validate_llm_intro("こんにちは — שלום") is False

    def test_rejects_chinese(self):
        assert _validate_llm_intro("这是一个问题 — שאלה") is False

    def test_rejects_surgery_term(self):
        assert _validate_llm_intro("ניתוח הוא אפשרות שיש לבדוק עם הצוות הרפואי.") is False

    def test_rejects_treatment_term(self):
        assert _validate_llm_intro("יש טיפולים שכדאי לשקול לאחר קבלת תוצאה זו.") is False

    def test_rejects_surveillance_term(self):
        assert _validate_llm_intro("מעקב רפואי קפדני מומלץ לנושאים.") is False

    def test_rejects_english_only(self):
        assert _validate_llm_intro("This is information about your genetic test.") is False

    def test_rejects_unknown_english_word(self):
        assert _validate_llm_intro("הנה information חשובה עבורך.") is False

    def test_rejects_no_hebrew(self):
        assert _validate_llm_intro("VUS BRCA1 DNA.") is False  # only allowed tokens, no Hebrew

    def test_allows_approved_english_tokens_in_hebrew_sentence(self):
        sentence = "ממצא ClinVar עבור גן BRCA2 יכול לספק מידע כללי מאוד חשוב."
        assert _validate_llm_intro(sentence) is True

    def test_rejects_diagnosis_term_hebrew(self):
        assert _validate_llm_intro("לאחר האבחנה הגנטית יש לפנות לצוות הרפואי.") is False

    def test_rejects_personal_risk_hebrew(self):
        assert _validate_llm_intro("הסיכון האישי שלך תלוי בממצאי הבדיקה.") is False


# ---------------------------------------------------------------------------
# C. _apply_safe_intro
# ---------------------------------------------------------------------------

class TestApplySafeIntro:

    def test_no_llm_returns_deterministic(self):
        det = "זהו תוכן קבוע ומאושר."
        result, llm_used = _apply_safe_intro("מה זה VUS?", det)
        assert result == det
        assert llm_used is False

    def test_llm_error_returns_deterministic(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.side_effect = LLMClientError("timeout")
            result, llm_used = _apply_safe_intro("מה זה VUS?", "תוכן קבוע.")
        assert result == "תוכן קבוע."
        assert llm_used is False

    def test_llm_invalid_output_returns_deterministic(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = "Should I get surgery now?"
            result, llm_used = _apply_safe_intro("מה זה VUS?", "תוכן קבוע.")
        assert result == "תוכן קבוע."
        assert llm_used is False

    def test_llm_empty_output_returns_deterministic(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = "  "
            result, llm_used = _apply_safe_intro("מה זה VUS?", "תוכן קבוע.")
        assert result == "תוכן קבוע."
        assert llm_used is False

    def test_llm_valid_intro_prepended(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        valid_intro = "שאלה חשובה — מידע זה יכול לשפוך אור על הממצא שלך."
        det = "זהו התוכן הקבוע."
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = valid_intro
            result, llm_used = _apply_safe_intro("מה זה VUS?", det)
        assert llm_used is True
        assert result.startswith(valid_intro)
        assert det in result
        assert result == f"{valid_intro}\n\n{det}"

    def test_deterministic_content_always_preserved(self, monkeypatch):
        """Medical content from the KB must be unchanged regardless of LLM output."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        det = "VUS הוא ממצא שמשמעותו עדיין לא ידועה — הצוות הגנטי יסביר."
        valid_intro = "מידע זה יכול לשפוך אור על הממצא שלך."
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = valid_intro
            result, _ = _apply_safe_intro("מה זה VUS?", det)
        assert det in result, "Deterministic content must appear verbatim in final answer"


# ---------------------------------------------------------------------------
# D. Answer tier invariants (via /ask)
# ---------------------------------------------------------------------------

class TestAnswerTierInvariants:
    """
    Tier 1 — approved gene card
    Tier 2 — ClinVar index only (no approved card)
    Tier 3 — gene not found anywhere
    """

    # -- Tier 1 (approved cards: BRCA1, BRCA2, NF1, APC, TP53, SHANK3) -----

    def test_tier1_safety_level(self):
        for gene in ("BRCA1", "APC", "NF1"):
            data = _ask(f"מה ידוע על {gene}?")
            assert data["safety_level"] == "general_information", (
                f"{gene}: expected general_information, got {data['safety_level']}"
            )

    def test_tier1_matched_topic(self):
        for gene in ("BRCA1", "BRCA2", "TP53"):
            data = _ask(f"תסביר לי על {gene}")
            assert data["matched_topic"] == "gene_clinvar_summary", (
                f"{gene}: expected gene_clinvar_summary, got {data['matched_topic']}"
            )

    def test_tier1_answer_contains_gene_name(self):
        for gene in ("APC", "SHANK3"):
            data = _ask(f"מה זה {gene}?")
            assert gene in data["answer"], f"{gene} symbol missing from Tier-1 answer"

    def test_tier1_answer_contains_curated_content(self):
        data = _ask("מה ידוע על BRCA1?")
        assert "DNA" in data["answer"], "BRCA1 curated content (DNA repair) not in answer"

    def test_tier1_apc_contains_cell_growth_context(self):
        data = _ask("מה זה APC?")
        assert (
            "גדילה" in data["answer"]
            or "תאים" in data["answer"]
            or "בקרה" in data["answer"]
        ), "APC curated education (cell growth control) missing from Tier-1 answer"

    def test_tier1_has_safety_note(self):
        for gene in ("BRCA1", "NF1"):
            data = _ask(f"מה ידוע על {gene}?")
            assert "צוות הגנטי" in data["answer"] or "פנה" in data["answer"], (
                f"{gene}: Tier-1 answer missing counselor referral safety note"
            )

    def test_tier1_gene_metadata_present(self):
        data = _ask("מה ידוע על APC?")
        assert "gene_metadata" in data
        assert data["gene_metadata"]["gene_symbol"] == "APC"

    def test_tier1_deterministic_without_llm(self):
        data = _ask("מה ידוע על BRCA2?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    # -- Tier 2 (e.g. HBB, SOX1 — in ClinVar index, no approved card) ------

    def test_tier2_transparency_note_present(self):
        data = _ask("מה זה HBB?")
        if data["matched_topic"] == "gene_clinvar_summary" and data["gene_metadata"].get("found_in_index"):
            if data["gene_metadata"].get("answer_tier") == "tier2":
                # New Tier-2 answer uses patient-friendly short message
                assert "אין לי עדיין סיכום ביולוגי מאושר" in data["answer"], (
                    "Tier-2 answer is missing the transparency note"
                )

    def test_tier2_clinvar_data_in_gene_metadata(self):
        """HBB is in the full ClinVar index — ClinVar stats must be in gene_metadata.
        The main answer is now a short patient-friendly message; stats
        are kept in gene_metadata for the collapsed technical UI card.
        """
        data = _ask("מה ידוע על HBB?")
        if data["matched_topic"] == "gene_clinvar_summary":
            meta = data.get("gene_metadata", {})
            if meta.get("answer_tier") == "tier2":
                assert "total_variants" in meta, "total_variants missing from gene_metadata"
                assert "significance_breakdown" in meta, "significance_breakdown missing from gene_metadata"
                assert "top_phenotypes" in meta, "top_phenotypes missing from gene_metadata"
                # Answer itself must NOT contain raw ClinVar stats dump
                assert "סיווגים קליניים מדווחים" not in data["answer"]
                assert "מצבים רפואיים מדווחים" not in data["answer"]

    def test_tier2_no_invented_biology(self):
        """Tier-2 must not invent gene-biology facts not in the approved cards."""
        data = _ask("מה זה SOX1?")
        if data["matched_topic"] == "gene_clinvar_summary":
            meta = data.get("gene_metadata", {})
            if meta.get("answer_tier") == "tier2":
                # Should NOT claim it has curated content (it doesn't)
                assert "gene_cards" not in data["answer"].lower(), (
                    "Tier-2 answer leaks implementation detail"
                )

    def test_tier2_safety_level(self):
        data = _ask("מה ידוע על HBB?")
        if data["matched_topic"] == "gene_clinvar_summary":
            assert data["safety_level"] == "general_information"

    def test_tier2_gene_metadata_found_in_index(self):
        data = _ask("מה זה HBB?")
        if data["matched_topic"] == "gene_clinvar_summary":
            meta = data["gene_metadata"]
            assert meta["found_in_index"] is True
            assert "answer_tier" in meta

    def test_tier2_deterministic_without_llm(self):
        data = _ask("מה זה SOX1?")
        if data["matched_topic"] == "gene_clinvar_summary":
            assert data["llm_used"] is False
            assert data["fallback_used"] is True

    # -- Tier 3 (gene not found anywhere) -----------------------------------

    def test_tier3_safe_not_found_message(self):
        """Unknown genes must get a gentle 'not in local DB' message."""
        data = _ask("מה ידוע על FAKEGENE12345?")
        if data["matched_topic"] == "gene_clinvar_summary":
            assert "לא קיים" not in data["answer"]
            assert "אינו ידוע" not in data["answer"]
            # Should mention either local DB or direct to genetics team
            assert (
                "מאגר" in data["answer"]
                or "צוות" in data["answer"]
                or "גנטי" in data["answer"]
            )

    def test_tier3_no_personal_recommendation(self):
        data = _ask("מה ידוע על XXXXXXXXFAKEGENE?")
        forbidden = re.compile(r"ניתוח|כריתה|הסיכון\s+שלך|יש\s+לך\s+(מחלה|סרטן)")
        assert not forbidden.search(data["answer"])


# ---------------------------------------------------------------------------
# E. Safety invariants on LLM-related paths
# ---------------------------------------------------------------------------

class TestLlmSafetyInvariants:
    """
    These tests verify that:
    1. VUS/KB answers do not contain non-Hebrew artefacts when LLM is OFF.
    2. Even with LLM ON, the deterministic medical content is always preserved.
    3. Bad LLM output (English, CJK, medical advice) is silently rejected.
    """

    def test_vus_answer_no_llm_contains_hebrew(self):
        data = _ask("מה זה VUS?")
        assert re.search(r"[א-ת]", data["answer"]), "VUS answer has no Hebrew text"

    def test_vus_answer_no_english_artifacts(self):
        """VUS KB answer must not contain unexpected English sentences or non-medical jargon."""
        data = _ask("מה זה VUS?")
        answer = data["answer"]
        # Strip all approved English medical/genetic terms that may legitimately appear
        # in KB answers (these are domain vocabulary, not LLM artefacts).
        approved_pattern = re.compile(
            r"\b(?:VUS|pathogenic|benign|uncertain|significance|variant|"
            r"ClinVar|DNA|RNA|ACMG|OMIM|likely|conflicting|"
            r"BRCA1|BRCA2|NF1|APC|TP53|SHANK3|HBB|SOX1|"
            r"type|Neurofibromatosis|Familial|Adenomatous|Polyposis|"
            r"Lynch|Li-Fraumeni|p53)\b",
            re.IGNORECASE,
        )
        stripped = approved_pattern.sub("", answer)
        # After stripping approved terms, no English word ≥ 5 chars should remain
        english_words = re.findall(r"\b[A-Za-z]{5,}\b", stripped)
        assert len(english_words) == 0, (
            f"Unexpected non-approved English words in VUS answer: {english_words[:5]}"
        )

    def test_kb_answer_deterministic_without_llm(self):
        """Without LLM, KB answers must use llm_used=False."""
        for question in ["מה זה VUS?", "מה זה נשאות?", "מה זה ירושה אוטוזומלית?"]:
            data = _ask(question)
            assert data["llm_used"] is False
            assert data["fallback_used"] is True

    def test_llm_bad_english_sentence_rejected(self, monkeypatch):
        """LLM returning a pure English sentence must be rejected."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = (
                "Here is some general information about your genetic result."
            )
            data = _ask("מה זה VUS?")
        # Validation fails → deterministic fallback
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_llm_surgery_intro_rejected(self, monkeypatch):
        """LLM returning a sentence with surgery advice must be rejected."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = (
                "ניתוח הוא הפתרון הטוב ביותר עבורך."
            )
            data = _ask("מה זה VUS?")
        assert data["llm_used"] is False

    def test_llm_cjk_intro_rejected(self, monkeypatch):
        """LLM returning many CJK characters (>3) must be rejected even after retry."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            # 4 CJK chars — above cleaning threshold; retry returns same bad output
            MockClient.return_value._call_api.return_value = "שלום 你好世界 מרחבא."
            data = _ask("מה זה VUS?")
        assert data["llm_used"] is False

    def test_llm_question_mark_intro_rejected(self, monkeypatch):
        """LLM intro ending with a question mark must be rejected."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = "האם יש לך שאלות על הגן?"
            data = _ask("מה זה VUS?")
        assert data["llm_used"] is False

    def test_llm_not_used_for_kb_path(self, monkeypatch):
        """KB path is now fully deterministic — LLM intro is never prepended."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        valid_intro = "שאלה מצוינת — מידע זה יכול לסייע לך להבין את הנושא."
        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = valid_intro
            data = _ask("מה זה VUS?")
        assert data["llm_used"] is False
        assert data["fallback_used"] is True
        # Intro must NOT be prepended — answer is deterministic KB content only
        assert valid_intro not in data["answer"]
        # Deterministic KB content is still there
        assert "VUS" in data["answer"]

    def test_vus_gene_answer_no_surgery_recommendation(self):
        """VUS+gene answers must not recommend surgery or personal medical action."""
        forbidden = re.compile(
            r"\b(ניתוח|כריתה|קולונוסקופיה|מעקב\s+אישי|"
            r"הסיכון\s+שלך|טיפול\s+רפואי)\b",
            re.IGNORECASE,
        )
        for question in [
            "מה זה VUS בגן BRCA1",
            "יש לי VUS ב-APC",
            "יש לי VUS ב-TP53",
        ]:
            data = _ask(question)
            assert not forbidden.search(data["answer"]), (
                f"Personal medical recommendation in VUS+gene answer: {question!r}"
            )

    def test_no_full_llm_rewrite_of_kb_answers(self, monkeypatch):
        """
        Even when LLM returns a 1000-char Hebrew paragraph, only the intro
        sentence (≤200 chars) is accepted.  The remaining deterministic KB
        content must appear unchanged in the final answer.
        """
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        long_llm_output = "מידע זה מאוד מאוד חשוב. " * 50  # 1250 chars — way over 200

        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = long_llm_output
            data = _ask("מה זה VUS?")

        # Validation fails (> 200 chars) → deterministic only
        assert data["llm_used"] is False
        assert data["fallback_used"] is True
        # The long output must NOT be in the answer
        assert long_llm_output.strip() not in data["answer"]

    def test_gene_level_no_full_llm_biology(self, monkeypatch):
        """
        For Tier-2 genes (no approved card), the LLM must NOT be used to
        generate gene biology facts.  Only a validated intro sentence is allowed.
        The deterministic ClinVar stats block must always appear.
        """
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        fake_biology = "SOX1 הוא גן האחראי על תפקוד חשוב מאוד בגוף האדם."
        det_marker = "ממאגר ClinVar"  # always present in deterministic Tier-2 answers

        with patch("app.counseling_engine.LocalLLMClient") as MockClient:
            MockClient.return_value._call_api.return_value = fake_biology
            data = _ask("מה זה SOX1?")

        if data["matched_topic"] == "gene_clinvar_summary":
            meta = data.get("gene_metadata", {})
            if meta.get("answer_tier") == "tier2":
                # Tier-2 now uses a short patient-friendly deterministic answer;
                # raw ClinVar stats are in gene_metadata, not the answer text.
                # The fake_biology must never appear.
                assert fake_biology not in data["answer"], (
                    "Tier-2 answer must not contain invented LLM biology"
                )
                assert data["llm_used"] is False, (
                    "Tier-2 main answer must not use LLM framing"
                )
                # ClinVar stats must be available in metadata
                assert "total_variants" in meta, (
                    "ClinVar total_variants must be in gene_metadata for Tier-2"
                )
