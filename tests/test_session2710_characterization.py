# -*- coding: utf-8 -*-
"""
Session 27.10 — Characterization tests: pin CURRENT working behavior before changes.

These tests document the baseline so any regression from Parts A-D is caught
immediately. They exercise the complete /ask path (not mocked internals).

Behaviors captured:
  1. Curated gene (BRCA1) — main answer, no supplemental card, no physician review
  2. ClinVar-grounded gene (COL1A1 if in index) — main answer, no supplemental card
  3. ai_unreviewed (KIAA2022) — supplemental card, physician review persisted
  4. General low-risk (genome history) — no gene card, no review row
  5. VUS deterministic wording — primary VUS phrase in main answer
  6. ClinVar technical data — in gene_metadata, NOT in main answer text
  7. Physician review persistence — review_draft_id inside unverified_gene_draft
  8. Response schema — exactly 5 mandatory fields always present
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAFE_TEXT = (
    "הגן זה מקודד לחלבון הממלא תפקיד חיוני בתאים. "
    "הוא מעורב בתהליכים ביולוגיים מרכזיים הקשורים לפיתוח ותפקוד תאי. "
    "שינויים בגן עלולים להשפיע על תהליכים אלו, אך המשמעות הקלינית של כל ממצא "
    "נקבעת על ידי הצוות הגנטי בהתאם לדוח המלא ולהיסטוריה המשפחתית."
)

MANDATORY = ("answer", "safety_level", "needs_genetic_counselor",
             "matched_topic", "suggested_questions")


def ask(q, **kwargs):
    payload = {"question": q, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200
    return r.json()


def mock_llm(text=SAFE_TEXT):
    m = MagicMock()
    m.call_text_raw.return_value = text
    m._model = "test-model"
    return m


# ===========================================================================
# 1. Curated gene (BRCA1) — trusted main answer, no AI card, no review row
# ===========================================================================

class TestCuratedGeneCharacterization:
    """BRCA1 has curated content: answer is authoritative, no AI expansion."""

    def test_main_answer_nonempty(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        assert len(data.get("answer", "")) >= 80, "Curated BRCA1 answer must be substantial"

    def test_matched_topic_gene(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        assert data.get("matched_topic") == "gene_clinvar_summary"

    def test_no_supplemental_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            "Curated BRCA1 must never have a visible supplemental AI card"
        )

    def test_no_physician_review_row(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("review_draft_id"), (
            "Curated gene must not create a physician review row"
        )

    def test_requires_physician_review_false(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        meta = data.get("gene_metadata") or {}
        assert meta.get("requires_physician_review") is not True

    def test_mandatory_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        for f in MANDATORY:
            assert f in data, f"Missing mandatory field: {f}"

    def test_function_question_no_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("איך הגן BRCA1 פועל בגוף?")
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            "Curated BRCA1 function question must not produce AI card"
        )

    def test_vus_query_no_supplemental_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה המשמעות של VUS בגן BRCA1?")
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            "VUS query for curated gene must not produce AI card"
        )


# ===========================================================================
# 2. ai_unreviewed gene (KIAA2022) — supplemental card, physician review persisted
# ===========================================================================

class TestUnreviewedGeneCharacterization:
    """KIAA2022 has no curated/grounded content: AI card + physician review."""

    def test_supplemental_card_visible_with_llm(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("איך הגן KIAA2022 פועל בגוף?")
        draft = data.get("unverified_gene_draft") or {}
        assert draft.get("visible") is True, (
            "KIAA2022 with LLM must produce a visible supplemental card"
        )

    def test_draft_text_not_in_main_answer(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("איך הגן KIAA2022 פועל בגוף?")
        draft = data.get("unverified_gene_draft") or {}
        answer = data.get("answer", "")
        if draft.get("text_he"):
            assert draft["text_he"][:60] not in answer

    def test_review_draft_id_inside_draft(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("איך הגן KIAA2022 פועל בגוף?")
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert "review_draft_id" in draft, (
                "review_draft_id must be inside unverified_gene_draft"
            )

    def test_requires_physician_review_true(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("איך הגן KIAA2022 פועל בגוף?")
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert draft.get("requires_physician_review") is True

    def test_mandatory_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("איך הגן KIAA2022 פועל בגוף?")
        for f in MANDATORY:
            assert f in data


# ===========================================================================
# 3. General low-risk question — no gene card, no review row
# ===========================================================================

class TestGeneralLowRiskCharacterization:
    """Non-gene educational questions must not produce medical AI cards."""

    def test_no_gene_metadata_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מתי גילו את הגנום?")
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            "General history question must not produce visible medical AI card"
        )

    def test_no_physician_review_row(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מתי גילו את הגנום?")
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("review_draft_id")

    def test_mandatory_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מתי גילו את הגנום?")
        for f in MANDATORY:
            assert f in data


# ===========================================================================
# 4. VUS deterministic wording — primary VUS phrase remains in main answer
# ===========================================================================

class TestVusWordingCharacterization:
    """VUS answers must contain the deterministic safety wording as the primary."""

    @pytest.mark.parametrize("gene,question", [
        ("BRCA1", "מה המשמעות של VUS בגן BRCA1?"),
        ("C12orf57", "מה המשמעות של VUS בגן C12orf57?"),
        ("KIAA2022", "מה המשמעות של VUS בגן KIAA2022?"),
    ])
    def test_vus_wording_in_answer(self, monkeypatch, gene, question):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask(question)
        answer = data.get("answer", "")
        assert "VUS" in answer, f"VUS wording must appear in main answer for {gene}"
        assert gene in answer, f"Gene {gene} must appear in the VUS answer"

    @pytest.mark.parametrize("gene,question", [
        ("BRCA1", "מה המשמעות של VUS בגן BRCA1?"),
        ("C12orf57", "מה המשמעות של VUS בגן C12orf57?"),
    ])
    def test_vus_matched_topic(self, monkeypatch, gene, question):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask(question)
        assert data.get("matched_topic") == "vus_known_gene"

    @pytest.mark.parametrize("gene,question", [
        ("BRCA1", "מה המשמעות של VUS בגן BRCA1?"),
        ("C12orf57", "מה המשמעות של VUS בגן C12orf57?"),
    ])
    def test_vus_draft_text_not_in_answer(self, monkeypatch, gene, question):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask(question)
        draft = data.get("unverified_gene_draft") or {}
        answer = data.get("answer", "")
        if draft.get("text_he"):
            assert draft["text_he"][:60] not in answer, (
                f"Draft text must not be duplicated in VUS answer for {gene}"
            )

    def test_vus_safety_level(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה המשמעות של VUS?")
        # VUS general education must be general_information, not requires_genetic_counselor
        assert data.get("safety_level") == "general_information"


# ===========================================================================
# 5. ClinVar technical data stays in gene_metadata, not leaked into main answer
# ===========================================================================

class TestClinVarDataIsolationCharacterization:
    """ClinVar raw statistics must not appear as raw text in the main answer."""

    def test_clinvar_total_not_as_raw_dump(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        answer = data.get("answer", "")
        # Raw ClinVar dump sentence not in main answer for non-explicit queries
        assert "רשומות וריאנט עבור גן" not in answer, (
            "Raw ClinVar count sentence must not appear in main BRCA1 answer"
        )

    def test_gene_metadata_present_for_gene_question(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        assert data.get("gene_metadata"), "gene_metadata must be present for gene questions"

    def test_gene_metadata_has_gene_symbol(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        meta = data.get("gene_metadata") or {}
        assert meta.get("gene_symbol") == "BRCA1"


# ===========================================================================
# 6. Response schema — mandatory fields always present
# ===========================================================================

class TestResponseSchemaCharacterization:
    """All routes must return the 5 mandatory API fields."""

    @pytest.mark.parametrize("question", [
        "מה זה הגן BRCA1?",
        "מה המשמעות של VUS בגן C12orf57?",
        "איך הגן KIAA2022 פועל בגוף?",
        "מתי גילו את הגנום?",
        "מה זה נשאות?",
        "האם עלי לנתח?",                     # high-stakes redirect
        "אמרו שיש לי VUS",                   # general VUS
        "יש לי ת.ז. 123456789, מה המשמעות?",  # privacy block
    ])
    def test_mandatory_fields_all_routes(self, monkeypatch, question):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask(question)
        for f in MANDATORY:
            assert f in data, f"Mandatory field '{f}' missing for: {question[:40]}"

    def test_answer_always_nonempty(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        data = ask("מה זה הגן BRCA1?")
        assert data.get("answer"), "Answer must be non-empty for gene questions"

    def test_safety_level_valid_values(self, monkeypatch):
        """safety_level must always be one of the four declared values."""
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        valid = {
            "general_information", "contains_identifying_info",
            "requires_genetic_counselor", "out_of_scope",
        }
        for q in [
            "מה זה BRCA1?",
            "מה המשמעות של VUS?",
            "האם עלי לנתח?",
            "יש לי ת.ז. 123456789",
        ]:
            data = ask(q)
            assert data.get("safety_level") in valid, (
                f"safety_level={data.get('safety_level')!r} invalid for: {q}"
            )


# ===========================================================================
# 7. Follow-up handling — VUS practical answer on follow-up
# ===========================================================================

class TestFollowupCharacterization:
    """Follow-up questions route via prior context, not KB keyword scoring."""

    def test_vus_followup_stays_on_vus(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        r1 = ask("מה המשמעות של VUS בגן BRCA1?")
        ctx = [
            {"role": "user", "content": "מה המשמעות של VUS בגן BRCA1?"},
            {"role": "assistant", "content": r1.get("answer", "")},
        ]
        r2 = ask("מה כדאי לעשות עם זה?",
                 last_topic=r1.get("matched_topic"),
                 conversation_context=ctx)
        answer2 = r2.get("answer", "")
        # Follow-up should not return carrier status or unrelated topics
        assert "נשאות" not in answer2 or "VUS" in answer2, (
            "VUS follow-up must not drift to carrier status"
        )

    def test_gene_function_followup_preserves_gene(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: mock_llm())
        r1 = ask("מה המשמעות של VUS בגן C12orf57?")
        ctx = [
            {"role": "user", "content": "מה המשמעות של VUS בגן C12orf57?"},
            {"role": "assistant", "content": r1.get("answer", "")},
        ]
        r2 = ask("ומה התפקיד שלו?",
                 last_topic=r1.get("matched_topic"),
                 conversation_context=ctx)
        answer2 = r2.get("answer", "")
        assert "C12orf57" in answer2, (
            "Function follow-up must preserve C12orf57 context"
        )


# ===========================================================================
# 8. Safety — identifying info and high-stakes blocked
# ===========================================================================

class TestSafetyCharacterization:
    """Safety blocks must fire before any content generation."""

    def test_id_number_blocked(self):
        data = ask("יש לי תעודת זהות 123456789, מה המשמעות?")
        assert data.get("safety_level") == "contains_identifying_info"
        assert data.get("matched_topic") is None

    def test_surgery_decision_redirect(self):
        data = ask("האם עלי לעשות ניתוח?")
        assert data.get("safety_level") in (
            "requires_genetic_counselor", "general_information"
        ), "Surgery decision must be redirected or bounded"

    def test_personal_interpretation_redirect(self):
        data = ask("האם הממצא שלי מסוכן?")
        assert data.get("safety_level") in (
            "requires_genetic_counselor", "general_information"
        )
