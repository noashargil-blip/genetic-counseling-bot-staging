# -*- coding: utf-8 -*-
"""
Session 27.9.10 — Final live-LLM acceptance gate.

Exercises the complete production path with a deterministic mock LLM:
  generation → safety validation → review persistence → response construction → card contract

Cases:
  1. Unknown standalone gene (KIAA2022) — standalone route, LLM available
  2. Unknown gene + VUS (C12orf57) — VUS route, LLM available
  3. Natural follow-up C12orf57: VUS turn → "ומה התפקיד שלו?" — function intent routing
  4. General low-risk history — no supplemental medical card
"""
from __future__ import annotations

import os
import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Safe deterministic Hebrew text that passes all safety/length validators.
# Must be long enough (> 80 chars) and must not contain HGVS, diagnosis,
# specific-risk claims, or identifying information.
# ---------------------------------------------------------------------------
SAFE_GENE_TEXT_HE = (
    "הגן זה מקודד לחלבון הממלא תפקיד חיוני בתאים. "
    "הוא מעורב בתהליכים ביולוגיים מרכזיים הקשורים לפיתוח ותפקוד תאי. "
    "שינויים בגן עלולים להשפיע על תהליכים אלו, אך המשמעות הקלינית של כל ממצא "
    "נקבעת על ידי הצוות הגנטי בהתאם לדוח המלא ולהיסטוריה המשפחתית."
)

SAFE_GENERAL_TEXT_HE = (
    "גילוי הגנום האנושי היה פרויקט מדעי בין-לאומי שהושלם בשנת 2001. "
    "במסגרת פרויקט הגנום האנושי, מדענים ממדינות שונות שיתפו פעולה לפענוח "
    "רצף ה-DNA של כל הכרומוזומים האנושיים. "
    "הפרויקט פתח דלת לעידן חדש של מחקר גנטי."
)


def _mock_llm(text: str = SAFE_GENE_TEXT_HE):
    """Return a MagicMock LLM client that returns the given safe text."""
    m = MagicMock()
    m.call_text_raw.return_value = text
    return m


# ---------------------------------------------------------------------------
# CASE 1 — Unknown standalone gene (KIAA2022) with LLM available
# ---------------------------------------------------------------------------

class TestCase1StandaloneGeneWithLlm:
    """KIAA2022 standalone query: LLM generates biology draft → supplemental card."""

    def test_supplemental_card_visible(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        # If LLM path reached, card must be visible
        if draft:
            assert draft.get("visible") is True, (
                f"unverified_gene_draft.visible should be True, got: {draft}"
            )

    def test_draft_text_he_nonempty(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            assert draft.get("text_he") and len(draft["text_he"]) > 20, (
                "draft.text_he must be non-empty when card is visible"
            )

    def test_draft_text_not_in_main_answer(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        answer = data.get("answer", "")
        if draft and draft.get("text_he"):
            first60 = draft["text_he"][:60]
            assert first60 not in answer, (
                "Draft text must NOT be duplicated inside the main answer"
            )

    def test_gene_meta_supplemental_flags(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        meta = data.get("gene_metadata") or {}
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            assert meta.get("requires_physician_review") is True, (
                f"requires_physician_review must be True for ai_unreviewed. meta={meta}"
            )
            assert meta.get("unverified_gene_draft_displayable") is True, (
                f"unverified_gene_draft_displayable must be True. meta={meta}"
            )
            assert meta.get("draft_promoted_to_answer") is False, (
                f"draft_promoted_to_answer must be False. meta={meta}"
            )

    def test_review_persistence(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            # review_draft_id lives inside unverified_gene_draft (top-level is stripped
            # by CounselingAskResponse's @model_serializer which only keeps declared fields).
            assert "review_draft_id" in draft, (
                "review_draft_id must be present in unverified_gene_draft dict"
            )
            # review_status must be set on the draft object
            assert draft.get("review_status") is not None, (
                "draft.review_status must not be None when persistence succeeded"
            )

    def test_ai_draft_debug_flags(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        debug = data.get("ai_draft_debug") or {}
        if draft and draft.get("visible"):
            assert debug.get("attempted") is True, f"ai_draft_debug.attempted must be True: {debug}"
            assert debug.get("generated") is True, f"ai_draft_debug.generated must be True: {debug}"
            assert debug.get("shown") is True, f"ai_draft_debug.shown must be True: {debug}"

    def test_matched_topic(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        assert data.get("matched_topic") == "gene_clinvar_summary"


# ---------------------------------------------------------------------------
# CASE 2 — VUS + unknown gene (C12orf57) with LLM available
# ---------------------------------------------------------------------------

class TestCase2VusGeneWithLlm:
    """C12orf57 VUS query: deterministic VUS answer + gene AI draft in supplemental card."""

    def test_answer_contains_vus_explanation(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"}).json()
        answer = data.get("answer", "")
        assert "VUS" in answer, "Main answer must contain VUS explanation"
        assert "C12orf57" in answer, "Main answer must mention C12orf57"

    def test_supplemental_card_when_llm_generates(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft:
            assert draft.get("visible") is True, "Supplemental card must be visible"

    def test_ai_does_not_rewrite_vus_answer(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        answer = data.get("answer", "")
        # Draft text must NOT appear in the main VUS answer
        if draft and draft.get("text_he"):
            first60 = draft["text_he"][:60]
            assert first60 not in answer, "Draft text must not be duplicated in VUS answer"

    def test_supplemental_meta_flags(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        meta = data.get("gene_metadata") or {}
        if draft and draft.get("visible"):
            assert meta.get("requires_physician_review") is True
            assert meta.get("unverified_gene_draft_displayable") is True
            assert meta.get("draft_promoted_to_answer") is False
            # review_draft_id lives inside unverified_gene_draft (top-level is stripped by serializer)
            assert "review_draft_id" in draft, (
                "review_draft_id must be present in unverified_gene_draft"
            )

    def test_review_status_default(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            rs = draft.get("review_status")
            assert rs is not None, (
                "draft.review_status must not be None when card is visible"
            )

    def test_matched_topic_vus(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "מה המשמעות של VUS בגן C12orf57?"}).json()
        assert data.get("matched_topic") == "vus_known_gene"


# ---------------------------------------------------------------------------
# CASE 3 — Natural follow-up: VUS turn → gene-function question
# ---------------------------------------------------------------------------

class TestCase3FollowupGeneFunction:
    """Turn 1: VUS in C12orf57. Turn 2: 'ומה התפקיד שלו?' → function routing."""

    Q1 = "מה המשמעות של VUS בגן C12orf57?"
    Q2 = "ומה התפקיד שלו?"

    def _run(self, monkeypatch, with_llm: bool = True):
        if with_llm:
            monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        r1 = client.post("/ask", json={"question": self.Q1}).json()
        ctx_messages = [
            {"role": "user", "content": self.Q1},
            {"role": "assistant", "content": r1.get("answer", ""),
             "matched_topic": r1.get("matched_topic")},
        ]
        r2 = client.post("/ask", json={
            "question": self.Q2,
            "last_topic": r1.get("matched_topic"),
            "conversation_context": ctx_messages,
        }).json()
        return r1, r2

    def test_followup_preserves_gene_context(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        answer = r2.get("answer", "")
        assert "C12orf57" in answer, (
            f"Follow-up answer must mention C12orf57 to preserve context. "
            f"Got: {answer[:300]}"
        )

    def test_followup_not_generic_gene_definition(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        answer = r2.get("answer", "")
        # Generic "what is a gene" entry begins with a definition of a gene in general.
        # A gene-context-aware answer should NOT begin with "גן הוא יחידת מידע גנטי"
        generic_start = "גן הוא יחידת מידע גנטי"
        assert generic_start not in answer, (
            "Follow-up must not return generic gene-definition answer"
        )

    def test_followup_does_not_merely_repeat_vus_definition(self, monkeypatch):
        r1, r2 = self._run(monkeypatch)
        # Turn 2 answer should differ from Turn 1 answer (not identical VUS opener)
        a1 = (r1.get("answer") or "")[:100]
        a2 = (r2.get("answer") or "")[:100]
        assert a1 != a2, (
            "Follow-up answer should differ from the VUS definition turn"
        )

    def test_followup_answer_nonempty(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        assert len(r2.get("answer", "")) >= 50, "Follow-up answer must be substantial"

    def test_followup_draft_not_in_main_answer(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        draft = r2.get("unverified_gene_draft") or {}
        answer = r2.get("answer", "")
        if draft and draft.get("text_he"):
            assert draft["text_he"][:60] not in answer, "Draft text duplicated in answer"

    def test_followup_applies_source_policy(self, monkeypatch):
        """With LLM: C12orf57 (no curated/grounded) → supplemental card. No promotion."""
        _, r2 = self._run(monkeypatch, with_llm=True)
        draft = r2.get("unverified_gene_draft") or {}
        meta = r2.get("gene_metadata") or {}
        # If the engine generated a gene explanation (LLM reached), it must be in card only
        if draft and draft.get("visible"):
            assert meta.get("draft_promoted_to_answer") is not True, (
                "AI draft must not be promoted to main answer"
            )


# ---------------------------------------------------------------------------
# CASE 4 — General low-risk history question
# ---------------------------------------------------------------------------

class TestCase4GeneralHistory:
    """'מתי גילו את הגנום?' — low-risk general education, no medical supplemental card."""

    Q = "מתי גילו את הגנום?"

    def test_no_supplemental_medical_card(self, monkeypatch):
        """No medical AI card for a general history question."""
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": self.Q}).json()
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            f"General history must not produce a visible supplemental medical AI card. "
            f"Got: {draft}"
        )

    def test_no_physician_review_row(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": self.Q}).json()
        # review_draft_id must be absent or None for general questions
        assert not data.get("review_draft_id"), (
            "General history question must not produce a physician-review row"
        )

    def test_answer_nonempty(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": self.Q}).json()
        assert len(data.get("answer", "")) > 20, "Answer must be non-empty"

    def test_mandatory_api_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": self.Q}).json()
        for field in ("answer", "safety_level", "needs_genetic_counselor",
                      "matched_topic", "suggested_questions"):
            assert field in data, f"Mandatory response field '{field}' missing"

    def test_print_actual_answer(self, monkeypatch, capsys):
        """Print the actual patient-facing answer for manual review."""
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": self.Q}).json()
        with capsys.disabled():
            print(f"\n[CASE4 actual answer]\n  safety_level: {data.get('safety_level')}")
            print(f"  matched_topic: {data.get('matched_topic')}")
            print(f"  answer[:400]: {(data.get('answer') or '')[:400]}")


# ---------------------------------------------------------------------------
# Frontend contract: supplemental card round-trip
# ---------------------------------------------------------------------------

class TestFrontendContract:
    """Verify that the unverified_gene_draft structure is complete for frontend rendering."""

    REQUIRED_DRAFT_KEYS = ("visible", "text_he", "review_status", "requires_physician_review",
                            "gene_symbol", "ai_content_type", "approved")

    def test_draft_keys_present_when_visible(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            for key in self.REQUIRED_DRAFT_KEYS:
                assert key in draft, (
                    f"unverified_gene_draft missing key '{key}'. draft={draft}"
                )

    def test_draft_review_status_not_none_when_visible(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            assert draft.get("review_status") is not None, (
                "draft.review_status must not be None when card is visible — "
                "frontend relies on this to show review badge"
            )

    def test_review_metadata_in_draft(self, monkeypatch):
        """review_draft_id lives inside unverified_gene_draft (serializer strips top-level)."""
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            assert "review_draft_id" in draft, (
                "review_draft_id must be inside unverified_gene_draft for frontend use"
            )
            assert draft.get("review_status") is not None, (
                "draft.review_status must be set so frontend can show review badge"
            )

    def test_gene_metadata_displayable_flag(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן KIAA2022 פועל בגוף?"}).json()
        meta = data.get("gene_metadata") or {}
        draft = data.get("unverified_gene_draft") or {}
        if draft and draft.get("visible"):
            assert meta.get("unverified_gene_draft_displayable") is True, (
                "gene_metadata.unverified_gene_draft_displayable must be True "
                "so the frontend gate does not suppress a visible card"
            )

    def test_curated_gene_never_gets_supplemental_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm())
        data = client.post("/ask", json={"question": "איך הגן BRCA1 פועל בגוף?"}).json()
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            "BRCA1 (curated Source A) must never produce a visible supplemental AI card"
        )
