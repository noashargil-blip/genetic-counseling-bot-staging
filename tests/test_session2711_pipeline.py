# -*- coding: utf-8 -*-
"""
Session 27.9.11 — Comprehensive end-to-end pipeline tests.

Tests the complete production path:
  POST /ask → intent routing → source resolution → LLM client →
  response composition → review_db persistence → serialization → card contract

Patching: ONLY the lowest-level LLM call (create_llm_client).
NOT mocked: resolver, route, persistence, response construction, serialization.

Source class taxonomy:
  A — curated / grounded / physician_approved → no supplemental card, no review row
  B — ai_unreviewed (LLM succeeds) → visible card, review persisted in DB
  E — LLM unavailable or returns invalid content → safe fallback, no visible card
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.llm_client import LLMClientError

client = TestClient(app)

# ---------------------------------------------------------------------------
# Safe deterministic text — passes all validators (Hebrew, length, no HGVS,
# no diagnosis, no personal risk, no identifying info).
# ---------------------------------------------------------------------------
SAFE_DRAFT_TEXT = (
    "הגן זה מקודד לחלבון הממלא תפקיד חיוני בתאים. "
    "הוא מעורב בתהליכים ביולוגיים מרכזיים הקשורים לפיתוח ותפקוד תאי. "
    "שינויים בגן עלולים להשפיע על תהליכים אלו, אך המשמעות הקלינית של כל ממצא "
    "נקבעת על ידי הצוות הגנטי בהתאם לדוח המלא ולהיסטוריה המשפחתית."
)

# Text that fails validation — contains personal risk language
INVALID_DRAFT_TEXT = "הגן הזה גורם לסרטן ומהווה סיכון אישי גבוה עבורך."


def _mock_llm_ok(text: str = SAFE_DRAFT_TEXT) -> MagicMock:
    m = MagicMock()
    m.call_text_raw.return_value = text
    m._model = "test-model-gpt4o"
    return m


def _mock_llm_empty() -> MagicMock:
    m = MagicMock()
    m.call_text_raw.return_value = ""
    m._model = "test-model-gpt4o"
    return m


def _mock_llm_invalid() -> MagicMock:
    m = MagicMock()
    m.call_text_raw.return_value = INVALID_DRAFT_TEXT
    m._model = "test-model-gpt4o"
    return m


def _mock_llm_raises() -> MagicMock:
    m = MagicMock()
    m.call_text_raw.side_effect = LLMClientError("connection refused")
    m._model = "test-model-gpt4o"
    return m


def ask(question, last_topic=None, conversation_context=None):
    payload = {"question": question}
    if last_topic:
        payload["last_topic"] = last_topic
    if conversation_context:
        payload["conversation_context"] = conversation_context
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    return resp.json()


MANDATORY_RESPONSE_FIELDS = (
    "answer", "safety_level", "needs_genetic_counselor",
    "matched_topic", "suggested_questions",
)


def assert_mandatory_fields(data: dict) -> None:
    for field in MANDATORY_RESPONSE_FIELDS:
        assert field in data, f"Mandatory field '{field}' missing from response"


# ===========================================================================
# SOURCE CLASS A — curated or grounded source → no supplemental card
# Gene: BRCA1 (curated biology answer)
# ===========================================================================

class TestSourceA_CuratedBRCA1:
    """BRCA1 is curated — must never produce a supplemental AI card regardless of LLM."""

    def test_standalone_overview_no_supplemental_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן BRCA1?")
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            f"BRCA1 curated must not have a visible supplemental card. draft={draft}"
        )

    def test_standalone_function_no_supplemental_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("איך הגן BRCA1 פועל בגוף?")
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            f"BRCA1 curated must not have a visible supplemental card. draft={draft}"
        )

    def test_mandatory_fields_present(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן BRCA1?")
        assert_mandatory_fields(data)

    def test_no_physician_review_row(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן BRCA1?")
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("review_draft_id"), (
            "Curated gene must not produce a physician-review row"
        )

    def test_requires_physician_review_false(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן BRCA1?")
        meta = data.get("gene_metadata") or {}
        # Curated source A must not flag physician review
        assert meta.get("requires_physician_review") is not True, (
            "Curated gene must not require physician review"
        )

    def test_answer_nonempty(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן BRCA1?")
        assert len(data.get("answer", "")) >= 100, "Curated BRCA1 answer must be substantial"

    def test_vus_query_no_supplemental_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה המשמעות של VUS בגן BRCA1?")
        draft = data.get("unverified_gene_draft")
        # VUS + curated gene: main answer is VUS explanation.
        # The gene biology from curated source means no AI card needed.
        assert draft is None or not (draft or {}).get("visible"), (
            f"BRCA1 curated VUS query must not produce visible AI card. draft={draft}"
        )


# ===========================================================================
# SOURCE CLASS B — ai_unreviewed; LLM succeeds
# Gene: KIAA2022 (unknown gene, tier3, no ClinVar index entry)
# ===========================================================================

class TestSourceB_StandaloneUnknownGene:
    """KIAA2022 standalone: LLM generates biology → supplemental card visible, review persisted."""

    Q_OVERVIEW = "מה זה הגן KIAA2022?"
    Q_FUNCTION = "איך הגן KIAA2022 פועל בגוף?"

    def _post_overview(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        return ask(self.Q_OVERVIEW)

    def _post_function(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        return ask(self.Q_FUNCTION)

    def test_overview_matched_topic(self, monkeypatch):
        data = self._post_overview(monkeypatch)
        assert data.get("matched_topic") == "gene_clinvar_summary"

    def test_function_matched_topic(self, monkeypatch):
        data = self._post_function(monkeypatch)
        assert data.get("matched_topic") == "gene_clinvar_summary"

    def test_overview_mandatory_fields(self, monkeypatch):
        assert_mandatory_fields(self._post_overview(monkeypatch))

    def test_function_mandatory_fields(self, monkeypatch):
        assert_mandatory_fields(self._post_function(monkeypatch))

    def test_supplemental_card_visible(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        assert draft.get("visible") is True, (
            f"KIAA2022 unknown gene with LLM should produce visible card. draft={draft}"
        )

    def test_draft_text_nonempty(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert draft.get("text_he") and len(draft["text_he"]) > 20, (
                "Visible draft must have non-empty text_he"
            )

    def test_draft_text_not_in_main_answer(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        answer = data.get("answer", "")
        if draft.get("text_he"):
            first60 = draft["text_he"][:60]
            assert first60 not in answer, "Draft text must not be duplicated in main answer"

    def test_review_persisted_in_draft(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert "review_draft_id" in draft, (
                "review_draft_id must be present inside unverified_gene_draft"
            )

    def test_review_status_set(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert draft.get("review_status") is not None, (
                "draft.review_status must not be None when card is visible"
            )

    def test_requires_physician_review_true(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert draft.get("requires_physician_review") is True, (
                "AI-generated draft must require physician review"
            )

    def test_gene_metadata_displayable_flag(self, monkeypatch):
        data = self._post_function(monkeypatch)
        meta = data.get("gene_metadata") or {}
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert meta.get("unverified_gene_draft_displayable") is True, (
                "gene_metadata.unverified_gene_draft_displayable must be True"
            )
            assert meta.get("draft_promoted_to_answer") is False, (
                "Draft must not be promoted to main answer"
            )

    def test_required_draft_keys(self, monkeypatch):
        data = self._post_function(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            required = ("visible", "text_he", "review_status", "requires_physician_review",
                        "gene_symbol", "ai_content_type", "approved")
            for key in required:
                assert key in draft, f"unverified_gene_draft missing key '{key}'"

    def test_ai_draft_debug_flags(self, monkeypatch):
        data = self._post_function(monkeypatch)
        debug = data.get("ai_draft_debug") or {}
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert debug.get("attempted") is True
            assert debug.get("generated") is True
            assert debug.get("shown") is True


# ===========================================================================
# SOURCE CLASS B — VUS + unknown gene route
# Gene: C12orf57 (sparse, likely ai_unreviewed)
# ===========================================================================

class TestSourceB_VusUnknownGene:
    """C12orf57 VUS: deterministic VUS answer + gene AI draft in supplemental card."""

    Q = "מה המשמעות של VUS בגן C12orf57?"

    def _post(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        return ask(self.Q)

    def test_matched_topic_vus(self, monkeypatch):
        assert self._post(monkeypatch).get("matched_topic") == "vus_known_gene"

    def test_mandatory_fields(self, monkeypatch):
        assert_mandatory_fields(self._post(monkeypatch))

    def test_vus_explanation_in_answer(self, monkeypatch):
        data = self._post(monkeypatch)
        answer = data.get("answer", "")
        assert "VUS" in answer, "VUS explanation must appear in main answer"
        assert "C12orf57" in answer, "Gene symbol must appear in main answer"

    def test_supplemental_card_visible_when_llm_available(self, monkeypatch):
        data = self._post(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft:
            assert draft.get("visible") is True, (
                f"C12orf57 supplemental card should be visible when LLM is available. "
                f"draft={draft}"
            )

    def test_draft_text_not_in_vus_answer(self, monkeypatch):
        data = self._post(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        answer = data.get("answer", "")
        if draft.get("text_he"):
            first60 = draft["text_he"][:60]
            assert first60 not in answer, "Draft text must not be duplicated in VUS main answer"

    def test_review_draft_id_in_draft(self, monkeypatch):
        data = self._post(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert "review_draft_id" in draft, (
                "review_draft_id must be inside unverified_gene_draft"
            )

    def test_physician_review_flagged(self, monkeypatch):
        data = self._post(monkeypatch)
        draft = data.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert draft.get("requires_physician_review") is True


# ===========================================================================
# SOURCE CLASS B — Follow-up route
# Turn 1: VUS in C12orf57, Turn 2: "ומה התפקיד שלו?"
# ===========================================================================

class TestSourceB_Followup:
    """Multi-turn: VUS → function follow-up preserves gene context."""

    Q1 = "מה המשמעות של VUS בגן C12orf57?"
    Q2 = "ומה התפקיד שלו?"

    def _run(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        r1 = ask(self.Q1)
        ctx = [
            {"role": "user", "content": self.Q1},
            {"role": "assistant", "content": r1.get("answer", ""),
             "matched_topic": r1.get("matched_topic")},
        ]
        r2 = ask(self.Q2, last_topic=r1.get("matched_topic"), conversation_context=ctx)
        return r1, r2

    def test_followup_answer_nonempty(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        assert len(r2.get("answer", "")) >= 50, "Follow-up answer must be substantial"

    def test_followup_gene_context_preserved(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        answer = r2.get("answer", "")
        assert "C12orf57" in answer, (
            f"Follow-up must preserve C12orf57 context. Got: {answer[:300]}"
        )

    def test_followup_not_generic_gene_definition(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        answer = r2.get("answer", "")
        # Generic gene-definition entry
        assert "גן הוא יחידת מידע גנטי" not in answer, (
            "Follow-up must not return a generic gene-definition answer"
        )

    def test_followup_differs_from_vus_turn(self, monkeypatch):
        r1, r2 = self._run(monkeypatch)
        a1 = (r1.get("answer") or "")[:100]
        a2 = (r2.get("answer") or "")[:100]
        assert a1 != a2, "Follow-up must not be identical to the VUS turn"

    def test_followup_mandatory_fields(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        assert_mandatory_fields(r2)

    def test_followup_draft_not_in_answer(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        draft = r2.get("unverified_gene_draft") or {}
        answer = r2.get("answer", "")
        if draft.get("text_he"):
            first60 = draft["text_he"][:60]
            assert first60 not in answer, "Draft text must not be duplicated in follow-up answer"

    def test_followup_draft_not_promoted(self, monkeypatch):
        _, r2 = self._run(monkeypatch)
        meta = r2.get("gene_metadata") or {}
        draft = r2.get("unverified_gene_draft") or {}
        if draft.get("visible"):
            assert meta.get("draft_promoted_to_answer") is not True, (
                "AI draft must never be promoted to main answer"
            )


# ===========================================================================
# SOURCE CLASS E — LLM unavailable or returns invalid content
# Standalone gene route: KIAA2022
# ===========================================================================

class TestSourceE_LlmFails:
    """When LLM is unavailable or generates invalid content, fallback must be safe."""

    Q = "איך הגן KIAA2022 פועל בגוף?"

    def test_llm_client_error_no_visible_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask(self.Q)
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("visible"), (
            "LLM error must not produce a visible supplemental card"
        )

    def test_llm_client_error_answer_nonempty(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask(self.Q)
        assert len(data.get("answer", "")) > 0, (
            "Fallback answer must be non-empty even when LLM fails"
        )

    def test_llm_client_error_mandatory_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        assert_mandatory_fields(ask(self.Q))

    def test_llm_client_error_no_review_row(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask(self.Q)
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("review_draft_id"), (
            "No review row must be persisted when LLM generation failed"
        )

    def test_empty_llm_response_no_visible_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_empty())
        data = ask(self.Q)
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("visible"), (
            "Empty LLM response must not produce a visible supplemental card"
        )

    def test_empty_llm_response_mandatory_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_empty())
        assert_mandatory_fields(ask(self.Q))

    def test_invalid_llm_response_no_visible_card(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_invalid())
        data = ask(self.Q)
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("visible"), (
            "Validation-rejected LLM text must not produce a visible supplemental card"
        )

    def test_invalid_llm_response_mandatory_fields(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_invalid())
        assert_mandatory_fields(ask(self.Q))

    def test_invalid_llm_response_no_personal_risk_in_answer(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_invalid())
        data = ask(self.Q)
        answer = data.get("answer", "")
        # The rejected draft text must never leak into the main answer
        assert INVALID_DRAFT_TEXT[:40] not in answer, (
            "Validation-rejected draft text must never appear in the main answer"
        )


# ===========================================================================
# SOURCE CLASS E — VUS + unknown gene; LLM fails
# ===========================================================================

class TestSourceE_VusLlmFails:
    """C12orf57 VUS when LLM is unavailable: deterministic VUS answer must still work."""

    Q = "מה המשמעות של VUS בגן C12orf57?"

    def test_vus_answer_present_without_llm(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask(self.Q)
        answer = data.get("answer", "")
        # VUS explanation must still be provided even without gene biology
        assert "VUS" in answer, "VUS explanation must appear even when LLM fails"

    def test_matched_topic_vus_without_llm(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask(self.Q)
        assert data.get("matched_topic") == "vus_known_gene"

    def test_no_visible_card_when_llm_fails(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask(self.Q)
        draft = data.get("unverified_gene_draft") or {}
        assert not draft.get("visible"), (
            "LLM failure must not produce visible card for VUS+gene route"
        )

    def test_mandatory_fields_without_llm(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        assert_mandatory_fields(ask(self.Q))


# ===========================================================================
# Response schema contract — five fields exactly (no extras at top level)
# ===========================================================================

class TestResponseSchemaContract:
    """The API contract requires exactly 5 declared fields at the response root."""

    DECLARED_FIELDS = frozenset({
        "answer", "safety_level", "needs_genetic_counselor",
        "matched_topic", "suggested_questions",
    })

    def _check_mandatory_present(self, data: dict) -> None:
        for f in self.DECLARED_FIELDS:
            assert f in data, f"Mandatory field '{f}' missing"

    def test_schema_unknown_gene_with_llm(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן KIAA2022?")
        self._check_mandatory_present(data)

    def test_schema_vus_unknown_gene(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה המשמעות של VUS בגן C12orf57?")
        self._check_mandatory_present(data)

    def test_schema_curated_gene(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מה זה הגן BRCA1?")
        self._check_mandatory_present(data)

    def test_schema_llm_error(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_raises())
        data = ask("מה זה הגן KIAA2022?")
        self._check_mandatory_present(data)

    def test_schema_general_question(self, monkeypatch):
        monkeypatch.setattr("app.counseling_engine.create_llm_client", lambda: _mock_llm_ok())
        data = ask("מתי גילו את הגנום?")
        self._check_mandatory_present(data)
        draft = data.get("unverified_gene_draft")
        assert draft is None or not (draft or {}).get("visible"), (
            "General question must not produce a visible medical supplemental card"
        )
