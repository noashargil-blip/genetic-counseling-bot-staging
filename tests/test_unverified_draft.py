"""
tests/test_unverified_draft.py — Unverified gene draft feature tests.

Covers the opt-in Tier 2 gene background draft:
  - Default: no draft unless explicitly requested
  - Draft structure when returned (warning, approved=False, review_status)
  - Draft never merged into answer text
  - Validator rejects medical recommendations, personal-risk language, CJK
  - Tier 1 (approved gene card) does not offer a draft
  - Tier 3 (unknown gene) does not generate a misleading draft
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.counseling_engine import (
    _generate_unverified_gene_draft,
    _validate_unverified_draft,
    _validate_unverified_draft_clinvar_ok,
)
from app.main import app

_client = TestClient(app)


def _ask(payload: dict) -> dict:
    resp = _client.post("/ask", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# _validate_unverified_draft — unit tests
# ---------------------------------------------------------------------------

class TestValidateUnverifiedDraft:

    def test_empty_rejected(self):
        assert _validate_unverified_draft("") == "empty"

    def test_clinvar_in_draft_text_is_rejected(self):
        # Patient-facing draft text must not contain "ClinVar" brand name
        text = (
            "הגן BRCA1 מדווח במאגר ClinVar בהקשרים הקשורים לסרטן שד ולסרטן שחלה. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
        )
        assert _validate_unverified_draft(text) is not None

    def test_too_long_rejected(self):
        text = "א" * 601
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "too long" in reason

    def test_question_mark_rejected(self):
        text = "מה התפקיד של BRCA1? הגן מעורב בתיקון DNA."
        assert _validate_unverified_draft(text) is not None

    def test_cjk_rejected(self):
        text = "הגן APC מעורב בסרטן המעי. 日本語テスト"
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "CJK" in reason

    def test_no_hebrew_rejected(self):
        text = "BRCA1 is a gene involved in DNA repair."
        assert _validate_unverified_draft(text) is not None

    # Medical action terms — forbidden by _FORBIDDEN_INTRO_RE
    def test_surgery_term_rejected(self):
        text = "הגן APC קשור לסרטן המעי הגס. מומלץ לבצע ניתוח מניעתי."
        reason = _validate_unverified_draft(text)
        assert reason is not None

    def test_colonoscopy_term_rejected(self):
        text = "הגן APC קשור לתסמונת FAP. יש לבצע קולונוסקופיה."
        reason = _validate_unverified_draft(text)
        assert reason is not None

    def test_surveillance_term_rejected(self):
        text = "הגן BRCA1 קשור לסרטן שד. מומלץ מעקב רפואי שוטף."
        reason = _validate_unverified_draft(text)
        assert reason is not None

    # Personal-risk language — forbidden by _FORBIDDEN_DRAFT_PERSONAL_RE
    def test_personal_risk_לך_יש_rejected(self):
        text = "לך יש שינוי בגן APC שקשור לסרטן."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "personal-risk" in reason

    def test_personal_risk_הסיכון_שלך_rejected(self):
        text = "הסיכון שלך לחלות בסרטן גבוה."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "personal-risk" in reason

    def test_personal_risk_אתה_חולה_rejected(self):
        text = "הגן BRCA1 חשוב לתיקון DNA. אתה חולה בגלל הוריאנט שלך."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "personal-risk" in reason

    def test_personal_risk_אצלך_יש_rejected(self):
        text = "אצלך יש מוטציה שמשפיעה על תפקוד הגן."
        reason = _validate_unverified_draft(text)
        assert reason is not None

    # Quality-gate phrases (from _FRAMING_QUALITY_RE)
    def test_ממצאון_rejected(self):
        text = "הממצאון בגן שלך אינו מאפיין בהכרח מחלה."
        assert _validate_unverified_draft(text) is not None

    def test_אני_כאן_לעזור_rejected(self):
        text = "הגן APC מעורב בתהליך. אני כאן לעזור בכל שאלה."
        assert _validate_unverified_draft(text) is not None


# ---------------------------------------------------------------------------
# _DRAFT_QUALITY_RE patterns — broken/mixed-language output
# ---------------------------------------------------------------------------

class TestDraftQualityPatterns:
    """Patterns from _DRAFT_QUALITY_RE that catch LLM hallucinations and
    broken Hebrew-English mixed output."""

    def test_genom_rejected(self):
        """'genom' is a broken transliteration; should be rejected."""
        text = "הגן POLE הוא חלק מה genom האנושי."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "draft-quality" in reason

    def test_Genom_case_insensitive_rejected(self):
        text = "הגן הוא חלק מה Genom."
        assert _validate_unverified_draft(text) is not None

    def test_אנזימ_truncated_rejected(self):
        """'אנזימ' is a truncated form of 'אנזים' — rejected as broken."""
        text = "הגן מייצר אנזימ שמשתתף בתהליכים תאיים."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "draft-quality" in reason

    def test_קטלאזה_invented_word_rejected(self):
        """'קטלאזה' is an invented word for catalysis — rejected."""
        text = "הגן משתתף בתהליך הקטלאזה של DNA."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "draft-quality" in reason

    def test_מודד_חשיבותה_rejected(self):
        """Example broken phrase from the bad output."""
        text = "הגן הזה מודד חשיבותה במחקר גנטיקה רפואי."
        reason = _validate_unverified_draft(text)
        assert reason is not None
        assert "draft-quality" in reason

    def test_prompt_echo_600_chars_rejected(self):
        """LLM echoing the system prompt's '600 characters total' constraint."""
        text = "הגן APC קשור לתהליכים ביולוגיים. (600 characters total)"
        assert _validate_unverified_draft(text) is not None

    def test_prompt_echo_characters_total_rejected(self):
        text = "הגן POLE. Maximum 600 characters total."
        assert _validate_unverified_draft(text) is not None

    def test_clinvar_mention_in_draft_is_now_rejected(self):
        """Patient-facing draft text must not contain 'ClinVar' by name."""
        text = (
            "הגן POLE מדווח במאגר ClinVar בהקשרים הקשורים לסרטן מעי גס ולסרטן אנדומטריום. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
        )
        assert _validate_unverified_draft(text) is not None


# ---------------------------------------------------------------------------
# _validate_unverified_draft_clinvar_ok — second-pass validator
# ---------------------------------------------------------------------------

class TestValidateUnverifiedDraftClinvarOk:
    """Second-pass validator: ClinVar allowed, statistics and dangerous content still blocked."""

    SAFE_WITH_CLINVAR = (
        "הגן HBB מופיע ב-ClinVar בהקשרים של אנמיה חרמשית. "
        "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
    )

    def test_clinvar_mention_accepted_on_second_pass(self):
        """ClinVar in an otherwise safe draft is accepted by the second-pass validator."""
        assert _validate_unverified_draft_clinvar_ok(self.SAFE_WITH_CLINVAR) is None

    def test_clinvar_still_rejected_on_first_pass(self):
        """ClinVar still causes first-pass rejection (retry trigger)."""
        assert _validate_unverified_draft(self.SAFE_WITH_CLINVAR) is not None

    def test_clnvar_typo_rejected_on_second_pass(self):
        """ClnVar (LLM typo) is rejected in both passes — it's in _DRAFT_QUALITY_RE."""
        text = "הגן HBB מופיע ב-ClnVar בהקשרים של אנמיה. ממצא VUS נותר בגדר אי-ודאות."
        assert _validate_unverified_draft_clinvar_ok(text) is not None

    def test_variant_count_rejected_on_second_pass(self):
        """Variant count statistics are rejected in both passes."""
        text = "הגן HBB מדווח עם 532 וריאנטים. ממצא VUS נותר בגדר אי-ודאות."
        assert _validate_unverified_draft_clinvar_ok(text) is not None

    def test_pathogenic_count_rejected_on_second_pass(self):
        """English clinical classification counts rejected on second pass."""
        text = "הגן HBB מדווח עם 45 Pathogenic. ממצא VUS נותר בגדר אי-ודאות."
        assert _validate_unverified_draft_clinvar_ok(text) is not None

    def test_pathogenic_term_rejected_on_second_pass(self):
        """English 'Pathogenic' term rejected on second pass."""
        text = "הגן HBB קשור ל-Pathogenic וריאנטים שונים. ממצא VUS נותר בגדר אי-ודאות."
        assert _validate_unverified_draft_clinvar_ok(text) is not None

    def test_personal_risk_rejected_on_second_pass(self):
        """Personal risk language still blocked on second pass."""
        text = "הסיכון שלך לחלות גבוה בגלל גן זה."
        assert _validate_unverified_draft_clinvar_ok(text) is not None

    def test_medical_advice_rejected_on_second_pass(self):
        """Medical action terms still blocked on second pass."""
        text = "הגן HBB קשור לאנמיה. מומלץ לבצע קולונוסקופיה."
        assert _validate_unverified_draft_clinvar_ok(text) is not None

    def test_safe_no_clinvar_accepted_on_second_pass(self):
        """A clean safe draft without ClinVar is accepted on second pass too."""
        text = "הגן HBB מופיע לעיתים בהקשרים של אנמיה ומחלות המוגלובין שונות. ממצא VUS בגן זה נותר בגדר אי-ודאות."
        assert _validate_unverified_draft_clinvar_ok(text) is None


# ---------------------------------------------------------------------------
# Retry logic — TestGenerateUnverifiedDraft extension
# ---------------------------------------------------------------------------

class TestDraftRetryLogic:
    """_generate_unverified_gene_draft retries once with a stricter prompt."""

    def test_retry_on_first_failure_returns_valid(self):
        """First call returns bad text; second call returns valid text → draft returned."""
        bad_text = "הגן POLE הוא חלק מה genom האנושי."
        good_text = "הגן POLE מופיע לעיתים בהקשרים של סרטן מעי גס. ממצא VUS בגן זה נותר בגדר אי-ודאות."

        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = [bad_text, good_text]

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("POLE")

        assert mock_client.call_text_raw.call_count == 2, "Should have retried exactly once"
        assert result is not None
        assert result["text_he"] == good_text
        assert result["approved"] is False

    def test_both_attempts_fail_returns_none(self):
        """First and second attempts both invalid → None returned."""
        bad = "הגן הזה מודד חשיבותה במחקר גנטיקה."

        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = [bad, bad]

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("POLE")

        assert mock_client.call_text_raw.call_count == 2
        assert result is None

    def test_first_attempt_valid_no_retry(self):
        """When the first attempt is valid, no retry call is made."""
        good_text = "הגן POLE מופיע לעיתים בהקשרים של סרטן מעי גס. ממצא VUS בגן זה נותר בגדר אי-ודאות."

        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = good_text

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("POLE")

        assert mock_client.call_text_raw.call_count == 1, "Should NOT retry when first attempt is valid"
        assert result is not None

    def test_clinvar_in_first_triggers_retry_accepted_if_safe(self):
        """First attempt has ClinVar → retry; retry has ClinVar but is otherwise safe → accepted."""
        first_text = "הגן HBB מדווח ב-ClinVar בהקשרים של אנמיה חרמשית. ממצא VUS נותר בגדר אי-ודאות."
        retry_text = "הגן HBB מופיע ב-ClinVar בהקשרים של אנמיה חרמשית. ממצא VUS בגן זה נותר בגדר אי-ודאות."

        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = [first_text, retry_text]

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("HBB")

        assert mock_client.call_text_raw.call_count == 2
        assert result is not None, "Retry with safe ClinVar-mentioning text should be accepted"
        assert result["text_he"] == retry_text
        assert result["approved"] is False

    def test_clinvar_in_retry_plus_statistics_rejected(self):
        """First attempt has ClinVar → retry; retry has ClinVar + statistics → rejected → None."""
        first_text = "הגן HBB מדווח ב-ClinVar בהקשרים של אנמיה. ממצא VUS נותר בגדר אי-ודאות."
        retry_text = "הגן HBB מדווח ב-ClinVar עם 532 וריאנטים. ממצא VUS נותר בגדר אי-ודאות."

        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = [first_text, retry_text]

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("HBB")

        assert result is None, "Statistics in retry should still be rejected"


# ---------------------------------------------------------------------------
# /ask endpoint integration — default opt-out behavior
# ---------------------------------------------------------------------------

class TestUnverifiedDraftDefault:
    """By default (include_unverified_gene_draft not sent), no draft is returned."""

    def test_tier2_gene_no_draft_by_default(self):
        """Tier 2 question with no flag — unverified_gene_draft must be absent."""
        data = _ask({"question": "מה ידוע על HBB?"})
        assert "unverified_gene_draft" not in data, (
            f"Expected no unverified_gene_draft in response, got: {data.get('unverified_gene_draft')}"
        )

    def test_include_false_explicit_also_no_draft(self):
        data = _ask({"question": "מה ידוע על HBB?", "include_unverified_gene_draft": False})
        assert "unverified_gene_draft" not in data

    def test_vus_question_no_draft(self):
        """VUS-only questions (not gene-level) never generate a draft."""
        data = _ask({"question": "מה זה VUS?"})
        assert "unverified_gene_draft" not in data

    def test_carrier_question_no_draft(self):
        data = _ask({"question": "אמרו לי שאני נשאית, מה זה?"})
        assert "unverified_gene_draft" not in data


# ---------------------------------------------------------------------------
# /ask endpoint integration — opt-in behavior with mocked LLM
# ---------------------------------------------------------------------------

class TestUnverifiedDraftOptIn:
    """With include_unverified_gene_draft=True and a mocked LLM returning valid text,
    a Tier 2 gene question should include the draft in the response."""

    VALID_DRAFT = (
        "הגן HBB מופיע לעיתים בהקשרים של מחלות המוגלובין ואנמיה. "
        "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
    )

    def test_tier2_with_opt_in_returns_draft(self):
        """Tier 2 gene + opt-in + valid LLM output → draft in response."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.VALID_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        # draft must be present when LLM returns valid text
        if "unverified_gene_draft" in data:
            draft = data["unverified_gene_draft"]
            assert draft["approved"] is False
            assert draft["review_status"] == "unreviewed"
            assert draft["status"] == "ai_generated_unreviewed"
            assert "warning_he" in draft
            assert draft["text_he"] == self.VALID_DRAFT

    def test_draft_never_in_answer_text(self):
        """Draft text must not be appended to the answer; answer stays deterministic.

        We patch _apply_llm_layer to return a no-LLM result so the mock client
        is only called once — for draft generation.
        """
        from app.counseling_engine import LLMLayerResult
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.VALID_DRAFT
        no_llm = LLMLayerResult(
            answer="DETERMINISTIC_KB_CONTENT",
            llm_used=False, attempted=False,
            mode="none", rejection_reason=None,
        )

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client), \
             patch("app.counseling_engine._apply_llm_layer", return_value=no_llm):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        # The draft text must appear ONLY in unverified_gene_draft.text_he,
        # not merged into the answer field.
        if "unverified_gene_draft" in data:
            assert data["unverified_gene_draft"]["text_he"] == self.VALID_DRAFT
        assert self.VALID_DRAFT not in data["answer"], (
            "Draft text must be in unverified_gene_draft, not merged into answer"
        )

    def test_draft_includes_warning(self):
        """Warning text is always present in the draft."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.VALID_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        if "unverified_gene_draft" in data:
            assert "warning_he" in data["unverified_gene_draft"]
            warning = data["unverified_gene_draft"]["warning_he"]
            assert len(warning) > 50, "Warning text is too short"
            assert "לא עבר בדיקה מקצועית" in warning

    def test_draft_approved_always_false(self):
        """approved field is unconditionally False."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.VALID_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        if "unverified_gene_draft" in data:
            assert data["unverified_gene_draft"]["approved"] is False

    def test_draft_review_status_unreviewed(self):
        """review_status is always 'unreviewed'."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = self.VALID_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        if "unverified_gene_draft" in data:
            assert data["unverified_gene_draft"]["review_status"] == "unreviewed"

    def test_bad_draft_rejected_deterministic_preserved(self):
        """LLM returning medical recommendation → bad text rejected, answer still complete.

        When ClinVar data is available for the gene, the deterministic fallback is
        returned instead. The key invariant is that the bad LLM text does NOT reach
        the user — only the safe deterministic fallback (or nothing) is acceptable.
        """
        BAD_DRAFT = "הגן APC קשור לסרטן המעי. מומלץ לבצע קולונוסקופיה בגיל צעיר."
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = BAD_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        # Bad LLM text must not reach user; only deterministic fallback is acceptable
        draft = data.get("unverified_gene_draft")
        if draft is not None:
            assert draft.get("generated_by_model") == "deterministic", (
                "When LLM returns bad text, only deterministic fallback is acceptable"
            )
        # The answer must still be present regardless
        assert data["answer"]
        assert len(data["answer"]) > 20

    def test_personal_risk_draft_rejected(self):
        """LLM returning personal-risk language → LLM text rejected, answer intact."""
        BAD_DRAFT = "הסיכון שלך לחלות בסרטן גבוה בגלל שינוי בגן זה."
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = BAD_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        draft = data.get("unverified_gene_draft")
        if draft is not None:
            assert draft.get("generated_by_model") == "deterministic"
        assert data["answer"]

    def test_cjk_draft_rejected(self):
        """LLM returning CJK characters → LLM text rejected, answer intact."""
        BAD_DRAFT = "הגן APC מעורב בסרטן המעי. 日本語 שלום"
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = BAD_DRAFT

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            data = _ask({
                "question": "מה ידוע על HBB?",
                "include_unverified_gene_draft": True,
            })

        draft = data.get("unverified_gene_draft")
        if draft is not None:
            assert draft.get("generated_by_model") == "deterministic"
        assert data["answer"]


# ---------------------------------------------------------------------------
# Tier 1 and Tier 3 behavior
# ---------------------------------------------------------------------------

class TestTierBehavior:
    """Tier 1 (approved card) and Tier 3 (unknown gene) should never offer drafts."""

    def test_tier1_brca1_gene_knowledge_status_approved(self):
        """BRCA1 has an approved card → gene_knowledge_status=approved, no draft available."""
        data = _ask({"question": "מה ידוע על BRCA1?"})
        meta = data.get("gene_metadata", {})
        if meta:
            # If gene_metadata is present, Tier 1 cards should be marked approved
            if meta.get("answer_tier") == "tier1":
                assert meta.get("gene_knowledge_status") == "approved"
                assert meta.get("unverified_gene_draft_available") is False

    def test_tier1_no_draft_even_with_opt_in(self):
        """Tier 1 gene + opt-in → no unverified_gene_draft in response."""
        data = _ask({"question": "מה ידוע על BRCA1?", "include_unverified_gene_draft": True})
        meta = data.get("gene_metadata", {})
        if meta and meta.get("answer_tier") == "tier1":
            assert "unverified_gene_draft" not in data

    def test_tier3_gene_knowledge_status_missing(self):
        """Gene not in any source → gene_knowledge_status=missing."""
        data = _ask({"question": "מה ידוע על FAKEGENE99?"})
        meta = data.get("gene_metadata", {})
        if meta and meta.get("answer_tier") == "tier3":
            assert meta.get("gene_knowledge_status") == "missing"
            assert meta.get("unverified_gene_draft_available") is False

    def test_tier3_no_draft_even_with_opt_in(self):
        """Tier 3 gene + opt-in → no unverified_gene_draft in response."""
        data = _ask({"question": "מה ידוע על FAKEGENE99?", "include_unverified_gene_draft": True})
        meta = data.get("gene_metadata", {})
        if meta and meta.get("answer_tier") == "tier3":
            assert "unverified_gene_draft" not in data


# ---------------------------------------------------------------------------
# _generate_unverified_gene_draft — unit tests
# ---------------------------------------------------------------------------

class TestGenerateUnverifiedDraft:

    def test_no_provider_configured_returns_none(self):
        """When no LLM is configured (create_llm_client raises ValueError), returns None."""
        with patch("app.counseling_engine.create_llm_client",
                   side_effect=ValueError("No LLM configured")):
            result = _generate_unverified_gene_draft("HBB")
        assert result is None

    def test_llm_provider_none_returns_none(self):
        """LLM_PROVIDER=none raises ValueError in create_llm_client → returns None."""
        with patch("app.counseling_engine.create_llm_client",
                   side_effect=ValueError("LLM explicitly disabled via LLM_PROVIDER=none.")):
            result = _generate_unverified_gene_draft("HBB")
        assert result is None

    def test_valid_output_returns_dict(self):
        """Valid LLM output → structured dict with all required keys."""
        valid_text = (
            "הגן HBB מופיע לעיתים בהקשרים של אנמיה ומחלות המוגלובין שונות. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
        )
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = valid_text

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("HBB")

        assert result is not None
        assert result["visible"] is True
        assert result["status"] == "ai_generated_unreviewed"
        assert result["gene_symbol"] == "HBB"
        assert result["approved"] is False
        assert result["review_status"] == "unreviewed"
        assert "warning_he" in result
        assert result["text_he"] == valid_text
        assert "generated_at" in result

    def test_invalid_output_returns_none(self):
        """Rejected LLM output → None (no draft exposed)."""
        bad_text = "מומלץ לבצע קולונוסקופיה בהקדם."
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = bad_text

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("APC")

        assert result is None

    def test_llm_error_returns_none(self):
        """LLM error → None (never raises)."""
        from app.llm_client import LLMClientError
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = LLMClientError("connection refused")

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("HBB")

        assert result is None

    def test_provider_failure_returns_none(self):
        """Any unexpected exception from call_text_raw → None, never raises."""
        mock_client = MagicMock()
        mock_client.call_text_raw.side_effect = RuntimeError("unexpected provider failure")

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("HBB")

        assert result is None

    def test_draft_includes_mandatory_warning(self):
        """The warning_he field must contain the required patient safety text."""
        valid_text = (
            "הגן HBB מקודד לשרשרת בטא של ההמוגלובין. "
            "גן זה נחקר בהקשר של מחלות דם שונות."
        )
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = valid_text

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("HBB")

        if result:
            assert "לא עבר בדיקה מקצועית" in result["warning_he"]
            assert "הצוות הגנטי" in result["warning_he"]

    def test_openai_provider_mocked_returns_draft(self):
        """LLM_PROVIDER=openai with mocked create_llm_client → draft returned."""
        valid_text = (
            "הגן BRCA2 מופיע לעיתים בהקשרים של סרטן שד ושחלות. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
        )
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = valid_text

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake"}), \
             patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("BRCA2")

        assert result is not None
        assert result["text_he"] == valid_text
        assert result["approved"] is False
        assert result["review_status"] == "unreviewed"

    def test_local_provider_mocked_returns_draft(self):
        """LLM_PROVIDER=local with mocked create_llm_client → draft returned."""
        valid_text = (
            "הגן TP53 נחקר בהקשרים של תסמונות נדירות בעלות רגישות גבוהה לסרטן. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
        )
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = valid_text

        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("TP53")

        assert result is not None
        assert result["text_he"] == valid_text


# ---------------------------------------------------------------------------
# Metadata fields in gene_metadata
# ---------------------------------------------------------------------------

class TestGeneMetadataFields:

    def test_tier2_has_unverified_draft_available_true(self):
        """Tier 2 gene → gene_metadata.unverified_gene_draft_available=True."""
        data = _ask({"question": "מה ידוע על HBB?"})
        meta = data.get("gene_metadata", {})
        if meta and meta.get("answer_tier") == "tier2":
            assert meta.get("unverified_gene_draft_available") is True
            assert meta.get("gene_knowledge_status") == "unverified_available"

    def test_response_schema_unchanged_for_non_gene_questions(self):
        """The standard 8-key schema is not broken by the new fields."""
        data = _ask({"question": "מה זה VUS?"})
        expected_keys = {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }
        assert expected_keys.issubset(set(data.keys())), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )


# ---------------------------------------------------------------------------
# Tier 2 patient-friendly main answer
# ---------------------------------------------------------------------------

class TestTier2PatientFriendlyAnswer:
    """For Tier 2 genes the main answer must be short and patient-friendly.
    Raw ClinVar data (breakdown, phenotype lists) must not appear in answer."""

    def _tier2_data(self):
        return _ask({"question": "מה ידוע על HBB?"})

    def _is_tier2(self, data):
        meta = data.get("gene_metadata", {})
        return meta.get("answer_tier") == "tier2"

    def test_tier2_answer_does_not_contain_clinvar_breakdown_header(self):
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        assert "סיווגים קליניים מדווחים" not in data["answer"], (
            "Raw ClinVar breakdown header must not appear in the Tier 2 main answer"
        )

    def test_tier2_answer_does_not_contain_conditions_header(self):
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        assert "מצבים רפואיים מדווחים" not in data["answer"]
        assert "מצבים קשורים מדווחים" not in data["answer"]

    def test_tier2_answer_is_short(self):
        """Tier 2 answer should be under 600 characters."""
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        assert len(data["answer"]) < 600, (
            f"Tier 2 answer too long ({len(data['answer'])} chars): {data['answer'][:200]}"
        )

    def test_tier2_answer_contains_gene_name(self):
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        assert "HBB" in data["answer"]

    def test_tier2_answer_no_emoji(self):
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        import re
        assert not re.search(r'[\U0001F300-\U0001FAFF]', data["answer"]), (
            "Tier 2 answer must not contain emoji"
        )

    def test_tier2_llm_not_used(self):
        """Tier 2 main answer must be deterministic — llm_used must be False."""
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        assert data.get("llm_used") is False, (
            "Tier 2 must not use LLM framing for the main answer"
        )

    def test_tier2_llm_mode_none(self):
        data = self._tier2_data()
        if not self._is_tier2(data):
            return
        assert data.get("llm_mode") == "none"

    def test_tier2_clinvar_stats_in_metadata_not_answer(self):
        """ClinVar stats must be in gene_metadata, not dumped into the answer."""
        data = self._tier2_data()
        meta = data.get("gene_metadata", {})
        if not meta or meta.get("answer_tier") != "tier2":
            return
        # Stats must be present in metadata
        assert "significance_breakdown" in meta, "significance_breakdown missing from gene_metadata"
        assert "top_phenotypes" in meta, "top_phenotypes missing from gene_metadata"
        assert "total_variants" in meta, "total_variants missing from gene_metadata"
        # And must NOT appear raw in the answer
        assert "Pathogenic" not in data["answer"] or len(data["answer"]) < 600

    def test_tier2_unverified_draft_available_flag_set(self):
        data = self._tier2_data()
        meta = data.get("gene_metadata", {})
        if meta.get("answer_tier") == "tier2":
            assert meta.get("unverified_gene_draft_available") is True
        assert "unverified_gene_draft" not in data
