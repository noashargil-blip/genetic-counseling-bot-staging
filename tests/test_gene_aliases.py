"""
Tests for gene alias normalisation (POLE / Pol-E / etc.) and
corrupted LLM framing rejection.

All tests run with the FastAPI TestClient — no live server needed.
"""
import re
import sys
import pathlib
import importlib

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on the path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_client():
    import app.retriever as retriever
    retriever._DB_AVAILABLE = False
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _ask(client, question, last_topic=None, conversation_context=None,
         include_unverified_gene_draft=False):
    """conversation_context must be None or List[{"role": str, "content": str}]."""
    payload = {"question": question}
    if last_topic:
        payload["last_topic"] = last_topic
    if conversation_context is not None:
        payload["conversation_context"] = conversation_context
    if include_unverified_gene_draft:
        payload["include_unverified_gene_draft"] = True
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


# ── Gene alias detection (unit) ───────────────────────────────────────────────

class TestPoleAliasDetection:
    """_detect_known_gene must return 'POLE' for every alias form."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from app.counseling_engine import _detect_known_gene
        self._detect = _detect_known_gene

    def _check(self, text):
        result = self._detect(text)
        assert result == "POLE", f"_detect_known_gene({text!r}) = {result!r}, expected 'POLE'"

    def test_pole_uppercase(self):
        self._check("יש לי VUS ב POLE")

    def test_pole_lowercase(self):
        self._check("יש לי VUS ב pole")

    def test_pol_hyphen_e_mixed(self):
        self._check("אמרו לי שיש לי VUS בגן Pol-E מה זה אומר")

    def test_pol_hyphen_e_upper(self):
        self._check("VUS ב POL-E")

    def test_pol_hyphen_e_lower(self):
        self._check("pol-e mutation")

    def test_pol_space_e_upper(self):
        self._check("VUS בגן POL E")

    def test_pol_space_e_mixed(self):
        self._check("VUS בגן Pol E")

    def test_pole_hyphen_e(self):
        self._check("gene Pole-E, what is it?")

    def test_pole_space_e(self):
        self._check("POLE E result")

    def test_pole_lower_hyphen(self):
        self._check("pole-e gene")

    def test_hebrew_phonetic(self):
        self._check("יש לי VUS בגן פול אי")


# ── VUS + POLE routing (API) ──────────────────────────────────────────────────

class TestVusPoleRouting:
    """VUS + POLE questions must return gene_metadata with gene_symbol=POLE."""

    @pytest.fixture(scope="class")
    def client(self):
        return _get_client()

    def test_vus_pole_uppercase_returns_gene_metadata(self, client):
        data = _ask(client, "יש לי VUS ב POLE מה זה אומר?")
        meta = data.get("gene_metadata")
        assert meta is not None, "gene_metadata missing for VUS+POLE question"
        assert meta["gene_symbol"] == "POLE"

    def test_vus_pole_hyphen_alias_returns_gene_metadata(self, client):
        data = _ask(client, "אמרו לי שיש לי VUS בגן Pol-E מה זה אומר")
        meta = data.get("gene_metadata")
        assert meta is not None, "gene_metadata missing for VUS+Pol-E question"
        assert meta["gene_symbol"] == "POLE"

    def test_vus_pole_hyphen_alias_matched_topic(self, client):
        data = _ask(client, "יש לי VUS בגן Pol-E")
        # Must be vus_known_gene (step 4) or gene_clinvar_summary (step 4.5)
        assert data["matched_topic"] in ("vus_known_gene", "gene_clinvar_summary"), (
            f"unexpected matched_topic: {data['matched_topic']!r}"
        )

    def test_vus_pole_tier_is_set(self, client):
        data = _ask(client, "יש לי VUS בגן Pol-E")
        meta = data.get("gene_metadata")
        assert meta is not None
        assert meta.get("answer_tier") in ("tier1", "tier2", "tier3"), (
            f"answer_tier missing or unknown: {meta.get('answer_tier')!r}"
        )

    def test_vus_pole_tier2_exposes_draft_flag(self, client):
        """If POLE is Tier-2 (in ClinVar index, no approved card), draft must be available."""
        data = _ask(client, "יש לי VUS בגן Pol-E")
        meta = data.get("gene_metadata")
        if meta and meta.get("answer_tier") == "tier2":
            assert meta.get("unverified_gene_draft_available") is True, (
                "Tier-2 VUS+gene answer must expose unverified_gene_draft_available=True"
            )

    def test_vus_pole_not_carrier_topic(self, client):
        """Pol-E VUS must NOT route to carrier_vs_affected."""
        data = _ask(client, "יש לי VUS בגן Pol-E, מה זה אומר?")
        assert data["matched_topic"] != "carrier_vs_affected", (
            "Pol-E VUS was misrouted to carrier_vs_affected"
        )

    def test_vus_pole_hebrew_alias_gene_metadata(self, client):
        data = _ask(client, "יש לי VUS בגן פול אי")
        meta = data.get("gene_metadata")
        assert meta is not None, "gene_metadata missing for Hebrew alias פול אי"
        assert meta["gene_symbol"] == "POLE"


# ── Follow-up after POLE question ─────────────────────────────────────────────

class TestPoleFollowUp:
    """Follow-up questions after a POLE VUS answer must resolve back to POLE."""

    @pytest.fixture(scope="class")
    def client(self):
        return _get_client()

    def test_pole_followup_stays_on_pole(self, client):
        # Turn 1: VUS + POLE
        t1 = _ask(client, "יש לי VUS בגן Pol-E")
        # Turn 2: follow-up — conversation_context as list of messages
        context = [
            {"role": "user", "content": "יש לי VUS בגן Pol-E"},
            {"role": "assistant", "content": t1["answer"]},
        ]
        t2 = _ask(
            client,
            "מה כדאי לעשות עם זה?",
            last_topic=t1.get("matched_topic"),
            conversation_context=context,
        )
        # The follow-up must not route to carrier_vs_affected
        assert t2["matched_topic"] != "carrier_vs_affected", (
            "POLE follow-up misrouted to carrier_vs_affected"
        )

    def test_pole_followup_general_what_is_it(self, client):
        t1 = _ask(client, "יש לי VUS בגן Pol-E")
        context = [
            {"role": "user", "content": "יש לי VUS בגן Pol-E"},
            {"role": "assistant", "content": t1["answer"]},
        ]
        t2 = _ask(
            client,
            "מה הגן הזה?",
            last_topic=t1.get("matched_topic"),
            conversation_context=context,
        )
        # Should produce a meaningful answer (not out_of_scope)
        assert t2["safety_level"] != "out_of_scope", (
            "Follow-up 'מה הגן הזה?' after POLE was treated as out_of_scope"
        )


# ── Corrupted framing rejection ───────────────────────────────────────────────

class TestCorruptedFramingRejected:
    """_FRAMING_QUALITY_RE / _validate_controlled_framing must catch corrupted text."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from app.counseling_engine import _validate_controlled_framing, _validate_intro_with_reason
        self._framing = _validate_controlled_framing
        self._intro = _validate_intro_with_reason

    def test_curly_brace_artifact_framing(self):
        text = "ת}ה ת}ק ת}מ ת}ך ת}ל ת}ר ת}ש ת}ת ת}ו ת}י ת}ן"
        reason = self._framing(text)
        assert reason is not None, "Curly-brace artifact must be rejected by _validate_controlled_framing"

    def test_new_keyword_artifact_framing(self):
        reason = self._framing("new VUS result ת}ה ת}ק")
        assert reason is not None, "'new' keyword + artifacts must be rejected"

    def test_opening_bracket_artifact_framing(self):
        reason = self._framing("הגן {POLE} הוא גן חשוב")
        assert reason is not None, "Curly braces must be rejected"

    def test_hebrew_close_brace_pattern_framing(self):
        reason = self._framing("ת)new שגיאה")
        assert reason is not None, "ת) pattern must be rejected"

    def test_square_bracket_artifact_framing(self):
        reason = self._framing("הגן [POLE] מקודד")
        assert reason is not None, "Square brackets must be rejected"

    def test_curly_brace_artifact_intro(self):
        text = "ת}ה ת}ק ת}מ ת}ך"
        reason = self._intro(text)
        assert reason is not None, "Curly-brace artifact must be rejected by _validate_intro_with_reason"

    def test_new_keyword_intro(self):
        reason = self._intro("new answer text here")
        assert reason is not None, "'new' keyword must be rejected by _validate_intro_with_reason"

    def test_clean_hebrew_sentence_still_valid_framing(self):
        # A normal Hebrew framing sentence must still pass
        text = "שמחתי לדעת שאת/ה מחפש/ת מידע כללי על ממצא גנטי."
        reason = self._framing(text)
        assert reason is None, f"Valid Hebrew sentence wrongly rejected: {reason!r}"

    def test_clean_hebrew_sentence_still_valid_intro(self):
        text = "מידע כללי על ממצא גנטי מופיע להלן."
        reason = self._intro(text)
        assert reason is None, f"Valid Hebrew intro wrongly rejected: {reason!r}"

    def test_corrupted_framing_not_prepended_to_answer(self):
        """When framing is corrupted, the answer must contain only deterministic text."""
        from unittest.mock import patch, MagicMock
        from app.counseling_engine import LLMLayerResult

        corrupted = "ת)new ת}ה ת}ק ת}מ ת}ך ת}ל"
        mock_result = MagicMock()
        mock_result.text = corrupted

        with patch("app.counseling_engine.LocalLLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance._call_api.return_value = mock_result
            from app.counseling_engine import _apply_llm_layer
            result = _apply_llm_layer(
                question="מה זה VUS?",
                deterministic_answer="VUS הוא ווריאנט בעל משמעות לא ידועה.",
                gene=None,
                topic="vus",
            )
        # Corrupted framing must not appear in the answer
        assert corrupted not in result.answer, (
            "Corrupted LLM framing was prepended to the answer despite validation"
        )
        # Deterministic part must still be present
        assert "VUS" in result.answer


# ── VUS + Tier-2 gene: no raw ClinVar dump in main answer ────────────────────

class TestVusTier2NoClinvarDump:
    """VUS+gene answer for Tier-2 genes must not expose raw ClinVar stats."""

    BAD_PHRASES = [
        "נתוני ClinVar",
        "פתוגניים / likely pathogenic",
        "שפירים / likely benign",
        "מצבים רפואיים קשורים",
    ]

    @pytest.fixture(scope="class")
    def client(self):
        return _get_client()

    def _pole_answer(self, client):
        return _ask(client, "אמרו לי שיש לי VUS בגן Pol-E מה זה אומר")

    def test_no_clinvar_dump_in_answer(self, client):
        data = self._pole_answer(client)
        for phrase in self.BAD_PHRASES:
            assert phrase not in data["answer"], (
                f"Raw ClinVar phrase {phrase!r} found in VUS+POLE main answer — "
                "must be in gene_metadata only"
            )

    def test_answer_is_short_and_patient_friendly(self, client):
        data = self._pole_answer(client)
        # The main answer should be a few sentences, not a full ClinVar dump
        assert len(data["answer"]) < 1200, (
            f"VUS+POLE answer is too long ({len(data['answer'])} chars) — "
            "expected short patient-friendly text"
        )

    def test_gene_metadata_still_has_clinvar_stats(self, client):
        data = self._pole_answer(client)
        meta = data.get("gene_metadata", {})
        assert meta.get("gene_symbol") == "POLE"
        assert meta.get("answer_tier") in ("tier2", "tier1")
        if meta.get("answer_tier") == "tier2":
            assert "total_variants" in meta, "total_variants must still be in gene_metadata"
            assert meta.get("unverified_gene_draft_available") is True

    def test_answer_mentions_vus_explanation(self, client):
        data = self._pole_answer(client)
        # VUS must still be explained in the short answer
        assert "VUS" in data["answer"]

    def test_answer_mentions_gene_name(self, client):
        data = self._pole_answer(client)
        assert "POLE" in data["answer"], "Gene symbol POLE must appear in the answer"

    def test_answer_has_no_total_count_line(self, client):
        data = self._pole_answer(client)
        import re as _re
        assert not _re.search(r"רשומות וריאנט", data["answer"]), (
            "Raw variant count phrase found in main answer"
        )


# ── Draft quality: RNA/mRNA/transliteration rejection ────────────────────────

class TestDraftQualityExtended:
    """Expanded _DRAFT_QUALITY_RE and _validate_unverified_draft tests."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from app.counseling_engine import _validate_unverified_draft
        self._validate = _validate_unverified_draft

    def _assert_rejected(self, text, label=""):
        reason = self._validate(text)
        assert reason is not None, f"Draft should be rejected [{label}]: {text!r}"

    def _assert_accepted(self, text, label=""):
        reason = self._validate(text)
        assert reason is None, f"Draft wrongly rejected [{label}]: {text!r} — reason: {reason!r}"

    def test_rejects_mrna_uppercase(self):
        self._assert_rejected(
            "הגן מקודד לחלבון שמעורב בעיבוד ה-mRNA.",
            "mRNA in sentence"
        )

    def test_rejects_rna_uppercase(self):
        self._assert_rejected(
            "גן פול-א הוא חלק מהדנ\"א שמשתתף בתהליך הקניית ה-RNA.",
            "RNA mention"
        )

    def test_rejects_mrna_lowercase(self):
        self._assert_rejected(
            "תהליך עיבוד mrna חשוב לתפקוד הגן.",
            "mrna lowercase"
        )

    def test_rejects_haqnayat(self):
        self._assert_rejected(
            "הגן אחראי על הקניית תהליכים גנטיים.",
            "הקניית — garbled transcription term"
        )

    def test_rejects_asimetria(self):
        self._assert_rejected(
            "POLE מודד את האסימטריה של המולקולה.",
            "אסימטריה"
        )

    def test_rejects_pul_aleph_transliteration(self):
        self._assert_rejected(
            "גן פול-א הוא אנזים חשוב בגוף.",
            "פול-א Hebrew transliteration"
        )

    def test_rejects_gn_pul_transliteration(self):
        self._assert_rejected(
            "גן פול הוא אנזים.",
            "גן פול transliteration"
        )

    def test_rejects_truncated_dna_phrase(self):
        self._assert_rejected(
            "הוא חלק מהדנ\"א ומשתתף בתהליכים.",
            "truncated 'הוא חלק מהדנ' phrase"
        )

    def test_accepts_valid_pole_dna_draft(self):
        self._assert_accepted(
            "הגן POLE מופיע לעיתים בהקשרים של סרטן מעי גס ולסרטן אנדומטריום. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה.",
            "valid POLE draft without ClinVar brand name"
        )

    def test_accepts_valid_hbb_draft(self):
        self._assert_accepted(
            "הגן HBB מופיע לעיתים בהקשרים של אנמיה ומחלות המוגלובין שונות. "
            "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה.",
            "valid HBB draft without ClinVar brand name"
        )

    def test_draft_fallback_when_both_attempts_fail(self):
        """When both LLM attempts fail, unverified_gene_draft must be None (no bad text shown)."""
        from unittest.mock import patch, MagicMock
        import os

        bad_draft = "גן פול-א הוא חלק מהדנ\"א שמשתתף בתהליך הקניית ה-RNA."
        mock_client = MagicMock()
        mock_client._call_api.return_value = bad_draft

        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:11434"}), \
             patch("app.counseling_engine.LocalLLMClient", return_value=mock_client):
            from app.counseling_engine import _generate_unverified_gene_draft
            result = _generate_unverified_gene_draft("POLE", "יש לי VUS בגן POLE")

        assert result is None, (
            "Both draft attempts failed validation but a draft was returned — "
            "bad drafts must never be shown"
        )
