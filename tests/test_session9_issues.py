# -*- coding: utf-8 -*-
"""
tests/test_session9_issues.py

Tests covering the four changes from Session 9:

1. Curated KB/FAQ answers are fully deterministic — no LLM framing prepended
2. Patient-facing draft text_he must not contain "ClinVar" brand name
3. Strengthened quality validation (_DRAFT_QUALITY_RE and _FRAMING_QUALITY_RE)
4. Human review/approval workflow CLI (scripts/review_drafts.py)
"""

import json
import os
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

QUEUE_PATH = ROOT / "app" / "data" / "draft_review_queue.json"
KB_PATH = ROOT / "app" / "data" / "gene_knowledge_base.json"


def _ask(payload: dict) -> dict:
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── Issue 1: Curated KB answers are always deterministic ─────────────────────

class TestKbAnswersDeterministic:
    """KB/FAQ answers must never have LLM intro prepended."""

    @pytest.fixture(autouse=True)
    def _no_llm(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    def test_vus_general_deterministic(self):
        data = _ask({"question": "מה זה VUS?"})
        assert data["llm_used"] is False
        assert data["fallback_used"] is True
        assert "VUS" in data["answer"]

    def test_carrier_deterministic(self):
        data = _ask({"question": "אמרו לי שאני נשאית, מה זה?"})
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_pathogenic_deterministic(self):
        data = _ask({"question": "מה זה וריאנט פתוגני?"})
        assert data["llm_used"] is False

    def test_brca1_gene_card_deterministic(self):
        data = _ask({"question": "מה ידוע על BRCA1?"})
        assert data["llm_used"] is False
        assert data["fallback_used"] is True
        assert "BRCA1" in data["answer"]

    def test_vus_plus_gene_deterministic(self):
        data = _ask({"question": "יש לי VUS ב-BRCA1, מה זה?"})
        assert data["llm_used"] is False
        assert data["fallback_used"] is True

    def test_llm_configured_still_not_used_for_kb(self, monkeypatch):
        """Even when LLM is configured, KB answers are always deterministic."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:9999")
        mock_intro = "שאלה נהדרת — ברוך הבא לשאלה שלך."
        with patch("app.counseling_engine.LocalLLMClient") as MockLLM:
            MockLLM.return_value._call_api.return_value = mock_intro
            data = _ask({"question": "מה זה VUS?"})
        assert data["llm_used"] is False
        assert mock_intro not in data["answer"]

    def test_llm_mode_is_none_for_kb_path(self):
        data = _ask({"question": "מה זה VUS?"})
        assert data.get("llm_mode") == "none"

    def test_llm_mode_is_none_for_gene_card(self):
        data = _ask({"question": "מה ידוע על NF1?"})
        assert data.get("llm_mode") == "none"

    def test_faq_answer_not_empty(self):
        for question in ["מה זה VUS?", "מה זה נשאות?", "מה זה ירושה אוטוזומלית דומיננטית?"]:
            data = _ask({"question": question})
            assert len(data["answer"]) > 50, f"Answer too short for: {question!r}"


# ── Issue 2: Patient-facing draft text_he must not contain "ClinVar" ─────────

class TestDraftTextNoClinVar:
    """Draft text_he returned to patients must never mention 'ClinVar' by name."""

    def test_deterministic_draft_no_clinvar_in_text_he(self):
        from app.counseling_engine import _build_deterministic_clinvar_draft
        result = _build_deterministic_clinvar_draft("HBB", ["אנמיה", "מחלות המוגלובין"])
        assert result is not None
        assert "ClinVar" not in result["text_he"], (
            "Deterministic draft text_he must not contain 'ClinVar'"
        )

    def test_deterministic_draft_based_on_unchanged(self):
        from app.counseling_engine import _build_deterministic_clinvar_draft
        result = _build_deterministic_clinvar_draft("HBB", ["אנמיה"])
        assert result is not None
        assert result["based_on"] == "clinvar_metadata"

    def test_source_note_he_no_clinvar(self):
        from app.counseling_engine import _UNVERIFIED_DRAFT_SOURCE_NOTE_HE
        assert "ClinVar" not in _UNVERIFIED_DRAFT_SOURCE_NOTE_HE

    def test_source_note_he_is_patient_friendly(self):
        from app.counseling_engine import _UNVERIFIED_DRAFT_SOURCE_NOTE_HE
        assert len(_UNVERIFIED_DRAFT_SOURCE_NOTE_HE) > 20
        import re
        assert re.search(r"[א-ת]", _UNVERIFIED_DRAFT_SOURCE_NOTE_HE)

    def test_generated_draft_via_llm_no_clinvar_accepted(self):
        """LLM draft text without ClinVar is accepted."""
        from app.counseling_engine import _generate_unverified_gene_draft
        valid_text = "הגן POLE מופיע לעיתים בהקשרים של סרטן מעי גס. ממצא VUS בגן זה נותר בגדר אי-ודאות."
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = valid_text
        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("POLE")
        assert result is not None
        assert "ClinVar" not in result["text_he"]

    def test_generated_draft_via_llm_with_statistics_rejected(self):
        """LLM draft text containing database statistics is rejected (hard block)."""
        from app.counseling_engine import _generate_unverified_gene_draft
        bad_text = "הגן POLE מדווח במאגר ClinVar עם 532 וריאנטים פתוגניים."
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = bad_text
        with patch("app.counseling_engine.create_llm_client", return_value=mock_client):
            result = _generate_unverified_gene_draft("POLE")
        # Statistics in draft → hard rejected in both passes → None
        assert result is None


# ── Issue 3: Strengthened quality validation ──────────────────────────────────

class TestStrengthened_DRAFT_QUALITY_RE:
    """_DRAFT_QUALITY_RE must reject ClinVar in patient-facing draft text."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.counseling_engine import _DRAFT_QUALITY_RE
        self.re = _DRAFT_QUALITY_RE

    def test_clinvar_rejected(self):
        from app.counseling_engine import _CLINVAR_IN_DRAFT_RE
        assert _CLINVAR_IN_DRAFT_RE.search("הגן POLE מדווח במאגר ClinVar בהקשרים שונים.")

    def test_clinvar_case_insensitive(self):
        from app.counseling_engine import _CLINVAR_IN_DRAFT_RE
        assert _CLINVAR_IN_DRAFT_RE.search("clinvar data was analyzed")

    def test_valid_draft_without_clinvar_not_rejected(self):
        text = "הגן POLE מופיע לעיתים בהקשרים של סרטן מעי גס. ממצא VUS בגדר אי-ודאות."
        assert not self.re.search(text)


class TestStrengthened_FRAMING_QUALITY_RE:
    """_FRAMING_QUALITY_RE must reject the newly added vague-filler phrases."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.counseling_engine import _FRAMING_QUALITY_RE
        self.re = _FRAMING_QUALITY_RE

    def test_rejects_tafsar(self):
        assert self.re.search("תפסיר של הגן ניתן לראות בדוח")

    def test_rejects_sheyecholim_lispek(self):
        assert self.re.search("מומחים שיכולים לספק מידע נוסף")

    def test_rejects_hine_meida_klali(self):
        assert self.re.search("הנה מידע כללי על הנושא")

    def test_rejects_ata_yachol_lishal_al(self):
        assert self.re.search("אתה יכול לשאול על התוצאה")

    def test_rejects_at_yechola_lishal_al(self):
        assert self.re.search("את יכולה לשאול על זה")

    def test_rejects_ze_yachol_lehiyot(self):
        assert self.re.search("זה יכול להיות קשור למחלה")

    def test_valid_intro_not_rejected(self):
        text = "מידע זה יכול לסייע לך להבין את הנושא בצורה טובה יותר."
        assert not self.re.search(text)

    def test_validate_intro_with_reason_rejects_hine_meida_klali(self):
        from app.counseling_engine import _validate_intro_with_reason
        reason = _validate_intro_with_reason("שאלה טובה — הנה מידע כללי שיכול לעזור.")
        assert reason is not None
        assert "quality-rejected" in reason

    def test_validate_intro_with_reason_rejects_tafsar(self):
        from app.counseling_engine import _validate_intro_with_reason
        reason = _validate_intro_with_reason("אני יכול לתפסיר את הממצא שלך.")
        assert reason is not None


# ── Issue 4: Human review/approval workflow CLI ───────────────────────────────

class TestReviewDraftsCLI:
    """scripts/review_drafts.py — review/approval workflow for gene drafts."""

    CLI = ROOT / "scripts" / "review_drafts.py"

    def _run(self, args: list[str]) -> tuple[int, str]:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(self.CLI)] + args,
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(ROOT),
        )
        return result.returncode, result.stdout + result.stderr

    def test_cli_exists(self):
        assert self.CLI.exists(), "scripts/review_drafts.py must exist"

    def test_list_empty_queue(self, tmp_path, monkeypatch):
        """--list --status approved shows empty message (no drafts have been approved yet)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(self.CLI), "--list", "--status", "approved"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "No" in result.stdout and "draft" in result.stdout.lower()

    def test_no_args_shows_usage(self):
        rc, output = self._run([])
        assert rc != 0  # argparse exits non-zero on missing required args

    def test_approve_requires_confirm(self, tmp_path):
        """--approve without --confirm must fail with a clear error."""
        rc, output = self._run(["--approve", "nonexistent-id", "--reviewer", "Dr. Test"])
        assert rc != 0
        assert "confirm" in output.lower()

    def test_reject_requires_reason(self):
        """--reject without --reason must fail."""
        rc, output = self._run(["--reject", "nonexistent-id"])
        assert rc != 0
        assert "reason" in output.lower()

    def test_approve_unknown_id_fails(self):
        rc, output = self._run([
            "--approve", "00000000-0000-0000-0000-000000000000",
            "--reviewer", "Dr. Test",
            "--confirm",
        ])
        assert rc != 0
        assert "not found" in output.lower()

    def test_reject_unknown_id_fails(self):
        rc, output = self._run([
            "--reject", "00000000-0000-0000-0000-000000000000",
            "--reason", "test",
        ])
        assert rc != 0
        assert "not found" in output.lower()

    def test_approve_writes_to_gene_knowledge_base(self, tmp_path):
        """Full approve flow: add to queue, approve, verify in KB."""
        import importlib
        import scripts.review_drafts as rd
        importlib.reload(rd)

        # Temporarily redirect paths
        tmp_queue = tmp_path / "queue.json"
        tmp_kb = tmp_path / "kb.json"
        tmp_queue.write_text("[]", encoding="utf-8")
        tmp_kb.write_text("[]", encoding="utf-8")

        draft_id = "test-draft-1234"
        queue = [{
            "draft_id": draft_id,
            "gene_symbol": "TTN",
            "draft_type": "general",
            "text_he": "הגן TTN מופיע לעיתים בהקשרים קרדיולוגיים. ממצא VUS בגדר אי-ודאות.",
            "source_context": "",
            "based_on": "clinvar_metadata",
            "source_note_he": "מידע כללי.",
            "generated_by_model": "test-model",
            "generated_at": "2026-07-03T00:00:00Z",
            "approved": False,
            "review_status": "unreviewed",
            "reviewed_by": None,
            "reviewed_at": None,
            "reviewer_notes": None,
            "edited_text_he": None,
            "source_1_name": None,
            "source_1_url_or_id": None,
            "source_2_name": None,
            "source_2_url_or_id": None,
        }]
        tmp_queue.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")

        # Monkeypatch the paths
        monkeypatch_paths = {
            "QUEUE_PATH": tmp_queue,
            "KB_PATH": tmp_kb,
        }
        orig_queue = rd.QUEUE_PATH
        orig_kb = rd.KB_PATH
        rd.QUEUE_PATH = tmp_queue
        rd.KB_PATH = tmp_kb
        try:
            class FakeArgs:
                approve = draft_id
                reviewer = "Dr. Test"
                confirm = True
                notes = None
            rc = rd.cmd_approve(FakeArgs())
            assert rc == 0
            kb_data = json.loads(tmp_kb.read_text(encoding="utf-8"))
            assert any(r["gene_symbol"] == "TTN" for r in kb_data), "TTN not in KB after approve"
            ttn = next(r for r in kb_data if r["gene_symbol"] == "TTN")
            assert ttn["approved"] is True
            assert ttn["reviewed_by"] == "Dr. Test"
        finally:
            rd.QUEUE_PATH = orig_queue
            rd.KB_PATH = orig_kb

    def test_reject_marks_draft_in_queue(self, tmp_path):
        """--reject marks the draft as rejected in the queue without touching KB."""
        import importlib
        import scripts.review_drafts as rd
        importlib.reload(rd)

        tmp_queue = tmp_path / "queue.json"
        tmp_kb = tmp_path / "kb.json"
        tmp_kb.write_text("[]", encoding="utf-8")

        draft_id = "test-reject-5678"
        queue = [{
            "draft_id": draft_id,
            "gene_symbol": "TTN2",
            "text_he": "הגן TTN2 מופיע לעיתים בהקשרים קרדיולוגיים.",
            "review_status": "unreviewed",
            "approved": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "reviewer_notes": None,
        }]
        tmp_queue.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")

        orig_queue = rd.QUEUE_PATH
        orig_kb = rd.KB_PATH
        rd.QUEUE_PATH = tmp_queue
        rd.KB_PATH = tmp_kb
        try:
            class FakeArgs:
                reject = draft_id
                reason = "Text contains inaccurate information"
                reviewer = None
            rc = rd.cmd_reject(FakeArgs())
            assert rc == 0
            queue_data = json.loads(tmp_queue.read_text(encoding="utf-8"))
            ttn2 = next(r for r in queue_data if r["draft_id"] == draft_id)
            assert ttn2["review_status"] == "rejected"
            assert ttn2["reviewer_notes"] == "Text contains inaccurate information"
            # KB must not be touched
            kb_data = json.loads(tmp_kb.read_text(encoding="utf-8"))
            assert kb_data == []
        finally:
            rd.QUEUE_PATH = orig_queue
            rd.KB_PATH = orig_kb

    def test_draft_review_queue_json_exists(self):
        """app/data/draft_review_queue.json must exist and be valid JSON."""
        assert QUEUE_PATH.exists(), "app/data/draft_review_queue.json must exist"
        data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, list)
