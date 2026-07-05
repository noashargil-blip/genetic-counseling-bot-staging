# -*- coding: utf-8 -*-
"""
tests/test_review_workflow.py

Tests for the physician/counselor review workflow:
  • app/draft_review.py  — enqueue helper
  • scripts/review_drafts.py — CLI commands
  • app/gene_knowledge.py   — approved_context_summary_he accessor

All file I/O uses tmp_path fixtures so production data is never touched.
"""

import json
import pathlib
import uuid

import pytest


# ---------------------------------------------------------------------------
# Fixtures: isolated queue + KB paths
# ---------------------------------------------------------------------------

@pytest.fixture()
def queue_path(tmp_path):
    p = tmp_path / "draft_review_queue.json"
    p.write_text("[]", encoding="utf-8")
    return p


@pytest.fixture()
def kb_path(tmp_path):
    p = tmp_path / "gene_knowledge_base.json"
    p.write_text("[]", encoding="utf-8")
    return p


@pytest.fixture()
def dr(queue_path, monkeypatch):
    """Return the draft_review module with _QUEUE_PATH patched to tmp."""
    import app.draft_review as dr_mod
    monkeypatch.setattr(dr_mod, "_QUEUE_PATH", queue_path)
    return dr_mod


@pytest.fixture()
def review_cli(queue_path, kb_path, monkeypatch):
    """Return the review_drafts module with both paths patched to tmp."""
    import scripts.review_drafts as cli_mod
    monkeypatch.setattr(cli_mod, "QUEUE_PATH", queue_path)
    monkeypatch.setattr(cli_mod, "KB_PATH", kb_path)
    return cli_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_draft(gene="BRCA1", draft_type="clinvar_context_summary", text=None):
    return {
        "gene_symbol": gene,
        "draft_type": draft_type,
        "text_he": text or (
            f"גן {gene} מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "based_on": "clinvar_metadata",
        "generated_by_model": "test_model",
        "created_from": "unverified_gene_draft",
        "approved": False,
        "review_status": "unreviewed",
    }


def _minimal_queue_record(draft_id="d1", gene="BRCA1",
                           draft_type="clinvar_context_summary",
                           status="needs_review"):
    return {
        "draft_id": draft_id,
        "gene_symbol": gene,
        "draft_type": draft_type,
        "text_he": (
            "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "review_status": status,
        "based_on": "clinvar_metadata",
        "approved": False,
        "reviewed_by": None,
        "reviewed_at": None,
        "reviewer_notes": None,
    }


# ---------------------------------------------------------------------------
# 1. enqueue_gene_draft_for_review — basic creation
# ---------------------------------------------------------------------------

class TestEnqueueBasic:
    def test_creates_needs_review_record(self, dr):
        record = dr.enqueue_gene_draft_for_review(_valid_draft())
        assert record["review_status"] == "needs_review"
        assert record["approved"] is False

    def test_record_has_required_fields(self, dr):
        record = dr.enqueue_gene_draft_for_review(_valid_draft())
        for field in (
            "draft_id", "gene_symbol", "draft_type", "text_he",
            "based_on", "review_status", "approved",
            "reviewed_by", "reviewed_at", "reviewer_notes",
            "content_hash", "enqueued_at",
        ):
            assert field in record, f"Missing field: {field}"

    def test_gene_symbol_uppercased(self, dr):
        record = dr.enqueue_gene_draft_for_review(_valid_draft(gene="brca1"))
        assert record["gene_symbol"] == "BRCA1"

    def test_draft_id_is_uuid(self, dr):
        record = dr.enqueue_gene_draft_for_review(_valid_draft())
        # Raises ValueError if not a valid UUID
        uuid.UUID(record["draft_id"])

    def test_persisted_to_queue_file(self, dr, queue_path):
        dr.enqueue_gene_draft_for_review(_valid_draft())
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 1
        assert queue[0]["gene_symbol"] == "BRCA1"

    def test_content_hash_computed(self, dr):
        record = dr.enqueue_gene_draft_for_review(_valid_draft())
        assert len(record["content_hash"]) == 16

    def test_multiple_genes_enqueued(self, dr, queue_path):
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA1"))
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA2"))
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 2


# ---------------------------------------------------------------------------
# 2. enqueue — safety: no patient data
# ---------------------------------------------------------------------------

class TestEnqueueSafety:
    def test_rejects_empty_gene_symbol(self, dr):
        d = _valid_draft()
        d["gene_symbol"] = ""
        with pytest.raises(ValueError, match="gene_symbol"):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_invalid_draft_type(self, dr):
        d = _valid_draft()
        d["draft_type"] = "patient_chat_answer"
        with pytest.raises(ValueError, match="draft_type"):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_empty_text(self, dr):
        d = _valid_draft()
        d["text_he"] = ""
        with pytest.raises(ValueError):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_too_short_text(self, dr):
        d = _valid_draft(text="קצר מדי.")
        with pytest.raises(ValueError, match="too short"):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_text_with_id_number(self, dr):
        d = _valid_draft(
            text="גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. 123456789 מספר זהות."
        )
        with pytest.raises(ValueError, match="identifying"):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_text_with_email(self, dr):
        d = _valid_draft(
            text="גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA. contact: user@example.com"
        )
        with pytest.raises(ValueError, match="identifying"):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_text_with_name_phrase(self, dr):
        d = _valid_draft(
            text="השם שלי שרה כהן. גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום."
        )
        with pytest.raises(ValueError, match="identifying"):
            dr.enqueue_gene_draft_for_review(d)

    def test_rejects_pre_approved_draft(self, dr):
        d = _valid_draft()
        d["approved"] = True
        with pytest.raises(ValueError, match="approved"):
            dr.enqueue_gene_draft_for_review(d)

    def test_does_not_save_raw_question_field(self, dr, queue_path):
        d = _valid_draft()
        d["user_question"] = "מה הממצא שלי אומר?"
        dr.enqueue_gene_draft_for_review(d)
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert "user_question" not in queue[0]

    def test_source_context_stored_as_summary(self, dr, queue_path):
        d = _valid_draft()
        ctx = {"total_variants": 5000, "phenotypes": ["Breast cancer"]}
        dr.enqueue_gene_draft_for_review(d, source_context=ctx)
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        stored = queue[0].get("source_context_summary", "")
        # Either stored as a repr snippet, or not stored at all — not the raw dict
        assert "total_variants" in stored or stored == ""


# ---------------------------------------------------------------------------
# 3. enqueue — deduplication
# ---------------------------------------------------------------------------

class TestEnqueueDeduplication:
    def test_same_content_not_duplicated(self, dr, queue_path):
        d = _valid_draft()
        dr.enqueue_gene_draft_for_review(d)
        dr.enqueue_gene_draft_for_review(d)
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 1

    def test_duplicate_returns_same_draft_id(self, dr):
        d = _valid_draft()
        r1 = dr.enqueue_gene_draft_for_review(d)
        r2 = dr.enqueue_gene_draft_for_review(d)
        assert r2["draft_id"] == r1["draft_id"]

    def test_duplicate_increments_seen_count(self, dr):
        d = _valid_draft()
        dr.enqueue_gene_draft_for_review(d)
        r2 = dr.enqueue_gene_draft_for_review(d)
        assert r2["seen_count"] == 2

    def test_different_gene_not_deduplicated(self, dr, queue_path):
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA1"))
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA2"))
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 2

    def test_different_draft_type_not_deduplicated(self, dr, queue_path):
        text = (
            "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        )
        dr.enqueue_gene_draft_for_review(_valid_draft(draft_type="clinvar_context_summary", text=text))
        dr.enqueue_gene_draft_for_review(_valid_draft(draft_type="gene_summary", text=text))
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 2

    def test_different_text_not_deduplicated(self, dr, queue_path):
        d1 = _valid_draft(text=(
            "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ))
        d2 = _valid_draft(text=(
            "גן BRCA1 הוא מדכא גידולים חשוב הקשור לסרטן שד ושחלה. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ))
        dr.enqueue_gene_draft_for_review(d1)
        dr.enqueue_gene_draft_for_review(d2)
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 2


# ---------------------------------------------------------------------------
# 4. list_queue
# ---------------------------------------------------------------------------

class TestListQueue:
    def test_empty_queue_returns_empty(self, dr):
        assert dr.list_queue(status_filter=["needs_review"]) == []

    def test_pending_drafts_returned(self, dr):
        dr.enqueue_gene_draft_for_review(_valid_draft())
        result = dr.list_queue(status_filter=["needs_review", "draft"])
        assert len(result) == 1

    def test_none_filter_returns_all(self, dr):
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA1"))
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA2"))
        result = dr.list_queue(status_filter=None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. CLI — cmd_list
# ---------------------------------------------------------------------------

class TestCmdList:
    def test_empty_queue_prints_message(self, review_cli, capsys):
        class Args:
            status = "pending"
        rc = review_cli.cmd_list(Args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "No" in out

    def test_pending_drafts_shown(self, review_cli, queue_path, capsys):
        queue = [_minimal_queue_record()]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            status = "pending"
        rc = review_cli.cmd_list(Args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "BRCA1" in out
        assert "d1" in out

    def test_all_status_shows_everything(self, review_cli, queue_path, capsys):
        queue = [
            _minimal_queue_record(draft_id="aaa", gene="BRCA1", status="approved"),
            _minimal_queue_record(draft_id="bbb", gene="BRCA2", status="rejected"),
        ]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            status = "all"
        rc = review_cli.cmd_list(Args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "BRCA1" in out
        assert "BRCA2" in out


# ---------------------------------------------------------------------------
# 6. CLI — cmd_preview
# ---------------------------------------------------------------------------

class TestCmdPreview:
    def _write_queue(self, queue_path, draft_id="test-draft-1"):
        queue = [{
            "draft_id": draft_id,
            "gene_symbol": "BRCA1",
            "draft_type": "clinvar_context_summary",
            "text_he": "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA.",
            "review_status": "needs_review",
            "based_on": "clinvar_metadata",
            "created_from": "unverified_gene_draft",
            "enqueued_at": "2026-07-01T00:00:00Z",
            "generated_by_model": "test_model",
            "content_hash": "abc123",
            "approved": False,
            "seen_count": 1,
        }]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

    def test_preview_shows_gene(self, review_cli, queue_path, capsys):
        self._write_queue(queue_path)

        class Args:
            preview = "test-draft-1"
        rc = review_cli.cmd_preview(Args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "BRCA1" in out

    def test_preview_shows_text_he(self, review_cli, queue_path, capsys):
        self._write_queue(queue_path)

        class Args:
            preview = "test-draft-1"
        review_cli.cmd_preview(Args())
        out = capsys.readouterr().out
        assert "תיקון DNA" in out

    def test_preview_unknown_id_returns_1(self, review_cli, capsys):
        class Args:
            preview = "does-not-exist"
        rc = review_cli.cmd_preview(Args())
        assert rc == 1


# ---------------------------------------------------------------------------
# 7. CLI — cmd_approve
# ---------------------------------------------------------------------------

class TestCmdApprove:
    def _write_queue(self, queue_path, draft_id="d1",
                     draft_type="clinvar_context_summary", status="needs_review"):
        queue = [dict(
            _minimal_queue_record(draft_id=draft_id, draft_type=draft_type, status=status),
            text_he=(
                "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. "
                "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
            )
        )]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

    def test_approve_requires_confirm(self, review_cli, queue_path):
        self._write_queue(queue_path)

        class Args:
            approve = "d1"; confirm = False; reviewer = "Dr. Test"
            notes = None; force = False

        rc = review_cli.cmd_approve(Args())
        assert rc == 1

    def test_approve_requires_reviewer(self, review_cli, queue_path):
        self._write_queue(queue_path)

        class Args:
            approve = "d1"; confirm = True; reviewer = None
            notes = None; force = False

        rc = review_cli.cmd_approve(Args())
        assert rc == 1

    def test_approve_writes_metadata_to_queue(self, review_cli, queue_path, kb_path):
        self._write_queue(queue_path)

        class Args:
            approve = "d1"; confirm = True; reviewer = "Dr. Cohen"
            notes = "Looks good"; force = False

        rc = review_cli.cmd_approve(Args())
        assert rc == 0

        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        rec = queue[0]
        assert rec["approved"] is True
        assert rec["reviewed_by"] == "Dr. Cohen"
        assert rec["reviewer_notes"] == "Looks good"
        assert rec["review_status"] == "approved"

    def test_clinvar_summary_goes_to_approved_context_summary_he(
        self, review_cli, queue_path, kb_path
    ):
        self._write_queue(queue_path, draft_type="clinvar_context_summary")

        class Args:
            approve = "d1"; confirm = True; reviewer = "Dr. Cohen"
            notes = None; force = False

        review_cli.cmd_approve(Args())
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        assert len(kb) == 1
        assert "approved_context_summary_he" in kb[0]
        assert "patient_summary_he" not in kb[0]

    def test_gene_summary_goes_to_patient_summary_he(
        self, review_cli, queue_path, kb_path
    ):
        self._write_queue(queue_path, draft_type="gene_summary")

        class Args:
            approve = "d1"; confirm = True; reviewer = "Dr. Cohen"
            notes = None; force = False

        review_cli.cmd_approve(Args())
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        assert "patient_summary_he" in kb[0]
        assert "approved_context_summary_he" not in kb[0]

    def test_approve_already_approved_blocked_without_force(
        self, review_cli, queue_path
    ):
        self._write_queue(queue_path, status="approved")

        class Args:
            approve = "d1"; confirm = True; reviewer = "Dr. Cohen"
            notes = None; force = False

        rc = review_cli.cmd_approve(Args())
        assert rc == 1

    def test_approve_already_approved_allowed_with_force(
        self, review_cli, queue_path, kb_path
    ):
        self._write_queue(queue_path, status="approved")

        class Args:
            approve = "d1"; confirm = True; reviewer = "Dr. Cohen"
            notes = None; force = True

        rc = review_cli.cmd_approve(Args())
        assert rc == 0

    def test_approve_preserves_existing_kb_fields(
        self, review_cli, queue_path, kb_path
    ):
        # Pre-populate KB with an existing BRCA1 record
        existing_kb = [{
            "gene_symbol": "BRCA1",
            "patient_summary_he": "Original patient summary.",
            "vus_note_he": "Original VUS note.",
            "review_status": "approved",
            "approved": True,
            "reviewed_by": "Dr. Original",
            "reviewed_at": "2026-01-01T00:00:00Z",
        }]
        kb_path.write_text(json.dumps(existing_kb), encoding="utf-8")

        # New clinvar_context_summary draft for the same gene
        queue = [{
            "draft_id": "d1",
            "gene_symbol": "BRCA1",
            "draft_type": "clinvar_context_summary",
            "text_he": (
                "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA ובשמירה על יציבות הגנום. "
                "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
            ),
            "review_status": "needs_review",
            "based_on": "clinvar_metadata",
            "approved": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "reviewer_notes": None,
        }]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            approve = "d1"; confirm = True; reviewer = "Dr. New"
            notes = None; force = False

        review_cli.cmd_approve(Args())
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        assert len(kb) == 1
        # Existing fields must be preserved
        assert kb[0]["patient_summary_he"] == "Original patient summary."
        assert kb[0]["vus_note_he"] == "Original VUS note."
        # New field added
        assert "approved_context_summary_he" in kb[0]


# ---------------------------------------------------------------------------
# 8. CLI — cmd_reject
# ---------------------------------------------------------------------------

class TestCmdReject:
    def _write_queue(self, queue_path, draft_id="d1"):
        queue = [_minimal_queue_record(draft_id=draft_id)]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

    def test_reject_requires_reason(self, review_cli, queue_path):
        self._write_queue(queue_path)

        class Args:
            reject = "d1"; reason = None; reviewer = None

        rc = review_cli.cmd_reject(Args())
        assert rc == 1

    def test_reject_sets_status(self, review_cli, queue_path):
        self._write_queue(queue_path)

        class Args:
            reject = "d1"; reason = "Phenotype list inaccurate"; reviewer = "Dr. Cohen"

        rc = review_cli.cmd_reject(Args())
        assert rc == 0
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        rec = queue[0]
        assert rec["review_status"] == "rejected"
        assert rec["approved"] is False
        assert rec["reviewer_notes"] == "Phenotype list inaccurate"

    def test_rejected_not_in_pending_list(self, review_cli, queue_path, capsys):
        queue = [_minimal_queue_record(status="rejected")]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            status = "pending"
        review_cli.cmd_list(Args())
        out = capsys.readouterr().out
        assert "BRCA1" not in out

    def test_rejected_not_written_to_kb(self, review_cli, queue_path, kb_path):
        self._write_queue(queue_path)

        class Args:
            reject = "d1"; reason = "Not suitable"; reviewer = None

        review_cli.cmd_reject(Args())
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        assert kb == []


# ---------------------------------------------------------------------------
# 9. CLI — cmd_edit (edited approval)
# ---------------------------------------------------------------------------

class TestCmdEdit:
    def _write_queue(self, queue_path, draft_id="d1"):
        queue = [{
            "draft_id": draft_id,
            "gene_symbol": "BRCA1",
            "draft_type": "gene_summary",
            "text_he": "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA. טקסט מקורי.",
            "review_status": "needs_review",
            "based_on": "clinvar_metadata",
            "approved": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "reviewer_notes": None,
            "original_text_he": None,
            "edited_text_he": None,
        }]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

    def test_edit_requires_confirm(self, review_cli, queue_path, tmp_path):
        self._write_queue(queue_path)
        edited_file = tmp_path / "edited.txt"
        edited_file.write_text("טקסט ערוך ומאושר על ידי הרופא.", encoding="utf-8")

        class Args:
            edit = "d1"; confirm = False; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = None

        rc = review_cli.cmd_edit(Args())
        assert rc == 1

    def test_edit_preserves_original_text(self, review_cli, queue_path, kb_path, tmp_path):
        self._write_queue(queue_path)
        edited_file = tmp_path / "edited.txt"
        edited_file.write_text("טקסט ערוך ומאושר על ידי הרופא.", encoding="utf-8")

        class Args:
            edit = "d1"; confirm = True; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = "Corrected phrasing"

        review_cli.cmd_edit(Args())
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        rec = queue[0]
        assert rec["original_text_he"] == "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA. טקסט מקורי."
        assert rec["edited_text_he"] == "טקסט ערוך ומאושר על ידי הרופא."
        assert rec["text_he"] == "טקסט ערוך ומאושר על ידי הרופא."

    def test_edit_writes_edited_text_to_kb(self, review_cli, queue_path, kb_path, tmp_path):
        self._write_queue(queue_path)
        edited_file = tmp_path / "edited.txt"
        edited_file.write_text("טקסט ערוך ומאושר על ידי הרופא.", encoding="utf-8")

        class Args:
            edit = "d1"; confirm = True; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = None

        review_cli.cmd_edit(Args())
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        assert kb[0]["patient_summary_he"] == "טקסט ערוך ומאושר על ידי הרופא."


# ---------------------------------------------------------------------------
# 10. gene_knowledge.py — get_gene_context_summary
# ---------------------------------------------------------------------------

class TestGeneKnowledgeContextSummary:
    def test_returns_none_for_unknown_gene(self):
        from app import gene_knowledge as gk
        assert gk.get_gene_context_summary("UNKNOWN_GENE_XYZ_99") is None

    def test_returns_none_for_unapproved_record(self, monkeypatch):
        from app import gene_knowledge as gk
        # monkeypatch.setitem auto-reverts after the test — no _RECORDS leak
        monkeypatch.setitem(gk._RECORDS, "TESTGENE_UNAPPROVED", {
            "gene_symbol": "TESTGENE_UNAPPROVED",
            "approved_context_summary_he": "some context text",
            "approved": False,
        })
        assert gk.get_gene_context_summary("TESTGENE_UNAPPROVED") is None

    def test_returns_context_summary_for_approved_record(self, monkeypatch):
        from app import gene_knowledge as gk
        monkeypatch.setitem(gk._RECORDS, "TESTGENE_APPROVED", {
            "gene_symbol": "TESTGENE_APPROVED",
            "approved_context_summary_he": "Approved context summary text.",
            "approved": True,
            "review_status": "approved",
        })
        result = gk.get_gene_context_summary("TESTGENE_APPROVED")
        assert result == "Approved context summary text."

    def test_returns_none_when_field_absent(self, monkeypatch):
        from app import gene_knowledge as gk
        monkeypatch.setitem(gk._RECORDS, "TESTGENE_NO_CTX", {
            "gene_symbol": "TESTGENE_NO_CTX",
            "patient_summary_he": "A patient summary.",
            "approved": True,
        })
        assert gk.get_gene_context_summary("TESTGENE_NO_CTX") is None


# ---------------------------------------------------------------------------
# 11. Approval gates — unapproved drafts do NOT become patient-facing answers
# ---------------------------------------------------------------------------

class TestApprovalGates:
    def test_has_approved_gene_knowledge_false_for_unapproved(self, monkeypatch):
        from app import gene_knowledge as gk
        monkeypatch.setitem(gk._RECORDS, "GATETEST_GENE", {
            "gene_symbol": "GATETEST_GENE",
            "patient_summary_he": "Patient summary text.",
            "approved": False,
            "review_status": "draft",
        })
        assert gk.has_approved_gene_knowledge("GATETEST_GENE") is False

    def test_get_patient_summary_returns_none_when_unapproved(self, monkeypatch):
        from app import gene_knowledge as gk
        monkeypatch.setitem(gk._RECORDS, "GATETEST2_GENE", {
            "gene_symbol": "GATETEST2_GENE",
            "patient_summary_he": "Patient summary text.",
            "approved": False,
        })
        assert gk.get_gene_patient_summary("GATETEST2_GENE") is None

    def test_unapproved_not_in_approved_list(self, monkeypatch):
        from app import gene_knowledge as gk
        monkeypatch.setitem(gk._RECORDS, "GATETEST3_GENE", {
            "gene_symbol": "GATETEST3_GENE",
            "approved": False,
        })
        approved = gk.list_approved_genes()
        assert "GATETEST3_GENE" not in approved

    def test_approved_gene_appears_in_approved_list(self, monkeypatch):
        from app import gene_knowledge as gk
        monkeypatch.setitem(gk._RECORDS, "GATETEST4_GENE", {
            "gene_symbol": "GATETEST4_GENE",
            "approved": True,
            "review_status": "approved",
        })
        approved = gk.list_approved_genes()
        assert "GATETEST4_GENE" in approved


# ---------------------------------------------------------------------------
# 12. pending_count utility
# ---------------------------------------------------------------------------

class TestPendingCount:
    def test_empty_queue_zero_count(self, dr):
        assert dr.pending_count() == 0

    def test_one_pending_draft(self, dr):
        dr.enqueue_gene_draft_for_review(_valid_draft())
        assert dr.pending_count() == 1

    def test_approved_not_counted_as_pending(self, dr, queue_path):
        dr.enqueue_gene_draft_for_review(_valid_draft())
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        queue[0]["review_status"] = "approved"
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        assert dr.pending_count() == 0

    def test_two_pending_different_genes(self, dr):
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA1"))
        dr.enqueue_gene_draft_for_review(_valid_draft(gene="BRCA2"))
        assert dr.pending_count() == 2


# ---------------------------------------------------------------------------
# 13. CLI hardening — edit guard on approved drafts
# ---------------------------------------------------------------------------

class TestCmdEditHardening:
    def _write_approved_queue(self, queue_path, draft_id="d-approved"):
        queue = [{
            "draft_id": draft_id,
            "gene_symbol": "BRCA1",
            "draft_type": "gene_summary",
            "text_he": (
                "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA. "
                "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
            ),
            "review_status": "approved",
            "based_on": "clinvar_metadata",
            "approved": True,
            "reviewed_by": "Dr. Cohen",
            "reviewed_at": "2026-07-01T00:00:00Z",
            "reviewer_notes": None,
            "original_text_he": None,
            "edited_text_he": None,
        }]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

    def test_edit_approved_without_force_fails(self, review_cli, queue_path, kb_path, tmp_path):
        self._write_approved_queue(queue_path)
        edited_file = tmp_path / "edit.txt"
        edited_file.write_text("עדכון.", encoding="utf-8")

        class Args:
            edit = "d-approved"; confirm = True; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = None; force = False

        rc = review_cli.cmd_edit(Args())
        assert rc == 1

    def test_edit_approved_without_force_does_not_touch_queue(
        self, review_cli, queue_path, kb_path, tmp_path
    ):
        self._write_approved_queue(queue_path)
        edited_file = tmp_path / "edit.txt"
        edited_file.write_text("עדכון.", encoding="utf-8")

        class Args:
            edit = "d-approved"; confirm = True; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = None; force = False

        review_cli.cmd_edit(Args())
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        # original text must be unchanged
        assert queue[0]["review_status"] == "approved"
        assert queue[0]["edited_text_he"] is None

    def test_edit_approved_with_force_succeeds(
        self, review_cli, queue_path, kb_path, tmp_path
    ):
        self._write_approved_queue(queue_path)
        edited_file = tmp_path / "edit.txt"
        edited_file.write_text(
            "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA — עדכון מאושר. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי.",
            encoding="utf-8",
        )

        class Args:
            edit = "d-approved"; confirm = True; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = "Re-approved after correction"; force = True

        rc = review_cli.cmd_edit(Args())
        assert rc == 0
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert queue[0]["edited_text_he"] is not None
        assert "עדכון מאושר" in queue[0]["edited_text_he"]

    def test_edit_approved_edited_status_without_force_fails(
        self, review_cli, queue_path, kb_path, tmp_path
    ):
        """approved_edited status is also blocked without --force."""
        queue = [{
            "draft_id": "d-approved-edited",
            "gene_symbol": "BRCA1",
            "draft_type": "gene_summary",
            "text_he": (
                "גן BRCA1 מקודד לחלבון המשתתף בתיקון DNA. "
                "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
            ),
            "review_status": "approved_edited",
            "approved": True,
            "reviewed_by": "Dr. Cohen",
            "reviewed_at": "2026-07-01T00:00:00Z",
            "reviewer_notes": None,
            "original_text_he": "Original.",
            "edited_text_he": None,
        }]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        edited_file = tmp_path / "edit.txt"
        edited_file.write_text("עדכון נוסף.", encoding="utf-8")

        class Args:
            edit = "d-approved-edited"; confirm = True; reviewer = "Dr. Cohen"
            from_file = str(edited_file); notes = None; force = False

        rc = review_cli.cmd_edit(Args())
        assert rc == 1


# ---------------------------------------------------------------------------
# 14. CLI hardening — archive command
# ---------------------------------------------------------------------------

class TestCmdArchive:
    def _write_pending_queue(self, queue_path, draft_id="d-pending", gene="POLE"):
        queue = [_minimal_queue_record(draft_id=draft_id, gene=gene)]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

    def test_archive_requires_reason(self, review_cli, queue_path):
        self._write_pending_queue(queue_path)

        class Args:
            archive = "d-pending"; reason = None; reviewer = None

        rc = review_cli.cmd_archive(Args())
        assert rc == 1

    def test_archive_sets_status(self, review_cli, queue_path):
        self._write_pending_queue(queue_path)

        class Args:
            archive = "d-pending"; reason = "Technical workflow test — not real content"
            reviewer = "Noa"

        rc = review_cli.cmd_archive(Args())
        assert rc == 0
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert queue[0]["review_status"] == "archived"
        assert queue[0]["approved"] is False
        assert "Technical workflow test" in queue[0]["reviewer_notes"]

    def test_archive_keeps_record_in_queue(self, review_cli, queue_path):
        self._write_pending_queue(queue_path)

        class Args:
            archive = "d-pending"; reason = "Superseded"; reviewer = None

        review_cli.cmd_archive(Args())
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue) == 1  # record still present

    def test_archived_not_in_pending_list(self, review_cli, queue_path, capsys):
        queue = [_minimal_queue_record(status="archived")]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            status = "pending"
        review_cli.cmd_list(Args())
        out = capsys.readouterr().out
        assert "BRCA1" not in out

    def test_archived_appears_with_status_all(self, review_cli, queue_path, capsys):
        queue = [_minimal_queue_record(status="archived")]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            status = "all"
        rc = review_cli.cmd_list(Args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "BRCA1" in out

    def test_archived_appears_with_status_archived(self, review_cli, queue_path, capsys):
        queue = [_minimal_queue_record(status="archived")]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            status = "archived"
        rc = review_cli.cmd_list(Args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "BRCA1" in out

    def test_archive_idempotent(self, review_cli, queue_path):
        """Archiving an already-archived draft is a no-op (SKIP)."""
        queue = [_minimal_queue_record(status="archived")]
        queue_path.write_text(json.dumps(queue), encoding="utf-8")

        class Args:
            archive = "d1"; reason = "Again"; reviewer = None

        rc = review_cli.cmd_archive(Args())
        assert rc == 0  # SKIP, not error
        queue_after = json.loads(queue_path.read_text(encoding="utf-8"))
        assert len(queue_after) == 1

    def test_archive_unknown_id_returns_1(self, review_cli, queue_path):
        class Args:
            archive = "nonexistent-id"; reason = "Test"; reviewer = None

        rc = review_cli.cmd_archive(Args())
        assert rc == 1

    def test_archive_does_not_write_to_kb(self, review_cli, queue_path, kb_path):
        self._write_pending_queue(queue_path)

        class Args:
            archive = "d-pending"; reason = "Technical test"; reviewer = None

        review_cli.cmd_archive(Args())
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        assert kb == []


# ---------------------------------------------------------------------------
# 15. Seed script — draft content and idempotency
# ---------------------------------------------------------------------------

class TestSeedScript:
    def test_seed_script_exists(self):
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        p = project_root / "scripts" / "seed_first_gene_review_drafts.py"
        assert p.exists(), "seed script must exist"

    def test_seeded_drafts_are_not_approved(self, dr, queue_path):
        import scripts.seed_first_gene_review_drafts as seed
        # Patch the queue path used inside the enqueue helper
        for draft in seed._DRAFTS:
            dr.enqueue_gene_draft_for_review(dict(draft))
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        for rec in queue:
            assert rec["approved"] is False, f"{rec['gene_symbol']} must not be approved"
            assert rec["review_status"] == "needs_review"

    def test_seeded_drafts_cover_required_genes(self, dr):
        import scripts.seed_first_gene_review_drafts as seed
        for draft in seed._DRAFTS:
            dr.enqueue_gene_draft_for_review(dict(draft))
        result = dr.list_queue(status_filter=["needs_review"])
        genes = {r["gene_symbol"] for r in result}
        assert {"BRCA1", "BRCA2", "APC", "POLE", "HBB"} <= genes

    def test_seeded_drafts_idempotent(self, dr, queue_path):
        import scripts.seed_first_gene_review_drafts as seed
        for draft in seed._DRAFTS:
            dr.enqueue_gene_draft_for_review(dict(draft))
        count_first = len(json.loads(queue_path.read_text(encoding="utf-8")))
        # Run again — should not create duplicates
        for draft in seed._DRAFTS:
            dr.enqueue_gene_draft_for_review(dict(draft))
        count_second = len(json.loads(queue_path.read_text(encoding="utf-8")))
        assert count_first == count_second

    def test_seeded_drafts_text_has_disclaimer(self, dr):
        import scripts.seed_first_gene_review_drafts as seed
        for draft in seed._DRAFTS:
            r = dr.enqueue_gene_draft_for_review(dict(draft))
            assert "המידע כללי ואינו מחליף ייעוץ רפואי אישי" in r["text_he"], \
                f"{r['gene_symbol']} draft missing standard disclaimer"

    def test_seeded_drafts_no_clinvar_in_patient_text(self, dr):
        import scripts.seed_first_gene_review_drafts as seed
        for draft in seed._DRAFTS:
            r = dr.enqueue_gene_draft_for_review(dict(draft))
            assert "ClinVar" not in r["text_he"], \
                f"{r['gene_symbol']} text_he must not mention ClinVar"

    def test_seeded_drafts_hebrew_only_patient_text(self, dr):
        """text_he must not contain raw English phenotype names."""
        import scripts.seed_first_gene_review_drafts as seed
        # These raw English phenotype strings should not appear in patient text
        banned = ["Breast cancer", "Ovarian cancer", "Colorectal cancer", "Hereditary"]
        for draft in seed._DRAFTS:
            r = dr.enqueue_gene_draft_for_review(dict(draft))
            for word in banned:
                assert word not in r["text_he"], \
                    f"{r['gene_symbol']} text_he contains raw English: {word!r}"

    def test_list_approved_genes_unchanged_after_seeding(self):
        from app import gene_knowledge as gk
        approved = gk.list_approved_genes()
        # HBB was approved by the instructor in Session 15; all other records remain drafts.
        unexpected = [g for g in approved if g != "HBB"]
        assert unexpected == [], f"Unexpected approved genes after seeding: {unexpected}"
