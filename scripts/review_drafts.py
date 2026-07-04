# -*- coding: utf-8 -*-
"""
scripts/review_drafts.py

Human review / approval workflow CLI for unverified gene drafts.

Physicians and genetic counselors use this tool to review gene-level drafts
queued in app/data/draft_review_queue.json and approve them into
app/data/gene_knowledge_base.json as Tier 1b Gene Knowledge.

Safety rules:
  • No draft can be auto-approved — --confirm flag always required.
  • Approval requires --reviewer (the approver's identity).
  • Rejected drafts are never shown as approved content.
  • A clinvar_context_summary draft is written to approved_context_summary_he,
    NOT to patient_summary_he, so it is not silently used as a full gene summary.
  • Already-approved drafts cannot be re-approved without --force.
  • Already-approved drafts cannot be edited without --force.
  • The original AI text is always preserved in original_text_he.
  • Archived drafts are kept for audit but removed from the pending list.
  • --force always prints a visible warning.

Usage:
  python scripts/review_drafts.py --list
  python scripts/review_drafts.py --list --status all
  python scripts/review_drafts.py --list --status archived
  python scripts/review_drafts.py --preview DRAFT_ID
  python scripts/review_drafts.py --approve DRAFT_ID --reviewer NAME --confirm
  python scripts/review_drafts.py --approve DRAFT_ID --reviewer NAME --confirm --notes "..."
  python scripts/review_drafts.py --reject  DRAFT_ID --reason TEXT [--reviewer NAME]
  python scripts/review_drafts.py --edit    DRAFT_ID --from-file PATH --reviewer NAME --confirm
  python scripts/review_drafts.py --archive DRAFT_ID --reason TEXT [--reviewer NAME]
"""

import argparse
import json
import sys
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "app" / "data" / "draft_review_queue.json"
KB_PATH    = ROOT / "app" / "data" / "gene_knowledge_base.json"

# Status values that appear in --list by default
_PENDING_STATUSES = {"needs_review", "draft"}
# Status values that mean "no longer under review"
_TERMINAL_STATUSES = {"approved", "approved_edited", "rejected", "archived"}
# Draft types that map to approved_context_summary_he instead of patient_summary_he
_CONTEXT_SUMMARY_TYPES = {"clinvar_context_summary"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_queue() -> list:
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def _save_queue(queue: list) -> None:
    QUEUE_PATH.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_kb() -> list:
    if not KB_PATH.exists():
        return []
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def _save_kb(kb: list) -> None:
    KB_PATH.write_text(
        json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _find_draft(queue: list, draft_id: str) -> dict | None:
    for d in queue:
        if d.get("draft_id") == draft_id:
            return d
    return None


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_to_kb(gene: str, draft: dict, approved_text: str, reviewer: str, notes: str | None) -> None:
    """Write an approved draft into gene_knowledge_base.json.

    clinvar_context_summary → approved_context_summary_he  (NOT patient_summary_he)
    gene_summary / vus_note → patient_summary_he
    Existing metadata (sources, other fields) is preserved.
    """
    kb = _load_kb()
    existing = next((r for r in kb if r.get("gene_symbol") == gene), None)

    draft_type = draft.get("draft_type", "gene_summary")
    now = _now_iso()

    if existing is None:
        record: dict = {
            "gene_symbol": gene,
            "based_on": draft.get("based_on", "clinvar_metadata"),
            "source_1_name": draft.get("source_1_name"),
            "source_1_url_or_id": draft.get("source_1_url_or_id"),
            "source_2_name": draft.get("source_2_name"),
            "source_2_url_or_id": draft.get("source_2_url_or_id"),
            "review_status": "approved",
            "approved": True,
            "reviewed_by": reviewer,
            "reviewed_at": now,
            "reviewer_notes": notes,
        }
        if draft_type in _CONTEXT_SUMMARY_TYPES:
            record["approved_context_summary_he"] = approved_text
        else:
            record["patient_summary_he"] = approved_text
        kb.append(record)
    else:
        # Preserve all existing fields; only update the relevant content field
        # and reviewer metadata.
        if draft_type in _CONTEXT_SUMMARY_TYPES:
            existing["approved_context_summary_he"] = approved_text
        else:
            existing["patient_summary_he"] = approved_text
        existing["review_status"] = "approved"
        existing["approved"] = True
        existing["reviewed_by"] = reviewer
        existing["reviewed_at"] = now
        existing["reviewer_notes"] = notes
        # Preserve sources if draft has them and existing doesn't
        for src_key in ("source_1_name", "source_1_url_or_id", "source_2_name", "source_2_url_or_id"):
            if draft.get(src_key) and not existing.get(src_key):
                existing[src_key] = draft[src_key]

    _save_kb(kb)


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_list(args) -> int:
    queue = _load_queue()

    status_arg = getattr(args, "status", "pending") or "pending"
    if status_arg == "all":
        filtered = queue
        label = "all"
    elif status_arg == "approved":
        filtered = [d for d in queue if d.get("review_status") in ("approved", "approved_edited")]
        label = "approved"
    elif status_arg == "rejected":
        filtered = [d for d in queue if d.get("review_status") == "rejected"]
        label = "rejected"
    elif status_arg == "archived":
        filtered = [d for d in queue if d.get("review_status") == "archived"]
        label = "archived"
    else:
        filtered = [d for d in queue if d.get("review_status") in _PENDING_STATUSES]
        label = "pending"

    if not filtered:
        print(f"No {label} drafts in queue.")
        return 0

    print(f"{'DRAFT_ID':<38} {'GENE':<8} {'TYPE':<28} {'STATUS':<14} {'ENQUEUED_AT'}")
    print("-" * 110)
    for d in filtered:
        print(
            f"{d.get('draft_id', '?'):<38} "
            f"{d.get('gene_symbol', '?'):<8} "
            f"{d.get('draft_type', '?'):<28} "
            f"{d.get('review_status', '?'):<14} "
            f"{d.get('enqueued_at') or d.get('generated_at', '?')}"
        )
    print(f"\n{len(filtered)} draft(s) shown ({label}).")
    return 0


def cmd_preview(args) -> int:
    queue = _load_queue()
    draft = _find_draft(queue, args.preview)
    if draft is None:
        print(f"ERROR: draft_id '{args.preview}' not found in queue.", file=sys.stderr)
        return 1

    print(f"=== Draft preview: {draft['draft_id']} ===")
    print(f"Gene:          {draft.get('gene_symbol', '?')}")
    print(f"Type:          {draft.get('draft_type', '?')}")
    print(f"Status:        {draft.get('review_status', '?')}")
    print(f"Approved:      {draft.get('approved', False)}")
    print(f"Based on:      {draft.get('based_on', '?')}")
    print(f"Created from:  {draft.get('created_from', '?')}")
    print(f"Enqueued at:   {draft.get('enqueued_at') or draft.get('generated_at', '?')}")
    print(f"Generated by:  {draft.get('generated_by_model', '?')}")
    print(f"Content hash:  {draft.get('content_hash', '?')}")
    print(f"Seen count:    {draft.get('seen_count', 1)}")
    if draft.get("source_1_name"):
        print(f"Source 1:      {draft['source_1_name']} — {draft.get('source_1_url_or_id', '')}")
    if draft.get("source_2_name"):
        print(f"Source 2:      {draft['source_2_name']} — {draft.get('source_2_url_or_id', '')}")
    print()
    print("--- text_he ---")
    print(draft.get("text_he", "(empty)"))
    print()
    if draft.get("source_note_he"):
        print("--- source_note_he ---")
        print(draft["source_note_he"])
        print()
    if draft.get("original_text_he"):
        print("--- original_text_he (before edit) ---")
        print(draft["original_text_he"])
        print()
    if draft.get("edited_text_he"):
        print("--- edited_text_he ---")
        print(draft["edited_text_he"])
        print()
    if draft.get("reviewer_notes"):
        print("--- reviewer_notes ---")
        print(draft["reviewer_notes"])
        print()
    if draft.get("reviewed_by"):
        print(f"Reviewed by: {draft['reviewed_by']} at {draft.get('reviewed_at', '?')}")
    return 0


def cmd_approve(args) -> int:
    if not args.confirm:
        print("ERROR: --confirm flag is required to approve a draft.", file=sys.stderr)
        return 1
    if not args.reviewer:
        print("ERROR: --reviewer NAME is required to approve a draft.", file=sys.stderr)
        return 1

    queue = _load_queue()
    draft = _find_draft(queue, args.approve)
    if draft is None:
        print(f"ERROR: draft_id '{args.approve}' not found in queue.", file=sys.stderr)
        return 1

    # Guard: already approved
    if draft.get("review_status") in ("approved", "approved_edited") and not getattr(args, "force", False):
        print(
            f"ERROR: draft '{args.approve}' is already approved (status: {draft['review_status']}). "
            "Use --force to re-approve (rare — requires explicit intent).",
            file=sys.stderr,
        )
        return 1

    # Guard: rejected — require explicit force
    if draft.get("review_status") == "rejected" and not getattr(args, "force", False):
        print(
            f"ERROR: draft '{args.approve}' was previously rejected (reason: "
            f"{draft.get('reviewer_notes', '?')}). Use --force to override.",
            file=sys.stderr,
        )
        return 1

    # Guard: archived — require explicit force
    if draft.get("review_status") == "archived" and not getattr(args, "force", False):
        print(
            f"ERROR: draft '{args.approve}' is archived. Use --force to approve an archived draft.",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "force", False):
        print(
            f"WARNING: --force used on draft '{args.approve}' "
            f"(previous status: {draft.get('review_status', '?')}). Proceeding.",
            file=sys.stderr,
        )

    text_he = (draft.get("text_he") or "").strip()
    if not text_he:
        print("ERROR: draft has empty text_he — cannot approve.", file=sys.stderr)
        return 1

    gene = draft.get("gene_symbol", "UNKNOWN")
    draft_type = draft.get("draft_type", "gene_summary")
    notes = getattr(args, "notes", None)

    _write_to_kb(gene, draft, text_he, args.reviewer, notes)

    # Update queue record
    draft["review_status"] = "approved"
    draft["approved"] = True
    draft["reviewed_by"] = args.reviewer
    draft["reviewed_at"] = _now_iso()
    if notes:
        draft["reviewer_notes"] = notes
    _save_queue(queue)

    field_written = "approved_context_summary_he" if draft_type in _CONTEXT_SUMMARY_TYPES else "patient_summary_he"
    print(f"OK: draft '{args.approve}' for gene {gene} ({draft_type}) approved.")
    print(f"    Written to gene_knowledge_base.json field: {field_written}")
    print(f"    Reviewer: {args.reviewer}")
    return 0


def cmd_reject(args) -> int:
    if not args.reason:
        print("ERROR: --reason TEXT is required to reject a draft.", file=sys.stderr)
        return 1

    queue = _load_queue()
    draft = _find_draft(queue, args.reject)
    if draft is None:
        print(f"ERROR: draft_id '{args.reject}' not found in queue.", file=sys.stderr)
        return 1

    draft["review_status"] = "rejected"
    draft["approved"] = False
    draft["reviewer_notes"] = args.reason
    draft["reviewed_at"] = _now_iso()
    if getattr(args, "reviewer", None):
        draft["reviewed_by"] = args.reviewer
    _save_queue(queue)

    gene = draft.get("gene_symbol", "?")
    print(f"OK: draft '{args.reject}' for gene {gene} rejected.")
    print(f"    Reason: {args.reason}")
    return 0


def cmd_edit(args) -> int:
    if not args.confirm:
        print("ERROR: --confirm flag is required to approve an edited draft.", file=sys.stderr)
        return 1
    if not args.reviewer:
        print("ERROR: --reviewer NAME is required.", file=sys.stderr)
        return 1
    if not args.from_file:
        print("ERROR: --from-file PATH is required.", file=sys.stderr)
        return 1

    edited_path = pathlib.Path(args.from_file)
    if not edited_path.exists():
        print(f"ERROR: file '{args.from_file}' not found.", file=sys.stderr)
        return 1

    edited_text = edited_path.read_text(encoding="utf-8").strip()
    if not edited_text:
        print("ERROR: edited text file is empty.", file=sys.stderr)
        return 1

    queue = _load_queue()
    draft = _find_draft(queue, args.edit)
    if draft is None:
        print(f"ERROR: draft_id '{args.edit}' not found in queue.", file=sys.stderr)
        return 1

    # Guard: editing already-approved content requires --force
    if draft.get("review_status") in ("approved", "approved_edited") and not getattr(args, "force", False):
        print(
            f"ERROR: draft '{args.edit}' has already been approved "
            f"(status: {draft['review_status']}). "
            "Editing approved content requires --force.",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "force", False):
        print(
            f"WARNING: --force used on draft '{args.edit}' "
            f"(previous status: {draft.get('review_status', '?')}). Proceeding with edit.",
            file=sys.stderr,
        )

    gene = draft.get("gene_symbol", "UNKNOWN")
    draft_type = draft.get("draft_type", "gene_summary")
    notes = getattr(args, "notes", "Approved with edits")

    # Preserve original text before overwriting
    if not draft.get("original_text_he"):
        draft["original_text_he"] = draft.get("text_he", "")

    draft["edited_text_he"] = edited_text
    draft["text_he"] = edited_text  # text_he now reflects what was approved
    draft["review_status"] = "approved_edited"
    draft["approved"] = True
    draft["reviewed_by"] = args.reviewer
    draft["reviewed_at"] = _now_iso()
    draft["reviewer_notes"] = notes
    _save_queue(queue)

    _write_to_kb(gene, draft, edited_text, args.reviewer, notes)

    field_written = "approved_context_summary_he" if draft_type in _CONTEXT_SUMMARY_TYPES else "patient_summary_he"
    print(f"OK: edited draft '{args.edit}' for gene {gene} ({draft_type}) approved.")
    print(f"    Written to gene_knowledge_base.json field: {field_written}")
    print(f"    Reviewer: {args.reviewer}")
    return 0


def cmd_archive(args) -> int:
    """Mark a draft as archived — removed from pending list but kept for audit.

    Use this to clean up test artifacts, superseded drafts, or any record that
    should no longer appear in the pending review queue but must not be deleted.
    Archived drafts are visible with --list --status archived or --status all.
    """
    if not args.reason:
        print("ERROR: --reason TEXT is required to archive a draft.", file=sys.stderr)
        return 1

    queue = _load_queue()
    draft = _find_draft(queue, args.archive)
    if draft is None:
        print(f"ERROR: draft_id '{args.archive}' not found in queue.", file=sys.stderr)
        return 1

    if draft.get("review_status") == "archived":
        print(f"SKIP: draft '{args.archive}' is already archived.")
        return 0

    prev_status = draft.get("review_status", "?")
    draft["review_status"] = "archived"
    draft["approved"] = False  # archived records are never approved
    draft["reviewer_notes"] = args.reason
    draft["reviewed_at"] = _now_iso()
    if getattr(args, "reviewer", None):
        draft["reviewed_by"] = args.reviewer
    _save_queue(queue)

    gene = draft.get("gene_symbol", "?")
    print(f"OK: draft '{args.archive}' for gene {gene} archived (was: {prev_status}).")
    print(f"    Reason: {args.reason}")
    return 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Human review/approval workflow CLI for gene drafts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/review_drafts.py --list
  python scripts/review_drafts.py --list --status all
  python scripts/review_drafts.py --list --status archived
  python scripts/review_drafts.py --preview <DRAFT_ID>
  python scripts/review_drafts.py --approve <DRAFT_ID> --reviewer "Dr. Cohen" --confirm
  python scripts/review_drafts.py --reject  <DRAFT_ID> --reason "Inaccurate phenotype list"
  python scripts/review_drafts.py --edit    <DRAFT_ID> --from-file edited.txt --reviewer "Dr. Cohen" --confirm
  python scripts/review_drafts.py --archive <DRAFT_ID> --reason "Technical workflow test"
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",    action="store_true",   help="List drafts (default: pending only)")
    group.add_argument("--preview", metavar="DRAFT_ID",    help="Preview a specific draft")
    group.add_argument("--approve", metavar="DRAFT_ID",    help="Approve a draft (requires --reviewer --confirm)")
    group.add_argument("--reject",  metavar="DRAFT_ID",    help="Reject a draft (requires --reason)")
    group.add_argument("--edit",    metavar="DRAFT_ID",    help="Approve an edited version (requires --from-file --reviewer --confirm)")
    group.add_argument("--archive", metavar="DRAFT_ID",    help="Archive a draft — remove from pending, keep for audit (requires --reason)")

    parser.add_argument("--reviewer",  metavar="NAME",  help="Reviewer name (required for approve/edit)")
    parser.add_argument("--confirm",   action="store_true", help="Required to execute approve/edit")
    parser.add_argument("--force",     action="store_true", help="Override approval/edit guard — always prints a warning")
    parser.add_argument("--reason",    metavar="TEXT",  help="Required for --reject and --archive")
    parser.add_argument("--from-file", metavar="PATH",  dest="from_file", help="Edited text file (for --edit)")
    parser.add_argument("--notes",     metavar="TEXT",  help="Optional reviewer notes (for approve/edit)")
    parser.add_argument("--status",    metavar="STATUS", default="pending",
                        help="Filter for --list: pending (default) | approved | rejected | archived | all")

    args = parser.parse_args()

    if args.list:
        return cmd_list(args)
    if args.preview:
        return cmd_preview(args)
    if args.approve:
        return cmd_approve(args)
    if args.reject:
        return cmd_reject(args)
    if args.edit:
        return cmd_edit(args)
    if args.archive:
        return cmd_archive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
