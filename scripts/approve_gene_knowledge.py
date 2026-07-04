#!/usr/bin/env python
"""
scripts/approve_gene_knowledge.py

Manual approval tool for the Gene Knowledge Base.

Usage:
    python scripts/approve_gene_knowledge.py --gene SYMBOL --reviewer NAME
    python scripts/approve_gene_knowledge.py --list          # show approval status
    python scripts/approve_gene_knowledge.py --list-all      # show all records

Safety invariants enforced by this script:
  1. NEVER auto-approves. Requires --gene and --reviewer explicitly.
  2. NEVER approves a record whose source_status is "source_missing".
     The reviewer must first update the record with a real source and set
     source_status = "needs_review" or "verified".
  3. Prints the full record for human confirmation before approving.
  4. Requires explicit --confirm flag to write. Without it, runs as a dry-run.

After approval the record gets:
  approved = true
  review_status = "approved"
  reviewed_by = <reviewer argument>
  reviewed_at = <ISO-8601 UTC timestamp>

To update a source before approval, edit data/gene_knowledge_base.json
directly and set source_status to "needs_review" or "verified".
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KB_PATH = _REPO_ROOT / "app" / "data" / "gene_knowledge_base.json"  # app/data/ — same as kb.py convention


def _load() -> list[dict]:
    if not _KB_PATH.exists():
        print(f"ERROR: {_KB_PATH} not found.")
        sys.exit(1)
    return json.loads(_KB_PATH.read_text(encoding="utf-8"))


def _save(records: list[dict]) -> None:
    _KB_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cmd_list(all_records: bool = False) -> None:
    records = _load()
    print(f"\n{'GENE':<10} {'SOURCE_STATUS':<18} {'REVIEW_STATUS':<16} {'APPROVED':<10} {'REVIEWER'}")
    print("-" * 75)
    for rec in sorted(records, key=lambda r: r.get("gene_symbol", "")):
        sym = rec.get("gene_symbol", "?")
        ss = rec.get("source_status", "?")
        rs = rec.get("review_status", "?")
        appr = str(rec.get("approved", False))
        reviewer = rec.get("reviewed_by") or "-"
        if all_records or rec.get("approved") is False:
            print(f"{sym:<10} {ss:<18} {rs:<16} {appr:<10} {reviewer}")
    print()


def cmd_approve(gene: str, reviewer: str, notes: str | None, confirm: bool) -> None:
    records = _load()
    gene_upper = gene.upper()
    target = None
    for rec in records:
        if (rec.get("gene_symbol") or "").upper() == gene_upper:
            target = rec
            break

    if target is None:
        print(f"ERROR: Gene '{gene_upper}' not found in {_KB_PATH}.")
        sys.exit(1)

    # Safety gate 1: already approved
    if target.get("approved") is True:
        print(f"SKIP: {gene_upper} is already approved (reviewed_by={target.get('reviewed_by')}).")
        sys.exit(0)

    # Safety gate 2: source_missing — cannot approve without a source
    source_status = target.get("source_status", "source_missing")
    if source_status == "source_missing":
        print(
            f"ERROR: Cannot approve {gene_upper} — source_status is 'source_missing'.\n"
            "       Update the record with a real source URL in data/gene_knowledge_base.json\n"
            "       and set source_status to 'needs_review' or 'verified', then re-run."
        )
        sys.exit(1)

    # Safety gate 3: empty patient_summary_he
    if not (target.get("patient_summary_he") or "").strip():
        print(
            f"ERROR: Cannot approve {gene_upper} — patient_summary_he is empty.\n"
            "       Add a Hebrew summary first."
        )
        sys.exit(1)

    # Print the full record for human review
    print(f"\n{'='*60}")
    print(f"Reviewing record for: {gene_upper}")
    print(f"{'='*60}")
    for k, v in target.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    if not confirm:
        print("DRY-RUN: No changes written. Add --confirm to actually approve.")
        return

    # Apply approval
    target["approved"] = True
    target["review_status"] = "approved"
    target["reviewed_by"] = reviewer
    target["reviewed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if notes:
        target["reviewer_notes"] = notes

    _save(records)
    print(f"OK: {gene_upper} approved by {reviewer} at {target['reviewed_at']}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual approval tool for the Gene Knowledge Base."
    )
    parser.add_argument("--gene", metavar="SYMBOL", help="Gene to approve.")
    parser.add_argument("--reviewer", metavar="NAME", help="Reviewer name or ID.")
    parser.add_argument("--notes", metavar="TEXT", help="Optional reviewer notes.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually write the approval. Without this, runs as dry-run.",
    )
    parser.add_argument("--list", action="store_true", help="List unapproved records.")
    parser.add_argument("--list-all", action="store_true", help="List all records.")
    args = parser.parse_args()

    if args.list_all:
        cmd_list(all_records=True)
    elif args.list:
        cmd_list(all_records=False)
    elif args.gene:
        if not args.reviewer:
            print("ERROR: --reviewer is required when approving a gene.")
            sys.exit(1)
        cmd_approve(args.gene, args.reviewer, args.notes, args.confirm)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
