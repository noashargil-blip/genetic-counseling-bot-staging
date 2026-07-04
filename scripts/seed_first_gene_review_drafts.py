# -*- coding: utf-8 -*-
"""
scripts/seed_first_gene_review_drafts.py

Enqueue the first batch of gene-level review drafts for physician/counselor review.

These are DRAFTS ONLY — not approved medical content.
Every record is created with:
  - approved=False
  - review_status="needs_review"
  - created_from="offline_draft_script"

A physician or genetic counselor must review and explicitly approve each draft
via scripts/review_drafts.py before it becomes Tier 1b Gene Knowledge.

Idempotent: running this script twice does not create duplicates.
Deduplication is handled by enqueue_gene_draft_for_review() using
gene_symbol + draft_type + content_hash.

Genes seeded: BRCA1, BRCA2, APC, POLE, HBB

Usage:
  python scripts/seed_first_gene_review_drafts.py [--dry-run]
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.draft_review import enqueue_gene_draft_for_review

# ---------------------------------------------------------------------------
# Draft definitions
#
# Each draft must be:
#   • Hebrew only, patient-friendly
#   • 2–4 sentences
#   • No personal interpretation
#   • No medical recommendations
#   • No raw ClinVar statistics
#   • No "ClinVar" in patient-facing text
#   • Ends with the neutral disclaimer
#   • draft_type=gene_summary because MedlinePlus/NCBI Gene sources exist
# ---------------------------------------------------------------------------

_DRAFTS = [
    {
        "gene_symbol": "BRCA1",
        "draft_type": "gene_summary",
        "text_he": (
            "גן BRCA1 מקודד לחלבון המשתתף בתיקון נזקי DNA ובשמירה על יציבות הגנום. "
            "שינויים פתוגניים בגן זה מוכרים בספרות כקשורים לנטייה תורשתית לסרטן שד וסרטן שחלה. "
            "VUS בגן BRCA1 אינו אבחנה ואינו בסיס לקבלת החלטות רפואיות — משמעותו תלויה בנסיבות "
            "הקליניות המלאות. המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "based_on": "MedlinePlus Genetics, NCBI Gene",
        "source_1_name": "MedlinePlus Genetics",
        "source_1_url_or_id": "https://medlineplus.gov/genetics/gene/brca1/",
        "source_2_name": "NCBI Gene",
        "source_2_url_or_id": "https://www.ncbi.nlm.nih.gov/gene/672",
        "generated_by_model": "offline_draft_script_v1",
        "created_from": "offline_draft_script",
    },
    {
        "gene_symbol": "BRCA2",
        "draft_type": "gene_summary",
        "text_he": (
            "גן BRCA2 מקודד לחלבון המשתתף בתיקון נזקי DNA בתהליך שחזור הומולוגי. "
            "שינויים פתוגניים בגן זה מוכרים בספרות כקשורים לנטייה תורשתית לסרטן שד, שחלה ולבלב. "
            "VUS בגן BRCA2 אינו אבחנה ואינו בסיס לקבלת החלטות רפואיות — משמעותו תלויה בנסיבות "
            "הקליניות המלאות. המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "based_on": "MedlinePlus Genetics, NCBI Gene",
        "source_1_name": "MedlinePlus Genetics",
        "source_1_url_or_id": "https://medlineplus.gov/genetics/gene/brca2/",
        "source_2_name": "NCBI Gene",
        "source_2_url_or_id": "https://www.ncbi.nlm.nih.gov/gene/675",
        "generated_by_model": "offline_draft_script_v1",
        "created_from": "offline_draft_script",
    },
    {
        "gene_symbol": "APC",
        "draft_type": "gene_summary",
        "text_he": (
            "גן APC מקודד לחלבון המשמש כמדכא גידולים ומשתתף בבקרה על חלוקת תאים. "
            "שינויים פתוגניים בגן זה מוכרים בספרות כקשורים לתסמונת פוליפוזיס אדנומטית "
            "משפחתית — מצב שבו מתפתחים פוליפים רבים בדופן המעי הגס. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "based_on": "MedlinePlus Genetics, NCBI Gene",
        "source_1_name": "MedlinePlus Genetics",
        "source_1_url_or_id": "https://medlineplus.gov/genetics/gene/apc/",
        "source_2_name": "NCBI Gene",
        "source_2_url_or_id": "https://www.ncbi.nlm.nih.gov/gene/324",
        "generated_by_model": "offline_draft_script_v1",
        "created_from": "offline_draft_script",
    },
    {
        "gene_symbol": "POLE",
        "draft_type": "gene_summary",
        "text_he": (
            "גן POLE מקודד לאנזים המשתתף בשכפול DNA ובתיקון שגיאות שכפול. "
            "שינויים מסוימים בגן זה מוכרים בספרות כקשורים לנטייה תורשתית לסוגי סרטן, "
            "בפרט סרטן מעי גס וסרטן רחם. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "based_on": "MedlinePlus Genetics, NCBI Gene",
        "source_1_name": "MedlinePlus Genetics",
        "source_1_url_or_id": "https://medlineplus.gov/genetics/gene/pole/",
        "source_2_name": "NCBI Gene",
        "source_2_url_or_id": "https://www.ncbi.nlm.nih.gov/gene/5426",
        "generated_by_model": "offline_draft_script_v1",
        "created_from": "offline_draft_script",
    },
    {
        "gene_symbol": "HBB",
        "draft_type": "gene_summary",
        "text_he": (
            "גן HBB מקודד לשרשרת הבטא של המוגלובין — החלבון הנושא חמצן בכדוריות הדם האדומות. "
            "שינויים גנטיים בגן זה מוכרים בספרות כקשורים למצבים כגון אנמיה חרמשית ובטא-תלסמיה. "
            "גן זה רלוונטי לעיתים גם בהקשר של נשאות רצסיבית, ולא רק בהקשר של תחלואה עצמית. "
            "המידע כללי ואינו מחליף ייעוץ רפואי אישי."
        ),
        "based_on": "MedlinePlus Genetics, NCBI Gene",
        "source_1_name": "MedlinePlus Genetics",
        "source_1_url_or_id": "https://medlineplus.gov/genetics/gene/hbb/",
        "source_2_name": "NCBI Gene",
        "source_2_url_or_id": "https://www.ncbi.nlm.nih.gov/gene/3043",
        "generated_by_model": "offline_draft_script_v1",
        "created_from": "offline_draft_script",
    },
]


def main(dry_run: bool = False) -> int:
    print("=" * 60)
    print("Seeding first batch gene review drafts")
    print("THESE ARE DRAFTS ONLY — approved=False for all")
    print("=" * 60)
    print()

    enqueued = 0
    skipped = 0
    errors = 0

    for draft in _DRAFTS:
        gene = draft["gene_symbol"]
        dtype = draft["draft_type"]
        try:
            if dry_run:
                print(f"  DRY-RUN: would enqueue {gene} / {dtype}")
                enqueued += 1
                continue

            record = enqueue_gene_draft_for_review(draft)
            seen = record.get("seen_count", 1)
            if seen > 1:
                print(f"  SKIP (duplicate): {gene} / {dtype} — already in queue (seen {seen}x)")
                skipped += 1
            else:
                print(f"  ENQUEUED: {gene} / {dtype} → draft_id={record['draft_id']}")
                enqueued += 1
        except ValueError as exc:
            print(f"  ERROR: {gene} / {dtype} — {exc}", file=sys.stderr)
            errors += 1

    print()
    print(f"Done. Enqueued: {enqueued}  Skipped (duplicate): {skipped}  Errors: {errors}")
    if errors:
        print("WARN: some drafts could not be enqueued — see errors above.", file=sys.stderr)
        return 1

    if not dry_run:
        print()
        print("Next step: review pending drafts with")
        print("  python scripts/review_drafts.py --list")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed first batch of gene review drafts (drafts only — not approved)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be enqueued without writing to disk",
    )
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
