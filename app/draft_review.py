# -*- coding: utf-8 -*-
"""
app/draft_review.py

Safe enqueue helper for the physician/genetic-counselor review workflow.

Drafts accepted here are reusable gene-level knowledge records — NOT one-off
patient chat answers.  Accepted draft types:

  gene_summary           — curated educational Hebrew summary of a gene
  vus_note               — gene-specific VUS context note
  clinvar_context_summary — summary derived from ClinVar aggregate metadata

Safety invariants (enforced here):
  • Raw patient questions are NEVER saved.
  • Identifying information (ID numbers, emails, names) is rejected.
  • Drafts with approved=True are rejected — only humans approve via CLI.
  • Duplicate gene+type+content records are silently skipped (idempotent).
  • All new records start with review_status="needs_review", approved=False.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_QUEUE_PATH = Path(__file__).parent / "data" / "draft_review_queue.json"

ALLOWED_DRAFT_TYPES = frozenset({
    "gene_summary",
    "vus_note",
    "clinvar_context_summary",
})

ALLOWED_CREATED_FROM = frozenset({
    "unverified_gene_draft",
    "offline_draft_script",
    "manual",
})

# ---------------------------------------------------------------------------
# Identifying-information guard (mirrors safety.py, but standalone)
# ---------------------------------------------------------------------------

_IDENTIFYING_RE = re.compile(
    r"(?<!\d)\d{9}(?!\d)"             # Israeli ID number (9 digits)
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"  # email
    r"|השם שלי|קוראים לי|שמי הוא"    # name disclosure
    r"|תעודת זהות|ת\.ז|מספר זהות"    # ID phrases
    r"|my name is|my id",
    re.IGNORECASE,
)

_MIN_TEXT_LENGTH = 30
_MAX_TEXT_LENGTH = 8000


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(text: str) -> str:
    """16-char SHA-256 prefix of normalized text — used for deduplication."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _sanitize_text(text: str) -> str:
    """Strip leading/trailing whitespace and normalize internal whitespace runs."""
    return " ".join(text.split())


def _load_queue() -> list:
    if not _QUEUE_PATH.exists():
        return []
    try:
        return json.loads(_QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("draft_review: failed to load queue: %s", exc)
        return []


def _save_queue(queue: list) -> None:
    _QUEUE_PATH.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_gene_draft_for_review(
    draft: dict,
    source_context: Optional[dict] = None,
) -> dict:
    """
    Safely enqueue a gene-level draft for physician/counselor review.

    Parameters
    ----------
    draft : dict
        A draft dict as produced by _generate_unverified_gene_draft() or
        assembled offline.  Required fields: gene_symbol, draft_type, text_he.
        Optional: generated_by_model, based_on, source_note_he, generated_at,
                  source_1_name, source_1_url_or_id, source_2_name, source_2_url_or_id,
                  created_from.

    source_context : dict | None
        Optional ClinVar metadata or other context dict.  Stored as
        source_context_summary (first 500 chars of repr) for audit purposes.
        Raw patient questions must NOT be passed here.

    Returns
    -------
    dict
        The newly created queue record (already persisted), or the existing
        record if this draft was already queued (idempotent).

    Raises
    ------
    ValueError
        If the draft fails any safety or schema check.
    """
    # ── 1. Required field validation ─────────────────────────────────────────
    gene = (draft.get("gene_symbol") or "").strip().upper()
    if not gene:
        raise ValueError("draft must include a non-empty gene_symbol")

    draft_type = (draft.get("draft_type") or "").strip()
    if draft_type not in ALLOWED_DRAFT_TYPES:
        raise ValueError(
            f"draft_type must be one of {sorted(ALLOWED_DRAFT_TYPES)}, got {draft_type!r}"
        )

    text_he = (draft.get("text_he") or "").strip()
    if not text_he:
        raise ValueError("draft text_he must not be empty")
    if len(text_he) < _MIN_TEXT_LENGTH:
        raise ValueError(
            f"draft text_he too short ({len(text_he)} chars, min {_MIN_TEXT_LENGTH})"
        )
    if len(text_he) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"draft text_he too long ({len(text_he)} chars, max {_MAX_TEXT_LENGTH})"
        )

    # ── 2. Identifying-information guard ─────────────────────────────────────
    if _IDENTIFYING_RE.search(text_he):
        raise ValueError(
            "draft text_he contains identifying information and cannot be saved"
        )

    # ── 3. Approved=True guard — drafts must start unapproved ────────────────
    if draft.get("approved") is True:
        raise ValueError(
            "draft.approved must not be True — drafts start unapproved and "
            "are approved only by explicit human action via the review CLI"
        )

    # ── 4. Deduplication by gene + draft_type + content_hash ─────────────────
    chash = _content_hash(text_he)
    queue = _load_queue()
    existing = next(
        (
            r for r in queue
            if r.get("gene_symbol") == gene
            and r.get("draft_type") == draft_type
            and r.get("content_hash") == chash
        ),
        None,
    )
    if existing:
        existing["last_seen_at"] = _now_iso()
        existing["seen_count"] = existing.get("seen_count", 1) + 1
        _save_queue(queue)
        logger.info(
            "draft_review: duplicate %s/%s — updated last_seen_at (id=%s)",
            gene, draft_type, existing.get("draft_id", "?"),
        )
        return existing

    # ── 5. Build the queue record ─────────────────────────────────────────────
    import uuid

    created_from = (draft.get("created_from") or "manual").strip()
    if created_from not in ALLOWED_CREATED_FROM:
        created_from = "manual"

    record: dict = {
        "draft_id": str(uuid.uuid4()),
        "gene_symbol": gene,
        "draft_type": draft_type,
        "text_he": text_he,
        "based_on": (draft.get("based_on") or "unknown").strip(),
        "source_note_he": (draft.get("source_note_he") or "").strip() or None,
        "generated_by_model": (draft.get("generated_by_model") or "unknown").strip(),
        "generated_at": (draft.get("generated_at") or _now_iso()).strip(),
        "created_from": created_from,
        "content_hash": chash,
        # Source provenance
        "source_1_name": draft.get("source_1_name") or None,
        "source_1_url_or_id": draft.get("source_1_url_or_id") or None,
        "source_2_name": draft.get("source_2_name") or None,
        "source_2_url_or_id": draft.get("source_2_url_or_id") or None,
        # Review state — always starts here
        "review_status": "needs_review",
        "approved": False,
        "reviewed_by": None,
        "reviewed_at": None,
        "reviewer_notes": None,
        "original_text_he": None,
        "edited_text_he": None,
        # Timestamps
        "enqueued_at": _now_iso(),
        "last_seen_at": _now_iso(),
        "seen_count": 1,
    }

    # Sanitized source_context summary (no patient data)
    if source_context is not None:
        ctx_repr = repr(source_context)[:500]
        if not _IDENTIFYING_RE.search(ctx_repr):
            record["source_context_summary"] = ctx_repr

    queue.append(record)
    _save_queue(queue)
    logger.info(
        "draft_review: enqueued %s/%s (id=%s)", gene, draft_type, record["draft_id"]
    )
    return record


def list_queue(status_filter: Optional[list] = None) -> list:
    """
    Return queue records filtered by review_status.

    Default filter shows pending records (needs_review, draft).
    Pass status_filter=None to get all records.
    """
    queue = _load_queue()
    if status_filter is None:
        return queue
    return [r for r in queue if r.get("review_status") in status_filter]


def get_by_draft_id(draft_id: str) -> Optional[dict]:
    """Return a single queue record by draft_id, or None."""
    for r in _load_queue():
        if r.get("draft_id") == draft_id:
            return r
    return None


def pending_count() -> int:
    """Return number of drafts awaiting review."""
    return len(list_queue(status_filter=["needs_review", "draft"]))
