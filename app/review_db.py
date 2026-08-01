# -*- coding: utf-8 -*-
"""
app/review_db.py

Persistent storage for the physician draft-review portal (Session 27).

STORAGE NOTICE:
  - SQLite is used locally and as a fallback in production.
  - In production on Render or similar platforms, set DATABASE_URL to a
    persistent PostgreSQL connection string.  Render's local filesystem is
    ephemeral — do NOT rely on SQLite there for production data.
  - No patient PII is stored.  See PRODUCT RULES below.

PRODUCT RULES:
  Only gene-level EDUCATIONAL draft text is stored, never:
    - Patient names, ID numbers, emails, phone numbers
    - Full uploaded genetic reports
    - HGVS variant strings supplied by users
    - Personal medical histories
    - Raw user messages that may contain identifying information

SUPPORTED BACKENDS:
  - SQLite  (default, local dev, tests)
  - PostgreSQL via DATABASE_URL  (production)

DEDUPLICATION:
  A new draft is only created when no existing pending/needs_revision record
  for the same gene + normalized_intent + content_hash already exists.
  Exact duplicates update last_seen_at and seen_count instead of inserting.

STATUSES:
  pending          — generated, not yet reviewed
  approved         — reviewer approved text as-is or with edits
  rejected         — reviewer rejected; never shown as approved
  needs_revision   — reviewer asked for changes; still pending
  superseded       — a newer approved version exists for this gene
"""

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path / URL
# ---------------------------------------------------------------------------

_DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
_DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "data" / "review_drafts.db"

_SQLITE_PATH = Path(
    os.environ.get("REVIEW_DB_SQLITE_PATH", str(_DEFAULT_SQLITE_PATH))
)

_USE_POSTGRES = _DATABASE_URL.startswith("postgres")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# review_drafts table is identical for both backends — only TEXT/INTEGER types
# that both engines handle identically.
_CREATE_DRAFTS = """
CREATE TABLE IF NOT EXISTS review_drafts (
    id                  TEXT PRIMARY KEY,
    draft_type          TEXT NOT NULL,
    gene_symbol         TEXT,
    normalized_intent   TEXT NOT NULL DEFAULT '',
    normalized_query    TEXT,
    original_ai_text    TEXT NOT NULL,
    physician_edited_text TEXT,
    review_status       TEXT NOT NULL DEFAULT 'pending',
    review_comment      TEXT,
    model_provider      TEXT,
    model_name          TEXT,
    prompt_version      TEXT,
    safety_policy_version TEXT,
    source_metadata     TEXT,
    content_hash        TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    reviewed_at         TEXT,
    reviewed_by         TEXT,
    revision            INTEGER NOT NULL DEFAULT 0,
    last_seen_at        TEXT NOT NULL,
    seen_count          INTEGER NOT NULL DEFAULT 1
)
"""

# review_audit primary-key auto-increment differs between engines.
# SQLite: INTEGER PRIMARY KEY AUTOINCREMENT
# PostgreSQL: AUTOINCREMENT is invalid — use BIGSERIAL (or BIGINT GENERATED AS IDENTITY)
_CREATE_AUDIT_SQLITE = """
CREATE TABLE IF NOT EXISTS review_audit (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id            TEXT NOT NULL,
    action              TEXT NOT NULL,
    prev_status         TEXT NOT NULL,
    new_status          TEXT NOT NULL,
    reviewer_identity   TEXT,
    timestamp           TEXT NOT NULL,
    review_comment      TEXT,
    text_was_edited     INTEGER NOT NULL DEFAULT 0,
    revision            INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_AUDIT_PG = """
CREATE TABLE IF NOT EXISTS review_audit (
    id                  BIGSERIAL PRIMARY KEY,
    draft_id            TEXT NOT NULL,
    action              TEXT NOT NULL,
    prev_status         TEXT NOT NULL,
    new_status          TEXT NOT NULL,
    reviewer_identity   TEXT,
    timestamp           TEXT NOT NULL,
    review_comment      TEXT,
    text_was_edited     INTEGER NOT NULL DEFAULT 0,
    revision            INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_drafts_status ON review_drafts (review_status)",
    "CREATE INDEX IF NOT EXISTS idx_drafts_gene ON review_drafts (gene_symbol, normalized_intent)",
    "CREATE INDEX IF NOT EXISTS idx_drafts_hash ON review_drafts (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_drafts_created ON review_drafts (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_draft_id ON review_audit (draft_id)",
]

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


@contextmanager
def _get_connection() -> Generator:
    """
    Yield a DB-API 2.0 connection; commit on success, rollback on error.

    PostgreSQL: connects with psycopg2.extras.RealDictCursor as the default
    cursor factory so that all rows support dict(row) uniformly.

    SQLite: sets row_factory = sqlite3.Row for the same dict(row) semantics.
    """
    if _USE_POSTGRES:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(_DATABASE_URL, cursor_factory=RealDictCursor)
        except Exception as exc:
            logger.error("review_db: PostgreSQL connection failed: %s", type(exc).__name__)
            raise
    else:
        _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_SQLITE_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _placeholder(n: int = 1) -> str:
    """Return %s for PostgreSQL, ? for SQLite — repeated n times."""
    token = "%s" if _USE_POSTGRES else "?"
    return ", ".join([token] * n)


def _ph() -> str:
    return "%s" if _USE_POSTGRES else "?"


def _row_to_dict(row) -> Optional[dict]:
    """
    Convert a DB row to a plain dict, regardless of backend.

    sqlite3.Row:   supports dict(row) via keys() + __iter__
    RealDictRow:   supports dict(row) natively
    """
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

_initialized = False


def init_db() -> bool:
    """
    Create tables and indexes if they don't exist.  Idempotent (IF NOT EXISTS).
    Returns True on success, False on failure.
    Never sets _initialized = True unless the schema commit succeeded.
    """
    global _initialized
    _create_audit = _CREATE_AUDIT_PG if _USE_POSTGRES else _CREATE_AUDIT_SQLITE
    try:
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(_CREATE_DRAFTS)
            cur.execute(_create_audit)
            for idx_sql in _CREATE_INDEXES:
                cur.execute(idx_sql)
        _initialized = True
        logger.info("review_db: schema ready (%s)",
                    "postgres" if _USE_POSTGRES else str(_SQLITE_PATH))
        return True
    except Exception as exc:
        logger.error("review_db: init failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CRUD — drafts
# ---------------------------------------------------------------------------

VALID_STATUSES = frozenset({"pending", "approved", "rejected", "needs_revision", "superseded"})

ALLOWED_TRANSITIONS = {
    "pending": {"approved", "rejected", "needs_revision"},
    "needs_revision": {"approved", "rejected"},
    "approved": {"superseded"},
    "rejected": set(),
    "superseded": set(),
}


def create_draft(
    *,
    draft_type: str,
    original_ai_text: str,
    gene_symbol: Optional[str] = None,
    normalized_intent: str = "",
    normalized_query: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    prompt_version: Optional[str] = None,
    safety_policy_version: Optional[str] = None,
    source_metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Create a new review draft record.

    Returns the created (or existing duplicate) record dict, or None on error.
    Never raises — all errors are caught so the caller's response is unaffected.
    """
    if not original_ai_text or len(original_ai_text.strip()) < 10:
        logger.warning("review_db: draft text too short, skipping")
        return None

    chash = _content_hash(original_ai_text)
    gene = (gene_symbol or "").strip().upper() or None
    now = _now_iso()
    ph = _ph()

    try:
        with _get_connection() as conn:
            cur = conn.cursor()

            # Deduplication: check for existing pending/needs_revision record.
            # Split execute() and fetchone() — psycopg2.cursor.execute() returns
            # None (not the cursor), so chaining .fetchone() on it would fail.
            cur.execute(
                f"""
                SELECT id, seen_count FROM review_drafts
                WHERE content_hash = {ph}
                  AND review_status IN ('pending', 'needs_revision')
                  AND (gene_symbol = {ph} OR ({ph} IS NULL AND gene_symbol IS NULL))
                LIMIT 1
                """,
                (chash, gene, gene),
            )
            existing = cur.fetchone()

            if existing:
                existing_dict = _row_to_dict(existing)
                cur.execute(
                    f"UPDATE review_drafts SET last_seen_at = {ph}, seen_count = {ph} WHERE id = {ph}",
                    (now, existing_dict["seen_count"] + 1, existing_dict["id"]),
                )
                logger.debug("review_db: duplicate draft for gene=%s, updated last_seen_at", gene)
                return get_draft(existing_dict["id"], _cur=cur)

            # Insert new record
            draft_id = str(uuid.uuid4())
            source_json = json.dumps(source_metadata, ensure_ascii=False) if source_metadata else None

            cur.execute(
                f"""
                INSERT INTO review_drafts
                    (id, draft_type, gene_symbol, normalized_intent, normalized_query,
                     original_ai_text, physician_edited_text, review_status, review_comment,
                     model_provider, model_name, prompt_version, safety_policy_version,
                     source_metadata, content_hash, created_at, updated_at,
                     reviewed_at, reviewed_by, revision, last_seen_at, seen_count)
                VALUES ({_placeholder(22)})
                """,
                (
                    draft_id, draft_type, gene, normalized_intent or "", normalized_query,
                    original_ai_text.strip(), None, "pending", None,
                    model_provider, model_name, prompt_version, safety_policy_version,
                    source_json, chash, now, now,
                    None, None, 0, now, 1,
                ),
            )
            logger.info("review_db: created draft %s for gene=%s", draft_id, gene)
            return get_draft(draft_id, _cur=cur)

    except Exception as exc:
        logger.error("review_db: create_draft failed for gene=%s: %s", gene, type(exc).__name__)
        return None


def get_draft(draft_id: str, *, _cur=None) -> Optional[dict]:
    """
    Return a single draft record by id, or None.

    _cur: optional open cursor to reuse within an existing transaction.
          Accepts a cursor (not a connection) so the call works on both
          SQLite and psycopg2 without relying on connection.execute() shortcut.
    """
    ph = _ph()

    def _fetch(cur):
        # Split execute + fetchone — psycopg2 cursor.execute() returns None.
        cur.execute(
            f"SELECT * FROM review_drafts WHERE id = {ph}",
            (draft_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = _row_to_dict(row)
        d["effective_text"] = d.get("physician_edited_text") or d.get("original_ai_text", "")
        d["physician_reviewed"] = d.get("reviewed_by") is not None
        d["physician_approved"] = d.get("review_status") == "approved"
        return d

    if _cur is not None:
        return _fetch(_cur)
    try:
        with _get_connection() as conn:
            cur = conn.cursor()
            return _fetch(cur)
    except Exception as exc:
        logger.error("review_db: get_draft(%s) failed: %s", draft_id, type(exc).__name__)
        return None


def list_drafts(
    *,
    status: Optional[str] = None,
    gene_symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    """
    Return draft records sorted by pending first, then created_at desc.
    status=None returns all statuses.
    """
    try:
        with _get_connection() as conn:
            ph = _ph()
            params: list = []
            where_clauses: list = []

            if status:
                where_clauses.append(f"review_status = {ph}")
                params.append(status)
            if gene_symbol:
                where_clauses.append(f"gene_symbol = {ph}")
                params.append(gene_symbol.upper())

            where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            # Use explicit cursor — psycopg2 Connection has no .execute() shortcut.
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT * FROM review_drafts
                {where}
                ORDER BY
                    CASE review_status WHEN 'pending' THEN 0 WHEN 'needs_revision' THEN 1 ELSE 2 END,
                    created_at DESC
                LIMIT {ph} OFFSET {ph}
                """,
                params + [limit, offset],
            )
            rows = cur.fetchall()

            result = []
            for row in rows:
                d = _row_to_dict(row)
                d["effective_text"] = d.get("physician_edited_text") or d.get("original_ai_text", "")
                d["physician_reviewed"] = d.get("reviewed_by") is not None
                d["physician_approved"] = d.get("review_status") == "approved"
                result.append(d)
            return result
    except Exception as exc:
        logger.error("review_db: list_drafts failed: %s", type(exc).__name__)
        return []


def update_draft_status(
    draft_id: str,
    *,
    new_status: str,
    reviewer_identity: str,
    review_comment: Optional[str] = None,
    physician_edited_text: Optional[str] = None,
) -> Optional[dict]:
    """
    Apply a status transition.  Returns updated record or None on error.
    Raises ValueError for invalid transitions or unknown drafts.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {new_status!r}")

    try:
        with _get_connection() as conn:
            ph = _ph()
            # Use explicit cursor — psycopg2 Connection has no .execute() shortcut.
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM review_drafts WHERE id = {ph}",
                (draft_id,),
            )
            row = cur.fetchone()

            if row is None:
                raise ValueError(f"Draft not found: {draft_id!r}")

            record = _row_to_dict(row)
            prev_status = record["review_status"]

            if new_status not in ALLOWED_TRANSITIONS.get(prev_status, set()):
                raise ValueError(
                    f"Invalid transition: {prev_status!r} → {new_status!r}"
                )

            now = _now_iso()
            new_revision = record["revision"] + 1
            text_was_edited = (
                physician_edited_text is not None
                and physician_edited_text.strip() != ""
            )

            # Build UPDATE without CASE WHEN integer — PostgreSQL requires
            # boolean operands in CASE WHEN, not integers.
            if text_was_edited:
                cur.execute(
                    f"""
                    UPDATE review_drafts SET
                        review_status = {ph},
                        review_comment = {ph},
                        physician_edited_text = {ph},
                        reviewed_at = {ph},
                        reviewed_by = {ph},
                        revision = {ph},
                        updated_at = {ph}
                    WHERE id = {ph}
                    """,
                    (
                        new_status, review_comment,
                        physician_edited_text,
                        now, reviewer_identity, new_revision, now,
                        draft_id,
                    ),
                )
            else:
                cur.execute(
                    f"""
                    UPDATE review_drafts SET
                        review_status = {ph},
                        review_comment = {ph},
                        reviewed_at = {ph},
                        reviewed_by = {ph},
                        revision = {ph},
                        updated_at = {ph}
                    WHERE id = {ph}
                    """,
                    (
                        new_status, review_comment,
                        now, reviewer_identity, new_revision, now,
                        draft_id,
                    ),
                )

            # Audit record
            cur.execute(
                f"""
                INSERT INTO review_audit
                    (draft_id, action, prev_status, new_status, reviewer_identity,
                     timestamp, review_comment, text_was_edited, revision)
                VALUES ({_placeholder(9)})
                """,
                (
                    draft_id, new_status, prev_status, new_status, reviewer_identity,
                    now, review_comment, 1 if text_was_edited else 0, new_revision,
                ),
            )

            return get_draft(draft_id, _cur=cur)

    except ValueError:
        raise
    except Exception as exc:
        logger.error("review_db: update_draft_status(%s) failed: %s", draft_id, type(exc).__name__)
        return None


def get_audit_trail(draft_id: str) -> List[dict]:
    """Return all audit events for a draft, oldest first."""
    try:
        ph = _ph()
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM review_audit WHERE draft_id = {ph} ORDER BY id ASC",
                (draft_id,),
            )
            rows = cur.fetchall()
            return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("review_db: get_audit_trail(%s) failed: %s", draft_id, type(exc).__name__)
        return []


def pending_count() -> int:
    """Return number of drafts awaiting review."""
    try:
        with _get_connection() as conn:
            cur = conn.cursor()
            # Alias the aggregate so both backends expose it by name, not
            # by index.  psycopg2 RealDictCursor returns {'cnt': N};
            # sqlite3.Row also supports key access as row["cnt"].
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM review_drafts"
                " WHERE review_status IN ('pending', 'needs_revision')"
            )
            row = cur.fetchone()
            return int(_row_to_dict(row)["cnt"]) if row else 0
    except Exception:
        return 0


def get_approved_draft(gene_symbol: str, draft_type: str = "gene_summary") -> Optional[dict]:
    """
    Return the most recently approved draft for a gene+type, or None.
    Used by the visibility-mode logic to show only approved content.
    """
    try:
        with _get_connection() as conn:
            ph = _ph()
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT * FROM review_drafts
                WHERE gene_symbol = {ph}
                  AND draft_type = {ph}
                  AND review_status = 'approved'
                ORDER BY reviewed_at DESC
                LIMIT 1
                """,
                (gene_symbol.upper(), draft_type),
            )
            row = cur.fetchone()
            if row is None:
                return None
            d = _row_to_dict(row)
            d["effective_text"] = d.get("physician_edited_text") or d.get("original_ai_text", "")
            d["physician_reviewed"] = True
            d["physician_approved"] = True
            return d
    except Exception as exc:
        logger.error("review_db: get_approved_draft(%s) failed: %s", gene_symbol, type(exc).__name__)
        return None


def get_approved_chromosome_draft(normalized_intent: str) -> Optional[dict]:
    """
    Return the most recently approved chromosome_education draft for the given
    normalized intent, or None.

    Unlike get_approved_draft(), this queries by normalized_intent (not
    gene_symbol), because chromosome education drafts are not gene-specific.
    """
    try:
        with _get_connection() as conn:
            ph = _ph()
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT * FROM review_drafts
                WHERE draft_type = {ph}
                  AND normalized_intent = {ph}
                  AND review_status = 'approved'
                ORDER BY reviewed_at DESC
                LIMIT 1
                """,
                ("chromosome_education", normalized_intent),
            )
            row = cur.fetchone()
            if row is None:
                return None
            d = _row_to_dict(row)
            d["effective_text"] = d.get("physician_edited_text") or d.get("original_ai_text", "")
            d["physician_reviewed"] = True
            d["physician_approved"] = True
            return d
    except Exception as exc:
        logger.error(
            "review_db: get_approved_chromosome_draft(%s) failed: %s",
            normalized_intent, type(exc).__name__,
        )
        return None
