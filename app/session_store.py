"""
app/session_store.py

Minimal persistent session store backed by SQLite.

STORAGE NOTICE — READ BEFORE DEPLOYING:
  Sessions are stored in a local SQLite database at data/sessions.db.
  Each session record contains the analyzed variant data derived from an
  uploaded genetic test file.

  RAW FILE BYTES ARE NEVER STORED.
  Filenames are stored as metadata only.

  THIS IS NOT PRODUCTION-GRADE PHI STORAGE.
  A production deployment handling patient genetic data requires:
    - Encrypted storage at rest (e.g. SQLCipher, filesystem encryption)
    - Encrypted transport (HTTPS only, HSTS)
    - Access controls and audit logging
    - Automatic session expiry and secure deletion
    - Compliance with applicable regulations (HIPAA, GDPR, etc.)
  Do not use this module as-is in any environment that may process
  protected health information (PHI).

SQLite schema (table: sessions):
  session_id  TEXT PRIMARY KEY  — UUID string
  created_at  TEXT NOT NULL     — ISO-8601 UTC timestamp
  filename    TEXT NOT NULL     — original uploaded filename (metadata, no bytes)
  variants    TEXT NOT NULL     — JSON-encoded list[{uploaded_variant, clinvar_result}]

Fallback:
  If SQLite is unavailable at startup or at write time, all operations fall
  back to an in-memory dict. The API continues to work; sessions are lost
  on server restart in that case.

# TODO: Add automatic session cleanup.
#       Suggested: DELETE FROM sessions WHERE created_at < datetime('now', '-24 hours')
#       Run on startup and/or as a periodic background task (e.g. APScheduler).
#       Use a shorter TTL (e.g. 1 hour) for shared or demo environments.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/sessions.db")

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename   TEXT NOT NULL,
    variants   TEXT NOT NULL
)
"""

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_sqlite_ok: bool = False
_MEMORY_FALLBACK: dict[str, list[dict]] = {}


def _init_sqlite() -> bool:
    """Create the sessions table if it does not exist. Returns True on success."""
    try:
        con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        con.execute(_CREATE_TABLE)
        con.commit()
        con.close()
        logger.info("SQLite session store ready at %s", _DB_PATH)
        return True
    except Exception as exc:
        logger.warning(
            "SQLite session store unavailable (%s). "
            "Sessions will be kept in memory only and lost on restart.",
            exc,
        )
        return False


_sqlite_ok = _init_sqlite()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_session_id() -> str:
    """Return a new random session UUID."""
    return str(uuid.uuid4())


def save_session(session_id: str, filename: str, variants: list[dict]) -> bool:
    """
    Persist a session. Always returns True.

    On SQLite failure, writes to in-memory fallback so the caller's response
    is never affected by storage errors.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    variants_json = json.dumps(variants, ensure_ascii=False)

    if _sqlite_ok:
        try:
            con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
            con.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, created_at, filename, variants)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, created_at, filename, variants_json),
            )
            con.commit()
            con.close()
            logger.debug("Session %s persisted to SQLite.", session_id)
            return True
        except Exception as exc:
            logger.warning(
                "SQLite write failed for session %s (%s). Falling back to memory.",
                session_id,
                exc,
            )

    _MEMORY_FALLBACK[session_id] = variants
    logger.debug("Session %s stored in memory fallback.", session_id)
    return True


def get_session(session_id: str) -> Optional[list[dict]]:
    """
    Return the variant list for a session, or None if not found.

    Checks SQLite first; falls back to the in-memory dict on failure.
    """
    if _sqlite_ok:
        try:
            con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
            row = con.execute(
                "SELECT variants FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            con.close()
            if row:
                return json.loads(row[0])
        except Exception as exc:
            logger.warning(
                "SQLite read failed for session %s (%s). Checking memory fallback.",
                session_id,
                exc,
            )

    return _MEMORY_FALLBACK.get(session_id)
