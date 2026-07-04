"""
Answer feedback storage.

Records helpful / not-helpful signals from users to logs/feedback.jsonl.
PRIVACY RULE: no question text, no answer text, no user-identifiable information
is ever stored.  Only aggregate-safe signals are kept:
  - was the answer helpful?
  - optional short reason (capped at 200 chars)
  - the matched_topic of the answer (topic classification, not content)
  - the safety_level of the answer
  - character length of the question (not the text)
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FEEDBACK_DIR  = Path("logs")
_FEEDBACK_FILE = _FEEDBACK_DIR / "feedback.jsonl"

# Reasons the user may select; free-text is also accepted (capped below).
PRESET_REASONS = [
    "התשובה לא הייתה רלוונטית",
    "התשובה לא הייתה מספיק ברורה",
    "התשובה לא ענתה על שאלתי",
    "מידע חסר",
    "אחר",
]


def record(
    *,
    helpful: bool,
    reason: Optional[str] = None,
    matched_topic: Optional[str] = None,
    safety_level: Optional[str] = None,
    question_length: Optional[int] = None,
) -> str:
    """
    Append one feedback entry to logs/feedback.jsonl.

    Returns a short feedback_id the client can echo back as confirmation.
    Never raises — logging errors are swallowed so a feedback failure never
    disrupts the user session.
    """
    feedback_id = uuid.uuid4().hex[:8]
    entry = {
        "feedback_id":    feedback_id,
        "timestamp_utc":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "helpful":        bool(helpful),
        "reason":         (reason or "")[:200] or None,
        "matched_topic":  matched_topic or None,
        "safety_level":   safety_level or None,
        "question_length": int(question_length) if question_length is not None else None,
    }

    try:
        _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with _FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Feedback recorded: id=%s helpful=%s topic=%s", feedback_id, helpful, matched_topic)
    except Exception as exc:
        logger.warning("Could not write feedback entry: %s", exc)

    return feedback_id


def read_recent(n: int = 100) -> list:
    """Return the n most recent feedback entries (for admin/debug use only)."""
    if not _FEEDBACK_FILE.exists():
        return []
    lines = _FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in reversed(lines[-n * 2:]):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        if len(entries) >= n:
            break
    return list(reversed(entries))
