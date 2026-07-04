"""
app/kb.py

Loader and lookup helpers for the approved genetic-counseling knowledge base
(app/data/genetic_counseling_kb.json).

This is the single source of truth for what the post-genetic-counseling
assistant is allowed to say. Nothing here interprets a personal test result —
every entry is a general, pre-approved explanation of a concept.

Matching has two tiers:
  1. Exact keyword scoring — a KB entry's keyword must appear verbatim
     (case-insensitive substring) in the question. High confidence.
  2. Fuzzy word-level fallback — only tried when tier 1 finds nothing.
     Tokenizes the question and fuzzy-matches each token against each
     entry's keyword tokens (catches typos / minor phrasing variants).
     Lower confidence, but per product requirement we should not
     immediately fall back to "out of scope" if a plausibly related KB
     entry exists.
"""

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent / "data" / "genetic_counseling_kb.json"

_ENTRIES: list[dict] = []
_BY_ID: dict[str, dict] = {}

# A few broadly useful topics to offer as generic suggestions when nothing
# matches at all (used by the "helpful fallback" message).
_GENERIC_TOPIC_IDS = ["vus", "pathogenic", "carrier", "family_testing"]

_TOKEN_RE = re.compile(r"[A-Za-z֐-׿]+")


def _load() -> None:
    """Load and index the knowledge base once at module import time."""
    global _ENTRIES, _BY_ID
    try:
        with open(_KB_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _ENTRIES = data.get("entries", [])
        _BY_ID = {e["id"]: e for e in _ENTRIES}
        logger.info("Loaded %d genetic-counseling KB entries from %s", len(_ENTRIES), _KB_PATH)
    except Exception as exc:
        logger.error("Failed to load genetic-counseling KB from %s: %s", _KB_PATH, exc)
        _ENTRIES = []
        _BY_ID = {}


_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def all_entries() -> list[dict]:
    """Return all KB entries (full records)."""
    return list(_ENTRIES)


def get_by_id(topic_id: str) -> Optional[dict]:
    """Return a single KB entry by its id, or None if unknown."""
    return _BY_ID.get(topic_id)


def list_topics() -> list[dict]:
    """Return a compact list of topics suitable for GET /topics."""
    return [
        {
            "id": e["id"],
            "topic": e["topic"],
            "title_he": e["title_he"],
            "requires_genetic_counselor": e.get("requires_genetic_counselor", False),
        }
        for e in _ENTRIES
    ]


def list_faq() -> list[dict]:
    """Return FAQ-style entries (no internal keyword list) for GET /faq."""
    return [
        {
            "id": e["id"],
            "topic": e["topic"],
            "title_he": e["title_he"],
            "approved_answer_he": e["approved_answer_he"],
            "suggested_questions": e.get("suggested_questions", []),
            "requires_genetic_counselor": e.get("requires_genetic_counselor", False),
        }
        for e in _ENTRIES
    ]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _exact_match(question: str) -> tuple[Optional[dict], int]:
    """Tier 1: exact keyword substring scoring. Returns (best_entry, score)."""
    q_lower = question.lower()
    best_entry: Optional[dict] = None
    best_score = 0
    for entry in _ENTRIES:
        score = sum(1 for kw in entry.get("keywords", []) if kw.lower() in q_lower)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry, best_score


def _fuzzy_match(question: str, cutoff: float = 0.8) -> Optional[dict]:
    """
    Tier 2: word-level fuzzy fallback, only used when tier 1 scores zero.
    For each word in the question, look for a close (typo-tolerant) match
    among each entry's keyword words; the entry with the most such matches
    wins. Returns None if nothing scores above zero.
    """
    q_tokens = _tokenize(question)
    if not q_tokens:
        return None

    best_entry: Optional[dict] = None
    best_score = 0
    for entry in _ENTRIES:
        kw_tokens: set[str] = set()
        for kw in entry.get("keywords", []):
            kw_tokens.update(_tokenize(kw))
        if not kw_tokens:
            continue
        score = 0
        for qt in q_tokens:
            if len(qt) < 3:
                continue  # too short to fuzzy-match meaningfully
            if difflib.get_close_matches(qt, kw_tokens, n=1, cutoff=cutoff):
                score += 1
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry if best_score > 0 else None


def match_question(question: str, topic_hint: Optional[str] = None) -> Optional[dict]:
    """
    Find the best matching KB entry for a free-text question.

    If topic_hint is a known KB id, it is used directly (the caller has
    already decided which topic this question belongs to, e.g. a UI button).
    Otherwise:
      1. Score entries by exact keyword substring matches (case-insensitive).
         Ties are broken by KB order (first entry defined wins).
      2. If that scores zero everywhere, try a fuzzy word-level fallback so a
         plausibly related entry is still found (typos, minor phrasing
         variants) instead of immediately giving up.
    Returns None only when neither tier finds anything.
    """
    if topic_hint:
        entry = _BY_ID.get(topic_hint.strip())
        if entry:
            return entry
        # Unknown topic hint — fall through to keyword scoring.

    best_entry, best_score = _exact_match(question)
    if best_score > 0:
        return best_entry

    return _fuzzy_match(question)


def suggest_topics(limit: int = 4) -> list[dict]:
    """Generic topic suggestions used to build the 'helpful fallback' message
    when nothing — not even fuzzy matching — relates to the question."""
    picks = [_BY_ID[i] for i in _GENERIC_TOPIC_IDS if i in _BY_ID]
    return picks[:limit]
