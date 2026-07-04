"""
app/gene_cards.py

Gene card loader.

Approved gene cards contain patient-friendly Hebrew educational summaries that
are safe to show directly to users. Only cards with ``approved=true`` are
returned by the public API.

Schema (gene_cards.json):
  gene_symbol   str   — canonical uppercase symbol (BRCA1, APC, …)
  summary_he    str   — patient-friendly Hebrew educational text
  approved      bool  — only True cards are surfaced in user-facing answers
  reviewed_by   str   — who signed off (e.g. "genetic-counseling-team")
  last_reviewed str   — ISO-8601 date of last review
  sources       list  — citation list (OMIM, ClinVar, GeneReviews, …)
  notes         str   — reviewer notes (not shown to users)

Loading priority:
  1. data/gene_cards.json — editable without redeploying Python
  2. Built-in Python fallback — same content; used when the JSON is absent

The fallback ensures the app works on a fresh clone before the JSON file is
uploaded, and gives a safe baseline for unit tests that don't mount the
data/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in fallback — same content that lived in counseling_engine._GENE_EDUCATION_HE.
# Used when gene_cards.json is absent (fresh deploy, unit-test isolation).
# ---------------------------------------------------------------------------
_BUILTIN_GENE_SUMMARIES: dict[str, str] = {
    "APC": (
        "APC הוא גן שקשור לבקרה על גדילה וחלוקה של תאים. "
        "שינויים מסוימים ב-APC יכולים להיות קשורים למצבים תורשתיים שבהם יש נטייה "
        "לריבוי פוליפים במעי, כמו Familial Adenomatous Polyposis. "
        "חשוב להבדיל בין העובדה שגן מסוים מוכר כקשור למצבים רפואיים מסוימים, "
        "לבין המשמעות של וריאנט אישי שנמצא אצלך. "
        "אם הממצא שסווג אצלך הוא VUS, המשמעות היא שהשינוי המסוים עדיין אינו ברור מספיק, "
        "ולכן לא מתייחסים אליו כמו לממצא פתוגני. "
        "המשמעות האישית תלויה בדוח הבדיקה, בסיפור המשפחתי ובהערכת הצוות הגנטי."
    ),
    "BRCA1": (
        "BRCA1 הוא גן שקשור לתיקון נזקים ב-DNA. "
        "שינויים פתוגניים מסוימים ב-BRCA1 קשורים לנטייה מוגברת לסוגים מסוימים של סרטן. "
        "חשוב להבדיל בין הגן עצמו לבין המשמעות של וריאנט ספציפי שנמצא בבדיקה. "
        "ממצא VUS ב-BRCA1 אינו שקול לממצא פתוגני — "
        "המשמעות של הממצא הספציפי שלך נקבעת על ידי הצוות הגנטי בהתאם לדוח ולהיסטוריה המשפחתית."
    ),
    "BRCA2": (
        "BRCA2 הוא גן הקשור לתיקון נזקים ב-DNA, ומכיל הוראות לייצור חלבון המסייע "
        "בתיקון שבירות בשרשראות ה-DNA. "
        "שינויים פתוגניים ב-BRCA2 קשורים לנטייה מוגברת לסוגים מסוימים של סרטן. "
        "ממצא VUS ב-BRCA2 אינו שקול לממצא פתוגני — "
        "המשמעות נקבעת על ידי הצוות הגנטי."
    ),
    "NF1": (
        "NF1 הוא גן שקשור לבקרה על גדילה תאית. "
        "שינויים פתוגניים ב-NF1 קשורים למחלה תורשתית הנקראת Neurofibromatosis type 1. "
        "ממצא VUS ב-NF1 אינו שקול לממצא פתוגני — "
        "המשמעות של הממצא הספציפי שלך נקבעת על ידי הצוות הגנטי."
    ),
    "TP53": (
        "TP53 הוא גן המכיל הוראות לייצור חלבון הנקרא p53, הממלא תפקיד מרכזי "
        "בפיקוח על גדילה תאית ובמניעת שינויים לא תקינים ב-DNA. "
        "שינויים פתוגניים מסוימים ב-TP53 קשורים לסינדרומים תורשתיים נדירים. "
        "ממצא VUS ב-TP53 אינו שקול לממצא פתוגני — "
        "המשמעות של הממצא הספציפי שלך נקבעת על ידי הצוות הגנטי."
    ),
    "SHANK3": (
        "SHANK3 הוא גן הקשור לתפקוד חיבורים בין תאי עצב במוח. "
        "שינויים בגן זה קשורים לעיתים למצבים המשפיעים על ההתפתחות הנוירולוגית. "
        "המשמעות של ממצא ספציפי ב-SHANK3 תלויה בסוג הממצא ובהקשר הקליני, "
        "ונקבעת על ידי הצוות הגנטי."
    ),
}

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
_CARDS_PATH = Path("data/gene_cards.json")
_APPROVED_CARDS: dict[str, dict] = {}
CARDS_AVAILABLE: bool = False


def _load() -> None:
    global CARDS_AVAILABLE
    try:
        raw = _CARDS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        loaded = 0
        for card in data:
            sym = (card.get("gene_symbol") or "").strip()
            if sym and card.get("approved") and card.get("summary_he"):
                _APPROVED_CARDS[sym] = card
                loaded += 1
        CARDS_AVAILABLE = bool(_APPROVED_CARDS)
        logger.info("gene_cards: loaded %d approved card(s) from %s", loaded, _CARDS_PATH)
    except FileNotFoundError:
        logger.info(
            "gene_cards: %s not found — loading %d built-in card(s)",
            _CARDS_PATH, len(_BUILTIN_GENE_SUMMARIES),
        )
        _load_builtin()
    except Exception as exc:
        logger.warning("gene_cards: failed to parse %s (%s) — loading built-in cards", _CARDS_PATH, exc)
        _load_builtin()


def _load_builtin() -> None:
    global CARDS_AVAILABLE
    for sym, summary in _BUILTIN_GENE_SUMMARIES.items():
        _APPROVED_CARDS[sym] = {
            "gene_symbol": sym,
            "summary_he": summary,
            "approved": True,
            "reviewed_by": "built-in",
            "last_reviewed": "2026-06-28",
            "sources": [],
            "notes": "Loaded from Python built-in fallback (gene_cards.json unavailable).",
        }
    CARDS_AVAILABLE = bool(_APPROVED_CARDS)


_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_approved_card(gene: str) -> Optional[dict]:
    """Return the full approved gene card dict, or None if the gene has no approved card."""
    return _APPROVED_CARDS.get(gene)


def get_approved_summary(gene: str) -> Optional[str]:
    """Return the approved patient-friendly Hebrew summary, or None."""
    card = _APPROVED_CARDS.get(gene)
    return card["summary_he"] if card else None


def has_approved_card(gene: str) -> bool:
    """Return True if the gene has an approved card available."""
    return gene in _APPROVED_CARDS


def list_approved_genes() -> list[str]:
    """Return a sorted list of all genes with approved cards."""
    return sorted(_APPROVED_CARDS.keys())
