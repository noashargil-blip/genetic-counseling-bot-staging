"""
app/safety.py

Safety classifier for the post-genetic-counseling assistant.

Runs BEFORE any knowledge-base lookup, ClinVar lookup, or LLM call.

1. contains_identifying_info() — Israeli ID-like numbers, phone numbers,
   email addresses, or phrases like "my name is" / "תעודת זהות שלי".
   If True, the question's content must never be answered or forwarded
   to an LLM/retriever; the caller returns a fixed privacy warning instead.
   This always takes priority over everything else.

2. is_personal_interpretation_request() — questions asking the bot to make
   a personal medical decision / give a personal risk estimate / recommend
   a clinical action (surgery, MRI, treatment, family testing) WITHOUT
   referencing a specific variant identifier. If True, the caller returns a
   fixed redirect to the genetic counselor — no evidence lookup is useful
   here since no variant was named.

3. contains_specific_variant() / extract_variant_query() — detects an
   actual variant identifier (HGVS cDNA/protein notation, rsID). These
   questions are NOT blocked outright: the caller may look up general
   ClinVar evidence and return it together with a safety boundary (see
   app/counseling_engine.py). This check is independent of (2) and is
   evaluated first, since naming a specific variant is a stronger, more
   useful signal than generic risk phrasing.
"""

import re

# ---------------------------------------------------------------------------
# Identifying information
# ---------------------------------------------------------------------------

# Bare 9-digit run (Israeli ID number length). Word boundaries prevent
# matching inside a longer digit run (e.g. a 10-digit phone number).
_ID_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")

# Israeli mobile/landline numbers (with or without separators) and 10-digit
# runs in general, plus international +972 format.
_PHONE_RE = re.compile(
    r"(?:\+972[-\s]?\d{1,2}[-\s]?\d{7})"   # +972-5X-XXXXXXX
    r"|(?:\b0\d{1,2}[-\s]?\d{7}\b)"        # 0X(X)-XXXXXXX  (9-10 digits)
    r"|(?<!\d)\d{10}(?!\d)"                 # bare 10-digit run
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_IDENTIFYING_PHRASES = [
    # Hebrew — name disclosure
    "השם שלי", "שמי הוא", "שמי ",   # "my name is" (with and without הוא)
    "קוראים לי",                      # "they call me"
    # Hebrew — ID / contact
    "תעודת זהות", "ת.ז", 'ת"ז', "מספר זהות", "מספר תעודת הזהות",
    "הטלפון שלי", "מספר הטלפון שלי",
    "האימייל שלי", "המייל שלי", "כתובת המייל שלי",
    # English
    "my id", "my name is", "my phone", "id number", "social security",
    "my email", "my e-mail",
]


def contains_identifying_info(text: str) -> bool:
    """Return True if the text appears to contain identifying information."""
    if _EMAIL_RE.search(text):
        return True
    if _ID_RE.search(text):
        return True
    if _PHONE_RE.search(text):
        return True
    lower = text.lower()
    return any(phrase in lower for phrase in _IDENTIFYING_PHRASES)


# ---------------------------------------------------------------------------
# Specific variant identifiers (HGVS notation, rsID) — NOT a block signal.
# These route to the variant-evidence-summary path in counseling_engine.py
# instead of being refused outright.
# ---------------------------------------------------------------------------

_HGVS_CDNA_RE = re.compile(r"\bc\.[\d_]+[A-Za-z>]+\d*", re.IGNORECASE)
_HGVS_PROTEIN_RE = re.compile(r"\bp\.[A-Za-z]{3}\d+[A-Za-z]*", re.IGNORECASE)
_RSID_RE = re.compile(r"\brs\d{2,}\b", re.IGNORECASE)


def contains_specific_variant(text: str) -> bool:
    """Return True if the text names a specific variant (HGVS notation or rsID)."""
    return bool(
        _HGVS_CDNA_RE.search(text)
        or _HGVS_PROTEIN_RE.search(text)
        or _RSID_RE.search(text)
    )


def extract_variant_query(text: str) -> dict:
    """
    Best-effort extraction of identifiers usable for a ClinVar lookup.
    Returns a dict suitable for retriever.match_uploaded_variant(), with
    only the keys that were actually found (e.g. {"rsid": "rs80357906"}).
    """
    query: dict = {}
    m = _RSID_RE.search(text)
    if m:
        query["rsid"] = m.group(0)
    m = _HGVS_CDNA_RE.search(text)
    if m:
        query["variant"] = m.group(0)
    m = _HGVS_PROTEIN_RE.search(text)
    if m:
        query["protein_change"] = m.group(0)
    return query


# ---------------------------------------------------------------------------
# Personal medical interpretation / action requests (no specific variant)
# ---------------------------------------------------------------------------

_PERSONAL_PHRASES = [
    # Hebrew — personal variant / finding references
    "הווריאנט שלי", "הוריאנט שלי",    # both common spellings of "my variant"
    "התוצאה שלי", "הממצא שלי",         # "my result / my finding"
    "הפרוגנוזה שלי",                   # "my prognosis"

    # Hebrew — personal risk and diagnosis
    "מסוכן לי", "האם זה מסוכן", "מה הסיכון שלי", "מה הסיכוי שלי",
    "כמה סיכוי יש לי", "כמה סיכוי יש לי לחלות", "כמה סיכויים יש לי",
    "אני חולה", "האם אני חולה",        # "I am sick / am I sick"
    "הילדים שלי יהיו חולים", "ילדיי יהיו חולים",
    "יש לי מחלה", "האם יש לי", "האם זה אומר שיש לי",

    # Hebrew — personal medical action requests
    "אני צריכה לעשות בדיקה", "אני צריך לעשות בדיקה",
    "אני צריכה כריתה", "אני צריך כריתה",
    "אני צריכה טיפול", "אני צריך טיפול",
    "אני צריכה מעקב", "אני צריך מעקב",
    "אני צריכה ניתוח", "אני צריך ניתוח", "צריכה ניתוח", "צריך ניתוח",
    "לעשות ניתוח",                      # "to have surgery" (covers "האם לעשות ניתוח?")
    "אני צריכה mri", "אני צריך mri", "צריכה mri", "צריך mri",
    "לעשות mri",                        # "to do MRI" (covers "לעשות MRI מעקב")
    "איזה טיפול לעשות", "איזה טיפול",
    "איזה תרופה",                       # "which medication"
    "כימותרפיה מתאימה", "כימותרפיה מתאים",  # "chemo suits me"

    # Hebrew — interpretation and phrasing bypass requests
    "תפרש לי", "תפרשי לי", "תסביר לי את התוצאה שלי", "תסבירי לי את התוצאה שלי",

    # English — personal variant / result references
    "my variant", "my result",
    "interpret my variant", "interpret my result", "what does my variant mean",
    "is my variant dangerous",

    # English — personal risk and diagnosis
    "what's my risk", "what is my risk", "what is my risk of", "my risk of",
    "my personal risk",                 # "my personal risk of cancer"
    "my prognosis",
    "do i have a disease",

    # English — personal medical action requests
    "should i get tested", "should i have surgery", "should i have a",
    "mastectomy",                       # always a surgical action request
]


def is_personal_interpretation_request(text: str) -> bool:
    """
    Return True if the text asks the bot to interpret the user's own
    result, estimate personal risk, or give a personal medical-action
    recommendation — WITHOUT naming a specific variant.

    Deliberately does not check for HGVS/rsID here: a question that names
    a specific variant is handled separately (see contains_specific_variant
    and app/counseling_engine.py), which may still provide a general
    evidence summary instead of an outright refusal.
    """
    lower = text.lower()
    return any(phrase in lower for phrase in _PERSONAL_PHRASES)
