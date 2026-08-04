"""
app/counseling_engine.py

Core logic for the Hebrew post-genetic-counseling assistant's POST /ask
endpoint.

Pipeline:
  1. Safety check — identifying information (blocks immediately, no LLM/KB/
     ClinVar lookup). Always takes priority over everything else.
  2. Specific-variant handling — if the question names an actual variant
     (HGVS cDNA/protein notation or rsID), look up general ClinVar evidence
     and return it together with a clear safety boundary. This is NOT a
     refusal: the bot may summarize publicly-available evidence, it just
     never says whether the variant is personally dangerous/benign for the
     user, and never gives a personal risk estimate or treatment advice.
  3. Personal medical interpretation / action request (no specific variant
     named) — fixed redirect, no KB/LLM/ClinVar lookup. Covers things like
     "should I get an MRI", "should my children be tested", etc.
  4. Gene-name + VUS handling — general "I got a VUS in gene X" questions
     (typo-tolerant gene detection), always general_information.
  5. Follow-up handling — short, vague continuation phrases ("can you
     elaborate?", "what does that mean?", "give me an example") are
     resolved using last_topic / conversation_context (sent by the
     frontend's in-memory session state — never persisted server-side)
     instead of being scored against the KB, where they would always
     score zero and fall back to "out of scope".
  6. Knowledge-base lookup — exact keyword scoring, with a fuzzy word-level
     fallback tier inside kb.match_question() so a plausibly related topic
     is still found instead of immediately giving up.
  7. If LOCAL_LLM_URL is configured, ask the LLM to phrase the answer using
     ONLY the matched KB content (or, for the variant-evidence/follow-up
     paths, ONLY the retrieved structured evidence / KB snippets). On any
     failure, fall back to the deterministic text.
  8. If nothing matches at all, return a helpful (not harsh) fallback that
     explains the scope and offers a few general topic suggestions —
     never an outright invented answer.

No chat history is stored anywhere in this module or the API layer above
it. conversation_context is supplied fresh by the caller on every request
(the frontend's in-memory browser session) and is never written to disk or
a database. No identifying information is ever forwarded to the LLM, the
ClinVar retriever, or used to resolve follow-up context.
"""

import logging
import os
import re
import traceback as _traceback
from typing import NamedTuple, Optional

from app import kb, retriever, safety, gene_index, gene_cards, gene_knowledge
from app.llm_client import LocalLLMClient, LLMClientError, create_llm_client

logger = logging.getLogger(__name__)

# AI_DRAFT_VISIBILITY_MODE controls what patients see when a gene has an AI draft.
#   "immediate"     — show pending (unreviewed) drafts with an "unverified" label.
#   "approved_only" — only show drafts that a physician has approved.
# The default is "immediate" so the system is useful before any reviews are done.
_AI_DRAFT_VISIBILITY_MODE = os.environ.get(
    "AI_DRAFT_VISIBILITY_MODE", "immediate"
).strip().lower()

# ---------------------------------------------------------------------------
# Gene-name + VUS handling (general education only — never a personal
# interpretation; this only fires for gene-name mentions WITHOUT a specific
# variant identifier — see _build_variant_evidence_answer below for the
# specific-variant path, which takes priority).
# ---------------------------------------------------------------------------

_VUS_TOKEN_RE = re.compile(r"\bvus\b", re.IGNORECASE)

# Tolerates common typos: "BRCA1"/"BRCA 1"/"BRACA1"/"BRACA 1"/"braca-1", etc.
_GENE_PATTERNS = [
    ("BRCA1",  re.compile(r"\bbra?ca[-\s]?1\b",  re.IGNORECASE)),
    ("BRCA2",  re.compile(r"\bbra?ca[-\s]?2\b",  re.IGNORECASE)),
    ("NF1",    re.compile(r"\bnf[-\s]?1\b",      re.IGNORECASE)),
    ("APC",    re.compile(r"\bapc\b",             re.IGNORECASE)),
    ("TP53",   re.compile(r"\btp53\b",            re.IGNORECASE)),
    ("SHANK3", re.compile(r"\bshank[-\s]?3\b",   re.IGNORECASE)),
    # HBB: handles plain "HBB" and Hebrew-prefix forms "בHBB", "ב-HBB" (no word boundary at Hebrew-Latin)
    ("HBB",    re.compile(r"(?<![A-Za-z])hbb(?![A-Za-z])", re.IGNORECASE)),
    # POLE: also recognises alias forms Pol-E / Pol E / Pole-E / Pole E / פול אי
    ("POLE",   re.compile(
        r"\bpole\b"           # POLE, pole
        r"|\bpol[-\s]+e\b"   # Pol-E, POL-E, pol-e, Pol E, POL E
        r"|\bpole[-\s]+e\b"  # Pole-E, POLE-E, pole-e, Pole E
        r"|\u05e4\u05d5\u05dc\s+\u05d0\u05d9",  # פול אי
        re.IGNORECASE,
    )),
]

VUS_KNOWN_GENE_TEMPLATE_HE = (
    "כאשר מתקבל VUS בגן {gene}, המשמעות היא שזוהה שינוי גנטי, אך עדיין "
    "אין מספיק ראיות מדעיות כדי לקבוע אם הוא pathogenic או benign. "
    "לכן אין להסיק מסקנות אישיות או לקבל החלטות רפואיות רק על סמך VUS."
)


def _compose_vus_practical_answer(gene: Optional[str]) -> str:
    """
    Compose a concise, conversational Hebrew follow-up answer about VUS.

    Called when the user asks a vague follow-up ("מה כדאי לעשות?",
    "מה ההשלכות?", "תסביר יותר") after an initial VUS answer — so this
    answer deliberately does NOT repeat the basic VUS definition.  Instead
    it opens practically ("בפועל...") and covers three points:

    1. Practical meaning: document it; not pathogenic; usually not the
       sole basis for medical decisions.
    2. Team considerations: clinical picture, family history, and the
       possibility of reclassification over time.
    3. Four suggested questions for the genetics team.

    Safety: no diagnosis, no personal risk, no treatment recommendation,
    no statement that the variant is dangerous or benign for this user.
    Target length: ~120–220 words, patient-friendly Hebrew.
    """
    # Paragraph 1 — practical opening, gene named once naturally
    if gene:
        p1 = (
            f"בפועל, VUS ב-{gene} אומר שיש ממצא שצריך לשמור בתיעוד, "
            f"אבל הוא עדיין לא ברור מספיק כדי להתייחס אליו כמו לממצא pathogenic. "
            f"לכן בדרך כלל לא מקבלים החלטות רפואיות רק על סמך VUS עצמו."
        )
    else:
        p1 = (
            "בפועל, VUS הוא ממצא שצריך לשמור בתיעוד, "
            "אבל הוא עדיין לא ברור מספיק כדי להתייחס אליו כמו לממצא pathogenic. "
            "לכן בדרך כלל לא מקבלים החלטות רפואיות רק על סמך VUS עצמו."
        )

    # Paragraph 2 — team considerations and reclassification possibility
    p2 = (
        "מה שכן חשוב הוא לברר עם הצוות הגנטי האם יש משהו בתמונה הקלינית "
        "או המשפחתית שמצריך התייחסות בלי קשר ל-VUS, "
        "והאם קיימת תוכנית לבדיקה חוזרת של הסיווג בעתיד — "
        "סיווג VUS עשוי להשתנות ככל שמצטברות ראיות מדעיות חדשות."
    )

    # Three patient-friendly questions for the genetics team — no ClinVar jargon
    questions = (
        "שאלות לצוות הגנטי:\n"
        "• האם יש ממצא קליני או משפחתי שמשנה את ההתייחסות?\n"
        "• האם הסיווג צפוי להתעדכן בעתיד?\n"
        "• האם הממצא אומר שיש לי מחלה, או שמשמעותו עדיין לא ידועה?"
    )

    return "\n\n".join([p1, p2, questions])


def _mentions_vus(text: str) -> bool:
    return bool(_VUS_TOKEN_RE.search(text)) or "וריאנט לא ידוע" in text or "משמעות לא ידועה" in text


def _detect_known_gene(text: str) -> Optional[str]:
    """Return the canonical gene symbol (e.g. 'BRCA1') if mentioned, typo-tolerant."""
    for canonical, pattern in _GENE_PATTERNS:
        if pattern.search(text):
            return canonical
    return None


# ---------------------------------------------------------------------------
# Shared persistence helper for patient-visible medical AI expansions (27.9.3)
# ---------------------------------------------------------------------------

def _persist_reviewable_ai_expansion(
    draft: dict,
    draft_type: str,
    normalized_intent: str,
    gene_symbol: "Optional[str]" = None,
    normalized_query: "Optional[str]" = None,
    source_metadata: "Optional[dict]" = None,
    model_provider: "Optional[str]" = None,
    prompt_version: str = "s2793",
) -> "Optional[dict]":
    """
    Persist a patient-visible medical AI expansion to the physician review queue.

    Returns the review_db record dict on success (new or existing duplicate),
    None on failure. Never raises — patient response must never be affected.

    Deduplication is handled inside review_db.create_draft():
    - Same content_hash + gene_symbol that is pending/needs_revision → update
      last_seen_at/seen_count and return the existing record.
    - Approved record found separately by the caller via get_approved_draft().

    Must NOT be called for GENERAL_LOW_RISK_AI or SOURCE_GROUNDED_AI expansions.
    """
    text_he = (draft or {}).get("text_he", "")
    if not text_he or len(text_he.strip()) < 10:
        logger.debug(
            "_persist_reviewable_ai_expansion: draft text too short, skip (%s / %s)",
            draft_type, gene_symbol,
        )
        return None
    try:
        from app import review_db as _rdb
        record = _rdb.create_draft(
            draft_type=draft_type,
            original_ai_text=text_he,
            gene_symbol=gene_symbol,
            normalized_intent=normalized_intent,
            normalized_query=normalized_query,
            model_provider=model_provider or draft.get("generated_by_provider"),
            model_name=draft.get("generated_by_model"),
            prompt_version=prompt_version,
            source_metadata=source_metadata,
        )
        if record:
            logger.debug(
                "_persist_reviewable_ai_expansion: persisted %s for %s, id=%s, status=%s",
                draft_type, gene_symbol, record.get("id"), record.get("review_status"),
            )
        return record
    except Exception as exc:
        logger.warning(
            "_persist_reviewable_ai_expansion: failed for %s/%s: %s",
            draft_type, gene_symbol, type(exc).__name__,
        )
        return None


def _attach_review_metadata(result: dict, record: "Optional[dict]", draft_key: str = "unverified_gene_draft") -> None:
    """
    Attach physician-review metadata from a review_db record to the API response.

    Sets top-level review_draft_id / review_status, and mirrors them into the
    draft sub-object (``draft_key``) so the frontend can read either location.
    When ``record`` is None (persistence failed), marks review_persistence_failed=True.
    Never mutates the record dict.
    """
    if record:
        result["review_draft_id"] = record.get("id")
        result["review_status"] = record.get("review_status")
        result["physician_reviewed"] = record.get("physician_reviewed", False)
        result["physician_approved"] = record.get("physician_approved", False)
        result["revision"] = record.get("revision", 0)
        if draft_key in result and isinstance(result[draft_key], dict):
            result[draft_key]["review_draft_id"] = record.get("id")
            result[draft_key]["review_status"] = record.get("review_status")
            result[draft_key]["physician_reviewed"] = record.get("physician_reviewed", False)
            result[draft_key]["physician_approved"] = record.get("physician_approved", False)
    else:
        result["review_persistence_failed"] = True


_PATHOGENIC_SIG_KEYS = frozenset({
    "pathogenic", "likely pathogenic", "pathogenic/likely pathogenic",
    "pathogenic/likely_pathogenic",
})
_VUS_SIG_KEYS = frozenset({
    "uncertain significance", "uncertain_significance", "vus",
    "variant of uncertain significance",
})
_BENIGN_SIG_KEYS = frozenset({
    "benign", "likely benign", "benign/likely benign",
    "benign/likely_benign",
})
_CLINVAR_NOTE_TRIVIAL_PHENOS = frozenset({
    "not specified", "not provided", "see cases", "not applicable",
    "all disease", "disease", "", "none provided", "not determined",
})


def _clean_clinvar_phenotypes(raw_phenotypes: list) -> list:
    """
    Clean raw ClinVar phenotype strings for patient-facing display.

    ClinVar often concatenates multiple conditions with semicolons in a single
    phenotype entry. This helper splits them, strips whitespace, deduplicates
    (case-insensitive), and removes trivial placeholder strings.
    """
    seen: set = set()
    cleaned: list = []
    for p in (raw_phenotypes or []):
        for part in p.split(";"):
            part = part.strip()
            if not part or len(part) <= 4:
                continue
            key = part.lower()
            if key in _CLINVAR_NOTE_TRIVIAL_PHENOS:
                continue
            if key not in seen:
                seen.add(key)
                cleaned.append(part)
    return cleaned


def _format_vus_significance_note(by_sig: dict) -> str:
    """Return a short Hebrew sentence describing which significance categories are present."""
    if not by_sig:
        return ""
    has_path = any(
        k.lower().strip() in _PATHOGENIC_SIG_KEYS or (
            "pathogenic" in k.lower() and "benign" not in k.lower()
        )
        for k, v in by_sig.items() if v
    )
    has_vus = any(
        k.lower().strip() in _VUS_SIG_KEYS or "uncertain" in k.lower()
        for k, v in by_sig.items() if v
    )
    has_benign = any(
        k.lower().strip() in _BENIGN_SIG_KEYS or (
            "benign" in k.lower() and "pathogenic" not in k.lower()
        )
        for k, v in by_sig.items() if v
    )
    categories = []
    if has_vus:
        categories.append("VUS")
    if has_benign:
        categories.append("likely benign")
    if has_path:
        categories.append("pathogenic")
    if not categories:
        return ""
    if len(categories) == 1:
        return f"הרשומות כוללות ממצאים שסווגו כ-{categories[0]}."
    if len(categories) == 2:
        return f"הרשומות כוללות ממצאים שסווגו כ-{categories[0]} ו-{categories[1]}."
    return f"הרשומות כוללות ממצאים שסווגו כ-{categories[0]}, {categories[1]} ו-{categories[2]}."


def _build_vus_clinvar_gene_note(gene: str, g_summary: Optional[dict]) -> Optional[str]:
    """
    Build a deterministic source-grounded Hebrew note about a gene's ClinVar profile.

    Uses ONLY trusted structured fields: total_variants, by_significance, phenotypes.
    Phenotype strings are cleaned via _clean_clinvar_phenotypes (splits semicolons,
    deduplicates, removes trivial entries).
    Does NOT invent biological mechanisms or pathway information.
    Returns None when no usable metadata is available (< 10 variants and no phenotypes),
    so the caller can fall through to alternative paths.
    Always appends a disclaimer that this describes DB records, not the user's variant.

    NOTE: This function must only be called for EXPLICIT ClinVar database queries
    (e.g. "כמה וריאנטים יש ב-ACE?").  It must NOT be called for generic VUS+gene
    questions — ClinVar technical data stays in gene_metadata / the technical card.
    """
    if not g_summary:
        return None
    total = g_summary.get("total_variants") or 0
    by_sig = g_summary.get("by_significance") or {}
    cleaned_phenotypes = _clean_clinvar_phenotypes(g_summary.get("phenotypes") or [])
    if total < 10 and not cleaned_phenotypes:
        return None
    sentences = []
    if total >= 10:
        sentences.append(f"במאגר ClinVar קיימות {total:,} רשומות הקשורות לגן {gene}.")
    else:
        sentences.append(f"הגן {gene} מופיע במאגר ClinVar.")
    sig_note = _format_vus_significance_note(by_sig)
    if sig_note:
        sentences.append(sig_note)
    if cleaned_phenotypes:
        if len(cleaned_phenotypes) == 1:
            sentences.append(f"בין המצבים המדווחים במאגר: {cleaned_phenotypes[0]}.")
        else:
            sentences.append(
                f"בין המצבים המדווחים במאגר: {cleaned_phenotypes[0]} ו-{cleaned_phenotypes[1]}."
            )
    sentences.append(
        "מידע זה מתאר את הרשומות במאגר ואינו מפרש את הממצא האישי שלך."
    )
    return " ".join(sentences)


_CLINVAR_EXPLICIT_QUERY_TOKENS = frozenset({
    "clinvar", "כמה וריאנטים", "כמה רשומות", "אילו סיווגים",
    "אילו מצבים", "מה יש במאגר", "ב-clinvar", "במאגר clinvar",
    "וריאנטים ב", "variants ב",
})


def _is_explicit_clinvar_db_query(question: str) -> bool:
    """
    Return True when the question explicitly asks about ClinVar database records
    (counts, classification breakdown, or reported conditions).

    Used inside _build_known_gene_answer() to decide whether to include a
    statistical ClinVar note in the VUS+gene response.  Generic VUS+gene
    questions ("מה המשמעות של VUS בגן ACE?") return False — ClinVar data
    stays in gene_metadata / the technical card for those questions.
    """
    q = question.lower()
    return any(token in q for token in _CLINVAR_EXPLICIT_QUERY_TOKENS)


def _resolve_gene_explanation_for_vus(
    gene: str,
    question: str = "",
    gene_summary: "Optional[dict]" = None,
) -> dict:
    """
    Universal priority chain for resolving a patient-friendly gene explanation.

    Used by _build_known_gene_answer so that BRCA1 and ACE (and every other
    recognized gene) go through the same pipeline — the response shape is
    determined by available content, not by gene identity.

    Priority (highest first):
      1. curated           — gene_cards approved text         → main answer, no review record
      2. grounded          — gene_knowledge KB text           → main answer, no review record
      3. grounded_clinvar  — source-grounded ClinVar summary  → main answer, no review record
                             (same path as standalone gene route; uses phenotypes + LLM)
      4. physician_approved — review_db approved draft        → main answer, no new record
      5. ai_unreviewed     — generate AI biology draft, persist → supplemental card only
      6. none              — LLM unavailable/failed           → VUS-only answer

    Passes ``gene_summary`` (the ClinVar index summary dict) to the grounded_clinvar
    step so it can use phenotype associations — exactly the same evidence used by
    the standalone gene-biology route (_build_gene_clinvar_answer).

    Returns:
      {
        "text_he":         Optional[str],
        "source_type":     "curated"|"grounded"|"grounded_clinvar"|
                           "physician_approved"|"ai_unreviewed"|"none",
        "review_draft_id": Optional[str],
        "review_status":   Optional[str],
        "approved":        Optional[bool],
        "vus_note_he":     Optional[str],  # gene_knowledge VUS addendum (grounded only)
      }
    """
    # 1. Curated gene card (pre-approved, zero review overhead)
    curated = gene_cards.get_approved_summary(gene)
    if curated:
        return {
            "text_he": curated,
            "source_type": "curated",
            "review_draft_id": None,
            "review_status": None,
            "approved": True,
            "vus_note_he": None,
        }

    # 2. Gene Knowledge Base (sourced, approved)
    gk_summary = gene_knowledge.get_gene_patient_summary(gene)
    if gk_summary:
        return {
            "text_he": gk_summary,
            "source_type": "grounded",
            "review_draft_id": None,
            "review_status": None,
            "approved": True,
            "vus_note_he": gene_knowledge.get_gene_vus_note(gene),
        }

    # 3. Source-grounded ClinVar summary — same pipeline as the standalone
    #    gene-biology route (_build_gene_clinvar_answer tier-2 grounded path).
    #    Uses ClinVar phenotype associations (≥3 non-trivial) and/or GK biology
    #    context to produce a short LLM-phrased patient-facing summary.
    #    No physician review required (AI_CONTENT_TYPE_GROUNDED classification).
    _has_grounded, _grounded_ctx = _has_sufficient_grounded_gene_context(gene, gene_summary)
    if _has_grounded:
        _grounded_text = _generate_source_grounded_gene_answer(gene, _grounded_ctx)
        if _grounded_text:
            return {
                "text_he": _grounded_text,
                "source_type": "grounded_clinvar",
                "review_draft_id": None,
                "review_status": None,
                "approved": True,
                "vus_note_he": None,
            }

    # 4. Physician-approved gene summary already in review_db
    try:
        from app import review_db as _rdb_vus
        _approved_rec = _rdb_vus.get_approved_draft(gene, "gene_summary")
    except Exception:
        _approved_rec = None
    if _approved_rec:
        _effective = (_approved_rec.get("effective_text") or "").strip()
        if _effective:
            return {
                "text_he": _effective,
                "source_type": "physician_approved",
                "review_draft_id": _approved_rec.get("id"),
                "review_status": "approved",
                "approved": True,
                "vus_note_he": None,
            }

    # 5. Generate AI gene biology summary, persist to physician review queue
    _ai_draft = _generate_unverified_gene_draft(
        gene,
        question=question,
        clinvar_context=None,
        use_lenient_validator=True,   # biology-function prompt, not ClinVar stats
    )
    if _ai_draft and _ai_draft.get("text_he"):
        _persist_rec = _persist_reviewable_ai_expansion(
            _ai_draft,
            draft_type="gene_summary",
            normalized_intent="vus_known_gene",
            gene_symbol=gene,
            prompt_version="s2796",
        )
        return {
            "text_he": _ai_draft["text_he"],
            "source_type": "ai_unreviewed",
            "review_draft_id": _persist_rec.get("id") if _persist_rec else None,
            "review_status": _persist_rec.get("review_status") if _persist_rec else None,
            "approved": False,
            "vus_note_he": None,
        }

    # 6. Nothing available (LLM not configured or failed)
    return {
        "text_he": None,
        "source_type": "none",
        "review_draft_id": None,
        "review_status": None,
        "approved": None,
        "vus_note_he": None,
    }


def _build_known_gene_answer(gene: str, question: str = "", include_unverified_gene_draft: bool = False) -> dict:
    """
    Build a warm, enriched answer for "I got a VUS in gene X" questions.

    Universal composition (27.9.6):
      1. Resolve gene explanation via _resolve_gene_explanation_for_vus —
         same priority pipeline for every gene (no gene-specific branches).
      2. VUS opening + gene explanation in main answer when source is
         curated / grounded / physician_approved.
      3. VUS opening + VUS-practical paragraph in main answer when source is
         ai_unreviewed (gene explanation goes to supplemental card) or none.
      4. ClinVar technical data stays in gene_metadata / technical card only.
      5. Explicit ClinVar database questions ("כמה וריאנטים יש ב-ACE?") may
         append a cleaned statistical note (27.9.5 rule, unchanged).

    Always general_information.  Safety.py already filtered out personal
    interpretation requests and identifying information before this is called.
    """
    # ClinVar stats for gene_metadata technical card — never in main answer by default
    g_summary = gene_index.get_gene_summary(gene) if gene_index._GENE_INDEX_AVAILABLE else None  # noqa: F821

    # Universal gene-explanation resolution (priority 1→6)
    # Pass g_summary so the grounded_clinvar step can use ClinVar phenotypes —
    # the same evidence used by the standalone gene-biology route.
    gene_expl = _resolve_gene_explanation_for_vus(gene, question=question, gene_summary=g_summary)
    expl_source: str = gene_expl["source_type"]
    expl_text: Optional[str] = gene_expl["text_he"]
    expl_vus_note: Optional[str] = gene_expl.get("vus_note_he")

    # Compose VUS answer
    if expl_source in ("curated", "grounded", "grounded_clinvar", "physician_approved"):
        # Concise VUS opening (gene explanation follows in main answer)
        opening = (
            f"קבלת תוצאה שמציינת VUS בגן {gene} יכולה להיות מבלבלת. "
            "VUS הוא ממצא שמשמעותו עדיין לא ידועה — הוא אינו שקול לממצא פתוגני. "
            "הסיווג עשוי להשתנות בעתיד ככל שמצטברות ראיות מדעיות חדשות."
        )
        parts = [opening, f"לגבי הגן {gene}:\n{expl_text}"]
        if expl_vus_note:
            parts.append(expl_vus_note)
        parts.append(
            "חשוב להבדיל בין התפקיד הכללי של הגן לבין המשמעות של ה-VUS הספציפי שנמצא בבדיקה."
        )
    else:
        # ai_unreviewed or none: fuller VUS paragraph in main answer;
        # gene explanation (if any) appears only in the supplemental AI card.
        opening = (
            f"קבלת תוצאה שמציינת VUS בגן {gene} יכולה להיות מבלבלת. "
            "VUS — Variant of Uncertain Significance — הוא ממצא שמשמעותו עדיין לא ידועה. "
            "חשוב לדעת: זה לא אומר שהשינוי מסוכן, אבל גם לא שהוא בהכרח תקין. "
            "פשוט אין עדיין מספיק ראיות מדעיות כדי לסווג אותו בוודאות."
        )
        vus_practical = (
            f"VUS ב-{gene} אינו מהווה ממצא פתוגני, ובדרך כלל לא מקבלים "
            "החלטות רפואיות על בסיסו בלבד. הסיווג עשוי להשתנות בעתיד "
            "ככל שמצטברות ראיות מדעיות חדשות."
        )
        parts = [opening, vus_practical]

    entry = kb.get_by_id("vus_known_gene")
    suggested = list(entry.get("suggested_questions", [])) if entry else list(_GENE_SUGGESTED_QUESTIONS)  # noqa: F821

    deterministic = "\n\n".join(parts)

    # Build gene_metadata (unified for all source types)
    _tier_map = {
        "curated": "tier1",
        "grounded": "tier1b",
        "grounded_clinvar": "tier2",  # ClinVar-phenotype-grounded, same as standalone tier2
        "physician_approved": "tier2p",
    }
    answer_tier = _tier_map.get(expl_source, "tier2" if g_summary is not None else "tier3")

    _status_map = {
        "curated": "approved",
        "grounded": "approved",
        "grounded_clinvar": "grounded",
        "physician_approved": "physician_approved",
        "ai_unreviewed": "ai_draft_pending",
        "none": "clinvar_index_no_expansion" if g_summary is not None else "missing",
    }
    gene_knowledge_status = _status_map.get(expl_source, "clinvar_index_no_expansion")

    _ds_map = {
        "curated": "Curated educational content" + (" + ClinVar" if g_summary else ""),
        "grounded": "Gene Knowledge Base",
        "grounded_clinvar": "ClinVar (NCBI) via local gene index",
        "physician_approved": "Physician-approved summary",
    }
    data_source = _ds_map.get(expl_source, "ClinVar (NCBI) via local gene index")

    # grounded_clinvar uses the same LLM-phrasing step as the standalone gene route
    _llm_used_for_expl = expl_source == "grounded_clinvar"

    gene_meta: dict = {
        "gene_symbol": gene,
        "data_source": data_source,
        "llm_used": _llm_used_for_expl,
        "fallback_used": not _llm_used_for_expl,
        "total_variants": g_summary.get("total_variants") if g_summary else None,
        "found_in_index": g_summary is not None,
        "answer_tier": answer_tier,
        "gene_knowledge_status": gene_knowledge_status,
        "gene_explanation_source": expl_source,
        "source_grounded": expl_source == "grounded_clinvar",
        "unverified_gene_draft_available": expl_source == "ai_unreviewed",
        "significance_breakdown": g_summary.get("by_significance") or {} if g_summary else {},
        "top_phenotypes": (g_summary.get("phenotypes") or [])[:6] if g_summary else [],
    }

    result: dict = {
        "answer": deterministic,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "vus_known_gene",
        "suggested_questions": suggested,
        "llm_used": _llm_used_for_expl,
        "fallback_used": not _llm_used_for_expl,
        "llm_mode": "source_grounded" if _llm_used_for_expl else "none",
        "gene_metadata": gene_meta,
    }

    # Attach AI gene explanation to supplemental card (ai_unreviewed path)
    if expl_source == "ai_unreviewed" and expl_text:
        result["unverified_gene_draft"] = {
            "visible": True,
            "status": "ai_generated_unreviewed",
            "gene_symbol": gene,
            "text_he": expl_text,
            "review_status": gene_expl.get("review_status"),
            "approved": False,
            "requires_physician_review": True,
            "ai_content_type": "medical_educational_ai_expansion",
            "based_on": "model_expansion_llm_knowledge",
        }
        result["review_draft_id"] = gene_expl.get("review_draft_id")
        result["review_status"] = gene_expl.get("review_status")
        gene_meta["ai_draft_attempted"] = True
        gene_meta["ai_draft_generated"] = True
        result["ai_draft_debug"] = {
            "attempted": True,
            "generated": True,
            "shown": False,
            "reason": "gene_explanation_ai_generated",
        }
    else:
        gene_meta["ai_draft_attempted"] = False
        gene_meta["ai_draft_generated"] = False

    # Explicit ClinVar database query: append cleaned statistical note
    if g_summary and _is_explicit_clinvar_db_query(question):
        _clinvar_note: Optional[str] = _build_vus_clinvar_gene_note(gene, g_summary)
        if _clinvar_note:
            result["answer"] += f"\n\n{_clinvar_note}"
            gene_meta["gene_knowledge_status"] = "clinvar_summarized"
            gene_meta["source_grounded"] = True
            result["ai_draft_debug"] = result.get("ai_draft_debug") or {}
            result["ai_draft_debug"].update({
                "reason": "clinvar_explicit_query_summarized",
            })

    return result

# ---------------------------------------------------------------------------
# Fixed Hebrew safety messages
# ---------------------------------------------------------------------------

PRIVACY_WARNING_HE = (
    "נראה שהוזן פרט מזהה. מטעמי פרטיות אין להזין שם, תעודת זהות, טלפון, "
    "אימייל או פרטים מזהים בצ׳אט. אפשר לשאול שוב בלי הפרטים האישיים."
)

PERSONAL_REDIRECT_HE = (
    "השאלה הזו תלויה בפרטי המקרה ובייעוץ הגנטי שקיבלת. הבוט אינו מפרש "
    "תוצאות אישיות ואינו מחליף גנטיקאי/ת. מומלץ לפנות לצוות הגנטי לקבלת "
    "תשובה מותאמת אישית."
)

_GENERIC_SUGGESTED_QUESTIONS = [
    "מה הסיווג המדויק של הממצא שלי?",
    "אילו בדיקות מעקב מומלצות עבורי?",
    "האם כדאי לבדוק קרובי משפחה, ומי?",
]

# ---------------------------------------------------------------------------
# Reproductive / abortion decision intent — high-stakes irreversible decision
# ---------------------------------------------------------------------------

REPRODUCTIVE_DECISION_HE = (
    "שאלות הנוגעות להפסקת הריון הן החלטות אישיות ורפואיות מורכבות מאוד. "
    "ממצא VUS לבדו אינו בסיס מספיק לקבלת החלטה בלתי הפיכה כזו — "
    "VUS פירושו שמשמעות הממצא עדיין לא ידועה, וסיווגו עשוי להשתנות בעתיד. "
    "על נושא זה חשוב לשוחח ישירות עם הצוות הגנטי ועם רופא/ת נשים, "
    "שיוכלו להתייחס לכל הנסיבות הרלוונטיות עבורך."
)

_REPRODUCTIVE_DECISION_TERMS = [
    "הפלה", "להפיל", "הפסקת הריון", "להפסיק הריון",
    "להפסיק את ההריון", "להפסיק את הריון",
    "להפסיק את ההיריון", "להפסיק את היריון",  # yod-variant spelling
    "לסיים הריון", "לסיים את ההריון", "לסיים את הריון",
    "לסיים את ההיריון", "לסיים את היריון",  # yod-variant spelling
    "סיום הריון", "הפסקה של הריון",
    "הפסקת היריון",  # yod-variant spelling
    "termination", "abortion",
]

# ה{0,2}(?:יר|ר)יון matches all Hebrew variants of "pregnancy":
# הריון, היריון (yod-variant), ההריון (with article), ההיריון (article + yod-variant)
_REPRODUCTIVE_DECISION_RE = re.compile(
    r"הפל[הת]|להפיל|הפסקת\s+ה{0,2}(?:יר|ר)יון|להפסיק\s+(?:את\s+)?ה{0,2}(?:יר|ר)יון"
    r"|לסיים\s+(?:את\s+)?ה{0,2}(?:יר|ר)יון|סיום\s+ה{0,2}(?:יר|ר)יון"
    r"|termination|abortion",
    re.IGNORECASE,
)


def _is_reproductive_decision_question(text: str) -> bool:
    """Return True if the question involves abortion or pregnancy termination intent."""
    return bool(_REPRODUCTIVE_DECISION_RE.search(text))


# ---------------------------------------------------------------------------
# Helpful fallback (replaces the old harsh "I have no approved info" message)
# ---------------------------------------------------------------------------

_FALLBACK_PREFIX_HE = (
    "אני כאן כדי להסביר מושגים גנטיים כלליים — כמו VUS, נשאות, תורשה, וסיווג ממצאים. "
    "לפרשנות של ממצא ספציפי יש לפנות לצוות הגנטי שמכיר את הפרטים האישיים."
)


def _build_helpful_fallback(question: str) -> dict:
    """
    Friendlier out_of_scope response: explain the scope, and — if any
    generic topics seem plausibly relevant — name them and suggest concrete
    questions, instead of a flat refusal.
    """
    suggestions = kb.suggest_topics(limit=3)
    if suggestions:
        topic_names = ", ".join(e["title_he"] for e in suggestions)
        example_questions = []
        for e in suggestions:
            qs = e.get("suggested_questions", [])
            if qs:
                example_questions.append(qs[0])
        examples = "; ".join(example_questions[:3])
        answer = (
            f"{_FALLBACK_PREFIX_HE} לפי מה ששאלת, אני לא בטוח/ה שיש לי מידע מאושר "
            f"שעונה במדויק על השאלה הזו, אבל אולי הכוונה היא לאחד מהנושאים האלה: "
            f"{topic_names}. אפשר לשאול למשל: {examples}. "
            "אם השאלה שלך נוגעת לתוצאה האישית שלך, מומלץ לפנות לצוות הגנטי שטיפל בך."
        )
        suggested_questions = example_questions[:3]
    else:
        answer = (
            f"{_FALLBACK_PREFIX_HE} אין לי מידע מאושר שעונה על השאלה הזו במאגר הנוכחי. "
            "מומלץ לפנות לצוות הגנטי שטיפל בך לקבלת תשובה מדויקת ומותאמת."
        )
        suggested_questions = list(_GENERIC_SUGGESTED_QUESTIONS)

    return {
        "answer": answer,
        "safety_level": "out_of_scope",
        "needs_genetic_counselor": True,
        "matched_topic": None,
        "suggested_questions": suggested_questions,
        "llm_used": False,
        "fallback_used": True,
    }


# ---------------------------------------------------------------------------
# Referral/disclaimer policy (Session 27.9 Part E)
# ---------------------------------------------------------------------------

def _should_include_clinical_referral(
    intent: str = "",
    is_personal: bool = False,
    asks_for_action: bool = False,
    asks_for_interpretation: bool = False,
    requires_report_details: bool = False,
) -> bool:
    """
    Return True only when referral language is genuinely warranted.

    Referral IS appropriate when at least one of:
    - personal result interpretation is requested (is_personal)
    - exact report/variant/coordinates are required (requires_report_details)
    - medical decision or next clinical action is requested (asks_for_action)
    - personal risk/prognosis (covered by is_personal)
    - the answer cannot be completed without clinician-held information

    Referral is NOT appropriate for:
    - general gene descriptions
    - general chromosome concepts
    - animal-genetics questions
    - basic inheritance explanations
    - general definitions
    - low-risk factual questions
    - source-grounded educational answers
    """
    if is_personal or asks_for_action or asks_for_interpretation or requires_report_details:
        return True
    _personal_intents = {
        "personal_high_stakes", "surgery_decision", "reproductive_block",
        "specific_variant",
    }
    return intent in _personal_intents


# ---------------------------------------------------------------------------
# Specific-variant evidence summary
# ---------------------------------------------------------------------------
# Triggered whenever the question names an actual variant identifier (HGVS
# cDNA/protein notation or rsID). Reuses the existing legacy ClinVar
# retriever (app/retriever.py) purely as a data source — the response shape
# returned to the caller is always the new 5-field counseling schema, never
# the legacy ClinVar response format.

VARIANT_SAFETY_BOUNDARY_HE = (
    "לא ניתן לקבוע מהמידע הזה אם הווריאנט מסוכן או תקין באופן אישי עבורך, "
    "וזה אינו תחליף לייעוץ גנטי. יש להתייעץ עם הצוות הגנטי לפני כל מסקנה אישית."
)

# Fallback constant used only if the "variant_interpretation_factors" KB
# entry is somehow missing — see _no_evidence_explanation_he().
_VARIANT_NO_EVIDENCE_FALLBACK_HE = (
    "הווריאנט שצוין לא נמצא במאגר המידע המאושר/המקומי שבו משתמש הבוט. "
    "כשגנטיקאים מעריכים וריאנט, הם בוחנים בדרך כלל כמה סוגי מידע: "
    "הסיווג הקליני של הווריאנט במאגרי מידע (כגון ClinVar), הקשר בין הגן "
    "למחלה הנדונה, דפוס התורשה (autosomal dominant / recessive / X-linked וכו'), "
    "שכיחות הווריאנט באוכלוסייה הכללית, האם הווריאנט 'מתפלג' (segregates) עם "
    "המחלה במשפחה, עדויות חישוביות ותפקודיות (in-silico ומעבדתיות), וההקשר "
    "הקליני המלא של המטופל. יש להתייחס למידע זה כמידע כללי בלבד; לא ניתן "
    "להסיק ממנו מסקנה אישית."
)

_VARIANT_SUGGESTED_QUESTIONS = [
    "מה ההבדל בין VUS לבין pathogenic variant?",
    "מה אפשר לשאול את הגנטיקאי/ת לגבי הממצא הזה?",
]

VARIANT_EVIDENCE_SYSTEM_PROMPT = (
    "You are a Hebrew genetics evidence-summary assistant. You are given ONLY "
    "structured ClinVar evidence retrieved for a variant the user mentioned, "
    "plus strict safety instructions. Summarize the evidence in Hebrew, calmly "
    "and factually: gene, variant/rsID/position, clinical significance labels, "
    "review status, related conditions, whether classifications conflict, last "
    "evaluated date, and number of matching records. If no evidence was "
    "retrieved, say so plainly and describe in general terms what geneticists "
    "usually consider (variant classification, gene-disease relationship, "
    "inheritance pattern, population frequency, segregation in the family, "
    "computational/functional evidence, clinical context) — without inventing "
    "any specific facts about this case. "
    "You must NEVER say the variant is dangerous or benign for the user "
    "personally, never give a personal risk estimate, never recommend "
    "treatment, surgery, or surveillance, and never tell the user their family "
    "must be tested. Always state clearly that this is general information "
    "only and does not replace genetic counseling. Use only the data provided "
    "below; do not add facts. Answer in Hebrew."
)

# Defensive output filter: if the LLM is used to phrase the evidence summary,
# reject (and fall back to the deterministic text) if any of these forbidden
# patterns slip through.
_FORBIDDEN_VARIANT_OUTPUT_RE = re.compile(
    r"מסוכן\s+(לך|עבורך|בשבילך)"
    r"|תקין\s+(לך|עבורך|בשבילך)"
    r"|בטוח\s+(לך|עבורך)"
    r"|את(ה|ם)?\s+צריכ[הי]?\s+(טיפול|ניתוח|מעקב|כריתה|בדיקה)"
    r"|המשפחה שלך (חייבת|צריכה) להיבדק"
    r"|הילדים שלך (יהיו|יחלו)",
    re.IGNORECASE,
)


# Draft-specific forbidden patterns: personal risk/diagnosis language that
# must never appear in a patient-visible unverified gene background draft.
_FORBIDDEN_DRAFT_PERSONAL_RE = re.compile(
    r"לך\s+יש"
    r"|אצלך\s+יש"
    r"|הסיכון\s+שלך"
    r"|יש\s+לך\s+מחלה"
    r"|יש\s+לך\s+סרטן"
    r"|אתה\s+חולה"
    r"|את\s+חולה"
    r"|כדאי\s+לך\s"
    r"|עליך\s+ל"
    r"|מומלץ\s+לך"
    r"|צריכה?\s+לך",
    re.IGNORECASE,
)

# Patterns that signal low-quality or broken draft output.
_DRAFT_QUALITY_RE = re.compile(
    r"\bGenom\b"
    r"|\bgenome\b"
    r"|אנזימ\b"
    r"|קטלאזה"
    r"|מודד.{0,5}חשיבותה"
    r"|600\s+characters"
    r"|characters\s+total"
    r"|Maximum\s+600"
    r"|[\U0001F300-\U0001FAFF]"
    # mRNA — biologically incorrect for many genes (e.g. POLE = DNA polymerase)
    r"|\bmrna\b"
    r"|\bmRNA\b"
    # Hallucinated terms found in low-quality drafts
    r"|הקניית"           # garbled transcription phrase
    r"|אסימטריה"         # "asymmetry" — not meaningful gene biology
    r"|הוא\s+חלק\s+מהדנ" # truncated/vague "is part of the DN..."
    # Malformed Hebrew transliteration of gene symbols
    r"|פול\s*-\s*א"    # "פול-א" instead of POLE
    r"|גן\s+פול"        # "גן פול" — Hebrew transliteration of "gene POLE"
    r"|\bRNA\b"
    # Hype / poetic language — never acceptable in a patient-facing draft
    r"|קסם"             # "magic" — poetic/hype language
    r"|מעורר\s+עניין"   # "interesting/evocative" — hype
    r"|מרתק"            # "fascinating" — hype
    r"|מדהי[מם]"        # "amazing" — hype (both regular מ and final-mem ם forms)
    # Unsupported biological mechanism claims
    # (only acceptable if approved Gene Knowledge context was supplied)
    r"|הגן\s+עושה"      # "the gene does"
    r"|הגן\s+אחראי"     # "the gene is responsible for"
    r"|מקודד\s+ל"       # "encodes [protein]"
    r"|משתתף\s+בתהליך"  # "participates in the process"
    r"|מייצר\s+חלבון"       # "produces a protein"
    r"|חלבון\s+ה[א-ת]"      # "the [X] protein" — biological role claim
    # Mixed Hebrew/Latin within a single token (no whitespace separator) —
    # catches LLM transliteration artefacts like "סינדromות", "פREDISפוזיצIONיות"
    r"|[א-ת][A-Za-z]"       # Hebrew letter immediately followed by Latin
    r"|[A-Za-z][א-ת]"       # Latin letter immediately followed by Hebrew
    # Bad Hebrew transliterations of English medical terms
    r"|ירכיים"              # wrong "colorectal" (lit. "thighs")
    r"|כרוניקליים"          # garbled "clinical"
    r"|פוליאפולויפוזיס"     # garbled "polyposis"
    r"|קולורטאלית"          # bad transliteration of "colorectal"
    r"|predis"              # English "predisposition" fragment (IGNORECASE catches PREDIS)
    r"|redis"              # REDIS/Redis artefact
    # "ClnVar" is a common LLM typo for ClinVar — always reject
    r"|\bClnVar\b"
    # Hebrew transliteration of ClinVar — always reject
    r"|קלינוואר"
    # Database statistics — variant/classification counts indicate LLM is summarising
    # a database, not writing a patient-friendly biological explanation
    r"|\d+\s*(?:וריאנטים|variants?)\b"
    r"|\d+\s*(?:pathogenic|benign|פתוגניים|שפירים)\b"
    r"|\b(?:pathogenic|benign|likely\s+pathogenic|uncertain\s+significance)\b"
    r"|\blikely\s+(?:pathogenic|benign)\b",
    re.IGNORECASE,
)

# Separate check for "ClinVar" — only rejected on the FIRST draft attempt.
# On retry the stricter prompt discourages ClinVar; if the retry output is
# otherwise safe (no statistics, no personal risk, no medical advice) we accept
# it rather than always falling back to the deterministic path.
_CLINVAR_IN_DRAFT_RE = re.compile(r"\bClinVar\b", re.IGNORECASE)

# Warning text prepended to every patient-visible unverified draft.
_UNVERIFIED_DRAFT_WARNING_HE = (
    "מידע AI לא מאומת — להסבר כללי בלבד."
)

# LLM system prompt for generating short general gene biology background.
# Draft generation is Tier 2 only; Tier 1 uses approved curated content.
_UNVERIFIED_DRAFT_SYSTEM_PROMPT = (
    "You are writing 1-2 short, patient-friendly Hebrew sentences about a gene's "
    "general biological role.\n\n"
    "AUDIENCE: A patient in Israel who just had genetic counseling. Use plain, clear Hebrew.\n\n"
    "STYLE EXAMPLE (HBB gene):\n"
    "  'גן HBB מכיל הוראות ליצירת שרשרת בטא של המוגלובין, חלבון שעוזר לכדוריות הדם "
    "האדומות לשאת חמצן. שינויים בגן זה יכולים להיות קשורים למצבים של המוגלובין, "
    "אך המשמעות של כל שינוי תלויה בפרטי הבדיקה ונקבעת על ידי הצוות הגנטי.'\n\n"
    "You MUST NOT:\n"
    "  - Mention 'ClinVar', 'ClnVar', or any database name.\n"
    "  - Include variant counts, pathogenic counts, or any statistics.\n"
    "  - Mention 'VUS' or 'וריאנט' unless the user's question explicitly refers to VUS.\n"
    "  - Mention RNA, mRNA, transcription, or gene expression.\n"
    "  - Say 'מקודד ל' (encodes) — prefer 'מכיל הוראות ליצירת'.\n"
    "  - Recommend any action (surgery, testing, surveillance, medication).\n"
    "  - Estimate personal risk or interpret test results.\n"
    "  - Diagnose any condition or imply the patient has a disease.\n"
    "  - Use emoji.\n"
    "  - Transliterate gene symbols into Hebrew phonetics "
       "(e.g. do NOT write 'פול-א' instead of POLE).\n"
    "  - Mix English letters inside Hebrew words.\n"
    "  - Use invented, truncated, vague, or uncertain Hebrew terminology.\n"
    "  - Include question marks.\n"
    "  - If you are not confident about this gene's role, output only a dash: -\n\n"
    "STRICT FORMAT:\n"
    "  - Hebrew ONLY. Gene symbols (BRCA1, POLE, HBB, etc.) and DNA are kept as-is.\n"
    "  - 1-2 sentences maximum. Maximum 350 characters.\n"
    "  - Output ONLY the sentences. No labels, no quotes, no preamble."
)

# Stricter retry prompt used when the first draft fails validation.
_UNVERIFIED_DRAFT_RETRY_SYSTEM_PROMPT = (
    "Write 1-2 short Hebrew sentences about the general biology of the given gene.\n\n"
    "STRICT RULES:\n"
    "  - Do NOT mention 'ClinVar', 'ClnVar', or any database name.\n"
    "  - Do NOT include variant counts, statistics, or classification terms.\n"
    "  - Use the official gene symbol (e.g. POLE) — never transliterate into Hebrew.\n"
    "  - Do NOT say 'מקודד ל' — use patient-friendly phrasing like 'מכיל הוראות ליצירת'.\n"
    "  - Do NOT mention RNA, mRNA, or transcription.\n"
    "  - Do NOT mix English into Hebrew words.\n"
    "  - Do NOT name diseases, recommend treatment, or estimate risk.\n"
    "  - Use ONLY terms you are confident about. If unsure, output a single dash: -\n"
    "  - No question marks, emoji, or disclaimers.\n\n"
    "FORMAT:\n"
    "  - Hebrew only. Gene symbols and DNA are allowed as-is.\n"
    "  - Maximum 350 characters.\n"
    "  - Output ONLY the sentences. Nothing else."
)

# ---------------------------------------------------------------------------
# ClinVar-context draft prompts — used when ClinVar metadata is available.
# These replace the biology-from-memory prompts (above) for Tier 2 genes.
# The LLM must summarise ONLY what is in the provided metadata block;
# it must NOT invent biological function, protein names, or pathways.
# ---------------------------------------------------------------------------
_UNVERIFIED_CLINVAR_DRAFT_SYSTEM_PROMPT = (
    "You are helping a patient in Israel understand a gene mentioned in their genetic "
    "counseling result.\n\n"
    "You will receive: a gene symbol and a list of clinical contexts already translated "
    "into Hebrew.\n\n"
    "TASK: Write 1-2 short, plain Hebrew sentences that:\n"
    "  - Briefly explain what this gene is generally associated with, using the Hebrew "
    "contexts provided.\n\n"
    "STYLE EXAMPLE (HBB gene, contexts: אנמיה חרמשית, תלסמיה):\n"
    "  'גן HBB קשור לעיתים למצבים של המוגלובין כגון אנמיה חרמשית ותלסמיה.'\n\n"
    "You MUST NOT:\n"
    "  - Mention 'ClinVar', 'ClnVar', 'מאגר', 'בסיס נתונים', or any database name.\n"
    "  - Include variant counts, pathogenic counts, or any statistics.\n"
    "  - Use English clinical terms (Pathogenic, Benign, Likely pathogenic, "
    "Uncertain significance).\n"
    "  - Mention 'VUS' or 'וריאנט' unless the user's question explicitly refers to VUS.\n"
    "  - Describe what the gene 'encodes', what protein it 'produces', or "
    "what pathway it participates in.\n"
    "  - Invent biological details not provided in the context.\n"
    "  - State or imply the patient has any condition.\n"
    "  - Estimate personal risk.\n"
    "  - Recommend any action, test, or consultation.\n"
    "  - Use hype or poetic language (e.g. 'קסם', 'מעורר עניין', 'מרתק', 'מדהים').\n"
    "  - Use emoji.\n"
    "  - Mix English letters inside Hebrew words.\n\n"
    "STRICT FORMAT:\n"
    "  - Hebrew ONLY. Gene symbols (e.g. POLE, HBB) are kept as-is.\n"
    "  - 1-2 sentences maximum. Maximum 350 characters.\n"
    "  - Output ONLY the sentences. No labels, no quotes, no preamble."
)

_UNVERIFIED_CLINVAR_DRAFT_RETRY_SYSTEM_PROMPT = (
    "Write 1-2 cautious Hebrew sentences explaining what a gene is associated with.\n\n"
    "RULES (absolute — failure means output a single dash: -):\n"
    "  - Use ONLY the Hebrew disease contexts provided. No external knowledge.\n"
    "  - Do NOT mention 'ClinVar', 'ClnVar', any database name, or any statistics.\n"
    "  - Do NOT use English clinical terms (Pathogenic, Benign, etc.).\n"
    "  - Do NOT describe the gene's biology, protein, or mechanism.\n"
    "  - Do NOT say the patient has any condition.\n"
    "  - Do NOT estimate risk or recommend action.\n"
    "  - Do NOT use hype words (קסם, מרתק, מעורר עניין, מדהים).\n"
    "  - Hebrew only. Gene symbols kept as-is.\n"
    "  - Maximum 350 characters.\n"
    "  - Output ONLY the sentence(s)."
)

# ---------------------------------------------------------------------------
# Gene education draft prompts — used for general "what is gene X?" questions.
# More tolerant than the ClinVar-context prompts: allows biological function,
# disease associations, and English biomedical terms.  Still blocks personal
# risk language, diagnosis, and treatment recommendations.
# ---------------------------------------------------------------------------
_GENE_EDUCATION_DRAFT_SYSTEM_PROMPT = (
    "You are a genetic counseling assistant writing a short Hebrew educational summary "
    "about a gene for a patient who just had genetic counseling in Israel.\n\n"
    "TASK: ענה בעברית פשוטה וקצרה, ב-3 עד 5 משפטים קצרים בלבד.\n"
    "  1. Explain the gene's main biological role (what protein it produces or what "
    "biological process it participates in).\n"
    "  2. If the gene has ONE very well-known and central clinical association, you may "
    "mention it cautiously using 'קשור ל...' or 'מוכר בהקשר של...' — not 'גורם ל...'.\n\n"
    "ALLOWED:\n"
    "  - Gene symbols in English (BRCA1, DMD, MSH2, APOE, etc.)\n"
    "  - English biomedical terms (dystrophin, mismatch repair, beta-globin, etc.)\n"
    "  - A single specific, named clinical association when highly central to this gene\n"
    "  - Cautious phrasing: 'קשור ל...', 'מוכר בהקשר של...', 'נמצא בהקשר של...'\n"
    "  - The word 'pathogenic' in a general non-personal context\n\n"
    "PROHIBITED:\n"
    "  - 'יש לך', 'אצלך', 'הסיכון שלך', 'הממצא שלך', 'התוצאה שלך'\n"
    "  - Definitive causation: 'גורם ל...' / 'מוביל ל...' — use 'קשור ל' instead\n"
    "  - Cancer predisposition framing: 'נטייה לסרטן', 'סוגי סרטן שונים'\n"
    "  - Vague broad disease lists: 'מצבים כמו X,Y,Z', 'מחלות רבות כמו', 'מגוון רחב של מחלות'\n"
    "  - Referral phrases: do NOT add 'לפנות לצוות הגנטי', 'המשמעות נקבעת על ידי', "
    "'לתשובה אישית', or similar\n"
    "  - Treatment, surgery, or screening recommendations\n"
    "  - Personal risk estimates or 'you should...'\n"
    "  - Question marks, emoji, or ClinVar statistics\n\n"
    "FORMAT:\n"
    "  - Hebrew ONLY for main text. Gene symbols and medical terms in English.\n"
    "  - 3-5 short sentences. Maximum 500 characters.\n"
    "  - Output ONLY the sentences. No labels, no preamble, no quotes."
)

_GENE_EDUCATION_DRAFT_RETRY_SYSTEM_PROMPT = (
    "Write 2-3 short Hebrew sentences about this gene.\n\n"
    "STRICT RULES:\n"
    "  - Hebrew ONLY for main text. Gene symbols and biomedical terms in English.\n"
    "  - Sentence 1: the gene's molecular/biological role only.\n"
    "  - Sentence 2 (optional): ONE specific, named clinical association if highly "
    "central to this gene — use 'קשור ל' not 'גורם ל'. Do NOT invent this.\n"
    "  - Do NOT use 'יש לך', 'אצלך', 'הסיכון שלך'.\n"
    "  - Do NOT say 'גורם ל...' or 'נטייה לסרטן'.\n"
    "  - Do NOT give lists of diseases.\n"
    "  - Do NOT say 'מצבים כמו...' or 'סוגי סרטן שונים'.\n"
    "  - Do NOT recommend surgery, treatment, or screening.\n"
    "  - Do NOT include ClinVar statistics or variant counts.\n"
    "  - No question marks, emoji, or disclaimers.\n"
    "  - Maximum 500 characters.\n"
    "  - Output ONLY the sentences."
)

# Source note appended to every ClinVar-context draft.
# Patient-friendly wording — does not mention "ClinVar" by name.
_UNVERIFIED_DRAFT_SOURCE_NOTE_HE = (
    "הטיוטה מבוססת על מידע כללי ממאגרי דיווחים גנטיים, ולא על בדיקה מקצועית של הממצא האישי."
)


def _no_evidence_explanation_he() -> str:
    """Single source of truth for the 'no ClinVar evidence found' education
    text — sourced from the KB entry so it stays consistent with the
    standalone 'variant_interpretation_factors' topic used elsewhere
    (e.g. in follow-up elaborations)."""
    entry = kb.get_by_id("variant_interpretation_factors")
    if entry:
        return entry["approved_answer_he"]
    return _VARIANT_NO_EVIDENCE_FALLBACK_HE


def _format_evidence_block(matches: list[dict]) -> str:
    """Render matched ClinVar records as a compact labelled block for the LLM."""
    if not matches:
        return "No ClinVar records were found for this variant."
    lines = []
    for i, m in enumerate(matches, 1):
        lines.append(f"Record {i}:")
        for key in (
            "gene_symbol", "clinical_significance", "review_status",
            "phenotype_list", "dbsnp_id", "chromosome", "start_pos",
            "stop_pos", "last_evaluated",
        ):
            val = m.get(key)
            if val not in (None, "", "nan"):
                lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def _summarize_clinvar_matches_he(clinvar_result: dict) -> str:
    """Deterministic Hebrew summary of matched ClinVar records (no LLM)."""
    matches = clinvar_result.get("matches", [])

    unique_genes = sorted({str(m.get("gene_symbol") or "").strip() for m in matches if m.get("gene_symbol")})
    unique_sigs = sorted({str(m.get("clinical_significance") or "").strip() for m in matches if m.get("clinical_significance")})
    unique_statuses = sorted({str(m.get("review_status") or "").strip() for m in matches if m.get("review_status")})
    unique_rsids = sorted({
        f"rs{int(m['dbsnp_id'])}" for m in matches
        if m.get("dbsnp_id") not in (None, "") and int(m.get("dbsnp_id") or 0) > 0
    })
    unique_positions = sorted({
        f"chr{m.get('chromosome')}:{m.get('start_pos')}" for m in matches
        if m.get("chromosome") and m.get("start_pos")
    })
    last_evaluated = sorted(
        {str(m.get("last_evaluated") or "").strip() for m in matches if m.get("last_evaluated")},
        reverse=True,
    )
    phenotypes: set[str] = set()
    for m in matches:
        for p in str(m.get("phenotype_list") or "").split("|"):
            p = p.strip()
            if p and p.lower() not in ("not provided", "not specified", ""):
                phenotypes.add(p)
    has_conflict = (
        any("conflicting" in s.lower() for s in unique_sigs)
        or (
            any("pathogenic" in s.lower() and "benign" not in s.lower() for s in unique_sigs)
            and any("benign" in s.lower() and "pathogenic" not in s.lower() for s in unique_sigs)
        )
    )

    lines = [f"באופן כללי, במאגר המידע המקומי (ClinVar) נמצאו {len(matches)} רשומות תואמות."]
    if unique_genes:
        lines.append(f"גן: {', '.join(unique_genes)}.")
    if unique_rsids:
        lines.append(f"rsID: {', '.join(unique_rsids)}.")
    if unique_positions:
        lines.append(f"מיקום גנומי: {', '.join(unique_positions)}.")
    if unique_sigs:
        lines.append(f"לפי הרשומות שנמצאו, הסיווגים המדווחים כוללים: {', '.join(unique_sigs)}.")
    if has_conflict:
        lines.append("קיימות בין הרשומות סיווגים שונים/לא עקביים (conflicting classifications) עבור וריאנט זה.")
    if unique_statuses:
        lines.append(f"סטטוס הבדיקה (review status): {', '.join(unique_statuses)}.")
    if phenotypes:
        shown = sorted(phenotypes)[:5]
        lines.append(f"מצבים/תופעות המוזכרים בתיעוד: {'; '.join(shown)}.")
    if last_evaluated:
        lines.append(f"תאריך העדכון האחרון הידוע: {last_evaluated[0]}.")
    lines.append(
        "מגבלות: המידע מבוסס על תמונת מצב מקומית של ClinVar, ועלול לא לשקף את "
        "כל המקורות הקיימים או את העדכון העדכני ביותר. יש להתייחס למידע זה "
        "כמידע כללי בלבד, ולא ניתן להסיק מכך מסקנה אישית ללא ייעוץ גנטי."
    )
    return " ".join(lines)


def _call_local_llm_for_variant_evidence(question: str, clinvar_result: dict) -> Optional[str]:
    """
    Optionally ask the local LLM to phrase the variant evidence summary,
    passing it ONLY the retrieved structured evidence (never the raw
    question's personal framing beyond what's needed for context) plus
    strict safety instructions. Falls back to None (deterministic text is
    used instead) on any failure or if the output trips the forbidden-output
    filter — the LLM is never trusted to invent evidence or break the
    safety boundary.
    """
    url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if not url:
        return None
    try:
        client = LocalLLMClient(url)
        evidence_block = _format_evidence_block(clinvar_result.get("matches", []))
        user_content = (
            f"User question (Hebrew): {question}\n\n"
            f"=== RETRIEVED CLINVAR EVIDENCE ===\n{evidence_block}\n\n"
            "Write the Hebrew evidence summary now, following the system instructions exactly."
        )
        raw = client._call_api(user_content, system_prompt=VARIANT_EVIDENCE_SYSTEM_PROMPT)
        text = raw.strip()
        if not text:
            return None
        if _FORBIDDEN_VARIANT_OUTPUT_RE.search(text):
            logger.warning("LLM variant-evidence answer tripped the forbidden-output filter; using deterministic fallback.")
            return None
        return text
    except LLMClientError as exc:
        logger.warning("Local LLM unavailable for variant evidence summary (%s); using deterministic fallback.", exc)
        return None
    except Exception as exc:  # defensive — never let LLM errors break /ask
        logger.warning("Unexpected error calling local LLM for variant evidence (%s); using deterministic fallback.", exc)
        return None


def _build_variant_evidence_answer(question: str) -> dict:
    """
    Build the response for a question that names a specific variant
    (HGVS notation or rsID). Always safety_level=requires_genetic_counselor
    and needs_genetic_counselor=True — a specific variant always needs
    personalized clinical interpretation, regardless of whether evidence
    was found.
    """
    variant_query = safety.extract_variant_query(question)
    try:
        clinvar_result = retriever.match_uploaded_variant(variant_query, limit=10)
    except Exception as exc:  # never let a retriever failure break /ask
        logger.warning("ClinVar lookup failed for variant query %s (%s); using no-evidence answer.", variant_query, exc)
        clinvar_result = {"match_confidence": "no_match", "matches": [], "warnings": []}

    matches = clinvar_result.get("matches", [])

    if matches:
        llm_summary = _call_local_llm_for_variant_evidence(question, clinvar_result)
        llm_used = llm_summary is not None
        body = llm_summary or _summarize_clinvar_matches_he(clinvar_result)
        answer = f"{VARIANT_SAFETY_BOUNDARY_HE}\n\n{body}"
    else:
        llm_used = False
        answer = f"{VARIANT_SAFETY_BOUNDARY_HE}\n\n{_no_evidence_explanation_he()}"

    return {
        "answer": answer,
        "safety_level": "requires_genetic_counselor",
        "needs_genetic_counselor": True,
        "matched_topic": "variant_evidence_summary",
        "suggested_questions": list(_VARIANT_SUGGESTED_QUESTIONS),
        "llm_used": llm_used,
        "fallback_used": not llm_used,
    }

# ---------------------------------------------------------------------------
# Gene-level ClinVar summary (pipeline step 4.5)
# ---------------------------------------------------------------------------
# Fires for questions like "What is known about BRCA1?" / "מה ידוע על NF1?"
# Requires gene_index._GENE_INDEX_AVAILABLE to be True.
# Placed BEFORE follow-up handling so "tell me about BRCA1" is not
# consumed as a vague follow-up phrase.

# Regex to find candidate gene-symbol tokens (uppercase, 2–10 chars).
# Uses lookahead/lookbehind on [A-Za-z] instead of \b so genes written with
# a Hebrew prefix directly attached ("בHBB", "ב-HBB") are still detected.
_GENE_SYMBOL_CANDIDATE_RE = re.compile(r"(?<![A-Za-z])([A-Z][A-Z0-9]{1,9})(?![a-zA-Z])")

# Regex to extract gene symbols from explicit "בגן X" or "הגן X" Hebrew phrases.
# Supports mixed-case HGNC-style symbols (e.g. C12orf57, KIAA2022) that are
# NOT matched by _GENE_SYMBOL_CANDIDATE_RE, which requires all-uppercase tokens.
# Conservative: 3–15 chars, starts with uppercase, letters and digits only.
_EXPLICIT_GENE_PHRASE_RE = re.compile(
    r"(?:בגן|הגן)\s+([A-Z][A-Za-z0-9]{2,14})(?=[^A-Za-z0-9]|$)",
    re.UNICODE,
)

# Well-known abbreviations that are NOT gene symbols — skip index lookup for these.
_NON_GENE_TOKENS: frozenset[str] = frozenset({
    "VUS", "DNA", "RNA", "MRNA", "CDNA", "GDNA", "SNP", "SNV",
    "HGVS", "ACMG", "OMIM", "HGMD", "NIH", "FDA", "PCR",
    "MRI", "CT", "AI", "LP", "LB", "VCV", "RCV", "HPO",
})


def _is_standalone_gene_query(text: str, gene: str) -> bool:
    """Return True when the input is essentially just a gene symbol (possibly with Hebrew prefix/suffix/question mark)."""
    stripped = text.strip().rstrip("?").strip()
    g = gene.upper()
    if stripped.upper() == g:
        return True
    # Hebrew prefix directly attached or with hyphen: "בCFTR", "ב-CFTR", "לPTEN", "ל-PTEN"
    if re.match(r'^[א-ת]-?' + re.escape(g) + r'$', stripped, re.IGNORECASE):
        return True
    return False


def _extract_gene_from_explicit_phrase(text: str) -> "Optional[str]":
    """
    Extract a gene symbol from explicit Hebrew phrases 'בגן X' or 'הגן X'.

    Supports HGNC-style mixed-case symbols such as C12orf57 (which contain
    lowercase letters and are NOT matched by _GENE_SYMBOL_CANDIDATE_RE).
    Does not require the gene to exist in the local ClinVar index.

    Returns the extracted symbol (preserving original casing) or None.
    """
    m = _EXPLICIT_GENE_PHRASE_RE.search(text)
    if not m:
        return None
    sym = m.group(1)
    if sym.upper() in _NON_GENE_TOKENS:
        return None
    # Require digit OR ≥2 uppercase letters to avoid false positives
    # (e.g. ordinary Hebrew words transliterated in Latin letters).
    has_digit = any(c.isdigit() for c in sym)
    uppercase_count = sum(1 for c in sym if c.isupper())
    if not (has_digit or uppercase_count >= 2):
        return None
    return sym


# Intent phrases: signal the user is asking about a gene's general profile.
_GENE_QUESTION_PHRASES: frozenset[str] = frozenset([
    # Hebrew — general gene-knowledge intent
    "מה ידוע", "ידוע על", "מה אפשר לדעת",
    "ספר לי על", "ספרי לי על", "ספר על",
    "תסביר לי על", "תסבירי לי על", "תסביר על", "תסביר",
    "הסבר על", "הסבר לי על",
    "מה הגן", "על הגן", "לגבי הגן",
    "אילו מצבים", "אילו מחלות",
    "מה מדווח", "מה מוכר",
    "מה קשור",
    "מה זה", "מה הוא", "מה היא",
    "clinvar",
    # Gene-explanation intent — e.g. "מה המשמעות של הגן APC", "מה התפקיד של SHANK3",
    # "איזה גן זה BRCA1". These fire step 4.5 BEFORE step 5 (follow-up), so they
    # correctly override prior VUS context when the user is now asking about the gene.
    "המשמעות של הגן", "מה המשמעות של",
    "מה התפקיד", "תפקיד של",
    "איזה גן", "הגן הזה",
    # English
    "what is known about", "what do you know about",
    "tell me about",
    "explain",
    "describe",
    "what conditions", "conditions associated",
    "variants in", "what variants",
    "what does", "what is", "what are", "what can",
    "how many",
    # Personal-discovery phrasings that name a specific gene — route to gene info,
    # not to a generic KB topic.
    "שינוי גנטי בגן",      # "I have a genetic change in gene X"
    "ממצא בגן",            # "finding in gene X"
    "גיליתי שיש לי",       # "I discovered I have…" (often followed by gene name)
    "נמצא לי בגן",         # "found in my gene X"
    "בגן ה",               # "in the gene …" (broad, but safe within gene context)
    # Mutation-question phrasings — "מה הבעיה במוטציה בגן X?", "שינוי בגן X"
    # Without these, step 4.5 is skipped and the question falls to KB fuzzy
    # lookup which returns the vus_known_gene entry with a hardcoded gene example.
    "מוטציה בגן",
    "שינוי בגן",
    "מה הבעיה",
    "הבעיה ב",
    "מה הסכנה",
    "מה הסיכון של הגן",
    "מה קורה בגן",
    # "What is THE mutation of gene X" — implies one mutation; triggers routing + explanation note
    "מה המוטציה",
    "המוטציה של",
    "המוטציה ב",
    "מה הוריאנט של",
    "מה הווריאנט של",
    "איזה מוטציה",
    "איזה וריאנט",
    "מוטציה של הגן",
    "וריאנט של הגן",
    # Association / disease-link phrasings — "לאיזה מצבים קליניים הגן APOE מקושר?"
    "לאיזה מצבים",
    "לאיזה מחלות",
    "מקושר",
    "קשורות ל",
    "מה הקשר",        # covers "מה הקשר ל", "מה הקשר של", "מה הקשר שלו"
    # English association phrasings
    "associated with",
    "disease association",
    "clinical conditions",
    # Gene-function intent — "איך הגן BRCA1 פועל בגוף?", "מה עושה הגן KIAA2022?"
    # Without these, the question falls to the generic what_is_gene KB entry
    # instead of routing to the gene-specific explanation pipeline.
    "איך הגן",         # "how does gene X work?" — explicit gene reference
    "מה עושה הגן",     # "what does gene X do?" — explicit gene reference
    "מה עושה ה",       # "what does the ... do?" — covers "מה עושה הגן"
])

# Phrases that signal the user is asking about "the mutation" (singular) of a gene.
# These trigger a preamble explaining that a gene can have many different variants.
_MUTATION_SPECIFIC_PHRASES: frozenset[str] = frozenset([
    "מה המוטציה",
    "המוטציה של",
    "המוטציה ב",
    "מה הוריאנט של",
    "מה הווריאנט של",
    "איזה מוטציה",
    "איזה וריאנט",
    "מוטציה של הגן",
    "וריאנט של הגן",
    "what mutation",
    "what variant",
    "mutation of",
])

# Preamble prepended when user asks about "the mutation of gene X" —
# corrects the misconception that a gene has one single mutation.
_MUTATION_GENE_PREAMBLE_HE = (
    "גן {gene} יכול לכלול מגוון רחב של וריאנטים ושינויים גנטיים שונים — "
    "אין וריאנט יחיד שמייצג את הגן. "
    "המשמעות של כל ממצא תלויה בווריאנט המדויק שנמצא בבדיקה הגנטית, "
    "בסיווגו ובהקשר הקליני.\n\n"
)


def _is_mutation_specific_question(text: str) -> bool:
    """Return True when the user asks 'what is THE mutation of gene X' (singular)."""
    lower = text.strip().lower()
    return any(phrase in lower for phrase in _MUTATION_SPECIFIC_PHRASES)


# ---------------------------------------------------------------------------
# Fuzzy gene-symbol matching — typo correction (XFTR → CFTR, SOCKS1 → SOX1)
# ---------------------------------------------------------------------------
_known_gene_set_cache: "Optional[frozenset]" = None


def _get_known_gene_set() -> frozenset:
    """Build lazily-cached set of all known gene symbols from all sources."""
    global _known_gene_set_cache
    if _known_gene_set_cache is not None:
        return _known_gene_set_cache
    genes: "set[str]" = set()
    for canon, _ in _GENE_PATTERNS:
        genes.add(canon)
    if gene_index._GENE_INDEX_AVAILABLE:
        try:
            for entry in gene_index.list_genes(limit=2000):
                sym = str(entry.get("gene_symbol", "")).strip().upper()
                if sym and len(sym) <= 10:
                    genes.add(sym)
        except Exception:
            pass
    _known_gene_set_cache = frozenset(g for g in genes if g and len(g) >= 2)
    return _known_gene_set_cache


def _fuzzy_match_gene_symbol(candidate: str) -> "Optional[str]":
    """
    Fuzzy-match a typo'd uppercase token to the nearest known gene.
    Returns corrected gene or None. Never matches reserved tokens.
    Only returns a result when exactly ONE gene has similarity >= 0.60.
    """
    import difflib
    if not candidate or not re.match(r'^[A-Z][A-Z0-9]{1,9}$', candidate):
        return None
    if candidate in _NON_GENE_TOKENS:
        return None
    known = _get_known_gene_set()
    if candidate in known:
        return None  # already exact — not a typo
    matches = difflib.get_close_matches(candidate, list(known), n=2, cutoff=0.60)
    if len(matches) == 1:
        return matches[0]
    return None  # ambiguous or no match


def _extract_gene_with_correction(text: str) -> "tuple[Optional[str], Optional[str]]":
    """
    Like _extract_gene_symbol_from_question but also returns the original token
    when a typo was corrected.

    Returns (gene_symbol_or_None, original_typo_token_or_None).
    """
    # First: exact / pattern-based detection (no correction needed)
    gene = _extract_gene_symbol_from_question(text)
    if gene:
        return gene, None
    # Second: fuzzy-match any uppercase-token candidate
    candidates = _GENE_SYMBOL_CANDIDATE_RE.findall(text)
    for c in candidates:
        if c in _NON_GENE_TOKENS:
            continue
        corrected = _fuzzy_match_gene_symbol(c)
        if corrected:
            return corrected, c
    return None, None


# ---------------------------------------------------------------------------
# ClinVar condition label hygiene
# ---------------------------------------------------------------------------

# Labels that are NOT meaningful condition names — strip before showing patients.
_CLINVAR_NON_PATIENT_CONDITION_RE = re.compile(
    r'^\d+\s+condition'                            # "8 conditions", "2 conditions"
    r'|^see\s+cases?$'                             # "See cases"
    r'|^not\s+(provided|specified|applicable)$'    # "not provided", etc.
    r'|^\d+$',                                     # purely numeric
    re.IGNORECASE,
)


def _filter_patient_conditions(phenotypes: list) -> list:
    """Remove non-patient-friendly phenotype labels from a gene index phenotype list."""
    out = []
    for p in phenotypes:
        p = (p or "").strip()
        if not p or len(p) < 4:
            continue
        if _CLINVAR_NON_PATIENT_CONDITION_RE.match(p):
            continue
        out.append(p)
    return out


def _build_brief_clinvar_stats(gene: str, summary: dict) -> Optional[str]:
    """
    Compact ClinVar statistics section for embedding in larger patient answers.
    Returns None if no meaningful data can be shown.
    Applies _filter_patient_conditions so non-patient-friendly labels are excluded.
    """
    total = summary.get("total_variants", 0)
    by_sig = summary.get("by_significance", {})
    phenotypes = _filter_patient_conditions(summary.get("phenotypes", []))

    lines = [f"נתוני ClinVar עבור גן {gene}:"]
    if total:
        lines.append(f"• סה\"כ {total:,} רשומות וריאנט במאגר")
    if by_sig:
        path_n   = sum(v for k, v in by_sig.items()
                       if "pathogenic" in k.lower() and "benign"      not in k.lower()
                       and "uncertain" not in k.lower() and "conflicting" not in k.lower())
        benign_n = sum(v for k, v in by_sig.items()
                       if "benign"     in k.lower() and "pathogenic"  not in k.lower()
                       and "uncertain" not in k.lower() and "conflicting" not in k.lower())
        vus_n    = sum(v for k, v in by_sig.items() if "uncertain" in k.lower())
        if path_n:
            lines.append(f"• {path_n:,} פתוגניים / likely pathogenic")
        if benign_n:
            lines.append(f"• {benign_n:,} שפירים / likely benign")
        if vus_n:
            lines.append(f"• {vus_n:,} VUS (משמעות לא ידועה)")
    if phenotypes:
        max_show = 6
        lines.append("מצבים רפואיים קשורים בתיעוד ClinVar:")
        for p in phenotypes[:max_show]:
            lines.append(f"  • {p}")
        if len(phenotypes) > max_show:
            lines.append(f"  (ועוד {len(phenotypes) - max_show} מצבים נוספים)")

    return "\n".join(lines) if len(lines) > 1 else None


# LLM system prompt for gene-level summaries.
GENE_SUMMARY_SYSTEM_PROMPT = (
    "You are a Hebrew post-genetic-counseling assistant. You help patients "
    "understand general genetic concepts after they have already met a genetic "
    "counselor. You are given structured aggregate statistics from ClinVar for "
    "a specific gene — this is population-level database data, NOT a personal "
    "variant result for this user.\n\n"
    "Write a clear, calm, patient-friendly Hebrew summary (150–250 words) covering:\n"
    "1. How many variant records are documented in ClinVar for this gene.\n"
    "2. The distribution of clinical significance categories "
    "(pathogenic, benign, VUS, conflicting, etc.).\n"
    "3. The main medical conditions/phenotypes associated with this gene in ClinVar.\n"
    "4. A clear statement that this is general information — not a personal result — "
    "and that personal interpretation requires consulting the genetics team.\n\n"
    "Rules:\n"
    "- Write entirely in Hebrew (keep gene symbols, pathogenic, benign, VUS, ClinVar in English).\n"
    "- Do NOT diagnose.\n"
    "- Do NOT estimate personal medical risk.\n"
    "- Do NOT recommend surgery, screening, treatment, or family testing.\n"
    "- Do NOT say the user has this gene variant or any disease.\n"
    "- Use only the data provided below; do not invent facts."
)

# Forbidden-output filter: reject LLM answers that include personal risk language.
_FORBIDDEN_GENE_OUTPUT_RE = re.compile(
    r"הסיכון\s+שלך"
    r"|מסוכן\s+(לך|עבורך)"
    r"|את[הן]?\s+צריכ[הי]?\s+(ניתוח|טיפול|מעקב|כריתה|בדיקה)"
    r"|המשפחה\s+שלך\s+(חייבת|צריכה)\s+להיבדק"
    r"|יש\s+לך\s+(מחלה|סרטן|גידול)"
    r"|אתה\s+חולה"
    r"|את\s+חולה"
    r"|you\s+(have|should|must|need)",
    re.IGNORECASE,
)

_GENE_SUMMARY_SAFETY_NOTE_HE = "המידע כללי ואינו מחליף ייעוץ רפואי אישי."

_GENE_SUGGESTED_QUESTIONS = [
    "מה ההבדל בין VUS לבין ממצא pathogenic?",
    "האם VUS יכול להשתנות בעתיד?",
    "למה בדרך כלל לא מקבלים החלטות רפואיות רק לפי VUS?",
]

# Shown for non-VUS gene questions (general biological curiosity)
_GENE_SUGGESTED_QUESTIONS_GENERAL = [
    "מה ההבדל בין VUS לבין ממצא pathogenic?",
    "האם ממצא בגן יכול להשתנות בסיווגו לאורך זמן?",
    "מה ההבדל בין גן לבין וריאנט?",
]


def _gene_suggested_questions(question: str, gene: str) -> list:
    """VUS questions when question asks about VUS; general questions otherwise."""
    if _mentions_vus(question):
        return [q.replace("בגן זה", f"ב-{gene}") for q in _GENE_SUGGESTED_QUESTIONS]
    return list(_GENE_SUGGESTED_QUESTIONS_GENERAL)


# ---------------------------------------------------------------------------
# Trisomy 21 / Down syndrome educational answer
# ---------------------------------------------------------------------------

_TRISOMY21_SIGNALS_RE = re.compile(
    r"טריזומי[הא]\s*21|trisomy\s*21"
    r"|כרומוזום\s*21\s*(?:עודף|נוסף|יותר)"
    r"|כרומוזום\s+עודף\s*21"
    r"|תסמונת\s+דאון|סינדרום\s+דאון|down\s*syndrome",
    re.IGNORECASE,
)

_TRISOMY21_EDUCATIONAL_HE = (
    "טריזומיה 21 (Trisomy 21) היא מצב גנטי שבו בתאי הגוף יש שלושה עותקים של כרומוזום 21 "
    "במקום שניים הרגילים. זהו הבסיס הגנטי של תסמונת דאון (Down syndrome).\n\n"
    "ברוב המקרים, כרומוזום 21 הנוסף מופיע כתוצאה מאי-הפרדה תקינה של כרומוזומים "
    "בתא הביצה (non-disjunction). בחלק קטן מהמקרים קיימת צורת פסיפס (מוזאיקה) — "
    "שם חלק מהתאים בלבד מכילים שלושה כרומוזומים 21.\n\n"
    "מבחינה קלינית, התסמונת קשורה למנעד רחב של ביטויים: רמות שונות של השפעה על "
    "ההתפתחות, מאפיינים גופניים מסוימים, ולעיתים ממצאים נוספים במערכות שונות כגון "
    "לב ומעיים. ביטויי התסמונת שונים מאדם לאדם. "
    "האבחנה מאושרת בדרך כלל על ידי בדיקת קריוטיפ (בדיקת כרומוזומים)."
)


def _detect_trisomy21(text: str) -> bool:
    """True when the question contains trisomy 21 / Down syndrome signals."""
    return bool(_TRISOMY21_SIGNALS_RE.search(text))


def _build_trisomy21_answer() -> dict:
    return {
        "answer": _TRISOMY21_EDUCATIONAL_HE,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "trisomy21",
        "suggested_questions": [
            "מה ההבדל בין טריזומיה 21 לבין פסיפס (מוזאיקה)?",
            "האם טריזומיה 21 ניתן לאבחן לפני הלידה?",
            "מה כדאי לשאול את הצוות הגנטי על טריזומיה 21?",
        ],
        "llm_used": False,
        "fallback_used": True,
    }


# ---------------------------------------------------------------------------
# Extra sex chromosome / chromosomal aneuploidy educational answer
# (XXY, XXX, XYY, extra X chromosome, extra Y chromosome)
# NOTE: trisomy21 ("כרומוזום 21 עודף") is handled separately above.
# ---------------------------------------------------------------------------

_EXTRA_CHROMOSOME_RE = re.compile(
    r"\bXXY\b|\bXXX\b|\bXYY\b|\bXXXY\b|\bXYYY\b"
    r"|Klinefelter|קליינפלטר|triple\s*x"
    r"|כרומוזום\s+(?:X|Y|מין)\s+(?:עודף|נוסף)"
    r"|(?:עודף|נוסף)\s+כרומוזום\s+(?:X|Y|מין)"
    r"|כרומוזום\s+(?:עודף|נוסף)\s+(?:X|Y)"
    r"|(?:X|Y)\s+(?:עודף|נוסף)\b",
    re.IGNORECASE,
)

_EXTRA_CHROMOSOME_EDUCATIONAL_HE = (
    "כרומוזום מין עודף פירושו שיש עותק נוסף של כרומוזום מין בחלק מהתאים או בכולם. "
    "המשמעות תלויה בהרכב הכרומוזומים המדויק — למשל XXY (תסמונת קליינפלטר), XXX, או XYY — "
    "ובשאלה אם הממצא נמצא בכל התאים (טריזומיה מלאה) או רק בחלקם (פסיפס/מוזאיקה).\n\n"
    "מצבים אלה יכולים להיות קשורים למנעד רחב של מאפיינים: חלקם קלים, חלקם משמעותיים יותר, "
    "ולעיתים לא קיים כלל ביטוי קליני משמעותי. הביטוי משתנה מאוד מאדם לאדם.\n\n"
    "האבחנה מאושרת בדרך כלל על ידי בדיקת קריוטיפ (בדיקת כרומוזומים). "
    "חשוב לדון עם הצוות הגנטי על ההקשר הספציפי של הממצא."
)


def _detect_extra_chromosome(text: str) -> bool:
    """True when the question contains sex chromosome aneuploidy signals (not trisomy21)."""
    if _detect_trisomy21(text):
        return False
    return bool(_EXTRA_CHROMOSOME_RE.search(text))


def _build_extra_chromosome_answer() -> dict:
    return {
        "answer": _EXTRA_CHROMOSOME_EDUCATIONAL_HE,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "chromosomal_finding",
        "suggested_questions": [
            "מה ההבדל בין טריזומיה לבין פסיפס (מוזאיקה)?",
            "האם ממצא כרומוזומי נבדק תמיד בכל התאים?",
            "מה כדאי לשאול את הצוות הגנטי על ממצא כרומוזומי?",
        ],
        "llm_used": False,
        "fallback_used": True,
    }


# ---------------------------------------------------------------------------
# Chromosome number extraction helper (used for follow-up routing context)
# ---------------------------------------------------------------------------

_CHROMOSOME_NUMBER_RE = re.compile(
    r"(?:כרומוזום|chromosome)\s*(\d{1,2}|X|Y)\b",
    re.IGNORECASE,
)


def _extract_chromosome_number(text: str) -> "Optional[str]":
    """Extract chromosome identifier from text like 'כרומוזום 21' → '21'."""
    m = _CHROMOSOME_NUMBER_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# AI content type classification (Session 27.8 Part A)
# ---------------------------------------------------------------------------

# SOURCE_GROUNDED_PHRASING: LLM only phrases supplied structured data
# (phenotypes, approved context summary).  No physician review required.
# Must NOT populate the physician review queue (Part I).
AI_CONTENT_TYPE_GROUNDED = "source_grounded_phrasing"

# AI_EXPANDED_UNVERIFIED: LLM adds biology beyond supplied data.  Must go
# to the physician review queue before appearing in patient-facing answers.
AI_CONTENT_TYPE_EXPANDED = "ai_expanded_unverified"


# ---------------------------------------------------------------------------
# General chromosomal / cytogenetic education
# (Does NOT overlap with trisomy21 or extra_chromosome handlers above;
#  those fire first in classify_question_intent.)
# ---------------------------------------------------------------------------

# Patterns for each sub-intent.  They fire only when the more-specific
# trisomy21 and extra_chromosome detectors have already returned False.

_CHROMOSOME_DELETION_RE = re.compile(
    r"מחיקה\s+בכרומוזום|מחיקה\s+כרומוזומית"
    r"|chromosomal\s+deletion|deletion.*chromosome"
    r"|חסר\s+בכרומוזום|קטע\s+חסר.*כרומוזום",
    re.IGNORECASE,
)

_CHROMOSOME_DUPLICATION_RE = re.compile(
    r"כפילות\s+בכרומוזום|כפילות\s+כרומוזומית"
    r"|chromosomal\s+duplication|duplication.*chromosome"
    r"|קטע\s+כפול.*כרומוזום",
    re.IGNORECASE,
)

_TRANSLOCATION_RE = re.compile(
    r"טרנסלוקציה|translocation|translocat"
    r"|העברה\s+(?:בין\s+)?כרומוזום",
    re.IGNORECASE,
)

_MOSAICISM_GENERAL_RE = re.compile(
    r"(?:מוזאיצ|מוזאיק|מוזאיקה|פסיפס)\b"
    r"|mosaicism|mosaic",
    re.IGNORECASE,
)

_CYTOGENETIC_TEST_RE = re.compile(
    r"קריוטיפ|karyotype|karyotyp"
    r"|מיקרואריי|microarray|array\s*CGH|chromosomal\s*microarray|\bCMA\b"
    r"|\bFISH\b|fluorescence\s+in\s+situ"
    r"|בדיקת\s+כרומוזומים",
    re.IGNORECASE,
)

_ANEUPLOIDY_GENERAL_RE = re.compile(
    r"אנאופלואידיה|aneuploidy|aneuploid"
    r"|מונוזומי[הא]|monosomy"
    r"|טריזומי[הא]\s+\d",   # "טריזומיה N" — trisomy 21 already handled before this
    re.IGNORECASE,
)

_CHROMOSOME_FINDING_GENERAL_RE = re.compile(
    # "בעיה בכרומוזום X" / "ממצא כרומוזומי" / "שינוי בכרומוזום"
    r"בעיה\s+בכרומוזום"
    r"|ממצא(?:ים?)?\s+כרומוזומי"
    r"|שינוי\s+בכרומוזום"
    r"|חריגה\s+בכרומוזום"
    r"|כרומוזום.*(?:בעיה|ממצא|שינוי)"
    r"|ניתוח\s+כרומוזומי",   # "chromosomal analysis" but not karyotype specifically
    re.IGNORECASE,
)

_CYTOGENETIC_KB: dict = {
    "chromosome_finding_general": {
        "answer_he": (
            "כשמדברים על \"שינוי בכרומוזום\", הכוונה יכולה להיות לסוגים שונים של ממצאים:\n\n"
            "• שינוי במספר הכרומוזומים — כרומוזום עודף (טריזומיה) או חסר (מונוזומיה).\n"
            "• שינוי במבנה הכרומוזום — מחיקה (חסרה של קטע), כפילות (הכפלה של קטע), "
            "או טרנסלוקציה (קטע שעבר מכרומוזום אחד לאחר).\n"
            "• ממצא פסיפס (מוזאיקה) — שינוי קיים רק בחלק מהתאים.\n\n"
            "המשמעות תלויה לחלוטין בסוג הממצא המדויק, במיקומו, בגודלו, "
            "ובסוג הבדיקה שנעשתה.\n\n"
            "מה כדאי לברר מהסיכום או מהדוח?\n"
            "• מה סוג הממצא — עודף כרומוזום, חסר, מחיקה, כפילות, או שינוי מבני אחר?\n"
            "• האם מדובר בכרומוזום שלם או בקטע ממנו?\n"
            "• מה גודל הממצא, ואילו גנים ידועים כלולים?\n"
            "• האם הממצא אושש בבדיקה נוספת?\n"
            "• מה סוג הבדיקה שגילתה את הממצא — סקר, אבחנה, קריוטיפ, מיקרואריי?"
        ),
        # Bot-clickable educational follow-up questions
        "suggested_questions": [
            "מה ההבדל בין מחיקה לבין כפילות כרומוזומית?",
            "מה זה פסיפס (מוזאיקה) גנטי?",
            "מה בדיקת קריוטיפ יכולה לגלות?",
        ],
        # Questions to ask the genetics team (non-clickable, shown under separate heading)
        "clinician_questions": [
            "מה סוג הממצא שנמצא בכרומוזום — עודף, חסר, מחיקה או שינוי מבני אחר?",
            "האם הממצא קיים בכל התאים או רק בחלקם?",
            "מה המשמעות הקלינית של הממצא הספציפי הזה?",
        ],
    },
    "chromosome_deletion_general": {
        "answer_he": (
            "מחיקה כרומוזומית היא מצב שבו קטע של כרומוזום חסר. "
            "המחיקה יכולה להיות קטנה (מיקרו-מחיקה, הנראית לעיתים רק במיקרואריי) "
            "או גדולה יותר (נראית בקריוטיפ).\n\n"
            "ההשפעה הקלינית תלויה בגודל המחיקה ובמיקומה — אילו גנים כלולים בקטע החסר. "
            "מחיקות קטנות בגנים לא-קריטיים עלולות להיות חסרות משמעות קלינית, "
            "בעוד שמחיקות גדולות יותר או במיקומים רגישים עשויות להיות קשורות לתסמינים שונים."
        ),
        "suggested_questions": [
            "מה ההבדל בין מחיקה לבין כפילות כרומוזומית?",
            "איך מזהים מחיקה — קריוטיפ או מיקרואריי?",
        ],
        "clinician_questions": [
            "מה המיקום המדויק של המחיקה (כרומוזום ואזור)?",
            "האם המחיקה כוללת גנים ידועים?",
            "האם ניתן לבדוק אם המחיקה נמצאת גם אצל ההורים?",
        ],
    },
    "chromosome_duplication_general": {
        "answer_he": (
            "כפילות כרומוזומית היא מצב שבו קטע של כרומוזום מופיע בעותק נוסף. "
            "בניגוד למחיקה (חסר), בכפילות יש חומר גנטי עודף.\n\n"
            "ההשפעה הקלינית תלויה בגודל הכפילות, במיקומה ובגנים שהיא כוללת. "
            "כפילות קטנות יכולות לעיתים להיות ניטרליות; כפילות גדולות יותר עשויות להיות "
            "קשורות לביטויים שונים בהתאם לאזור הכרומוזום המדובר."
        ),
        "suggested_questions": [
            "מה ההבדל בין מחיקה לבין כפילות כרומוזומית?",
            "מה זה VUS בהקשר של כפילות כרומוזומית?",
        ],
        "clinician_questions": [
            "מה המיקום המדויק של הכפילות?",
            "האם הכפילות כוללת גנים ידועים?",
            "מה ידוע על ממצאים דומים במאגרים הגנטיים?",
        ],
    },
    "translocation_general": {
        "answer_he": (
            "טרנסלוקציה היא שינוי מבני שבו קטע כרומוזומי עבר ממקומו המקורי לכרומוזום אחר.\n\n"
            "קיימים שני סוגים עיקריים:\n"
            "• טרנסלוקציה מאוזנת — החומר הגנטי עבר מקומו אך לא חסר ולא עודף. "
            "נשאים של טרנסלוקציה מאוזנת לרוב בריאים, אך עשויים להיות בסיכון מוגבר לבעיות בפריון "
            "או להעביר טרנסלוקציה לא מאוזנת לצאצאים.\n"
            "• טרנסלוקציה לא מאוזנת — יש חסר או עודף של חומר גנטי. "
            "סוג זה עשוי להיות קשור לביטויים קליניים, תלוי בגנים המעורבים."
        ),
        "suggested_questions": [
            "מה ההבדל בין טרנסלוקציה מאוזנת ללא מאוזנת?",
            "האם ניתן להיות נשא של טרנסלוקציה ללא תסמינים?",
        ],
        "clinician_questions": [
            "האם הטרנסלוקציה מאוזנת או לא מאוזנת?",
            "אילו כרומוזומים מעורבים ואיפה הנקודות?",
            "האם כדאי לבדוק את שאר בני המשפחה?",
        ],
    },
    "mosaicism_general": {
        "answer_he": (
            "פסיפס (מוזאיקה) הוא מצב שבו אדם מכיל שתי אוכלוסיות תאים לפחות, "
            "בעלות הרכב כרומוזומי שונה. לדוגמה, חלק מהתאים עשויים להכיל את המספר הרגיל "
            "של כרומוזומים, בעוד שתאים אחרים מכילים כרומוזום עודף או חסר.\n\n"
            "ההשפעה הקלינית תלויה בשניים:\n"
            "• אחוז התאים הנגועים (\"אחוז המוזאיקה\").\n"
            "• אילו רקמות ואיברים מכילים את התאים המשתנים.\n\n"
            "פסיפס יכול לגרום לביטוי קל יותר בהשוואה לאותה הפרעה במצב מלא, "
            "או לכלל ללא ביטוי קליני — אך זה משתנה מאוד מאדם לאדם ומממצא לממצא."
        ),
        "suggested_questions": [
            "מה ההבדל בין ממצא מלא לבין פסיפס?",
            "האם פסיפס תמיד מורש מההורים?",
        ],
        "clinician_questions": [
            "מה אחוז התאים עם הממצא הכרומוזומי?",
            "באילו רקמות בוצעה הבדיקה?",
            "מה ידוע על ביטוי קליני בדרגת פסיפס דומה?",
        ],
    },
    "cytogenetic_test_general": {
        "answer_he": (
            "ישנן כמה בדיקות מרכזיות לניתוח כרומוזומים:\n\n"
            "• קריוטיפ (Karyotype) — מסדר את כל הכרומוזומים ומאפשר לזהות שינויים גדולים "
            "במספרם (למשל טריזומיה) או במבנם (כמו טרנסלוקציות גדולות). "
            "רזולוציה: ~5-10 מיליון בסיסים.\n"
            "• מיקרואריי כרומוזומלי (Chromosomal Microarray / CMA) — מזהה "
            "מחיקות וכפילויות קטנות שאינן נראות בקריוטיפ. רזולוציה גבוהה הרבה יותר.\n"
            "• FISH — בדיקה ממוקדת לאזור כרומוזומי ספציפי.\n\n"
            "הבדיקה שנבחרה מותאמת לשאלה הקלינית הספציפית."
        ),
        "suggested_questions": [
            "מה ההבדל בין קריוטיפ לבין מיקרואריי?",
            "מה זה מחיקה כרומוזומית?",
        ],
        "clinician_questions": [
            "איזו בדיקת כרומוזומים בוצעה — קריוטיפ, מיקרואריי, או אחר?",
            "מה הרזולוציה של הבדיקה שנעשתה?",
            "האם יש צורך בבדיקה נוספת כדי לאשש את הממצא?",
        ],
    },
    "aneuploidy_general": {
        "answer_he": (
            "אנאופלואידיה פירושה שמספר הכרומוזומים בתא שונה מהמספר הצפוי (46 בבני אדם). "
            "קיימות שתי צורות עיקריות:\n\n"
            "• טריזומיה — כרומוזום עודף (47 כרומוזומים). "
            "דוגמאות: טריזומיה 21 (תסמונת דאון), טריזומיה 18 (תסמונת אדוורדס), טריזומיה 13.\n"
            "• מונוזומיה — כרומוזום חסר (45 כרומוזומים). "
            "הדוגמה הנפוצה ביותר: מונוזומיה X (תסמונת טרנר).\n\n"
            "המשמעות הקלינית שונה מאוד בין כרומוזומים שונים ובין מצבים שונים. "
            "הצוות הגנטי יפרט את הממצא הספציפי ואת המשמעות הרלוונטית לכם."
        ),
        "suggested_questions": [
            "מה ההבדל בין טריזומיה למונוזומיה?",
            "מה זה פסיפס (מוזאיקה)?",
        ],
        "clinician_questions": [
            "איזה כרומוזום מעורב ומה הממצא המדויק?",
            "האם הממצא נמצא בכל התאים או רק בחלקם (פסיפס)?",
            "מה המשמעות הקלינית של הממצא הספציפי הזה?",
        ],
    },
}

_CHROMOSOME_EDUCATION_SUGGESTED_QUESTIONS_DEFAULT = [
    "מה סוג הממצא שנמצא בכרומוזום?",
    "מה המשמעות הקלינית של הממצא הספציפי הזה?",
    "מה כדאי לשאול את הצוות הגנטי לאחר בדיקת כרומוזומים?",
]


def _detect_chromosome_education(text: str) -> Optional[str]:
    """
    Return a chromosomal-education sub-intent string, or None.

    Only called AFTER _detect_trisomy21 and _detect_extra_chromosome have
    returned False, so we do not need to guard against those overlaps here.
    """
    if _TRANSLOCATION_RE.search(text):
        return "translocation_general"
    if _CHROMOSOME_DELETION_RE.search(text):
        return "chromosome_deletion_general"
    if _CHROMOSOME_DUPLICATION_RE.search(text):
        return "chromosome_duplication_general"
    if _MOSAICISM_GENERAL_RE.search(text):
        return "mosaicism_general"
    if _CYTOGENETIC_TEST_RE.search(text):
        return "cytogenetic_test_general"
    if _ANEUPLOIDY_GENERAL_RE.search(text):
        return "aneuploidy_general"
    if _CHROMOSOME_FINDING_GENERAL_RE.search(text):
        return "chromosome_finding_general"
    return None


# ---------------------------------------------------------------------------
# Chromosome follow-up routing (Session 27.8 Part F)
# ---------------------------------------------------------------------------

_CHROMOSOME_TOPIC_INTENTS: frozenset = frozenset({
    "chromosome_finding_general",
    "chromosome_deletion_general",
    "chromosome_duplication_general",
    "translocation_general",
    "mosaicism_general",
    "cytogenetic_test_general",
    "aneuploidy_general",
})

# Whitelisted active topics for SafeSessionContext.
# This is the single canonical source — main.py imports it so the two sets
# cannot drift apart.  _build_session_context_out() uses it to whitelist
# the topics it writes into session_context_out.
_SAFE_CONTEXT_ALLOWED_ACTIVE_TOPICS: frozenset = frozenset({
    # Chromosome education sub-intents
    "chromosome_finding_general", "chromosome_deletion_general",
    "chromosome_duplication_general", "translocation_general",
    "mosaicism_general", "cytogenetic_test_general", "aneuploidy_general",
    # Chromosome special cases
    "trisomy21_education", "extra_chromosome_education", "chromosomal_finding",
    # Gene answers
    "gene_clinvar_summary", "gene_info",
    # VUS variants
    "vus", "vus_known_gene", "vus_general",
    # Carrier status
    "carrier", "carrier_vs_affected", "carrier_general",
    # Inheritance patterns
    "inheritance", "inheritance_general",
    "autosomal_dominant", "autosomal_recessive", "x_linked",
    # Testing and family
    "genetic_test_general", "family_testing",
    # Specific variant educational answer
    "specific_variant",
})

# Short keywords that map a brief follow-up message to a chromosome sub-intent.
_CHROMOSOME_FOLLOWUP_PATTERNS: list = [
    ("chromosome_deletion_general",    ["מחיקה", "deletion", "חסר", "חסרה"]),
    ("chromosome_duplication_general", ["כפילות", "duplication", "כפול", "עודף"]),
    ("translocation_general",          ["טרנסלוקציה", "translocation"]),
    ("mosaicism_general",              ["פסיפס", "מוזאיקה", "mosaic"]),
    ("cytogenetic_test_general",       ["קריוטיפ", "מיקרואריי", "karyotype", "microarray"]),
    ("aneuploidy_general",             ["אנאופלואידיה", "aneuploidy", "טריזומיה", "מונוזומיה"]),
]

# Context-aware openers prepended to the KB answer when chromosome_number is known (Part H).
# Directly addresses the chromosome the patient mentioned rather than being fully generic.
_CHROMOSOME_CONTEXT_OPENERS: dict = {
    "chromosome_finding_general": (
        "כרומוזום {chromosome_number} הוא אחד מהכרומוזומים האנושיים ומכיל גנים רבים. "
        "\"שינוי בכרומוזום {chromosome_number}\" יכול להתייחס לסוגים שונים של ממצאים — "
        "מחיקה, כפילות, שינוי במספר העותקים, טרנסלוקציה, או ממצא פסיפס."
    ),
    "chromosome_deletion_general": (
        "אם הכוונה היא למחיקה בכרומוזום {chromosome_number}, "
        "מדובר בחסר של קטע מסוים מכרומוזום {chromosome_number}."
    ),
    "chromosome_duplication_general": (
        "אם הכוונה היא לכפילות בכרומוזום {chromosome_number}, "
        "מדובר בעותק עודף של קטע מסוים מכרומוזום {chromosome_number}."
    ),
    "translocation_general": (
        "אם מדובר בממצא בכרומוזום {chromosome_number}, "
        "מדובר בשינוי מבני שבו קטע מכרומוזום {chromosome_number} עבר למיקום אחר."
    ),
    "mosaicism_general": (
        "אם מדובר בפסיפס בכרומוזום {chromosome_number}, "
        "המשמעות היא שהשינוי בכרומוזום {chromosome_number} קיים רק בחלק מהתאים."
    ),
    "aneuploidy_general": (
        "אם מדובר בשינוי במספר עותקי כרומוזום {chromosome_number}, "
        "הממצא מתאר מצב שבו מספר העותקים של כרומוזום {chromosome_number} שונה מהרגיל."
    ),
}

# Context-aware clinician questions when chromosome_number is known (Part J).
_CHROMOSOME_CONTEXT_CLINICIAN_QUESTIONS: dict = {
    "chromosome_finding_general": [
        "מה סוג הממצא שנמצא בכרומוזום {chromosome_number} — עודף, חסר, מחיקה, כפילות, או שינוי מבני?",
        "האם הממצא בכרומוזום {chromosome_number} קיים בכל התאים או רק בחלקם?",
        "מה המשמעות הקלינית של הממצא הספציפי בכרומוזום {chromosome_number}?",
    ],
    "chromosome_deletion_general": [
        "מה הגודל והמיקום המדויק של המחיקה בכרומוזום {chromosome_number}?",
        "האם המחיקה בכרומוזום {chromosome_number} כוללת גנים ידועים?",
        "האם ניתן לבדוק אם המחיקה בכרומוזום {chromosome_number} נמצאת גם אצל ההורים?",
    ],
    "chromosome_duplication_general": [
        "מה הגודל והמיקום המדויק של הכפילות בכרומוזום {chromosome_number}?",
        "האם הכפילות בכרומוזום {chromosome_number} כוללת גנים ידועים?",
        "מה ידוע על ממצאים דומים בכרומוזום {chromosome_number} במאגרים הגנטיים?",
    ],
    "translocation_general": [
        "אילו כרומוזומים מעורבים בטרנסלוקציה, ומה תפקיד כרומוזום {chromosome_number}?",
        "האם הטרנסלוקציה הכוללת כרומוזום {chromosome_number} מאוזנת או לא מאוזנת?",
        "האם ניתן לבדוק את ההורים לגבי טרנסלוקציה בכרומוזום {chromosome_number}?",
    ],
    "mosaicism_general": [
        "איזה אחוז מהתאים נושא את הממצא בכרומוזום {chromosome_number}?",
        "באיזו בדיקה זוהה הפסיפס בכרומוזום {chromosome_number}?",
        "מה המשמעות הקלינית של פסיפס בכרומוזום {chromosome_number}?",
    ],
    "aneuploidy_general": [
        "מה מספר העותקים שנמצאו של כרומוזום {chromosome_number}?",
        "האם הממצא בכרומוזום {chromosome_number} קיים בכל התאים או רק בחלקם?",
        "מה המשמעות הקלינית של שינוי זה בכרומוזום {chromosome_number}?",
    ],
}


def _resolve_chromosome_followup(
    text: str,
    session_context: "Optional[dict]",
) -> "Optional[dict]":
    """
    Detect short chromosome finding-type follow-ups using session context.

    Returns {"sub_intent": ..., "chromosome_number": ...} or None.

    Fires only when:
      - session_context.active_topic is a chromosome education sub-intent, AND
      - The message is ≤100 characters, AND
      - The message contains a known finding-type keyword.

    This must be called AFTER gene routing (E/F) and BEFORE general follow-up
    detection (step 5) so that finding-type keywords like 'מחיקה' route to
    the chromosome handler rather than the KB fuzzy lookup.
    """
    if not session_context:
        return None
    active_topic = (session_context.get("active_topic") or "")
    if active_topic not in _CHROMOSOME_TOPIC_INTENTS:
        return None
    if len(text.strip()) > 100:
        return None
    lower = text.lower()
    for sub_intent, keywords in _CHROMOSOME_FOLLOWUP_PATTERNS:
        if any(kw in lower for kw in keywords):
            return {
                "sub_intent": sub_intent,
                "chromosome_number": session_context.get("chromosome_number"),
                "finding_type": sub_intent,  # Part I: finding_type IS the sub_intent
            }
    return None


# ---------------------------------------------------------------------------
# Chromosome education AI draft — safety guard, generation, and lifecycle
# ---------------------------------------------------------------------------

# ISCN notation: specific cytogenetic coordinates / report strings that indicate
# the patient is describing a specific report rather than asking a general question.
# Draft generation is suppressed when these are detected.
_ISCN_RE = re.compile(
    # karyotype formula: "46,XX" / "47,XY" / "45,X0"
    r"\b(?:4[2-9]|5[0-5])\s*,\s*(?:XX|XY|X0|XYY|XXY|XO)\b"
    # structural abnormality notation
    r"|del\s*\(\d+\)\s*\("
    r"|dup\s*\(\d+\)\s*\("
    r"|inv\s*\(\d+\)\s*\("
    # translocation t(N;M)
    r"|\bt\s*\(\s*\d+\s*;\s*\d+\s*\)"
    # mosaic karyotype
    r"|mos\s+\d+\s*,\s*(?:XX|XY)",
    re.IGNORECASE,
)

# Personal interpretation or prognosis phrases that must suppress draft generation.
_CHROMOSOME_PERSONAL_SIGNALS = [
    "הסיכון שלי", "מה הסיכוי שלי", "מה ה outcome שלי",
    "מה הפרוגנוזה שלי", "האם אצטרך", "מה קורה לי",
    "מה ה פרוגנוזה", "מה יקרה לי",
]


def _is_safe_for_chromosome_draft(text: str) -> bool:
    """
    Return False when the question must NOT trigger chromosome AI draft generation:
    ISCN notation, personal interpretation signals, or PII.
    (Safety routing already blocks termination decisions upstream, but
    this is an extra guard specifically for draft generation.)
    """
    if _ISCN_RE.search(text):
        return False
    lower = text.lower()
    if any(sig in lower for sig in _CHROMOSOME_PERSONAL_SIGNALS):
        return False
    return True


_CHROMOSOME_EDUCATION_DRAFT_SYSTEM_PROMPT = """You are a genetic counseling educational assistant. Generate a short Hebrew educational expansion about a chromosomal finding concept for a patient who has already met with a genetic counselor.

MANDATORY RULES — never violate:
- Write ONLY general education about the chromosomal concept. Never diagnose.
- Do NOT assume which specific syndrome the patient has (e.g., "chromosome 21 problem" ≠ trisomy 21).
- Do NOT recommend surgery, treatment, surveillance, or termination.
- Do NOT give personal risk estimates or prognosis.
- Do NOT interpret ISCN strings, specific reports, or coordinate notation.
- Do NOT add a generic referral sentence such as 'יש לפנות לצוות הגנטי' or 'המידע כללי'. End naturally.
- Write in Hebrew. Maximum 3 short paragraphs.
- If you cannot generate safe content, respond with: NO_SAFE_CONTENT"""

_CHROMOSOME_EDUCATION_DRAFT_RETRY_SYSTEM_PROMPT = (
    _CHROMOSOME_EDUCATION_DRAFT_SYSTEM_PROMPT
    + "\n\nIMPORTANT: Previous attempt was rejected. "
    "Ensure the response contains ONLY general educational text in Hebrew. "
    "Do NOT include any personal interpretation, specific syndrome names as conclusions, or medical advice."
)


def _generate_chromosome_education_draft(
    question: str,
    sub_intent: str,
    _debug: "Optional[dict]" = None,
) -> "Optional[dict]":
    """
    Generate an optional AI educational expansion for a chromosomal concept.

    The deterministic KB answer is ALWAYS the main answer — this draft is
    supplemental only (never replaces the KB text).

    Returns a dict with text_he or None. Never raises.
    """
    from datetime import datetime, timezone

    def _dbg(**kw: object) -> None:
        if isinstance(_debug, dict):
            _debug.update(kw)

    try:
        client = create_llm_client()
    except ValueError:
        _dbg(attempted=False, provider="none", reason="llm_not_configured")
        return None

    provider_name = type(client).__name__
    _dbg(attempted=True, provider=provider_name)

    user_content = (
        f"Chromosomal concept: {sub_intent.replace('_', ' ')}\n"
        f"Patient question context (no personal details): {question[:200]}\n"
        f"Task: Provide a general educational expansion in Hebrew about this chromosomal concept."
    )

    try:
        raw = client.call_text_raw(user_content, system_prompt=_CHROMOSOME_EDUCATION_DRAFT_SYSTEM_PROMPT)
        text = (raw or "").strip()
        rejection = _validate_gene_education_draft(text) if text else "empty"
        if rejection == "no_safe_content" or "NO_SAFE_CONTENT" in text:
            _dbg(generated=False, rejection_code="no_safe_content")
            return None
        if rejection:
            raw2 = client.call_text_raw(user_content, system_prompt=_CHROMOSOME_EDUCATION_DRAFT_RETRY_SYSTEM_PROMPT)
            text = (raw2 or "").strip()
            rejection = _validate_gene_education_draft(text) if text else "empty"
            if rejection:
                _dbg(generated=False, rejection_code=rejection)
                return None

        _dbg(generated=True)
        return {
            "text_he": text,
            "sub_intent": sub_intent,
            "generated_by_model": provider_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "review_status": "pending",
            "approved": False,
            # Session 27.9 Part I: updated patient-facing pending warning (short)
            "warning_he": "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי.",
            # Expansion label shown in UI card header
            "expansion_label": "הסבר נוסף שנוצר באמצעות AI",
            # ai_content_type for policy routing
            "ai_content_type": "medical_educational_ai_expansion",
            "intended_use": "supplemental_medical_expansion",
            "risk_class": "medical_educational",
        }
    except Exception as exc:
        logger.debug("chromosome draft generation failed: %s", type(exc).__name__)
        _dbg(generated=False, rejection_code=str(type(exc).__name__))
        return None


def _build_chromosome_education_answer(
    question: str,
    sub_intent: str,
    chromosome_number: "Optional[str]" = None,
) -> dict:
    """
    Return a chromosomal/cytogenetic educational answer.

    Main answer: always the deterministic KB entry (never replaced by a draft).
    When chromosome_number is known (from context or question), a context-specific
    opener is prepended so the answer explicitly references the relevant chromosome
    (Part H).  Clinician questions are also adapted to mention the chromosome (Part J).

    Supplemental draft card: optional AI expansion shown in immediate mode;
    approved chromosome drafts are also supplemental (not main-answer replacements).
    """
    # Extract/confirm chromosome number early so it informs both answer text and
    # clinician questions (Part H: chromosome number used in answer text).
    _chr_number = chromosome_number or _extract_chromosome_number(question)

    kb_entry = _CYTOGENETIC_KB.get(sub_intent) or _CYTOGENETIC_KB["chromosome_finding_general"]
    main_answer = kb_entry["answer_he"]
    suggested = kb_entry.get("suggested_questions", _CHROMOSOME_EDUCATION_SUGGESTED_QUESTIONS_DEFAULT)
    clinician_qs = list(kb_entry.get("clinician_questions", []))

    # When chromosome_number is known and sub_intent has a specific opener, prepend it
    # so the answer explicitly mentions the chromosome the patient asked about (Part H).
    if _chr_number and sub_intent in _CHROMOSOME_CONTEXT_OPENERS:
        opener = _CHROMOSOME_CONTEXT_OPENERS[sub_intent].format(chromosome_number=_chr_number)
        main_answer = opener + "\n\n" + main_answer

    # Use context-aware clinician questions when chromosome_number is known (Part J).
    if _chr_number and sub_intent in _CHROMOSOME_CONTEXT_CLINICIAN_QUESTIONS:
        clinician_qs = [
            q.format(chromosome_number=_chr_number)
            for q in _CHROMOSOME_CONTEXT_CLINICIAN_QUESTIONS[sub_intent]
        ]

    # Optional AI draft expansion (supplemental only).
    _chr_draft_debug: dict = {}
    chr_draft: "Optional[dict]" = None
    if _is_safe_for_chromosome_draft(question):
        chr_draft = _generate_chromosome_education_draft(
            question, sub_intent, _debug=_chr_draft_debug,
        )

    chr_draft_available = chr_draft is not None

    # Check for physician-approved chromosome draft (still supplemental, not main).
    _approved_chr_draft: "Optional[dict]" = None
    try:
        from app import review_db as _rdb
        _approved_chr_draft = _rdb.get_approved_chromosome_draft(sub_intent)
    except Exception:
        pass

    # Supplemental card visibility — deterministic KB is always the main answer.
    _chr_draft_text = (chr_draft or {}).get("text_he", "")
    if _AI_DRAFT_VISIBILITY_MODE == "approved_only":
        chr_draft_displayable = False
        _chr_draft_hidden_reason: "Optional[str]" = (
            "approved_only_mode" if chr_draft_available else "no_draft"
        )
    else:
        chr_draft_displayable = (
            chr_draft_available
            and _draft_adds_meaningful_information(_chr_draft_text, main_answer)
        )
        _chr_draft_hidden_reason = (
            None if chr_draft_displayable
            else "no_draft" if not chr_draft_available
            else "identical_to_main_answer"
        )

    # Auto-enqueue new draft to physician review DB via shared helper (27.9.3).
    # Uses _persist_reviewable_ai_expansion so review metadata is returned
    # and attached to the response.
    _chr_persist_record: "Optional[dict]" = None
    if chr_draft_available and chr_draft:
        _chr_persist_record = _persist_reviewable_ai_expansion(
            chr_draft,
            draft_type="chromosome_education",
            normalized_intent=sub_intent,
            gene_symbol=None,
            model_provider=_chr_draft_debug.get("provider"),
            prompt_version="s279",  # session when chromosome prompt was introduced
            source_metadata={
                "sub_intent": sub_intent,
                "ai_content_type": "medical_educational_ai_expansion",
                "source_basis": "model_chromosome_education",
                "source_grounded": False,
                "requires_physician_review": True,
                "risk_class": "medical_educational",
                "intended_use": "supplemental_medical_expansion",
            },
        )

    result: dict = {
        "answer": main_answer,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": sub_intent,
        "suggested_questions": list(suggested),
        "clinician_questions": list(clinician_qs),
        "llm_used": chr_draft_available,
        "fallback_used": False,
        "chromosome_draft_metadata": {
            "sub_intent": sub_intent,
            "draft_available": chr_draft_available,
            "draft_displayable": chr_draft_displayable,
            "draft_hidden_reason": _chr_draft_hidden_reason,
            # Approved chromosome drafts are SUPPLEMENTAL (never replace the
            # deterministic KB main answer).
            "approved_draft_available": _approved_chr_draft is not None,
            "approved_draft_promoted": False,  # always False for chromosome education
            # Retained for session-context follow-up routing (Part F).
            "chromosome_number_detected": _chr_number,
            # Session 27.9 Part I: UI label for the supplemental card
            "expansion_label_pending": "הסבר נוסף שנוצר באמצעות AI",
            "expansion_label_approved": "מידע נוסף שנבדק ואושר",
            "expansion_pending_warning": "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי.",
        },
    }
    if chr_draft_available:
        result["unverified_chromosome_draft"] = chr_draft
        _attach_review_metadata(result, _chr_persist_record, draft_key="unverified_chromosome_draft")

    return result


# ---------------------------------------------------------------------------
# Source-grounded gene answer generation (Session 27.8 Parts B-C)
# ---------------------------------------------------------------------------

_SOURCE_GROUNDED_GENE_SYSTEM_PROMPT = """\
You are a genetic counseling educational assistant. Write 2-3 patient-friendly Hebrew
sentences summarizing the following structured information about gene {gene}.

MANDATORY RULES — violating any rule causes your response to be discarded:
1. Use ONLY information present in the supplied context below.
   Do NOT use your own knowledge of the gene, even if you have it.
   Do NOT add disease mechanisms, molecular pathways, protein functions,
   receptor roles, enzyme activity, or metabolic pathways unless they
   appear word-for-word in the supplied context.
2. Do NOT give risk estimates, prognosis, treatment, or surveillance recommendations.
3. Do NOT use personal pronouns directed at the patient (e.g. 'שלך', 'שלכם').
4. Do NOT add a generic referral sentence such as 'לפנות לצוות הגנטי' or
   'המידע כללי ואינו מחליף ייעוץ רפואי'. End the answer naturally.
5. Maximum 100 words. Write in Hebrew only.
6. If the context is insufficient for a meaningful patient-facing answer,
   respond with the single word: INSUFFICIENT_CONTEXT

Supplied context for gene {gene}:
{context}"""

# Association-only prompt — used when only ClinVar phenotype associations are available.
# The LLM may state that the gene is associated with conditions but must NOT add
# biological function, enzyme activity, protein products, or molecular mechanisms.
_SOURCE_GROUNDED_ASSOCIATION_SYSTEM_PROMPT = """\
You are a genetic counseling educational assistant. Write 1-2 patient-friendly Hebrew
sentences about gene {gene} based ONLY on the disease associations listed below.

MANDATORY RULES — violating any rule causes your response to be discarded:
1. Use ONLY the condition names listed in "ClinVar reported associations" below.
   Do NOT describe molecular function, enzyme activity, protein products,
   metabolic pathways, or biological mechanisms — that data is not supplied.
2. State only that the gene has been associated with certain medical conditions.
   Pattern: "{gene} קשור למספר מצבים רפואיים, כולל [condition1] ו-[condition2]."
3. Do NOT give risk estimates, prognosis, treatment, or surveillance recommendations.
4. Do NOT use personal pronouns directed at the patient (e.g. 'שלך', 'שלכם').
5. Do NOT add a generic referral sentence such as 'לפנות לצוות הגנטי' or
   'המידע כללי ואינו מחליף ייעוץ רפואי'. End the answer naturally after
   stating the associations.
6. Maximum 60 words. Write in Hebrew only.
7. If the associations are too few or too generic for a meaningful answer,
   respond with the single word: INSUFFICIENT_CONTEXT

Supplied associations for gene {gene}:
{context}"""

# Terms that indicate the LLM fabricated biological mechanism beyond supplied facts.
# Used to validate grounded output when context contains no molecular biology.
_FABRICATED_BIOLOGY_PATTERNS = re.compile(
    r"(?:מסלול מולקולרי|נתיב ביוכימי|חילוף חומרים של שומנים|ליפופרוטאין|"
    r"ריצפטור ל|קולטן ל|אנזים ה|פעילות אנזימטית|מנגנון ביולוגי|"
    r"ביטוי גנים|קינאז|פרוטאז|טרנספורמינג|פקטור גדילה)",
    re.IGNORECASE,
)


def _has_sufficient_grounded_gene_context(
    gene: str,
    clinvar_summary: "Optional[dict]",
) -> "tuple[bool, list]":
    """
    Return (has_sufficient_local_source, context_parts) for source-grounded gene phrasing.

    Session 27.8.2 Parts C/D update:
    - Layer 2.5: gene_knowledge draft biology text (approved=False records) is now
      included as a biology source.  These are curated project texts awaiting physician
      approval — they supply factual context for LLM phrasing without being served raw.
    - Phenotype associations (≥3 non-trivial) now qualify for an association-only
      grounded answer.  The generation function uses a restricted association-only
      prompt when only phenotypes are present.  Biology mechanism claims remain
      forbidden in that path.

    Evidence classes:
      curated_gene_description  — physician-approved patient summary; has_biology=True
      approved_context          — physician-approved ClinVar context summary; has_biology=True
      draft_biology_text        — gene_knowledge draft text (approved=False); has_biology=True
      phenotype_associations    — ClinVar phenotype names; has_biology=False
                                  (association wording only — no mechanisms or functions)
      clinvar_counts            — variant counts; has_biology=False; never sufficient alone

    Returns True when:
      - at least one biology part is present (curated, approved_context, or draft), OR
      - at least one phenotype_associations part is present (≥3 non-trivial phenotypes).
    """
    context_parts: list = []

    # 1. Physician-approved patient summary — explicit curated gene biology.
    gk_patient = gene_knowledge.get_gene_patient_summary(gene)
    if gk_patient and len(gk_patient.strip()) > 30:
        context_parts.append({
            "type": "curated_gene_description",
            "text": gk_patient.strip(),
            "has_biology": True,
            "approved": True,
        })

    # 2. Physician-approved ClinVar context summary.
    gk_context = gene_knowledge.get_gene_context_summary(gene)
    if gk_context and len(gk_context.strip()) > 30:
        context_parts.append({
            "type": "approved_context",
            "text": gk_context.strip(),
            "has_biology": True,
            "approved": True,
        })

    # 2.5 Gene Knowledge draft biology text — curated project text, pending physician
    #     approval for direct patient-facing serving.  Supplied as LLM input context only.
    #     Only added when no approved biology text was found in layers 1 or 2.
    if not any(p.get("has_biology") for p in context_parts):
        draft_bio = gene_knowledge.get_gene_knowledge_biology_text(gene)
        if draft_bio:
            context_parts.append({
                "type": "draft_biology_text",
                "text": draft_bio,
                "has_biology": True,
                "approved": False,
            })

    # 3. ClinVar phenotypes — association evidence only; NEVER qualifies as biology.
    #    Included for association-only grounded answers (≥3 non-trivial phenotypes).
    if clinvar_summary:
        _TRIVIAL = frozenset({
            "not specified", "not provided", "see cases", "not applicable",
            "all disease", "disease", "",
        })
        phenotypes = [
            p.strip() for p in ((clinvar_summary.get("phenotypes") or [])[:8])
            if p and p.strip().lower() not in _TRIVIAL and len(p.strip()) > 5
        ]
        if len(phenotypes) >= 3:
            context_parts.append({
                "type": "phenotype_associations",
                "phenotypes": phenotypes[:6],
                "has_biology": False,
            })

    # Sufficient when at least one biology part OR at least one phenotype association
    # part is present.  Counts alone never qualify (neither biology nor association).
    has_biology = any(p.get("has_biology") for p in context_parts)
    has_associations = any(p.get("type") == "phenotype_associations" for p in context_parts)
    return has_biology or has_associations, context_parts


def _generate_source_grounded_gene_answer(
    gene: str,
    context_parts: list,
    _debug: "Optional[dict]" = None,
) -> "Optional[str]":
    """
    Generate a source-grounded gene answer using only the supplied context parts.

    Classification: AI_CONTENT_TYPE_GROUNDED — no physician review required,
    must NOT populate the physician review queue (Part I).
    Returns plain Hebrew text or None on failure.  Never raises.
    """
    if not context_parts:
        return None

    def _dbg(**kw: object) -> None:
        if isinstance(_debug, dict):
            _debug.update(kw)

    # Determine what evidence is available and choose appropriate prompt.
    # Session 27.8.2: phenotype-only context is now allowed via association-only prompt.
    has_biology = any(p.get("has_biology") for p in context_parts)
    has_associations = any(p.get("type") == "phenotype_associations" for p in context_parts)

    if not has_biology and not has_associations:
        _dbg(attempted=False, reason="no_grounded_evidence")
        return None

    # Biology prompt when biology context is present; association-only when phenotypes only.
    use_biology_prompt = has_biology
    _dbg(prompt_type="biology_grounded" if use_biology_prompt else "association_only")

    context_lines: list = []
    for part in context_parts:
        ptype = part.get("type")
        if ptype in ("curated_gene_description", "draft_biology_text"):
            label = "Approved gene description" if ptype == "curated_gene_description" else "Gene biology context"
            context_lines.append(f"{label}: {part['text']}")
        elif ptype == "approved_context":
            context_lines.append(f"Physician-approved context: {part['text']}")
        elif ptype == "phenotype_associations":
            phenotypes = part.get("phenotypes", [])
            context_lines.append(f"ClinVar reported associations: {', '.join(phenotypes)}")

    if not context_lines:
        return None

    context_text = "\n".join(context_lines)
    prompt_template = (
        _SOURCE_GROUNDED_GENE_SYSTEM_PROMPT if use_biology_prompt
        else _SOURCE_GROUNDED_ASSOCIATION_SYSTEM_PROMPT
    )
    system_prompt = prompt_template.format(gene=gene, context=context_text)
    user_content = (
        f"Gene: {gene}\n"
        f"Task: Write 2-3 patient-friendly Hebrew sentences about this gene "
        f"using only the above supplied context."
    )

    try:
        client = create_llm_client()
    except ValueError:
        _dbg(attempted=False, reason="llm_not_configured")
        return None

    _dbg(attempted=True, provider=type(client).__name__, ai_content_type=AI_CONTENT_TYPE_GROUNDED)

    try:
        raw = client.call_text_raw(user_content, system_prompt=system_prompt)
        text = (raw or "").strip()
        if not text or "INSUFFICIENT_CONTEXT" in text:
            _dbg(generated=False, reason="insufficient_context")
            return None
        if len(text) > 600:
            _dbg(generated=False, reason="too_long")
            return None
        lower = text.lower()
        _personal = ["הסיכון שלך", "הפרוגנוזה שלך", "הסיכוי שלך", "הסיכון שלכם"]
        if any(p in lower for p in _personal):
            _dbg(generated=False, reason="personal_language_detected")
            return None
        # Must have substantial Hebrew content.
        if sum(1 for c in text if "א" <= c <= "ת") < 20:
            _dbg(generated=False, reason="insufficient_hebrew")
            return None
        # For association-only answers: biology terms in the output are fabricated
        # (the prompt forbids them).  Reject outright if any appear.
        # For biology-grounded answers: reject only if the terms don't appear in context.
        if _FABRICATED_BIOLOGY_PATTERNS.search(text):
            if not use_biology_prompt:
                _dbg(generated=False, reason="fabricated_biology_in_association_answer")
                return None
            context_text_lower = context_text.lower()
            found_terms = _FABRICATED_BIOLOGY_PATTERNS.findall(text)
            unsupported = [t for t in found_terms
                           if t.lower() not in context_text_lower]
            if unsupported:
                _dbg(generated=False, reason="fabricated_biology_detected",
                     unsupported_terms=unsupported)
                return None
        _dbg(generated=True)
        return text
    except Exception as exc:
        _dbg(generated=False, reason=str(type(exc).__name__))
        return None


# ---------------------------------------------------------------------------
# Out-of-domain detection — clearly non-genetics/medicine questions
# ---------------------------------------------------------------------------

_OUT_OF_DOMAIN_RE = re.compile(
    r"מגדל\s+אייפל|eiffel\s+tower"
    r"|ראש\s+ממשלת?\s+(?:ישראל|בריטניה|צרפת|גרמניה|אמריקה|ארה.?ב)"
    r"|מחיר\s+ה?(?:דירה|מכונית|רכב|דלק)\b"
    r"|מנות\s+(?:מזון|אוכל)|מתכון\s+ל"
    r"|מזג\s+(?:האוויר|אוויר)\b"
    r"|פסל\s+החירות|statue\s+of\s+liberty"
    r"|מי\s+ניצח|מי\s+זכה",
    re.IGNORECASE,
)

_OUT_OF_DOMAIN_HE = (
    "אני מיועד/ת להסברים בתחום גנטיקה, בדיקות גנטיות ומושגים רפואיים קשורים."
)


def _detect_out_of_domain(text: str) -> bool:
    """True when the question is clearly outside the genetics/medicine domain."""
    return bool(_OUT_OF_DOMAIN_RE.search(text))


# ---------------------------------------------------------------------------
# VUS options — practical patient answer for "what are my options?" + VUS
# ---------------------------------------------------------------------------

_VUS_OPTIONS_KEYWORDS = frozenset([
    "אפשרויות",
    "מה לעשות",
    "מה עושים",
    "מה כדאי לעשות",
    "מה הצעדים",
    "מה עלי לעשות",
    "מה ניתן לעשות",
    "מה אפשר לעשות",
    "next steps",
    "what can i do",
    "what are my options",
])

_VUS_OPTIONS_HE = (
    "כשמתקבל VUS, יש כמה דברים שכדאי לדעת:\n\n"
    "1. VUS אינו שקול לממצא pathogenic — הוא לא אישור למחלה ולא ממצא תקין לחלוטין. "
    "פשוט אין עדיין מספיק ראיות מדעיות לסיווג ברור.\n\n"
    "2. בדרך כלל לא מקבלים החלטות רפואיות משמעותיות רק על בסיס VUS — הצוות הגנטי "
    "מתייחס לתמונה הקלינית המלאה.\n\n"
    "3. חשוב לשמור תיעוד של הממצא — סיווג VUS יכול להתעדכן בעתיד ככל שמצטברות "
    "ראיות מדעיות חדשות, ולכן כדאי לשמור קשר עם הצוות הגנטי.\n\n"
    "4. לעיתים, בדיקת קרובי משפחה יכולה לתרום לסיווג מדויק יותר — זה נקרא בדיקת "
    "segregation.\n\n"
    "5. כדאי לעקוב אם הסיווג מתעדכן בעתיד ולשמור את פרטי הממצא בצורה מסודרת."
)


def _is_vus_options_request(text: str) -> bool:
    """True when text asks for practical options/steps in the context of VUS."""
    lower = text.lower()
    return any(kw in lower for kw in _VUS_OPTIONS_KEYWORDS)


def _build_vus_options_answer(gene: Optional[str] = None) -> dict:
    gene_prefix = f"לגבי VUS בגן {gene}:\n\n" if gene else ""
    return {
        "answer": gene_prefix + _VUS_OPTIONS_HE,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "vus_known_gene",
        "suggested_questions": list(_GENE_SUGGESTED_QUESTIONS),
        "llm_used": False,
        "fallback_used": True,
    }


# ---------------------------------------------------------------------------
# Educational personal context detection
# ---------------------------------------------------------------------------

_EDUCATIONAL_PERSONAL_PHRASES: tuple = (
    "אמרו לי",
    "אמרה לי",
    "אמר לי",
    "הרופא אמר",
    "הרופאה אמרה",
    "הצוות אמר",
    "הגנטיקאי אמר",
    "הגנטיקאית אמרה",
    "נמצא אצלי",
    "נמצאה אצלי",
    "נמצא אצלנו",
    "לתינוק שלי",
    "לילד שלי",
    "לילדה שלי",
    "לבן שלי",
    "לבת שלי",
)

_EDUCATIONAL_PERSONAL_DECISION_BLOCK: tuple = (
    "ניתוח",
    "כריתה",
    "כריתת",   # construct form of כריתה (e.g. "כריתת שד") — different char sequence
    "כימותרפיה",
    "הפלה",
    "מה הסיכון שלי",
    "כמה אחוז",
    "כמה סיכוי",
    "מה הסיכוי שלי",
    "האם יהיה לי",
    "האם יהיו לי",
    # Disease/diagnosis claims — also covers "אם אני חולה", "האם אני חולה"
    "אני חולה",
    "יש לי מחלה",
    "יש לי סרטן",
    "האם יש לי",
)


def _is_educational_personal_context(text: str) -> bool:
    """True when text has personal educational phrasing without decision/risk seeking."""
    lower = text.lower()
    has_personal = any(phrase in lower for phrase in _EDUCATIONAL_PERSONAL_PHRASES)
    if not has_personal:
        return False
    has_decision = any(phrase in lower for phrase in _EDUCATIONAL_PERSONAL_DECISION_BLOCK)
    return not has_decision


# ---------------------------------------------------------------------------
# Surgery / major-procedure decision detection
# ---------------------------------------------------------------------------

_SURGERY_DECISION_RE = re.compile(
    # "האם לעשות/לעבור" + major surgery term — decision-seeking pattern
    r"האם\s+ל(?:עשות|עבור)\s+(?:ניתוח|כריתה|כריתת)"
    # "לעבור ניתוח" standalone (used without "האם" prefix in some phrasings)
    r"|לעבור\s+ניתוח"
    # "כריתה מניעתית" — standalone preventive mastectomy, always a major decision
    r"|כריתה\s+מניע"
    r"|mastectomy"
    r"|preventive\s+surgery",
    re.IGNORECASE,
)


def _is_surgery_decision_question(text: str) -> bool:
    return bool(_SURGERY_DECISION_RE.search(text))


def _build_surgery_decision_answer(
    text: str, gene_symbol: Optional[str] = None
) -> dict:
    """Contextual high-stakes answer for surgery/major-procedure decision questions."""
    lower = text.lower()
    has_vus = _mentions_vus(lower)

    if "כריתת שד" in lower or "mastectomy" in lower:
        action_he = "כריתת שד"
    elif "כריתה" in lower or "כריתת" in lower:
        action_he = "כריתה מניעתית" if "מניע" in lower else "כריתה"
    else:
        action_he = "ניתוח"

    gene_str = f" ב{gene_symbol}" if gene_symbol else ""

    if has_vus:
        answer = (
            f"VUS{gene_str} אינו שקול לממצא פתוגני, ובדרך כלל לא מקבלים החלטות "
            f"ניתוחיות משמעותיות כמו {action_he} על סמך VUS בלבד. "
            f"{action_he} היא החלטה שתלויה בדוח הגנטי המלא, בהיסטוריה האישית "
            f"והמשפחתית ובשיקולים קליניים נוספים. "
            f"הבוט יכול להסביר את ההבדל בין VUS לממצא פתוגני, אך לא להמליץ על ניתוח."
        )
    else:
        answer = (
            f"שאלה בנוגע ל{action_he} היא החלטה רפואית שתלויה בדוח הגנטי המלא, "
            f"בהיסטוריה האישית והמשפחתית ובשיקולים קליניים נוספים. "
            f"הצוות הגנטי הוא שמכיר את המקרה שלך ויסביר מה המשמעות הקלינית."
        )

    return {
        "answer": answer,
        "safety_level": "requires_genetic_counselor",
        "needs_genetic_counselor": True,
        "matched_topic": "surgery_decision",
        "suggested_questions": [
            "מה ההבדל בין VUS לממצא פתוגני?",
            "אילו גורמים שוקלים לפני ניתוח מניעתי?",
        ],
    }


# ---------------------------------------------------------------------------
# Intro-only LLM system
# ---------------------------------------------------------------------------
# Curated gene education now lives in app/gene_cards.py (loaded from
# data/gene_cards.json at startup). The LLM is restricted to generating a
# single validated warm Hebrew intro sentence that is prepended to the
# deterministic answer.  Medical content is NEVER generated by the LLM.

class LLMLayerResult:
    """Result from the LLM intro/framing layer."""
    __slots__ = ("answer", "llm_used", "attempted", "mode", "rejection_reason",
                 "repaired", "repair_reason", "retry_used")
    def __init__(self, answer, llm_used, attempted, mode, rejection_reason,
                 repaired=False, repair_reason=None, retry_used=False):
        self.answer = answer
        self.llm_used = llm_used
        self.attempted = attempted
        self.mode = mode
        self.rejection_reason = rejection_reason
        self.repaired = repaired
        self.repair_reason = repair_reason
        self.retry_used = retry_used


# English tokens permitted in a Hebrew intro sentence (gene symbols, clinical terms).
_ALLOWED_INTRO_TOKENS: frozenset = frozenset({
    "VUS", "DNA", "RNA", "ClinVar", "NCBI", "ACMG", "OMIM",
    "BRCA1", "BRCA2", "NF1", "APC", "TP53", "SHANK3", "HBB", "SOX1",
    "CFTR", "ATM", "MLH1", "PALB2", "CHEK2", "STK11", "PTEN", "RB1",
    "MSH2", "MSH6", "EPCAM", "MUTYH", "BRIP1", "RAD51C", "RAD51D",
    "NBN", "CDH1", "VHL", "POLE", "NTHL1", "BMPR1A", "SMAD4",
    "pathogenic", "Pathogenic", "benign", "Benign", "likely", "Likely",
    "uncertain", "Uncertain", "significance", "Significance",
    "Neurofibromatosis", "Familial", "Adenomatous", "Polyposis",
    "Lynch", "Li-Fraumeni",
})

# Forbidden phrases that must not appear in an LLM-generated intro.
# English terms use \b (works correctly for ASCII). Hebrew terms use substring
# match without \b because Hebrew word boundaries are unreliable with Python
# regex when common prefixes (ה, ל, ב, ו, ש, מ) are attached to the root.
_FORBIDDEN_INTRO_RE = re.compile(
    r"\b(?:surgery|treatment|surveillance|medication|colonoscopy|"
    r"diagnosis|prognosis|family\s+testing|personal\s+risk)\b"
    r"|ניתוח|טיפול|קולונוסקופיה|תרופה|מעקב\s+רפואי|"
    r"בדיקות?\s+משפחה|אבחנ|פרוגנוזה|"
    r"סיכון.{0,10}אישי",
    re.IGNORECASE,
)

# Quality-rejection phrases for controlled_framing mode.
# Rejects malformed Hebrew, chatbot closing lines, and VUS-defining sentences
# that belong in the deterministic answer, not in the framing layer.
# All matches are case-insensitive substrings (Hebrew word boundaries unreliable).
_FRAMING_QUALITY_RE = re.compile(
    # Malformed / invented Hebrew words
    r"ממצאון"
    # Informal or unprofessional phrases
    r"|סוגיה\s+שלך"
    r"|מאפיין\s+בהכרח"
    r"|מחלה\s+ספציפית"
    r"|אני\s+כאן\s+לעזור"
    r"|שאלות\s+נוספות"
    r"|אם\s+יש\s+לך\s+שאלות"
    # VUS-defining phrases — framing must not explain what VUS means
    r"|VUS\s+מציין"
    r"|VUS\s+הוא"
    r"|VUS\s+פירושו"
    r"|VUS\s+מייצג"
    r"|VUS\s+מגדיר"
    r"|VUS\s+מסמן"
    r"|פירוש\s+VUS"
    # Curly/square brace artifacts — produced by broken LLM template outputs
    r"|[{}\[\]]"
    # "new" keyword — appears in corrupted tokenizer output
    r"|\bnew\b"
    # Hebrew letter immediately followed by a closing bracket/brace
    r"|[\u05d0-\u05ea][})\]]"
    # Repeated malformed token pattern like "ת}ה ת}ק"
    r"|[\u05d0-\u05ea][})\]][\u05d0-\u05ea]"
    # Incorrect/invented Hebrew words
    r"|תפסיר"                          # wrong word — not valid Hebrew
    # Poor-grammar LLM filler phrases that add no value
    r"|שיכולים\s+לספק"                 # poor grammar — "that can provide"
    r"|הנה\s+מידע\s+כללי"              # vague filler intro
    r"|אתה\s+יכול\s+לשאול\s+על"       # vague filler — "you can ask about"
    r"|את\s+יכולה\s+לשאול\s+על"       # same, feminine form
    r"|זה\s+יכול\s+להיות",             # vague filler — "it could be"
    re.IGNORECASE,
)

_INTRO_SYSTEM_PROMPT = (
    "You are a warm assistant for a Hebrew post-genetic-counseling chatbot.\n\n"
    "Write EXACTLY ONE short, warm Hebrew sentence (15–30 words, max 200 characters) "
    "as an opening introduction to the response that follows.\n"
    "The sentence should acknowledge the user's question and indicate that helpful "
    "general information follows.\n\n"
    "STRICT RULES:\n"
    "- Write in Hebrew ONLY. Gene symbols (BRCA1, BRCA2, NF1, APC, TP53, etc.), "
    "medical terms (VUS, DNA), and ClinVar are allowed as-is.\n"
    "- Do NOT include question marks.\n"
    "- Do NOT give medical advice, diagnoses, treatment recommendations, "
    "or personal risk estimates.\n"
    "- Do NOT repeat or summarize the information that follows.\n"
    "- Write EXACTLY ONE sentence ending with a period.\n"
    "- Output only the sentence itself — no labels, no quotes, no preamble."
)

_CONTROLLED_FRAMING_SYSTEM_PROMPT = (
    "You are a professional assistant for a Hebrew post-genetic-counseling chatbot.\n\n"
    "Write EXACTLY ONE short warm Hebrew sentence (max 200 characters) that opens "
    "the educational response below.\n\n"
    "The sentence MAY:\n"
    "  - Acknowledge that genetic terminology can feel confusing or overwhelming.\n"
    "  - Indicate that general educational information follows.\n"
    "  - Note that the genetics team should be consulted for personal interpretation.\n\n"
    "The sentence MUST NOT:\n"
    "  - Define or explain what VUS means.\n"
    "  - Describe what any gene does, causes, or is associated with.\n"
    "  - Mention disease, risk, prognosis, management, or treatment.\n"
    "  - Include medical interpretation or personal medical meaning.\n"
    "  - Add chatbot phrases such as 'I am here to help', 'if you have questions', "
    "or 'additional questions'.\n"
    "  - Use informal or invented Hebrew words.\n"
    "  - Include question marks.\n\n"
    "FORMAT — violation causes your output to be discarded:\n"
    "  - Hebrew ONLY. Gene symbols (BRCA1, BRCA2, NF1, APC, TP53, etc.), "
    "VUS, ClinVar, DNA, ACMG are allowed unchanged.\n"
    "  - Output ONLY the single sentence. No preamble, no labels, no quotes.\n"
    "  - End the sentence with a period."
)

_TIER2_FRAMING_SYSTEM_PROMPT = (
    "You are a warm assistant for a Hebrew post-genetic-counseling chatbot.\n\n"
    "Write 1-2 SHORT Hebrew sentences (max 400 characters) that describe "
    "ClinVar statistical data for a gene in patient-friendly terms.\n\n"
    "STRICT RULES — violation causes your output to be discarded:\n"
    "- Write in Hebrew ONLY (gene symbols and ClinVar/VUS/pathogenic/benign/ACMG are allowed).\n"
    "- Do NOT describe the gene's biological function.\n"
    "- Do NOT state that this gene causes any condition or disease.\n"
    "- Do NOT give medical advice or personal risk estimates.\n"
    "- Do NOT include question marks.\n"
    "- Only describe the statistical picture (how many records, rough distribution).\n"
    "- Total output must be ≤ 400 characters.\n"
    "- Output only the sentences — no labels, no quotes, no preamble."
)

_CJK_RETRY_SYSTEM_PROMPT = (
    "You are a professional assistant for a Hebrew post-genetic-counseling chatbot.\n\n"
    "CRITICAL: your previous response was rejected because it contained non-Hebrew scripts.\n\n"
    "Write EXACTLY ONE short warm Hebrew sentence (max 200 characters).\n\n"
    "MANDATORY LANGUAGE RULES:\n"
    "- Hebrew ONLY. No Chinese. No Japanese. No Korean. No Arabic. No Russian.\n"
    "- No Cyrillic, no CJK, no Arabic script at all.\n"
    "- The ONLY allowed non-Hebrew terms: gene symbols (BRCA1, BRCA2, NF1, APC, TP53, etc.), "
    "ClinVar, VUS, pathogenic, benign, likely pathogenic, likely benign, DNA, ACMG.\n\n"
    "CONTENT RULES:\n"
    "- Do NOT give medical advice, diagnoses, personal risk estimates, or treatment recommendations.\n"
    "- Do NOT define what VUS means or describe gene biology.\n"
    "- Do NOT include chatbot phrases like 'I am here to help' or 'additional questions'.\n"
    "- Do NOT include question marks.\n"
    "- Total output ≤ 200 characters.\n"
    "- Return ONLY the single sentence. No explanations. No labels. No preamble."
)


def _validate_llm_intro(text: str) -> bool:
    """
    Validate a single LLM-generated Hebrew intro sentence for safety and format.

    Accepts iff ALL of the following hold:
    - Non-empty and ≤ 200 characters
    - Contains Hebrew characters
    - Hebrew ≥ 50 % of all alphabetic characters
    - No question marks
    - No CJK characters
    - No forbidden medical-advice terms
    - No English words longer than 3 characters that are not in _ALLOWED_INTRO_TOKENS

    Returns True if safe; False otherwise.
    """
    return _validate_intro_with_reason(text) is None


def _validate_intro_with_reason(text: str) -> Optional[str]:
    """
    Validate a single LLM-generated Hebrew intro sentence for safety and format.

    Returns None if valid, or a string describing the rejection reason.
    Max length checked is 200 characters (intro_only constraint).
    """
    text = text.strip()
    if not text:
        return "empty"
    if len(text) > 200:
        return f"too long ({len(text)} chars > 200)"
    if "?" in text or "？" in text:
        return "contains question mark"
    if re.search(r"[一-鿿぀-ゟ゠-ヿ]", text):
        return "contains CJK characters"
    if _FORBIDDEN_INTRO_RE.search(text):
        return "contains forbidden medical-advice term"
    m = _FRAMING_QUALITY_RE.search(text)
    if m:
        return f"quality-rejected phrase: {m.group(0)!r}"
    hebrew_chars = re.findall(r"[֐-׿יִ-ﭏ]", text)
    if not hebrew_chars:
        return "no Hebrew characters"
    total_alpha = len(re.findall(r"[A-Za-z֐-׿יִ-ﭏ]", text))
    if total_alpha and len(hebrew_chars) / total_alpha < 0.5:
        return f"Hebrew ratio too low ({len(hebrew_chars)}/{total_alpha})"
    allowed_upper = {t.upper() for t in _ALLOWED_INTRO_TOKENS}
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*\b", text):
        if len(word) > 3 and word.upper() not in allowed_upper:
            logger.debug("LLM intro (with_reason) rejected — unknown English token %r", word)
            return f"unknown English token {word!r}"
    return None


def _validate_controlled_framing(text: str) -> Optional[str]:
    """
    Validate a controlled-framing LLM output (1 sentence, max 600 chars).

    Applies safety checks plus _FRAMING_QUALITY_RE, which rejects malformed
    Hebrew words, chatbot closing phrases, and VUS-defining sentences.
    Returns None if valid, or a string describing the rejection reason.
    """
    text = text.strip()
    if not text:
        return "empty"
    if len(text) > 600:
        return f"too long ({len(text)} chars > 600)"
    if "?" in text or "？" in text:
        return "contains question mark"
    if re.search(r"[一-鿿぀-ゟ゠-ヿ]", text):
        return "contains CJK characters"
    if _FORBIDDEN_INTRO_RE.search(text):
        return "contains forbidden medical-advice term"
    m = _FRAMING_QUALITY_RE.search(text)
    if m:
        return f"quality-rejected phrase: {m.group(0)!r}"
    hebrew_chars = re.findall(r"[֐-׿יִ-ﭏ]", text)
    if not hebrew_chars:
        return "no Hebrew characters"
    total_alpha = len(re.findall(r"[A-Za-z֐-׿יִ-ﭏ]", text))
    if total_alpha and len(hebrew_chars) / total_alpha < 0.5:
        return f"Hebrew ratio too low ({len(hebrew_chars)}/{total_alpha})"
    allowed_upper = {t.upper() for t in _ALLOWED_INTRO_TOKENS}
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*\b", text):
        if len(word) > 3 and word.upper() not in allowed_upper:
            logger.debug("Controlled framing rejected — unknown English token %r", word)
            return f"unknown English token {word!r}"
    return None


def _validate_tier2_framing(text: str, gene: str) -> Optional[str]:
    """
    Validate a Tier-2 statistical framing LLM output.

    First applies _validate_controlled_framing checks (max 600 chars).
    Additionally rejects if the text makes biological function claims about the gene.
    Returns None if valid, or a string describing the rejection reason.
    """
    reason = _validate_controlled_framing(text)
    if reason is not None:
        return reason
    bio_claim_re = re.compile(
        r"(?:הגן\s+" + re.escape(gene) + r"|" + re.escape(gene) + r"\s+הוא\s+גן)\s*"
        r"(?:אחראי|גורם|מקודד|פועל|ממלא|שולט)",
        re.IGNORECASE,
    )
    if bio_claim_re.search(text):
        return f"claims gene biological function for Tier-2 gene {gene!r}"
    return None


# Pattern for all-uppercase gene symbols (POLE, BRCA1, TP53, HBB …)
_GENE_SYMBOL_PAT = re.compile(r"^[A-Z][A-Z0-9]+$")
# Latin words that are approved regardless (not gene symbols but still OK)
_APPROVED_LATIN_WORDS: "frozenset[str]" = frozenset({"ClinVar"})
# Word boundary scan for Latin tokens
_LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")


def _count_non_gene_latin_words(text: str) -> int:
    """Count Latin words that are NOT all-caps gene symbols or approved terms."""
    count = 0
    for tok in _LATIN_TOKEN_RE.findall(text):
        if tok in _APPROVED_LATIN_WORDS:
            continue
        if _GENE_SYMBOL_PAT.match(tok):
            continue
        count += 1
    return count


def _validate_unverified_draft(text: str) -> "Optional[str]":
    """
    Strict first-pass validator for patient-visible unverified AI gene background drafts.

    Rejects personal-risk language, medical action terms, broken/hype output,
    database statistics, and ClinVar mentions.  A ClinVar mention triggers a
    retry with a stricter prompt; other hard failures skip the retry and go
    directly to the deterministic fallback.

    Returns None if valid, or a string describing the rejection reason.
    See also _validate_unverified_draft_clinvar_ok for the relaxed second-pass
    variant that accepts ClinVar if everything else is safe.
    """
    text = text.strip()
    if not text:
        return "empty"
    if len(text) > 600:
        return f"too long ({len(text)} chars > 600)"
    if "?" in text or "？" in text:
        return "contains question mark"
    if re.search(r"[一-鿿぀-ゟ゠-ヿ]", text):
        return "contains CJK characters"
    if _FORBIDDEN_INTRO_RE.search(text):
        return "contains forbidden medical-advice term"
    m = _FRAMING_QUALITY_RE.search(text)
    if m:
        return f"quality-rejected phrase: {m.group(0)!r}"
    m = _DRAFT_QUALITY_RE.search(text)
    if m:
        return f"draft-quality phrase: {m.group(0)!r}"
    if _CLINVAR_IN_DRAFT_RE.search(text):
        return "draft-quality phrase: 'ClinVar'"
    m = _FORBIDDEN_DRAFT_PERSONAL_RE.search(text)
    if m:
        return f"personal-risk phrase: {m.group(0)!r}"
    non_gene_latin = _count_non_gene_latin_words(text)
    if non_gene_latin > 1:
        return f"too many non-gene Latin words ({non_gene_latin})"
    heb = re.findall("[א-׿]", text)
    if not heb:
        return "no Hebrew characters"
    total_alpha = len(re.findall("[A-Za-zא-׿]", text))
    if total_alpha and len(heb) / total_alpha < 0.4:
        return f"Hebrew ratio too low ({len(heb)}/{total_alpha})"
    return None


def _validate_unverified_draft_clinvar_ok(text: str) -> "Optional[str]":
    """
    Relaxed second-pass validator for unverified AI gene background drafts.

    Identical to _validate_unverified_draft except that a ClinVar mention is
    NOT treated as a failure.  If the retry prompt already discouraged ClinVar
    and the output still mentions it but is otherwise safe (no statistics, no
    personal risk, no medical action), we accept the draft rather than always
    falling back to the deterministic path.

    'ClnVar' (LLM typo), קלינוואר, database statistics, English clinical
    classification terms, and all real safety gates still apply.
    """
    text = text.strip()
    if not text:
        return "empty"
    if len(text) > 600:
        return f"too long ({len(text)} chars > 600)"
    if "?" in text or "？" in text:
        return "contains question mark"
    if re.search(r"[一-鿿぀-ゟ゠-ヿ]", text):
        return "contains CJK characters"
    if _FORBIDDEN_INTRO_RE.search(text):
        return "contains forbidden medical-advice term"
    m = _FRAMING_QUALITY_RE.search(text)
    if m:
        return f"quality-rejected phrase: {m.group(0)!r}"
    m = _DRAFT_QUALITY_RE.search(text)  # includes ClnVar and stats; ClinVar not included
    if m:
        return f"draft-quality phrase: {m.group(0)!r}"
    # _CLINVAR_IN_DRAFT_RE intentionally NOT checked: ClinVar allowed on second pass
    m = _FORBIDDEN_DRAFT_PERSONAL_RE.search(text)
    if m:
        return f"personal-risk phrase: {m.group(0)!r}"
    non_gene_latin = _count_non_gene_latin_words(text)
    if non_gene_latin > 1:
        return f"too many non-gene Latin words ({non_gene_latin})"
    heb = re.findall("[א-׿]", text)
    if not heb:
        return "no Hebrew characters"
    total_alpha = len(re.findall("[A-Za-zא-׿]", text))
    if total_alpha and len(heb) / total_alpha < 0.4:
        return f"Hebrew ratio too low ({len(heb)}/{total_alpha})"
    return None


# Blocks for gene education drafts: ONLY personal risk, diagnosis, medical actions.
_FORBIDDEN_GENE_EDUCATION_DRAFT_RE = re.compile(
    # Personal risk language (but not "יש לך שאלות" — a suggestions phrase)
    r"יש\s+לך\s+(?:מחלה|סרטן|גידול|בעיה|סיכון|שינוי\s+שמוביל)"
    r"|אצלך\s+יש"
    r"|הסיכון\s+שלך"
    r"|הממצא\s+שלך\s+אומר"
    r"|התוצאה\s+שלך\s+מצביעה"
    # Medical action directives (personal)
    r"|(?:את|אתה)\s+(?:צריכה|צריך)\s+(?:ניתוח|טיפול|כריתה|MRI|מעקב)"
    r"|עליך\s+ל"
    r"|כדאי\s+לך\s+ל"
    r"|מומלץ\s+לך\s+ל"
    r"|צריכ[אה]?\s+לך"
    r"|עליכם?\s+לעבור"
    # Diagnosis / abortion
    r"|אובחנת"
    r"|אבחנת"
    r"|הגן\s+גורם\s+לך"
    r"|הפלה|להפיל|הפסקת\s+הריון",
    re.IGNORECASE,
)

# Broken/hype output patterns rejected even in lenient mode.
_GENE_EDUCATION_BROKEN_RE = re.compile(
    r"[\U0001F300-\U0001FAFF]"   # emoji ranges
    r"|[\U0001F000-\U0001F9FF]"
    r"|600\s+characters"
    r"|Maximum\s+600"
    r"|\bClnVar\b"                # common LLM typo for ClinVar
    r"|\bClanVar\b"               # another typo variant
    r"|קלינוואר"                  # Hebrew transliteration of ClinVar
    r"|\[.*?\]"                   # square-bracket template artifacts
    r"|\{.*?\}"                   # curly-brace template artifacts
    r"|קסם"                       # "magic" — hype
    r"|מרתק"                      # "fascinating" — hype
    r"|מדהי[מם]",                 # "amazing" — hype
    re.IGNORECASE,
)

# Rejects drafts that are primarily ClinVar-count summaries rather than biology.
# These appear when the conflicting context block was passed to the biology prompt.
_GENE_EDUCATION_CLINVAR_COUNT_RE = re.compile(
    r"(?:\d[\d,]*)\s+וריאנטים?\s+(?:פתוגניים|pathogenic|ב-?ClinVar|במאגר)"
    r"|(?:\d[\d,]*)\s+וריאנטים?\s+ב-?ClinVar"
    r"|ב-?ClinVar\s+(?:יש|נמצא|מתועד|קיים)"
    r"|יש\s+לו\s+(?:\d[\d,]*)\s+וריאנטים"
    r"|נמצאו\s+(?:\d[\d,]*)\s+וריאנטים",
    re.IGNORECASE,
)


# Disease/cancer association language — hallucination risk in unapproved gene drafts.
# Allows specific single named associations ("קשור למחלת Duchenne", "קשור לסרטן המעי הגס").
# Blocks vague/list patterns that invite hallucination.
_GENE_EDUCATION_DISEASE_ASSOC_RE = re.compile(
    # Definitive causation — always block; use "קשור ל" instead.
    # Covers all inflected forms: גורם (m.sg, final-mem ם), גורמת (f.sg),
    # גורמים (m.pl), גורמות (f.pl). The previous גורמ(ת|ים|ות)? missed גורם
    # because ם (final-mem U+05DD) ≠ מ (medial-mem U+05DE).
    r"גור(?:ם|מת|מים|מות)\s+ל(?:מחלה|מחלות|סרטן|תסמונת|הפרעה)"
    # Cancer predisposition framing — personal risk framing.
    r"|נטייה\s+לסרטן"
    # "various cancer types" — invitation to invent a list.
    r"|סוגי\s+סרטן\s+שונים"
    # "various/many diseases" — vague list indicator without specific type qualifier.
    # Blocks: "מחלות שונות של מערכת העצבים", "מחלות רבות".
    # Allows: "מחלות נוירולוגיות כמו אלצהיימר" (specific adjective, not "שונות/רבות").
    r"|מחלות\s+(?:שונות|רבות|מגוונות)"
    # Broad hallucination-prone list phrases — "wide range of diseases" / "many conditions".
    r"|מגוון\s+רחב\s+של\s+מחלות"
    r"|מצבים\s+רבים\s+(?:כגון|כמו)"
    # "conditions like X, Y, Z" / "diseases like X, Y, Z" — vague list without named qualifier.
    # NOTE: "מחלות [specific adjective] כמו" (e.g. "מחלות נוירולוגיות כמו") does NOT match
    # \s+כמו because \s+ requires IMMEDIATE whitespace between מחלות and כמו.
    r"|מצבים\s+כמו"
    r"|מחלות\s+כמו",
    re.IGNORECASE,
)


def _validate_gene_education_draft(text: str) -> "Optional[str]":
    """
    Lenient validator for general gene education drafts.

    Blocks only:
    - Personal risk / diagnosis / medical action language
    - Broken output (hype, emoji, template artifacts)
    - CJK characters
    - Empty / too short
    - No Hebrew characters

    Allows biological function, English biomedical terms, gene symbols, and VUS
    in general educational context.

    Returns None if valid, or a string rejection reason.
    """
    text = text.strip()
    if not text or text == "-":
        return "empty"
    if len(text) > 600:
        return f"too long ({len(text)} > 600)"
    if re.search(r"[一-鿿぀-ゟ゠-ヿ]", text):
        return "contains CJK characters"
    m = _FORBIDDEN_GENE_EDUCATION_DRAFT_RE.search(text)
    if m:
        return f"personal/action phrase: {m.group(0)!r}"
    m = _GENE_EDUCATION_DISEASE_ASSOC_RE.search(text)
    if m:
        return f"disease_assoc phrase: {m.group(0)!r}"
    m = _GENE_EDUCATION_BROKEN_RE.search(text)
    if m:
        return f"broken/hype output: {m.group(0)!r}"
    m = _GENE_EDUCATION_CLINVAR_COUNT_RE.search(text)
    if m:
        return f"clinvar_count_summary (not biology): {m.group(0)!r}"
    heb = re.findall("[א-׿]", text)
    if not heb:
        return "no Hebrew characters"
    return None


# ---------------------------------------------------------------------------
# ClinVar phenotype normalization — maps raw English ClinVar phenotype strings
# to clean, patient-friendly Hebrew labels before they reach the LLM.
# Ordered most-specific first so the first matching keyword wins.
# ---------------------------------------------------------------------------
_PHENOTYPE_NORM: "list[tuple[str, str | None]]" = [
    # Specific syndromes — must come before generic cancer patterns
    ("Polymerase proofreading-related adenomatous polyposis",
     "פוליפוזיס אדנומטוטית הקשורה למנגנון הגהה של שכפול DNA"),
    ("Familial adenomatous polyposis",   "פוליפוזיס אדנומטוטית משפחתית"),
    ("Adenomatous polyposis",            "פוליפוזיס אדנומטוטית"),
    ("Hereditary breast and ovarian cancer",
     "נטייה תורשתית לסרטן שד ושחלה"),
    ("Hereditary breast",                "נטייה תורשתית לסרטן שד"),
    ("Lynch syndrome",                   "תסמונת לינץ׳"),
    ("Li-Fraumeni syndrome",             "תסמונת לי-פראומני"),
    ("Cowden syndrome",                  "תסמונת קאודן"),
    ("Peutz-Jeghers syndrome",           "תסמונת פויץ-יאגרס"),
    ("Von Hippel-Lindau",                "תסמונת פון היפל-לינדאו"),
    ("Multiple endocrine neoplasia",     "ניאופלזיה אנדוקרינית מרובה"),
    ("Neurofibromatosis",                "נוירופיברומטוזיס"),
    # Cancer predisposition — general
    ("Hereditary cancer-predisposing syndrome",
     "נטייה תורשתית למצבים סרטניים"),
    ("Hereditary cancer",                "נטייה תורשתית למצבים סרטניים"),
    ("Cancer predisposition",            "נטייה תורשתית למצבים סרטניים"),
    ("Tumor predisposition",             "נטייה תורשתית לגידולים"),
    # Specific cancer sites
    ("Familial colorectal cancer",       "סרטן מעי גס משפחתי"),
    ("Colorectal cancer",                "נטייה לסרטן המעי הגס"),
    ("Colon cancer",                     "נטייה לסרטן המעי הגס"),
    ("Rectal cancer",                    "נטייה לסרטן פי הטבעת"),
    ("Endometrial cancer",               "נטייה לסרטן הרחם"),
    ("Uterine cancer",                   "נטייה לסרטן הרחם"),
    ("Ovarian cancer",                   "נטייה לסרטן השחלה"),
    ("Breast cancer",                    "נטייה לסרטן השד"),
    ("Prostate cancer",                  "נטייה לסרטן הערמונית"),
    ("Pancreatic cancer",                "נטייה לסרטן הלבלב"),
    ("Gastric cancer",                   "נטייה לסרטן הקיבה"),
    ("Stomach cancer",                   "נטייה לסרטן הקיבה"),
    ("Lung cancer",                      "נטייה לסרטן הריאה"),
    ("Thyroid cancer",                   "נטייה לסרטן בלוטת התריס"),
    ("Melanoma",                         "נטייה למלנומה"),
    ("Glioma",                           "גידולי מוח"),
    ("Medulloblastoma",                  "גידולי מוח"),
    ("Wilms tumor",                      "גידול וילמס"),
    # Cardiac
    ("Dilated cardiomyopathy",           "קרדיומיופתיה מורחבת"),
    ("Hypertrophic cardiomyopathy",      "קרדיומיופתיה היפרטרופית"),
    ("Left ventricular non-compaction",  "קרדיומיופתיה לא-קומפקטית"),
    ("Arrhythmogenic right ventricular", "קרדיומיופתיה אריתמוגנית"),
    ("Long QT syndrome",                 "תסמונת QT ממושך"),
    ("Brugada syndrome",                 "תסמונת ברוגדה"),
    ("Cardiomyopathy",                   "קרדיומיופתיה"),
    # Blood / hemoglobin
    ("Sickle cell",                      "מחלת תאי חרמש"),
    ("Beta-thalassemia",                 "בטא-תלסמיה"),
    ("Thalassemia",                      "תלסמיה"),
    ("Hemoglobinopathy",                 "מחלות המוגלובין"),
    ("Hemoglobin",                       "מצבים הקשורים להמוגלובין"),
    ("Anemia",                           "אנמיה"),
    # Respiratory
    ("Cystic fibrosis",                  "סיסטיק פיברוזיס"),
    # Neuro / developmental
    ("Autism spectrum",                  "מצבים בספקטרום האוטיסטי"),
    ("Intellectual disability",          "מוגבלות אינטלקטואלית"),
    ("Developmental delay",              "עיכוב התפתחותי"),
    # Uninformative entries — map to None so they are skipped
    ("not provided",                     None),
    ("not specified",                    None),
    ("not supplied",                     None),
    ("see cases",                        None),
    ("all of the above",                 None),
]


def _map_phenotype_to_hebrew(raw: str, gene: str = "") -> "Optional[str]":
    """
    Return a Hebrew patient-facing label for one ClinVar phenotype string,
    or None to indicate the entry should be skipped.
    """
    lower = raw.lower().strip()

    # Gene-specific "related / disorder / associated" pattern (uses caller-supplied gene)
    if gene:
        g_lower = gene.lower()
        if g_lower in lower and any(
            w in lower for w in ("related", "disorder", "associated", "syndrome")
        ):
            return f"מצבים הקשורים לגן {gene}"

    for keyword, label_he in _PHENOTYPE_NORM:
        if keyword.lower() in lower:
            return label_he  # may be None → skip

    return None  # unrecognised — skip


def _normalize_clinvar_phenotypes_for_patient(
    top_phenotypes: list, gene: str = ""
) -> "list[str]":
    """
    Map raw English ClinVar phenotype strings to clean Hebrew labels.

    * Strings containing ";" are split; each part is normalized individually.
    * Parts longer than 120 characters without a recognised mapping are skipped.
    * Duplicate Hebrew labels are deduplicated (order preserved).
    * Returns at most 5 unique Hebrew labels.
    """
    out: list = []
    seen: set = set()

    for raw in (top_phenotypes or []):
        parts = [p.strip() for p in str(raw).split(";")]
        for part in parts:
            if not part:
                continue
            if len(part) > 120:
                continue
            label = _map_phenotype_to_hebrew(part, gene)
            if not label:
                continue
            if label in seen:
                continue
            seen.add(label)
            out.append(label)
            if len(out) >= 5:
                return out
    return out


# ---------------------------------------------------------------------------
# Deterministic ClinVar fallback draft — used when both LLM attempts fail
# but normalised ClinVar contexts are available.  Not LLM-generated.
# ---------------------------------------------------------------------------
def _build_deterministic_clinvar_draft(
    gene: str, normalized_contexts: "list[str]"
) -> "Optional[dict]":
    """
    Build a safe, fully deterministic unverified draft from normalised contexts.
    Returns None when normalized_contexts is empty (nothing to say).
    approved=False and review_status='unreviewed' are set unconditionally.
    """
    from datetime import datetime, timezone

    if not normalized_contexts:
        return None

    if len(normalized_contexts) == 1:
        ctx_text = normalized_contexts[0]
    elif len(normalized_contexts) == 2:
        ctx_text = f"{normalized_contexts[0]} ו{normalized_contexts[1]}"
    else:
        # Up to 3 contexts, joined naturally
        ctx_text = (
            f"{normalized_contexts[0]}, {normalized_contexts[1]}"
            f" ו{normalized_contexts[2]}"
        )

    text = (
        f"באופן כללי, שינויים בגן {gene} מופיעים לעיתים בהקשרים של {ctx_text}. "
        "חשוב להדגיש שזהו מידע כללי בלבד, ואינו אומר שהממצא האישי שלך "
        "גורם למחלה או שיש לו משמעות פתוגנית."
    )

    return {
        "visible": True,
        "status": "deterministic_clinvar_summary",
        "gene_symbol": gene,
        "warning_he": _UNVERIFIED_DRAFT_WARNING_HE,
        "text_he": text,
        "generated_by_model": "deterministic",
        "review_status": "unreviewed",
        "approved": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "based_on": "clinvar_metadata",
        "source_note_he": _UNVERIFIED_DRAFT_SOURCE_NOTE_HE,
    }


# ---------------------------------------------------------------------------
# Staging-only AI draft debug mode (temporary; NOT for customer-facing deploy)
# ---------------------------------------------------------------------------

# Genes whose rejected AI drafts may be exposed in debug metadata when
# APP_ENV=staging/development and AI_DRAFT_DEBUG_SHOW_REJECTED=true.
# Only general education genes with no high-stakes phenotype ambiguity.
_DEBUG_SAFE_GENES: frozenset = frozenset({"APOE", "CFTR", "TLR3", "ABO", "PTEN"})

# Phrases that mark a question as high-stakes personal; raw AI text must never
# be exposed even in debug mode when these appear.
_HIGH_STAKES_DEBUG_PHRASES: tuple = (
    "סרטן", "cancer", "סיכון", "risk", "ניתוח", "surgery",
    "טיפול", "treatment", "תרופה", "medication", "הפסקת הריון",
    "abortion", "האם יש לי", "האם אני חולה", "diagnos",
    "האם עלי", "האם כדאי", "האם צריך",
)


def _ai_draft_debug_mode_active() -> bool:
    """True only in staging/development with AI_DRAFT_DEBUG_SHOW_REJECTED=true."""
    env = os.environ.get("APP_ENV", "production").strip().lower()
    if env not in ("staging", "development"):
        return False
    flag = os.environ.get("AI_DRAFT_DEBUG_SHOW_REJECTED", "").strip().lower()
    return flag in ("1", "true", "yes")


def _question_has_high_stakes_phrases(question: str) -> bool:
    q = question.lower()
    return any(phrase.lower() in q for phrase in _HIGH_STAKES_DEBUG_PHRASES)


# ---------------------------------------------------------------------------
# General education AI fallback (Session 19)
# Env: AI_GENERAL_EDUCATION_FALLBACK_ENABLED=true (default: false)
# Only active when APP_ENV=staging or development.
# Fires at pipeline step 6 when KB finds no match and the question is a
# safe general concept question (not personal/high-stakes).
# ---------------------------------------------------------------------------

def _ai_general_education_fallback_enabled() -> bool:
    """True when OPENAI_API_KEY is set, or in staging/dev with flag enabled."""
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return True
    env = os.environ.get("APP_ENV", "production").strip().lower()
    if env in ("staging", "development"):
        flag = os.environ.get("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "").strip().lower()
        if flag in ("1", "true", "yes"):
            return True
    return False


# Positive signals: the question is asking to EXPLAIN a concept.
_GENERAL_EDU_INTENT_PHRASES: tuple = (
    "מה זה",
    "מה הם ",
    "מהי ",
    "מהו ",
    "מה ה",
    "הסבר לי",
    "הסבירי לי",
    "תסביר לי",
    "תסבירי לי",
    "תסביר ",
    "תסבירי ",
    "הגדר",
    "תגדיר",
    "מה פירוש",
    "מה המשמעות של",
    "מה ההבדל בין",
    "מה הקשר בין",
    "מה נחשב",
    # Medical/disease questions — "do people die from...", "what happens with..."
    "מתים מ",
    "מה קורה כש",
    "מה קורה ל",
    "איך עובד",
    "איך עובדת",
    "מה גורם ל",
    "מה מאפיין",
    "מה הסיבה ל",
    "מה הגן ל",   # "what gene is responsible for albinism?"
    "מה הגן של",
    "איזה גן",    # "which gene..."
    "כיצד",      # "how..."
    "what is ",
    "what are ",
    "explain ",
    "define ",
    "what does",
    "what's the difference",
    # Historical / discovery questions — "מתי גילו את הגנום?" must reach
    # general educational AI, not KB gene-count entries.
    "מתי ",
)

# Additional personal / high-stakes signals NOT caught by safety.py step 3.
# Block the general AI fallback when any of these appear.
# NOTE: "שלי" and "אצלי" were intentionally removed — they are too broad and
# block legitimate educational questions ("לתינוק שלי", "נמצא אצלי VUS").
_GENERAL_EDU_EXTRA_BLOCK_PHRASES: tuple = (
    "יהיה לי",
    "יהיו לי",
    "יהיו לך",
    "הממצא שלי",
    "התוצאה שלי",
    "הבדיקה שלי",
    "הגן שלי",
    "הסיכון שלי",
    "מסוכן עבורי",
    "מסוכן לי",
    "מה אני צריכה",
    "מה אני צריך",
    "מה עלי לעשות",
    "כדאי לי",
    "כדאי לך",
    "מומלץ לי",
)


# Non-human biology organism signals — these questions are always low-risk general education
# (e.g. "כמה גנים יש לצב ים?", "כמה כרומוזומים יש לכלב?").
_NON_HUMAN_BIOLOGY_ORGANISMS = frozenset([
    "צב ים", "צב", "כלב", "חתול", "דג", "דגים", "ציפור", "שימפנזה", "קוף", "קופ",
    "עכבר", "עכברים", "חיידק", "חיידקים", "וירוס", "וירוסים", "צמח", "צמחים",
    "בעלי חיים", "בעל חיים",  # without definite article
    "בעלי החיים", "בעל החיים",  # with definite article (ה-prefix)
    "dog", "cat", "fish", "bird", "mouse", "bacteria", "virus", "plant",
    "chimpanzee", "turtle", "sea turtle", "animals",
])

# Broad genetics vocabulary that is always educational (not personal/clinical)
_BROAD_GENETICS_EDU_SIGNALS = frozenset([
    "כמה גנים", "כמה כרומוזומים", "כמה בסיסים", "כמה נוקלאוטידים",
    "גן לאלל", "מה זה אלל", "מה הם אללים",
    "מוטציה סומטית", "מוטציה נבטית", "מוטציה גרמינלית",
    "מהו גנום", "מה זה גנום", "מה הגנום", "מהי גנומיקה",
    "שני עותקים של", "שני כרומוזומים", "כרומוזום הומולוגי",
    "dna ל-rna", "dna ל rna", "dna לrna",
    "how many genes", "how many chromosomes",
])


# ---------------------------------------------------------------------------
# Carrier screening / ancestry-related genetic testing (Session 27.9.1 Part D)
# ---------------------------------------------------------------------------

_ANCESTRY_SIGNALS: frozenset = frozenset([
    "אשכנזי", "אשכנזייה", "אשכנזים",
    "ספרדי", "ספרדייה", "ספרדים",
    "מזרחי", "מזרחייה",
    "ממוצא",  # "of ancestry"
    "לפי מוצא",
])

_CARRIER_SCREENING_SIGNALS_D: frozenset = frozenset([
    "סקר נשאות",
    "בדיקות נשאות",
    "בדיקת נשאות",
    "carrier screening",
    "expanded carrier",
    "לפני היריון",
    "לפני הריון",
    "pre-pregnancy",
    "prenatal screening",
])

_CARRIER_SCREENING_TESTING_PHRASES: tuple = (
    "איזה בדיקות גנטיות",
    "אילו בדיקות גנטיות",
    "בדיקות גנטיות כדאי",
    "בדיקות גנטיות מומלצות",
    "בדיקות גנטיות לזוג",
    "אילו בדיקות נשאות",
    "איזה בדיקות נשאות",
    "בדיקות לפי מוצא",
    "בדיקות גנטיות לפני",
)


def _detect_carrier_screening_intent(text: str) -> "Optional[str]":
    """
    Detect carrier-screening / ancestry-related genetic testing questions.

    Returns:
      'carrier_screening_personal' — personal phrasing present (what should I/we do)
      'carrier_screening_general'  — impersonal / general question
      None                         — no carrier-screening signal found

    Fires before the KB lookup so ancestry questions never misroute to penetrance.
    """
    lower = text.lower()
    has_ancestry = any(s in lower for s in _ANCESTRY_SIGNALS)
    has_cs_signal = any(s in lower for s in _CARRIER_SCREENING_SIGNALS_D)
    has_testing_phrase = any(p in lower for p in _CARRIER_SCREENING_TESTING_PHRASES)

    if not (has_ancestry or has_cs_signal or has_testing_phrase):
        return None

    # Require a genetics/testing context word so we don't fire on unrelated uses
    has_genetics = any(
        w in lower
        for w in ["גנטי", "גנטיות", "גנטיקה", "בדיקה", "בדיקות",
                   "נשאות", "carrier", "screening", "genetic"]
    )
    if not has_genetics:
        return None

    _personal_forms = (
        "אם אני", "כדאי לי", "עלי לעשות", "מה אני צריכ",
        "שאעשה", "אם אנחנו", "כדאי לנו", "עלינו",
        "כדאי לעשות",  # "כדאי לי לעשות" is covered by this
    )
    is_personal = any(p in lower for p in _personal_forms)
    return "carrier_screening_personal" if is_personal else "carrier_screening_general"


# Carrier-screening answer content (Session 27.9.1 Part D)

_CARRIER_SCREENING_GENERAL_HE = (
    "סקר נשאות גנטית הוא בדיקה שנועדה לזהות אנשים שנושאים שינויים גנטיים "
    "הקשורים למחלות תורשתיות. נשאים בדרך כלל בריאים לחלוטין, אך עלולים להעביר "
    "את השינוי לילדיהם.\n\n"
    "פאנלים נפוצים:\n"
    "• פאנל אשכנזי קלאסי — כולל מחלות בתדירות מוגברת בקרב יהודי אשכנז "
    "(כגון Gaucher, Tay-Sachs, Canavan, ניוון שרירים של Duchenne בגרסה מסוימת).\n"
    "• פאנל מורחב (Expanded Carrier Screening) — מכסה מאות מחלות ללא תלות במוצא; "
    "כיום מוצע לכלל הזוגות.\n\n"
    "בחירת הפאנל המתאים תלויה בהיסטוריה המשפחתית, במוצא, בהקשר הרפואי "
    "ובהנחיות מערכת הבריאות."
)

_CARRIER_SCREENING_PERSONAL_HE = (
    "סקר נשאות גנטית מזהה נשאים לווריאנטים שעלולים להיות רלוונטיים לילדים עתידיים.\n\n"
    "בהקשר מוצא אשכנזי:\n"
    "• פאנל אשכנזי קלאסי — מכסה מחלות בתדירות מוגברת באוכלוסייה זו.\n"
    "• פאנל מורחב (Expanded Carrier Screening) — אינו תלוי במוצא, מוצע כיום לכלל הזוגות.\n\n"
    "איזה פאנל מתאים? זה תלוי בהיסטוריה המשפחתית, בבדיקות שכבר בוצעו, "
    "בשלב הרפואי ובהנחיות קופת החולים. לא ניתן לתת המלצה מדויקת ללא פגישה קלינית."
)

_CARRIER_SCREENING_PERSONAL_CLINICIAN_QUESTIONS: list = [
    "האם יש במשפחה מחלות גנטיות ידועות שצריכות להשפיע על בחירת הפאנל?",
    "האם כבר בוצע סקר נשאות לבן/בת הזוג?",
    "האם הפאנל המוצע הוא פאנל אשכנזי ספציפי או פאנל מורחב (expanded)?",
    "מה ההמלצות העדכניות של קופת החולים לגבי סקר נשאות לפי המצב שלי?",
]


def _build_carrier_screening_answer(text: str, is_personal: bool = False) -> dict:
    """
    Return a carrier-screening / ancestry-related testing educational answer.

    is_personal=True  → explains principles; avoids issuing a personal test list;
                         includes targeted clinician_questions.
    is_personal=False → general educational answer about carrier screening panels.
    """
    if is_personal:
        answer_text = _CARRIER_SCREENING_PERSONAL_HE
        clinician_qs = list(_CARRIER_SCREENING_PERSONAL_CLINICIAN_QUESTIONS)
        topic_id = "carrier_screening_personal"
        needs_gc = True
    else:
        answer_text = _CARRIER_SCREENING_GENERAL_HE
        clinician_qs = []
        topic_id = "carrier_screening_general"
        needs_gc = False

    return {
        "answer": answer_text,
        "safety_level": "general_information",
        "needs_genetic_counselor": needs_gc,
        "matched_topic": topic_id,
        "suggested_questions": [
            "מה ההבדל בין פאנל אשכנזי לסקר נשאות מורחב?",
            "מה קורה אם שני בני הזוג נשאים לאותה מחלה?",
        ],
        "clinician_questions": clinician_qs,
        "llm_used": False,
        "fallback_used": False,
    }


def _classify_general_question(question: str) -> str:
    """
    Classify a question that didn't match any KB entry.

    Returns one of:
    - "safe_general_education": concept question, AI draft allowed
    - "personal_or_high_stakes": additional personal signal found, no AI
    - "out_of_scope": unclear; use standard helpful fallback
    """
    lower = question.strip().lower()

    # Guard: VUS + plausible gene-symbol must NEVER classify as safe general
    # education — these queries belong in the VUS + gene pipeline.
    # This is a belt-and-suspenders check; step E of classify_question_intent
    # should catch these first via _extract_gene_from_explicit_phrase.
    if _mentions_vus(question) and (
        _detect_known_gene(question) or _extract_gene_from_explicit_phrase(question)
    ):
        return "out_of_scope"

    # Out-of-domain: clearly non-genetics/medicine → skip AI, use short message
    if _detect_out_of_domain(question):
        return "out_of_scope"

    # Extra personal/high-stakes guard (beyond safety.py step 3)
    for phrase in _GENERAL_EDU_EXTRA_BLOCK_PHRASES:
        if phrase in lower:
            return "personal_or_high_stakes"

    # Non-human biology or broad genetics vocabulary → always safe general education
    for signal in _NON_HUMAN_BIOLOGY_ORGANISMS:
        if signal in lower:
            return "safe_general_education"
    for signal in _BROAD_GENETICS_EDU_SIGNALS:
        if signal in lower:
            return "safe_general_education"
    # "כמה" + biology term at start (not personal) — e.g. "כמה גנים יש לצב ים?"
    if lower.startswith("כמה ") and any(
        word in lower for word in ["גנים", "כרומוזומים", "בסיסים", "נוקלאוטידים", "חלבונים", "אללים"]
    ):
        return "safe_general_education"

    # Positive educational-intent check
    for phrase in _GENERAL_EDU_INTENT_PHRASES:
        if lower.startswith(phrase) or f" {phrase}" in lower:
            return "safe_general_education"

    return "out_of_scope"


_GENERAL_EDUCATION_SYSTEM_PROMPT = (
    "You are a helpful genetic counseling assistant in Israel, writing a short "
    "educational answer in Hebrew for a patient who recently had genetic counseling.\n\n"
    "SCOPE: Answer ONLY questions about genetics, biology, medicine, or related "
    "biomedical concepts. If the question is about something else (geography, politics, "
    "food, sports, etc.) — output a single dash (-) only.\n\n"
    "TASK: ענה בעברית פשוטה, ב-2 עד 5 משפטים קצרים בלבד. "
    "Explain the genetics or biology concept clearly and concisely.\n\n"
    "UNCERTAINTY: When the scientific estimate is not exact (e.g. gene count varies by "
    "species or assembly), use cautious phrasing such as 'ההערכה הנוכחית היא...' or "
    "'מספר הגנים משתנה בין מינים ועשוי להגיע ל-...' — do NOT invent precise numbers.\n\n"
    "ALLOWED:\n"
    "  - General biological or medical concept explanations\n"
    "  - Non-human biology (sea turtle gene count, dog chromosomes, etc.)\n"
    "  - Disease category definitions (general, not personal)\n"
    "  - English biomedical terms when needed (mismatch repair, penetrance, "
    "beta-globin, etc.)\n"
    "  - General examples (e.g., 'מחלות כגון...')\n"
    "  - The word 'pathogenic' in a general, non-personal context\n\n"
    "PROHIBITED - output a single dash (-) if any of these apply:\n"
    "  - The question is not about genetics, biology, or medicine\n"
    "  - 'יש לך', 'אצלך', 'הסיכון שלך', 'הממצא שלך', 'התוצאה שלך'\n"
    "  - Diagnosis claims (stating the patient has a disease)\n"
    "  - Treatment, surgery, or medication recommendations\n"
    "  - Personal risk estimates or 'you should...' instructions\n"
    "  - Urgent clinical instructions\n"
    "  - Referral phrases (do not add 'יש לפנות לצוות הגנטי' or similar)\n"
    "  - Generic disclaimers ('המידע כללי ואינו מחליף ייעוץ רפואי')\n"
    "  - Question marks, emoji, or ClinVar statistics\n\n"
    "FORMAT:\n"
    "  - Hebrew mainly; English biomedical terms allowed.\n"
    "  - 2-5 short sentences. Maximum 500 characters.\n"
    "  - Output ONLY the sentences. No labels, no preamble, no quotes."
)

_GENERAL_EDUCATION_RETRY_SYSTEM_PROMPT = (
    "Write 2-3 short Hebrew sentences explaining the genetics or biology concept "
    "in the question.\n\n"
    "STRICT RULES:\n"
    "  - Hebrew ONLY for main text; English technical terms allowed.\n"
    "  - Do NOT mention 'יש לך', 'אצלך', 'הסיכון שלך'.\n"
    "  - Do NOT diagnose, recommend treatment, or estimate risk.\n"
    "  - No question marks, emoji, or disclaimers.\n"
    "  - Maximum 600 characters.\n"
    "  - Output ONLY the sentences."
)

_GENERAL_EDUCATION_WARNING_HE = "מידע AI לא מאומת — להסבר כללי בלבד."

_GENERAL_EDUCATION_SOURCE_NOTE_HE = "נוצר אוטומטית, לא עבר בדיקה מקצועית."


def _validate_general_education_draft(text: str) -> "tuple[bool, str]":
    """Validate a general education AI draft. Returns (is_valid, rejection_reason)."""
    if not text or not text.strip():
        return False, "empty"
    stripped = text.strip()
    if stripped == "-":
        return False, "model_unsure"
    if len(stripped) < 30:
        return False, "too_short"
    if len(stripped) > 700:
        return False, "too_long"
    hebrew_chars = sum(1 for c in stripped if "א" <= c <= "ת")
    if hebrew_chars < 15:
        return False, "not_hebrew"
    lower = stripped.lower()
    personal_in_answer = [
        "יש לך", "אצלך", "הסיכון שלך", "הממצא שלך", "התוצאה שלך",
        "אתה חולה", "את חולה", "אתה צריך", "את צריכה",
        "עלייך", "כדאי לך", "מומלץ לך",
        "your risk", "you have cancer", "you should",
    ]
    for phrase in personal_in_answer:
        if phrase in lower:
            return False, "personal_language"
    for term in ("ניתוח", "כריתה", "כימותרפיה", "הפלה", "surgery", "chemotherapy"):
        if term in lower:
            return False, "treatment_term"
    return True, ""


def _generate_general_education_draft(question: str) -> "tuple[Optional[str], dict]":
    """
    Call the LLM to generate a general education explanation.
    Returns (text_or_None, debug_dict). Never raises.
    """
    ai_debug: dict = {"attempted": True}
    try:
        client = create_llm_client()
    except (ValueError, Exception):
        ai_debug["generated"] = False
        ai_debug["rejection_code"] = "llm_not_configured"
        return None, ai_debug

    provider_name = type(client).__name__
    ai_debug["provider"] = provider_name

    text: Optional[str] = None
    try:
        text = client.call_text_raw(
            question,
            system_prompt=_GENERAL_EDUCATION_SYSTEM_PROMPT,
        )
    except Exception as exc:
        ai_debug["generated"] = False
        ai_debug["rejection_code"] = "generation_error"
        ai_debug["error_type"] = type(exc).__name__
        return None, ai_debug

    is_valid, reason = _validate_general_education_draft(text or "")
    if not is_valid:
        # One retry with a stricter prompt
        try:
            text2 = client.call_text_raw(
                question,
                system_prompt=_GENERAL_EDUCATION_RETRY_SYSTEM_PROMPT,
            )
            is_valid2, reason2 = _validate_general_education_draft(text2 or "")
            if is_valid2:
                text = text2
                is_valid = True
            else:
                reason = reason2
        except Exception:
            pass

    if not is_valid:
        ai_debug["generated"] = False
        ai_debug["validation_passed"] = False
        ai_debug["rejection_code"] = reason
        return None, ai_debug

    ai_debug["generated"] = True
    return (text or "").strip(), ai_debug


def _build_general_education_answer(question: str) -> "tuple[Optional[dict], dict]":
    """
    Generate and return (result_dict_or_None, ai_debug).
    ai_debug is ALWAYS returned so callers can attach it to the fallback
    response — staging responses must always include ai_general_debug.

    Session 27.9 Part B: GENERAL_LOW_RISK_AI class — no physician review required,
    no unverified-AI label shown to the patient, no generic disclaimer, no queue entry.
    The question classifier already blocked personal/high-stakes questions upstream.
    """
    text, ai_debug = _generate_general_education_draft(question)
    if not text:
        return None, ai_debug

    return {
        "answer": text,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "general_education_ai",
        "suggested_questions": [],
        "llm_used": True,
        "fallback_used": False,
        "llm_mode": "general_education_draft",
        # ai_content_type = GENERAL_LOW_RISK_AI: no physician review, no queue entry
        "ai_content_type": "general_low_risk_ai",
        "requires_physician_review": False,
        "ai_general_debug": ai_debug,
    }, ai_debug


def _build_clinvar_context_block(gene: str, clinvar_context: dict) -> str:
    """
    Build a compact metadata block to pass as context to the ClinVar-based draft LLM.
    clinvar_context keys: total_variants, phenotypes / top_phenotypes,
    significance_breakdown / by_significance.
    """
    total = clinvar_context.get("total_variants") or clinvar_context.get("total")
    phenotypes = (
        clinvar_context.get("top_phenotypes")
        or clinvar_context.get("phenotypes")
        or []
    )
    by_sig = (
        clinvar_context.get("significance_breakdown")
        or clinvar_context.get("by_significance")
        or {}
    )
    lines = [f"Gene: {gene}"]
    if total:
        lines.append(f"Total variants in ClinVar: {total:,}")
    if by_sig:
        path_n = sum(
            v for k, v in by_sig.items()
            if "pathogenic" in k.lower()
            and "uncertain" not in k.lower()
            and "conflicting" not in k.lower()
            and "benign" not in k.lower()
        )
        vus_n = sum(v for k, v in by_sig.items() if "uncertain" in k.lower())
        if path_n:
            lines.append(f"Pathogenic/Likely pathogenic variants: {path_n:,}")
        if vus_n:
            lines.append(f"VUS (uncertain significance) variants: {vus_n:,}")
    # Normalise raw English phenotype strings to clean Hebrew patient-facing labels.
    # Raw English names are intentionally NOT passed to the LLM \u2014 doing so caused
    # garbled mixed Hebrew/English output (e.g. "\u05e1\u05d9\u05e0\u05d3rom\u05d5\u05ea", "\u05e4REDIS\u05e4\u05d5\u05d6\u05d9\u05e6ION\u05d9\u05d5\u05ea").
    normalized_he = _normalize_clinvar_phenotypes_for_patient(phenotypes, gene)
    if normalized_he:
        lines.append("\u05d4\u05e7\u05e9\u05e8\u05d9\u05dd \u05e7\u05dc\u05d9\u05e0\u05d9\u05d9\u05dd \u05de\u05d3\u05d5\u05d5\u05d7\u05d9\u05dd (\u05d1\u05e2\u05d1\u05e8\u05d9\u05ea):")
        for ctx in normalized_he:
            lines.append(f"  - {ctx}")
    lines.append(
        "\nInstruction: Write 1-2 Hebrew sentences summarizing the clinical contexts "
        "listed above (in Hebrew). "
        "\u05e0\u05e1\u05d7 \u05e8\u05e7 \u05e2\u05dc \u05d1\u05e1\u05d9\u05e1 \u05d4\u05d4\u05e7\u05e9\u05e8\u05d9\u05dd \u05d4\u05e2\u05d1\u05e8\u05d9\u05d9\u05dd \u05e9\u05e1\u05d5\u05e4\u05e7\u05d5. "
        "\u05d0\u05dc \u05ea\u05ea\u05e8\u05d2\u05dd \u05de\u05d7\u05d3\u05e9 \u05de\u05d5\u05e0\u05d7\u05d9\u05dd \u05d1\u05d0\u05e0\u05d2\u05dc\u05d9\u05ea \u05d2\u05d5\u05dc\u05de\u05d9\u05d9\u05dd. "
        "Do NOT describe the gene's biology, protein, or mechanism."
    )
    return "\n".join(lines)


def _generate_unverified_gene_draft(
    gene: str,
    question: str = "",
    clinvar_context: "Optional[dict]" = None,
    use_lenient_validator: bool = False,
    _debug: "Optional[dict]" = None,
) -> "Optional[dict]":
    """
    Generate an AI-written gene draft for patient opt-in display.

    When ``clinvar_context`` is provided (Tier 2 genes with ClinVar data), the
    draft summarises ONLY the supplied metadata \u2014 the LLM is explicitly forbidden
    from inventing biological function, protein names, or pathways.

    When ``clinvar_context`` is None (legacy / gene index unavailable), the
    biology-from-memory prompt is used, with the same strict validation.

    Returns a structured dict or None (LLM unavailable / output fails validation).
    Never raises \u2014 all failures are caught and logged.

    The draft is NEVER written into approved gene cards or the KB.
    approved=False and review_status='unreviewed' are set unconditionally.

    ``_debug``: if a dict is passed, it is mutated with safe diagnostic fields
    (no API keys, no prompts, no personal data) explaining what happened.
    """
    from datetime import datetime, timezone

    def _dbg(**kw: object) -> None:
        if isinstance(_debug, dict):
            _debug.update(kw)

    try:
        client = create_llm_client()
    except ValueError as exc:
        logger.debug(
            "LLM not configured - skipping unverified draft for %r: %s",
            gene, type(exc).__name__,
        )
        _dbg(attempted=False, provider="none", reason="llm_not_configured")
        return None

    provider_name = type(client).__name__
    _dbg(attempted=True, provider=provider_name)
    logger.debug("Generating unverified draft for %r via %s", gene, provider_name)

    use_clinvar_context = bool(clinvar_context)

    try:
        if use_clinvar_context and use_lenient_validator:
            # Biology-first path for Tier-2 explicit gene education questions.
            # ClinVar counts are already in gene_metadata for the collapsed UI card.
            # Passing _build_clinvar_context_block() here conflicted with the biology
            # system prompt (that block ends with "Do NOT describe the gene's biology"),
            # causing OpenAI to return ClinVar-count summaries instead of function.
            user_content = (
                f"Gene: {gene}\n"
                f"Task: Explain the biological role of this gene in Hebrew for a patient."
            )
            system_prompt_1 = _GENE_EDUCATION_DRAFT_SYSTEM_PROMPT
            system_prompt_2 = _GENE_EDUCATION_DRAFT_RETRY_SYSTEM_PROMPT
        elif use_clinvar_context:
            # ClinVar clinical-context path (non-biology, strict validator).
            context_block = _build_clinvar_context_block(gene, clinvar_context)
            user_content = context_block
            if question:
                user_content += f"\n\nUser question context: {question[:200]}"
            system_prompt_1 = _UNVERIFIED_CLINVAR_DRAFT_SYSTEM_PROMPT
            system_prompt_2 = _UNVERIFIED_CLINVAR_DRAFT_RETRY_SYSTEM_PROMPT
        else:
            user_content = f"Gene symbol: {gene}"
            if question:
                user_content += f"\nUser context: {question[:200]}"
            if use_lenient_validator:
                system_prompt_1 = _GENE_EDUCATION_DRAFT_SYSTEM_PROMPT
                system_prompt_2 = _GENE_EDUCATION_DRAFT_RETRY_SYSTEM_PROMPT
            else:
                system_prompt_1 = _UNVERIFIED_DRAFT_SYSTEM_PROMPT
                system_prompt_2 = _UNVERIFIED_DRAFT_RETRY_SYSTEM_PROMPT

        raw = client.call_text_raw(user_content, system_prompt=system_prompt_1)
        text = raw.strip()
        if use_lenient_validator:
            rejection = _validate_gene_education_draft(text) if text else "empty"
        else:
            rejection = _validate_unverified_draft(text) if text else "empty"
        if rejection:
            try:
                logger.info(
                    "Unverified gene draft for %r rejected (%s) - retrying.",
                    gene, str(rejection)[:120],
                )
            except Exception:
                pass
            raw2 = client.call_text_raw(user_content, system_prompt=system_prompt_2)
            text = raw2.strip()
            if use_lenient_validator:
                rejection = _validate_gene_education_draft(text) if text else "empty"
            else:
                # Second-pass uses relaxed validator: ClinVar allowed if otherwise safe.
                rejection = _validate_unverified_draft_clinvar_ok(text) if text else "empty"
            if rejection:
                try:
                    logger.info(
                        "Unverified gene draft for %r rejected after retry (%s) - silent fallback.",
                        gene, str(rejection)[:120],
                    )
                except Exception:
                    pass
                _dbg(generated=False, validation_passed=False, rejection_code=rejection)
                # Staging debug mode: expose raw rejected text for safe genes only.
                if (
                    text
                    and _ai_draft_debug_mode_active()
                    and gene.upper() in _DEBUG_SAFE_GENES
                    and not _question_has_high_stakes_phrases(question or "")
                ):
                    _dbg(
                        raw_rejected_text_he=text[:500],
                        raw_rejected_status="ai_generated_rejected_debug_only",
                        raw_rejected_warning=(
                            "DEBUG ONLY - this draft failed validation "
                            "and must not be shown to real users."
                        ),
                        raw_rejection_reason=str(rejection)[:120],
                    )
                if not use_lenient_validator and use_clinvar_context:
                    phenos = (
                        clinvar_context.get("top_phenotypes")
                        or clinvar_context.get("phenotypes")
                        or []
                    )
                    normalized = _normalize_clinvar_phenotypes_for_patient(phenos, gene)
                    if normalized:
                        logger.info(
                            "Returning deterministic ClinVar draft for %r (%d contexts).",
                            gene, len(normalized),
                        )
                        det = _build_deterministic_clinvar_draft(gene, normalized)
                        _dbg(generated=True, validation_passed=True, rejection_code=None,
                             fallback="deterministic_clinvar")
                        return det
                return None

        model_name = (
            getattr(client, "_model", None)
            or os.environ.get("LOCAL_LLM_MODEL", None)
            or provider_name
        )
        _dbg(generated=True, validation_passed=True, model=model_name)
        result = {
            "visible": True,
            "status": "ai_generated_unreviewed",
            "gene_symbol": gene,
            "warning_he": _UNVERIFIED_DRAFT_WARNING_HE,
            "text_he": text,
            "generated_by_model": model_name,
            "review_status": "unreviewed",
            "approved": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if use_clinvar_context:
            # Part E (27.9.3): explicit provenance — the model generates biology
            # explanation from its own knowledge with ClinVar gene context as a hint.
            # ClinVar supplies only variant counts / phenotype associations, NOT
            # protein function, mechanism, or pathway — do not claim it does.
            result["based_on"] = "model_expansion_with_clinvar_gene_context"
            result["source_grounded"] = False
            result["requires_physician_review"] = True
            result["ai_content_type"] = "medical_educational_ai_expansion"
            result["source_note_he"] = _UNVERIFIED_DRAFT_SOURCE_NOTE_HE
        else:
            result["based_on"] = "model_expansion_llm_knowledge"
            result["source_grounded"] = False
            result["requires_physician_review"] = True
            result["ai_content_type"] = "medical_educational_ai_expansion"
        return result

    except LLMClientError as exc:
        logger.info("LLM unavailable for unverified draft for %r (%s).", gene, type(exc).__name__)
        _dbg(generated=False, validation_passed=False, rejection_code="llm_client_error",
             error_type=type(exc).__name__)
        return None
    except Exception as exc:
        tb_frames = _traceback.extract_tb(exc.__traceback__)
        last_frame = tb_frames[-1] if tb_frames else None
        err_func = getattr(last_frame, "name", "unknown") if last_frame else "unknown"
        err_line = getattr(last_frame, "lineno", 0) if last_frame else 0
        err_type = type(exc).__name__
        try:
            logger.warning(
                "Unexpected error generating draft for %r: %s in %s (line %d)",
                gene, err_type, err_func, err_line,
            )
        except Exception:
            pass  # never let logging crash the draft function
        _dbg(
            generated=False,
            rejection_code="unexpected_error",
            error_type=err_type,
            error_func=err_func,
            error_line=err_line,
        )
        return None

_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ]")
_CJK_ARTIFACT_MAX = 3


def _strip_tiny_cjk_artifacts(text: str) -> tuple:
    """
    Remove CJK characters from text only if total count is ≤ _CJK_ARTIFACT_MAX.
    Returns (cleaned_text, was_cleaned: bool).
    If CJK count > _CJK_ARTIFACT_MAX or zero, returns (text, False) unchanged.
    Never removes Hebrew medical content — only isolated stray CJK characters.
    """
    cjk_chars = _CJK_RE.findall(text)
    if not cjk_chars or len(cjk_chars) > _CJK_ARTIFACT_MAX:
        return text, False
    cleaned = _CJK_RE.sub("", text).strip()
    return cleaned, True


def _attempt_cjk_recovery(
    output: str,
    validator,
    llm_client,
    user_content: str,
    deterministic_answer: str,
    effective_mode: str,
) -> "LLMLayerResult":
    """
    Attempt to recover from a CJK-only rejection via two strategies:
    1. Artifact cleaning: remove ≤ _CJK_ARTIFACT_MAX stray CJK chars and re-validate.
    2. Single retry: call the LLM once more with an explicit Hebrew-only prompt.

    validator is a callable(str) -> Optional[str] matching the active mode.
    Never retries more than once. Always returns a LLMLayerResult.
    """
    # Strategy 1: tiny artifact cleaning
    cleaned, was_cleaned = _strip_tiny_cjk_artifacts(output)
    if was_cleaned:
        clean_rejection = validator(cleaned)
        if clean_rejection is None:
            logger.info("Tiny CJK artifacts removed (%d chars) — output accepted after cleaning.",
                        len(output) - len(cleaned))
            return LLMLayerResult(
                answer=f"{cleaned}\n\n{deterministic_answer}", llm_used=True,
                attempted=True, mode=effective_mode, rejection_reason=None,
                repaired=True, repair_reason="tiny_cjk_artifacts_removed", retry_used=False,
            )

    # Strategy 2: single retry with strict Hebrew-only prompt
    try:
        raw_retry = llm_client._call_api(user_content, system_prompt=_CJK_RETRY_SYSTEM_PROMPT)
        output_retry = (raw_retry or "").strip()
        retry_rejection = validator(output_retry) if output_retry else "empty"
        if retry_rejection is None:
            logger.info("CJK retry succeeded — output accepted after strict retry.")
            return LLMLayerResult(
                answer=f"{output_retry}\n\n{deterministic_answer}", llm_used=True,
                attempted=True, mode=effective_mode, rejection_reason=None,
                repaired=False, repair_reason=None, retry_used=True,
            )
        logger.info("CJK retry also rejected (%s) — deterministic fallback.", retry_rejection)
        return LLMLayerResult(
            answer=deterministic_answer, llm_used=False, attempted=True,
            mode=effective_mode, rejection_reason=f"CJK retry failed: {retry_rejection}",
            repaired=False, repair_reason=None, retry_used=True,
        )
    except Exception as exc:
        logger.info("CJK retry call raised (%s) — deterministic fallback.", exc)
        return LLMLayerResult(
            answer=deterministic_answer, llm_used=False, attempted=True,
            mode=effective_mode, rejection_reason=f"CJK retry error: {exc}",
            repaired=False, repair_reason=None, retry_used=True,
        )


def _build_tier2_framing_prompt(gene: str, context_fields: dict) -> str:
    """Build a user-content prompt for Tier-2 statistical framing."""
    total = context_fields.get("total_variants", 0)
    sig = context_fields.get("significance_breakdown", {})
    phenotypes = context_fields.get("top_phenotypes", [])
    sig_summary = "; ".join(f"{k}: {v:,}" for k, v in list(sig.items())[:4]) if sig else "—"
    pheno_summary = ", ".join(phenotypes[:4]) if phenotypes else "—"
    return (
        f"Gene: {gene}\n"
        f"Total ClinVar records: {total:,}\n"
        f"Significance distribution (top categories): {sig_summary}\n"
        f"Top reported conditions: {pheno_summary}\n\n"
        "Write 1-2 short Hebrew sentences that describe this statistical picture "
        "in a patient-friendly way. DO NOT describe what this gene does biologically. "
        "DO NOT state which conditions this gene causes. Only describe what the numbers show. "
        "Note that no approved Hebrew biological summary exists for this gene."
    )


def _generate_llm_intro_sentence(
    question: str,
    gene: Optional[str] = None,
    topic: Optional[str] = None,
) -> Optional[str]:
    """
    Call the local LLM for a single warm Hebrew intro sentence.

    Returns the validated sentence, or None on any failure (including when
    LOCAL_LLM_URL is not set, on connection errors, or when validation fails).
    Never raises.
    """
    url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if not url:
        return None
    try:
        llm_client = LocalLLMClient(url)
        gene_hint = f" (הגן {gene})" if gene else ""
        user_content = f"שאלת המשתמש: {question}{gene_hint}"
        raw = llm_client._call_api(user_content, system_prompt=_INTRO_SYSTEM_PROMPT)
        intro = raw.strip()
        if not _validate_llm_intro(intro):
            logger.info(
                "LLM intro failed validation — using deterministic answer only "
                "(question=%r, intro=%r)", question[:60], intro[:120]
            )
            return None
        return intro
    except LLMClientError as exc:
        logger.info("LLM intro unavailable (%s) — deterministic fallback.", exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error in LLM intro (%s) — deterministic fallback.", exc)
        return None


def _apply_safe_intro(
    question: str,
    deterministic_answer: str,
    gene: Optional[str] = None,
    topic: Optional[str] = None,
) -> tuple:
    """
    Backward-compatible wrapper around _apply_llm_layer.
    Returns (final_answer, llm_used). Tests import this directly.
    """
    result = _apply_llm_layer(question, deterministic_answer, gene=gene, topic=topic)
    return result.answer, result.llm_used


def _apply_llm_layer(
    question: str,
    deterministic_answer: str,
    gene: Optional[str] = None,
    topic: Optional[str] = None,
    mode: Optional[str] = None,
    context_fields: Optional[dict] = None,
) -> "LLMLayerResult":
    """
    Apply the controlled LLM layer to a deterministic answer.

    mode selects the LLM output type:
      "intro_only"        — single validated sentence ≤200 chars (default)
      "controlled_framing" — 1-3 sentences ≤500 chars
      "tier2_framing"     — Tier-2 statistical framing paragraph ≤400 chars

    When LOCAL_LLM_URL is unset, mode is forced to "none".
    Returns LLMLayerResult with answer, llm_used, attempted, mode, rejection_reason.
    """
    url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if not url:
        return LLMLayerResult(
            answer=deterministic_answer, llm_used=False, attempted=False,
            mode="none", rejection_reason=None,
        )

    # Determine effective mode
    effective_mode = mode or os.environ.get("LLM_MODE", "intro_only").strip()
    if effective_mode not in ("intro_only", "controlled_framing", "tier2_framing"):
        effective_mode = "intro_only"

    try:
        llm_client = LocalLLMClient(url)

        if effective_mode == "tier2_framing" and context_fields and gene:
            # Tier-2 structured framing
            user_content = _build_tier2_framing_prompt(gene, context_fields)
            raw = llm_client._call_api(user_content, system_prompt=_TIER2_FRAMING_SYSTEM_PROMPT)
            output = raw.strip()
            rejection = _validate_tier2_framing(output, gene) if output else "empty"
            if rejection:
                if rejection == "contains CJK characters":
                    return _attempt_cjk_recovery(
                        output,
                        lambda t: _validate_tier2_framing(t, gene),
                        llm_client, user_content, deterministic_answer, effective_mode,
                    )
                logger.info(
                    "Tier-2 LLM framing rejected (%s) for gene %r — deterministic fallback.",
                    rejection, gene,
                )
                return LLMLayerResult(
                    answer=deterministic_answer, llm_used=False, attempted=True,
                    mode=effective_mode, rejection_reason=rejection,
                )
            # Tier-2: LLM framing replaces the transparency note, stats remain unchanged
            # Structure: [LLM framing]\n\n[deterministic stats block]
            return LLMLayerResult(
                answer=f"{output}\n\n{deterministic_answer}", llm_used=True, attempted=True,
                mode=effective_mode, rejection_reason=None,
            )

        elif effective_mode == "controlled_framing":
            gene_hint = f" (הגן {gene})" if gene else ""
            user_content = f"שאלת המשתמש: {question}{gene_hint}"
            raw = llm_client._call_api(user_content, system_prompt=_CONTROLLED_FRAMING_SYSTEM_PROMPT)
            output = raw.strip()
            rejection = _validate_controlled_framing(output) if output else "empty"
            if rejection:
                if rejection == "contains CJK characters":
                    return _attempt_cjk_recovery(
                        output,
                        _validate_controlled_framing,
                        llm_client, user_content, deterministic_answer, effective_mode,
                    )
                logger.info(
                    "Controlled framing rejected (%s) — intro_only fallback.", rejection
                )
                # Fallback to intro_only within the same call
                gene_hint2 = f" (הגן {gene})" if gene else ""
                raw2 = llm_client._call_api(
                    f"שאלת המשתמש: {question}{gene_hint2}",
                    system_prompt=_INTRO_SYSTEM_PROMPT,
                )
                output2 = raw2.strip()
                rejection2 = _validate_intro_with_reason(output2) if output2 else "empty"
                if rejection2:
                    return LLMLayerResult(
                        answer=deterministic_answer, llm_used=False, attempted=True,
                        mode=effective_mode, rejection_reason=rejection,
                    )
                return LLMLayerResult(
                    answer=f"{output2}\n\n{deterministic_answer}", llm_used=True, attempted=True,
                    mode="intro_only", rejection_reason=None,
                )
            return LLMLayerResult(
                answer=f"{output}\n\n{deterministic_answer}", llm_used=True, attempted=True,
                mode=effective_mode, rejection_reason=None,
            )

        else:
            # intro_only (default)
            gene_hint = f" (הגן {gene})" if gene else ""
            user_content = f"שאלת המשתמש: {question}{gene_hint}"
            raw = llm_client._call_api(user_content, system_prompt=_INTRO_SYSTEM_PROMPT)
            output = raw.strip()
            rejection = _validate_intro_with_reason(output) if output else "empty"
            if rejection:
                if rejection == "contains CJK characters":
                    return _attempt_cjk_recovery(
                        output,
                        _validate_intro_with_reason,
                        llm_client, user_content, deterministic_answer, effective_mode,
                    )
                logger.info(
                    "LLM intro rejected (%s) — deterministic fallback. "
                    "(question=%r, output=%r)", rejection, question[:60], output[:120]
                )
                return LLMLayerResult(
                    answer=deterministic_answer, llm_used=False, attempted=True,
                    mode=effective_mode, rejection_reason=rejection,
                )
            return LLMLayerResult(
                answer=f"{output}\n\n{deterministic_answer}", llm_used=True, attempted=True,
                mode=effective_mode, rejection_reason=None,
            )

    except LLMClientError as exc:
        logger.info("LLM unavailable (%s) — deterministic fallback.", exc)
        return LLMLayerResult(
            answer=deterministic_answer, llm_used=False, attempted=True,
            mode=effective_mode, rejection_reason=f"LLMClientError: {exc}",
        )
    except Exception as exc:
        logger.warning("Unexpected LLM error (%s) — deterministic fallback.", exc)
        return LLMLayerResult(
            answer=deterministic_answer, llm_used=False, attempted=True,
            mode=effective_mode, rejection_reason=f"unexpected error: {exc}",
        )


def _llm_debug_fields(result: "LLMLayerResult") -> dict:
    """Return debug fields to include in the response when LLM_DEBUG=1."""
    if os.environ.get("LLM_DEBUG", "").strip() != "1":
        return {}
    out = {
        "llm_attempted": result.attempted,
        "llm_rejected_reason": result.rejection_reason,
    }
    if result.retry_used:
        out["llm_retry_used"] = True
    if result.repaired:
        out["llm_repaired"] = True
        out["llm_repair_reason"] = result.repair_reason
    return out


_GENE_EDUCATION_FALLBACK_HE = (
    "{gene} הוא גן שמכיל הוראות לייצור חלבון מסוים בגוף. "
    "שינויים שונים בגן זה עשויים להיות בעלי משמעות קלינית שונה — "
    "חלקם נחשבים תקינים וחלקם קשורים למצבים רפואיים מסוימים. "
    "המשמעות של ממצא ספציפי בגן זה — כולל ממצא VUS — "
    "תלויה בסוג הממצא, בדוח הבדיקה ובסיפור המשפחתי, "
    "ונקבעת על ידי הצוות הגנטי."
)

_GENE_INFO_SUGGESTED_QUESTIONS = [
    "מה ההבדל בין VUS לבין ממצא פתוגני?",
    "האם VUS יכול להשתנות בעתיד?",
    "מה כדאי לשאול את הצוות הגנטי על הממצא?",
    "האם הממצא הזה אומר שיש לי מחלה?",
]


def _extract_gene_symbol_from_question(text: str) -> Optional[str]:
    """
    Extract the single most likely gene symbol from the question.

    Priority:
    1. Typo-tolerant patterns for BRCA1, BRCA2, NF1 via _detect_known_gene().
    2. Exact uppercase-token matching against the live gene index.

    Returns None when no gene is found or when multiple distinct gene symbols
    appear (comparison questions are better handled by the KB).
    """
    known = _detect_known_gene(text)
    if known:
        return known

    if not gene_index._GENE_INDEX_AVAILABLE:
        # Fallback: extract from "בגן X" / "הגן X" Hebrew phrases (supports
        # mixed-case HGNC symbols such as C12orf57).
        return _extract_gene_from_explicit_phrase(text)

    candidates = _GENE_SYMBOL_CANDIDATE_RE.findall(text)
    found: list[str] = []
    for c in candidates:
        if c in _NON_GENE_TOKENS:
            continue
        if gene_index.get_gene_summary(c) is not None:
            found.append(c)
    if len(found) == 1:
        return found[0]
    # Extended fallback: "בגן X" / "הגן X" — for mixed-case symbols not matched
    # by _GENE_SYMBOL_CANDIDATE_RE (which requires all-uppercase tokens).
    return _extract_gene_from_explicit_phrase(text)


def _is_gene_level_question(text: str) -> bool:
    """
    Return True when the question signals intent to learn generally about a gene
    (its ClinVar profile, associated conditions, significance distribution, etc.)
    rather than asking about the user's personal result.
    """
    lower = text.strip().lower()
    return any(phrase in lower for phrase in _GENE_QUESTION_PHRASES)


def _format_gene_evidence_block(gene: str, summary: dict) -> str:
    """Structured text block passed to the LLM for gene summaries."""
    parts = [
        f"Gene: {gene}",
        f"Total ClinVar variant records: {summary.get('total_variants', 0):,}",
    ]
    by_sig = summary.get("by_significance", {})
    if by_sig:
        sig_lines = "\n".join(
            f"  - {k}: {v:,}"
            for k, v in sorted(by_sig.items(), key=lambda x: -x[1])[:10]
        )
        parts.append(f"Clinical significance distribution:\n{sig_lines}")
    by_review = summary.get("by_review_status", {})
    if by_review:
        rev_lines = "\n".join(
            f"  - {k}: {v:,}"
            for k, v in sorted(by_review.items(), key=lambda x: -x[1])[:5]
        )
        parts.append(f"Review status distribution:\n{rev_lines}")
    phenotypes = summary.get("phenotypes", [])
    if phenotypes:
        parts.append(
            "Associated conditions (from ClinVar):\n  "
            + "\n  ".join(phenotypes[:10])
        )
    var_types = summary.get("variant_types", {})
    if var_types:
        vt_lines = "\n".join(
            f"  - {k}: {v:,}"
            for k, v in sorted(var_types.items(), key=lambda x: -x[1])[:5]
        )
        parts.append(f"Variant types:\n{vt_lines}")
    dr = summary.get("date_range", {})
    if dr.get("latest"):
        parts.append(f"Latest record date: {dr['latest']}")
    return "\n\n".join(parts)


def _build_gene_clinvar_deterministic_answer(gene: str, summary: dict) -> str:
    """
    Build a structured Hebrew gene-level answer without an LLM.
    Covers: total count, significance breakdown, top phenotypes, safety note.
    """
    total = summary.get("total_variants", 0)
    by_sig = summary.get("by_significance", {})
    phenotypes = summary.get("phenotypes", [])

    lines = [f"מידע כללי על גן {gene} ממאגר ClinVar", ""]
    lines.append(f"במאגר ClinVar מתועדות {total:,} רשומות וריאנט עבור גן {gene}.")

    if by_sig:
        lines.append("")
        lines.append("סיווגים קליניים מדווחים:")
        path_n  = sum(v for k, v in by_sig.items()
                      if "pathogenic" in k.lower()
                      and "benign" not in k.lower()
                      and "uncertain" not in k.lower()
                      and "conflicting" not in k.lower())
        benign_n = sum(v for k, v in by_sig.items()
                       if "benign" in k.lower()
                       and "pathogenic" not in k.lower()
                       and "uncertain" not in k.lower()
                       and "conflicting" not in k.lower())
        vus_n    = sum(v for k, v in by_sig.items() if "uncertain" in k.lower())
        conf_n   = sum(v for k, v in by_sig.items() if "conflicting" in k.lower())
        other_n  = total - path_n - benign_n - vus_n - conf_n
        if path_n:
            lines.append(f"• Pathogenic / Likely pathogenic: {path_n:,}")
        if benign_n:
            lines.append(f"• Benign / Likely benign: {benign_n:,}")
        if vus_n:
            lines.append(f"• Uncertain significance (VUS): {vus_n:,}")
        if conf_n:
            lines.append(f"• Conflicting classifications: {conf_n:,}")
        if other_n > 0:
            lines.append(f"• אחר / לא סווג: {other_n:,}")

    clean_phenotypes = _filter_patient_conditions(phenotypes)
    if clean_phenotypes:
        lines.append("")
        lines.append("מצבים רפואיים מדווחים בתיעוד ClinVar:")
        for p in clean_phenotypes[:8]:
            lines.append(f"• {p}")
        if len(clean_phenotypes) > 8:
            lines.append(f"(ועוד {len(clean_phenotypes) - 8} מצבים נוספים במאגר)")

    return "\n".join(lines)


def _call_local_llm_for_gene_summary(
    question: str, gene: str, summary: dict
) -> Optional[str]:
    """
    Ask the local LLM to phrase the gene summary in patient-friendly Hebrew.
    Returns None (deterministic fallback) on any failure or forbidden output.
    """
    url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if not url:
        return None
    try:
        client = LocalLLMClient(url)
        evidence_block = _format_gene_evidence_block(gene, summary)
        user_content = (
            f"User question (Hebrew): {question}\n\n"
            f"=== GENE DATA FROM CLINVAR ===\n{evidence_block}\n\n"
            "Write the Hebrew summary now, following the system instructions exactly."
        )
        raw = client._call_api(user_content, system_prompt=GENE_SUMMARY_SYSTEM_PROMPT)
        text = raw.strip()
        if not text:
            return None
        if _FORBIDDEN_GENE_OUTPUT_RE.search(text):
            logger.warning(
                "LLM gene-summary answer tripped the forbidden-output filter; "
                "using deterministic fallback."
            )
            return None
        return text
    except LLMClientError as exc:
        logger.warning(
            "Local LLM unavailable for gene summary (%s); using deterministic fallback.", exc
        )
        return None
    except Exception as exc:  # defensive — never let LLM errors break /ask
        logger.warning(
            "Unexpected error calling local LLM for gene summary (%s); "
            "using deterministic fallback.", exc
        )
        return None


def _draft_adds_meaningful_information(draft_text: str, main_answer: str) -> bool:
    """True when the draft provides content meaningfully different from main_answer.

    Suppresses the patient card only when draft ≈ main answer (>65% word
    overlap), which in practice only happens if the draft was somehow used
    verbatim as the main answer.  For normal Tier-2 answers (ClinVar reference
    sentence) the draft biology text will always be displayable.
    """
    if not draft_text:
        return False
    if not main_answer:
        return True
    draft_words = set(draft_text.split())
    answer_words = set(main_answer.split())
    if not draft_words:
        return False
    overlap = len(draft_words & answer_words) / len(draft_words)
    return overlap < 0.65


def _build_gene_clinvar_answer(question: str, gene: str, include_unverified_gene_draft: bool = False, corrected_from: "Optional[str]" = None) -> Optional[dict]:
    """
    Build the full /ask response for a gene-level ClinVar question.

    Returns None only when the gene_index is unavailable (caller falls through).
    When the gene is not found in the index, returns a safe "not found" response.
    Always uses the deterministic formatter; optionally layers an LLM phrasing pass.

    The returned dict includes a ``gene_metadata`` key that the Pydantic response
    model (`CounselingAskResponse`) serializes conditionally — it appears in the
    JSON only for gene-level responses, keeping the standard 5-field schema intact
    for all other answer types.
    """
    summary = gene_index.get_gene_summary(gene)
    if summary is None:
        # Gene not found in the local ClinVar index.
        # Check approved sources in priority order: Tier 1a → Tier 1b → Tier 3.
        card_summary = gene_cards.get_approved_summary(gene)
        if card_summary:
            # Tier 1a (gene_cards card only — no ClinVar stats available).
            preamble_1a = _MUTATION_GENE_PREAMBLE_HE.format(gene=gene) if _is_mutation_specific_question(question) else ""
            det = preamble_1a + card_summary
            return {
                "answer": det,
                "safety_level": "general_information",
                "needs_genetic_counselor": False,
                "matched_topic": "gene_clinvar_summary",
                "suggested_questions": _gene_suggested_questions(question, gene),
                "llm_used": False,
                "fallback_used": True,
                "llm_mode": "none",
                "gene_metadata": {
                    "gene_symbol": gene,
                    "data_source": "Curated educational content",
                    "llm_used": False,
                    "fallback_used": True,
                    "total_variants": None,
                    "found_in_index": False,
                    "answer_tier": "tier1",
                    "gene_knowledge_status": "approved",
                    "unverified_gene_draft_available": False,
                },
            }
        # Tier 1b: Gene Knowledge Base approved record (no ClinVar stats).
        gk_patient = gene_knowledge.get_gene_patient_summary(gene)
        gk_vus = gene_knowledge.get_gene_vus_note(gene)
        if gk_patient:
            preamble_1b = _MUTATION_GENE_PREAMBLE_HE.format(gene=gene) if _is_mutation_specific_question(question) else ""
            parts_1b = [preamble_1b + gk_patient]
            if gk_vus and _mentions_vus(question):
                parts_1b.append(gk_vus)
            det = "\n\n".join(parts_1b)
            return {
                "answer": det,
                "safety_level": "general_information",
                "needs_genetic_counselor": False,
                "matched_topic": "gene_clinvar_summary",
                "suggested_questions": _gene_suggested_questions(question, gene),
                "llm_used": False,
                "fallback_used": True,
                "llm_mode": "none",
                "gene_metadata": {
                    "gene_symbol": gene,
                    "data_source": "Gene Knowledge Base",
                    "llm_used": False,
                    "fallback_used": True,
                    "total_variants": None,
                    "found_in_index": False,
                    "answer_tier": "tier1b",
                    "gene_knowledge_status": "approved",
                    "unverified_gene_draft_available": False,
                },
            }
        # Tier 3: gene not in any approved source and not in local ClinVar index.
        _correction_note_t3 = (
            f"ייתכן שהתכוונת לגן {gene} (מתיקון אוטומטי של '{corrected_from}').\n\n"
            if corrected_from else ""
        )
        return {
            "answer": _correction_note_t3 + f"אין עדיין מידע על הגן {gene} במאגר המקומי.",
            "safety_level": "general_information",
            "needs_genetic_counselor": False,
            "matched_topic": "gene_clinvar_summary",
            "suggested_questions": _gene_suggested_questions(question, gene),
            "llm_used": False,
            "fallback_used": True,
            "gene_metadata": {
                "gene_symbol": gene,
                "data_source": "ClinVar (NCBI) via local gene index",
                "llm_used": False,
                "fallback_used": True,
                "total_variants": None,
                "found_in_index": False,
                "answer_tier": "tier3",
                "gene_knowledge_status": "missing",
                "unverified_gene_draft_available": False,
            },
        }

    # Gene IS in the ClinVar index.
    # Priority order: Tier 1a (gene_cards) → Tier 1b (gene_knowledge) → Tier 2 (ClinVar-only).
    curated = gene_cards.get_approved_summary(gene)
    if curated:
        # Tier 1a: approved gene card — curated patient education.
        # ClinVar stats go to gene_metadata only (no ClinVar dump in main answer).
        preamble = _MUTATION_GENE_PREAMBLE_HE.format(gene=gene) if _is_mutation_specific_question(question) else ""
        det = preamble + curated
        return {
            "answer": det,
            "safety_level": "general_information",
            "needs_genetic_counselor": False,
            "matched_topic": "gene_clinvar_summary",
            "suggested_questions": _gene_suggested_questions(question, gene),
            "llm_used": False,
            "fallback_used": True,
            "llm_mode": "none",
            "gene_metadata": {
                "gene_symbol": gene,
                "data_source": "Curated educational content + ClinVar",
                "llm_used": False,
                "fallback_used": True,
                "total_variants": summary.get("total_variants"),
                "found_in_index": True,
                "answer_tier": "tier1",
                "gene_knowledge_status": "approved",
                "unverified_gene_draft_available": False,
                "significance_breakdown": summary.get("by_significance") or {},
                "top_phenotypes": (summary.get("phenotypes") or [])[:6],
            },
        }

    # Tier 1b: Gene Knowledge Base approved record + ClinVar stats in metadata.
    gk_patient_ci = gene_knowledge.get_gene_patient_summary(gene)
    gk_vus_ci = gene_knowledge.get_gene_vus_note(gene)
    if gk_patient_ci:
        preamble_ci = _MUTATION_GENE_PREAMBLE_HE.format(gene=gene) if _is_mutation_specific_question(question) else ""
        parts_ci = [preamble_ci + gk_patient_ci]
        if gk_vus_ci and _mentions_vus(question):
            parts_ci.append(gk_vus_ci)
        det_ci = "\n\n".join(parts_ci)
        return {
            "answer": det_ci,
            "safety_level": "general_information",
            "needs_genetic_counselor": False,
            "matched_topic": "gene_clinvar_summary",
            "suggested_questions": _gene_suggested_questions(question, gene),
            "llm_used": False,
            "fallback_used": True,
            "llm_mode": "none",
            "gene_metadata": {
                "gene_symbol": gene,
                "data_source": "Gene Knowledge Base + ClinVar",
                "llm_used": False,
                "fallback_used": True,
                "total_variants": summary.get("total_variants"),
                "found_in_index": True,
                "answer_tier": "tier1b",
                "gene_knowledge_status": "approved",
                "unverified_gene_draft_available": False,
                "significance_breakdown": summary.get("by_significance") or {},
                "top_phenotypes": (summary.get("phenotypes") or [])[:6],
            },
        }

    # Tier 2: no approved gene card or knowledge base record, but gene is in ClinVar index.
    # ClinVar details (significance_breakdown, top_phenotypes) remain in
    # gene_metadata for the collapsed technical UI card.
    _correction_prefix_t2 = (
        f"ייתכן שהתכוונת לגן {gene} (מתיקון אוטומטי של '{corrected_from}').\n\n"
        if corrected_from else ""
    )
    tier2_fallback_answer = (
        _correction_prefix_t2 +
        f"עדיין אין לי סיכום ביולוגי זמין לגן {gene}. "
        f"ניתן לראות פרטים טכניים ממאגר ClinVar בכרטיס המידע."
    )
    suggested = _gene_suggested_questions(question, gene)

    # Session 27.8 Part B: check for source-grounded phrasing FIRST.
    # Grounded phrasing uses only supplied context (ClinVar phenotypes /
    # approved_context_summary) and does NOT require physician review.
    # ClinVar variant counts alone are NOT sufficient (Part B invariant).
    _grounded_debug: dict = {}
    has_grounded, grounded_ctx = _has_sufficient_grounded_gene_context(gene, summary)
    grounded_answer: "Optional[str]" = None
    if has_grounded:
        grounded_answer = _generate_source_grounded_gene_answer(
            gene, grounded_ctx, _debug=_grounded_debug
        )

    # Session 27.8 Part C: gene answer cascade
    # Tier 2 priority: grounded phrasing → approved physician draft → fallback+unverified.
    # Only generate unverified draft when grounded phrasing is unavailable (Part I).
    _draft_debug: dict = {}
    unverified_draft = None
    draft_available = False
    _approved_db_draft = None
    _draft_promoted = False

    if grounded_answer:
        # Source-grounded: LLM phrases supplied context; no review queue; no unverified card.
        main_answer = _correction_prefix_t2 + grounded_answer
        _answer_source = AI_CONTENT_TYPE_GROUNDED
    else:
        # Session 27.7 Part B: look for a physician-approved draft in ALL visibility
        # modes.  Approved (physician-vetted) content fills the curated-content gap
        # regardless of whether pending/unreviewed drafts are shown.
        try:
            from app import review_db as _rdb
            _approved_db_draft = _rdb.get_approved_draft(gene, draft_type="gene_summary")
        except Exception:
            pass  # degraded silently — show fallback

        if _approved_db_draft:
            main_answer = _correction_prefix_t2 + _approved_db_draft["effective_text"]
            _answer_source = "approved"
            _draft_promoted = True
            if _AI_DRAFT_VISIBILITY_MODE == "approved_only":
                draft_available = False
        else:
            # No grounded context, no approved draft: generate unverified expansion.
            # PENDING drafts NEVER become the main answer (Session 27.6.1 invariant).
            unverified_draft = _generate_unverified_gene_draft(
                gene, question, clinvar_context=summary, use_lenient_validator=True,
                _debug=_draft_debug,
            )
            draft_available = unverified_draft is not None
            if _AI_DRAFT_VISIBILITY_MODE == "approved_only":
                main_answer = tier2_fallback_answer
                _answer_source = "fallback"
                draft_available = False
            else:
                main_answer = tier2_fallback_answer
                _answer_source = "fallback"

    # Supplemental card visibility — only for unverified expanded drafts.
    # Grounded answers (main) and approved-promoted answers never show a second card.
    _draft_text_he = (unverified_draft or {}).get("text_he", "")
    if grounded_answer or _draft_promoted:
        draft_displayable = False
        _draft_hidden_reason: "Optional[str]" = (
            "grounded_answer_is_main" if grounded_answer else "approved_text_is_main_answer"
        )
    elif _AI_DRAFT_VISIBILITY_MODE == "approved_only":
        draft_displayable = False
        _draft_hidden_reason = "approved_only_mode" if draft_available else "no_draft"
    else:
        draft_displayable = (
            draft_available
            and _draft_adds_meaningful_information(_draft_text_he, main_answer)
        )
        _draft_hidden_reason = (
            None if draft_displayable
            else "no_draft" if not draft_available
            else "identical_to_main_answer"
        )

    result: dict = {
        "answer": main_answer,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "gene_clinvar_summary",
        "suggested_questions": suggested,
        "llm_used": grounded_answer is not None or draft_available,
        "fallback_used": grounded_answer is None and not draft_available,
        "llm_mode": (
            "source_grounded" if grounded_answer
            else "draft_openai" if draft_available
            else "none"
        ),
        "gene_metadata": {
            "gene_symbol": gene,
            "data_source": "ClinVar (NCBI) via local gene index",
            "llm_used": grounded_answer is not None or draft_available,
            "fallback_used": grounded_answer is None and not draft_available,
            "total_variants": summary.get("total_variants"),
            "found_in_index": True,
            "answer_tier": "tier2",
            "gene_knowledge_status": (
                "approved" if _draft_promoted
                else "grounded" if grounded_answer
                else "unverified_available" if draft_available
                else "missing"
            ),
            "unverified_gene_draft_available": draft_available,
            # True only when the draft adds content not already in the main answer.
            "unverified_gene_draft_displayable": draft_displayable,
            "draft_hidden_reason": _draft_hidden_reason,
            # True when a physician-approved draft text is the main answer.
            "draft_promoted_to_answer": _draft_promoted,
            # Session 27.8 Part A: AI content type classification.
            "ai_content_type": (
                AI_CONTENT_TYPE_GROUNDED if grounded_answer
                else AI_CONTENT_TYPE_EXPANDED if draft_available
                else None
            ),
            "source_grounded": grounded_answer is not None,
            "requires_physician_review": draft_available and not grounded_answer,
            "ai_draft_attempted": _draft_debug.get("attempted", False),
            "ai_draft_generated": draft_available,
            "significance_breakdown": summary.get("by_significance") or {},
            "top_phenotypes": (summary.get("phenotypes") or [])[:6],
        },
    }
    if draft_available:
        # Include unverified_gene_draft for API consumers and tests, but the
        # frontend card is suppressed via draft_promoted_to_answer=True.
        result["unverified_gene_draft"] = unverified_draft
        result["ai_draft_debug"] = {
            "attempted": True,
            "generated": True,
            "shown": True,
            "provider": _draft_debug.get("provider", "unknown"),
        }
    else:
        result["ai_draft_debug"] = _draft_debug or {
            "attempted": False,
            "generated": False,
            "shown": False,
            "reason": "llm_not_configured_or_unknown",
        }
        result["ai_draft_debug"].setdefault("shown", False)

    # Auto-enqueue to physician review DB for AI_EXPANDED_UNVERIFIED only.
    # SOURCE_GROUNDED_PHRASING must NOT populate the physician review queue (Part I).
    if draft_available and unverified_draft and not grounded_answer:
        _gene_clinvar_persist_record = _persist_reviewable_ai_expansion(
            unverified_draft,
            draft_type="gene_summary",
            normalized_intent="gene_biology_expansion",
            gene_symbol=gene,
            model_provider=_draft_debug.get("provider"),
            prompt_version="s26",  # session when gene biology prompt was introduced
            source_metadata={
                "ai_content_type": "medical_educational_ai_expansion",
                "source_basis": "model_expansion_with_clinvar_gene_context",
                "source_grounded": False,
                "requires_physician_review": True,
                "answer_tier": "tier2",
                "total_variants": summary.get("total_variants"),
            },
        )
        _attach_review_metadata(result, _gene_clinvar_persist_record)

    return result


def _build_gene_education_fallback(gene: str) -> dict:
    """
    Educational gene-level answer for when the ClinVar gene index is unavailable
    but the gene is recognised by _GENE_PATTERNS.  Uses the approved gene card
    entry for the gene if one exists, otherwise the generic template.

    Safe invariants: no personal interpretation, no diagnosis, no treatment advice.
    """
    body = (
        gene_cards.get_approved_summary(gene)
        or gene_knowledge.get_gene_patient_summary(gene)
        or _GENE_EDUCATION_FALLBACK_HE.format(gene=gene)
    )
    return {
        "answer": body,
        "safety_level": "general_information",
        "needs_genetic_counselor": False,
        "matched_topic": "gene_info",
        "suggested_questions": list(_GENE_INFO_SUGGESTED_QUESTIONS),
        "llm_used": False,
        "fallback_used": True,
    }


# ---------------------------------------------------------------------------
# Conversation context (in-memory only, supplied fresh by the caller on
# every request — never persisted, never written to disk or a database).
# ---------------------------------------------------------------------------

_FOLLOWUP_PHRASES = [
    "אפשר לפרט", "תוכל לפרט", "תוכלי לפרט", "אתה יכול לפרט", "את יכולה לפרט",
    "תסביר יותר", "תסבירי יותר", "הסבר נוסף", "הסברה נוספת",
    "מה הכוונה", "מה זה אומר בפועל", "מה המשמעות",
    "אפשר דוגמה", "תן דוגמה", "תני דוגמה", "אפשר לתת דוגמה",
    "לא הבנתי", "ומה לגבי זה", "מה לגבי זה",
    "איך זה קשור אליי", "איך זה קשור לי",
    "תוכל להרחיב", "תוכלי להרחיב", "אפשר להרחיב", "פרט יותר", "פרטי יותר",
    # Implications / "tell me more" patterns — the gene+VUS step (step 4) fires
    # BEFORE step 5, so these phrases cannot accidentally short-circuit an
    # initial gene+VUS question; they only resolve via prior context.
    "מה ההשלכות",
    "ההשלכות של",
    "בכל זאת מה",
    "ספר לי",
    "תספר לי",
    "תוכל לספר",
    "יכול לספר",
    "יכולה לספר",
    "מה אפשר לומר",
    "מה ניתן לומר",
    "יותר על זה",
    "עוד על זה",
    # Action-oriented follow-ups ("what should I/we do with this?").
    # Step 4 (gene+VUS detection) fires BEFORE step 5, so these phrases
    # only activate as follow-ups when prior context exists — they cannot
    # accidentally short-circuit an initial gene+VUS question.
    "מה כדאי לעשות",
    "מה עושים עם",
    "מה לעשות עם",
    "מה הצעד הבא",
    "מה עושים",
    "what should i do",
    "what do we do",
    "next steps",
    # Additional "give me more" patterns
    "תרחיב",
    "תרחיבי",
    "מידע נוסף",
    "פרטים נוספים",
    "ספר עוד",
    "ספרי עוד",
    "ומה עוד",
    "יש עוד",
    "מה נוסף",
    # English
    "can you elaborate", "tell me more", "what do you mean", "give me an example",
    "i don't understand", "i didn't understand",
    "what are the implications",
    "what can you tell me",
]


def _is_followup_question(text: str) -> bool:
    """Detect short, vague continuation phrases that depend on prior context."""
    lower = text.strip().lower()
    return any(p in lower for p in _FOLLOWUP_PHRASES)


def _sanitize_context(conversation_context: Optional[list]) -> list:
    """
    Defense in depth: drop any context message whose content looks like it
    contains identifying information, BEFORE it is used for anything
    (topic/gene resolution, or — if ever passed along — an LLM prompt).
    conversation_context is supplied fresh by the caller every request and
    is never stored server-side; this function only filters what is used
    within the current request.
    """
    if not conversation_context:
        return []
    safe = []
    for msg in conversation_context:
        try:
            content = str(msg.get("content") or "")
        except AttributeError:
            continue
        if safety.contains_identifying_info(content):
            continue
        safe.append(msg)
    return safe


def _resolve_followup_context(
    conversation_context: list, last_topic: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """
    Determine (topic_id, gene) to elaborate on for a follow-up question,
    preferring the explicit last_topic param, then scanning the (already
    sanitized) conversation context backwards for the most recent
    assistant matched_topic and any gene name mentioned.
    """
    topic_id = last_topic
    gene = None
    for msg in reversed(conversation_context):
        content = str(msg.get("content") or "")
        if topic_id is None and msg.get("role") == "assistant":
            mt = msg.get("matched_topic")
            if mt:
                topic_id = mt
        if gene is None:
            g = _detect_known_gene(content)
            if g:
                gene = g
            elif gene_index._GENE_INDEX_AVAILABLE:
                # Extended lookup for genes not covered by _detect_known_gene
                # (e.g. SHANK3, TP53, CFTR).  Scans only one message at a time,
                # so the per-message DB overhead is bounded.
                g = _extract_gene_symbol_from_question(content)
                if g:
                    gene = g
        if topic_id and gene:
            break
    return topic_id, gene


def _build_followup_answer(topic_id: str, gene: Optional[str]) -> Optional[dict]:
    """
    Build an expanded answer for a follow-up question about a previously
    discussed topic. Pulls in related KB entries (referenced by the
    original topic's own suggested_questions, or a fixed related set for
    the dynamic pseudo-topics) so the response genuinely elaborates rather
    than repeating the same single paragraph. Returns None if topic_id
    can't be resolved to anything — the caller then falls through to
    normal handling of the literal follow-up text.
    """
    if topic_id in ("vus_known_gene", "vus"):
        # Use the structured 5-section practical answer instead of raw KB text
        # concatenation. For vus_known_gene we have a gene; for general vus we don't.
        effective_gene = gene if topic_id == "vus_known_gene" else gene
        answer = _compose_vus_practical_answer(effective_gene)
        e = kb.get_by_id(topic_id)
        suggested = list(e.get("suggested_questions", [])) if e else []
        return {
            "answer": answer,
            "safety_level": "general_information",
            "needs_genetic_counselor": False,
            "matched_topic": topic_id,
            "suggested_questions": suggested,
            "llm_used": False,
            "fallback_used": True,
        }

    if topic_id == "carrier":
        # Carrier follow-up: the carrier entry's suggested_questions all
        # contain the word "נשאות"/"נשא", so generic kb.match_question() always
        # re-matches the carrier entry itself (already in `seen`). Pull
        # related entries by ID instead.
        carrier_entry = kb.get_by_id("carrier")
        parts = [carrier_entry["approved_answer_he"]] if carrier_entry else []
        for related_id in ("carrier_vs_affected", "carrier_partner_testing", "autosomal_recessive"):
            e = kb.get_by_id(related_id)
            if e:
                parts.append(e["approved_answer_he"])
        suggested = list(carrier_entry.get("suggested_questions", [])) if carrier_entry else []
        return {
            "answer": "\n\n".join(parts),
            "safety_level": "general_information",
            "needs_genetic_counselor": False,
            "matched_topic": "carrier",
            "suggested_questions": suggested,
            "llm_used": False,
            "fallback_used": True,
        }

    if topic_id == "variant_evidence_summary":
        # Elaborating on a previous variant-evidence answer must NOT re-run
        # a ClinVar lookup keyed off a vague follow-up phrase — just restate
        # the safety boundary plus the general interpretation factors.
        return {
            "answer": f"{VARIANT_SAFETY_BOUNDARY_HE}\n\n{_no_evidence_explanation_he()}",
            "safety_level": "requires_genetic_counselor",
            "needs_genetic_counselor": True,
            "matched_topic": "variant_evidence_summary",
            "suggested_questions": list(_VARIANT_SUGGESTED_QUESTIONS),
            "llm_used": False,
            "fallback_used": True,
        }

    if topic_id == "gene_clinvar_summary":
        # Follow-up on a gene-level ClinVar answer: rebuild the deterministic
        # summary for the same gene (no LLM needed for a follow-up).
        if gene and gene_index._GENE_INDEX_AVAILABLE:
            summary = gene_index.get_gene_summary(gene)
            if summary:
                answer = _build_gene_clinvar_deterministic_answer(gene, summary)
                suggested = [q.replace("בגן זה", f"ב-{gene}") for q in _GENE_SUGGESTED_QUESTIONS]
                return {
                    "answer": answer,
                    "safety_level": "general_information",
                    "needs_genetic_counselor": False,
                    "matched_topic": "gene_clinvar_summary",
                    "suggested_questions": suggested,
                    "llm_used": False,
                    "fallback_used": True,
                }
        # Gene not found in context or index unavailable — fall through
        return None

    if topic_id == "gene_info":
        # Follow-up on a gene-education answer (index-unavailable path)
        if gene:
            return _build_gene_education_fallback(gene)
        return None

    entry = kb.get_by_id(topic_id)
    if not entry:
        return None

    parts = [entry["approved_answer_he"]]
    seen = {entry["id"]}
    for sq in entry.get("suggested_questions", [])[:4]:
        related = kb.match_question(sq)
        if related and related["id"] not in seen:
            parts.append(related["approved_answer_he"])
            seen.add(related["id"])
        if len(parts) >= 4:
            break

    requires_gc = bool(entry.get("requires_genetic_counselor", False))
    return {
        "answer": "\n\n".join(parts),
        "safety_level": "requires_genetic_counselor" if requires_gc else "general_information",
        "needs_genetic_counselor": requires_gc,
        "matched_topic": entry["id"],
        "suggested_questions": list(entry.get("suggested_questions", [])),
        "llm_used": False,
        "fallback_used": True,
    }

# ---------------------------------------------------------------------------
# LLM prompt (KB-grounded, Hebrew-only output) — used for the general
# knowledge-base lookup path, not the variant-evidence path above.
# ---------------------------------------------------------------------------

COUNSELING_SYSTEM_PROMPT = (
    "You are a Hebrew post-genetic-counseling assistant. You help patients "
    "understand general genetic concepts after they already met a genetic "
    "counselor. You do not interpret personal genetic results, do not "
    "diagnose, do not calculate personal risk, and do not give medical "
    "recommendations. Use only the approved knowledge-base content provided "
    "below, plus the minimal recent conversation context provided (if any). "
    "If the approved content is not a perfect match, you may still give a "
    "general explanation grounded in it, or ask one brief clarifying "
    "question — but you must never invent medical facts beyond it. "
    "Answer in Hebrew, clearly and calmly. Keep technical terms such as "
    "VUS, pathogenic, likely pathogenic, benign, carrier, BRCA1, NF1, ACMG, "
    "and Franklin in English."
)


def _format_recent_context(conversation_context: list, max_messages: int = 4) -> str:
    """Render a short, already-sanitized slice of recent turns for the LLM prompt."""
    if not conversation_context:
        return "(no prior context)"
    recent = conversation_context[-max_messages:]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior context)"


def _build_user_prompt(question: str, entry: dict, conversation_context: Optional[list] = None) -> str:
    safety_notes = "; ".join(entry.get("safety_notes", [])) or "none"
    context_block = _format_recent_context(conversation_context or [])
    return (
        "Approved knowledge-base content (Hebrew):\n"
        f"Topic: {entry['title_he']}\n"
        f"Approved answer: {entry['approved_answer_he']}\n"
        f"Safety notes: {safety_notes}\n\n"
        f"Recent conversation context (for tone/continuity only, not a new source of facts):\n{context_block}\n\n"
        f"User question (Hebrew): {question}\n\n"
        "Using ONLY the approved answer above, write a clear, calm Hebrew "
        "reply to the user's question. Do not add facts that are not in "
        "the approved content. If the approved content does not actually "
        "answer the question, say there is not enough approved information "
        "and recommend contacting the genetic counselor."
    )


def _call_local_llm(question: str, entry: dict, conversation_context: Optional[list] = None) -> Optional[str]:
    """
    Ask the configured local LLM to phrase the KB answer. Returns None
    (never raises) if LOCAL_LLM_URL is unset or the call fails for any
    reason, so the caller can fall back to the deterministic KB answer.
    """
    url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if not url:
        return None
    try:
        client = LocalLLMClient(url)
        raw = client._call_api(
            _build_user_prompt(question, entry, conversation_context),
            system_prompt=COUNSELING_SYSTEM_PROMPT,
        )
        text = raw.strip()
        return text or None
    except LLMClientError as exc:
        logger.warning("Local LLM unavailable for counseling assistant (%s); using KB fallback.", exc)
        return None
    except Exception as exc:  # defensive — never let LLM errors break /ask
        logger.warning("Unexpected error calling local LLM (%s); using KB fallback.", exc)
        return None


# ---------------------------------------------------------------------------
# Intent classification — single routing pass
# ---------------------------------------------------------------------------

# Explicit Hebrew pronouns / possessives that signal "this question is about
# the entity mentioned in the PREVIOUS turn."
# Standalone questions like "מה זה המוגלובין?" must NEVER contain these.
# Follow-up questions like "מה התפקיד שלו?" MUST contain at least one.
_GENE_FOLLOWUP_PRONOUN_SIGNALS: frozenset[str] = frozenset([
    "שלו",           # "מה התפקיד שלו?"
    "אליו",          # "קשור אליו"
    "בו",            # "מה קורה בו?"
    "לגביו",         # "ומה לגביו?"
    "עליו",          # "מה אפשר לדעת עליו?"
    "ממנו",          # "שמעת ממנו?"
    "הוא קשור",      # "לאיזה מצבים הוא קשור?"
    "הוא מקושר",
    "הוא עושה",      # "מה הוא עושה?"
    "הוא מייצר",
    "הוא ממוקם",
    "תפקידו",        # "מה תפקידו?"
    "הקשר אליו",
    "המשמעות שלו",
    "התפקיד שלו",
    "הקשר שלו",
    "לאיזה מצבים הוא",
    "לאיזה מחלות הוא",
    "מה הוא עשוי",
])


def _has_gene_followup_signal(text: str) -> bool:
    """
    Return True ONLY when text has explicit pronoun/possessive signals that
    indicate the question is a follow-up about a previously mentioned gene.

    Conservative by design — ambiguous questions fall through to general
    concept handling, NOT to the previous gene.

    Examples:
      "מה התפקיד שלו?"     → True  (possessive "שלו")
      "לאיזה מצבים הוא קשור?" → True  (pronoun reference "הוא קשור")
      "מה זה המוגלובין?"   → False (standalone concept)
      "מה זה כרומוזום?"    → False (standalone concept)
      "מה זה אלצהיימר?"    → False (standalone concept)
    """
    lower = text.strip().lower()
    return any(sig in lower for sig in _GENE_FOLLOWUP_PRONOUN_SIGNALS)


def classify_question_intent(
    question: str,
    last_gene_symbol: Optional[str] = None,
    topic: Optional[str] = None,
) -> dict:
    """
    Single-pass intent router for answer_question.

    Routing priority (first match wins):
      A. privacy_identifier   — identifying info detected
      B. reproductive_block   — abortion / pregnancy termination decision
      C. specific_variant     — named variant (HGVS/rsID) — gets evidence summary
      D. personal_high_stakes — personal medical interpretation / action request
      E. explicit_gene_question — gene symbol found in the CURRENT text
      F. gene_followup        — pronoun signals + last_gene_symbol present
      G. unclear              — fall through to KB → AI fallback → helpful fallback

    Returns:
      {
        "intent":      str,
        "gene_symbol": Optional[str],  # gene to route to (from text or context)
        "reason":      str,            # human-readable explanation
      }

    NOTE: step F (gene_followup) requires BOTH:
      - last_gene_symbol is set (previous turn had a gene answer)
      - _has_gene_followup_signal(text) is True (pronoun/possessive present)
    Standalone concept questions ("מה זה המוגלובין?", "מה זה כרומוזום?") will
    never reach step F even if last_gene_symbol is set.
    """
    text = question.strip()

    # A. Privacy identifiers — block before any other logic
    if safety.contains_identifying_info(text):
        return {"intent": "privacy_identifier", "gene_symbol": None,
                "reason": "identifying_info_detected"}

    # A.5. Out-of-domain — clearly non-genetics/medicine; fires before KB lookup
    if _detect_out_of_domain(text):
        return {"intent": "out_of_domain", "gene_symbol": None,
                "reason": "out_of_domain_detected"}

    # B. Reproductive / abortion decision — irreversible, special boundary
    if _is_reproductive_decision_question(text):
        return {"intent": "reproductive_block", "gene_symbol": None,
                "reason": "reproductive_decision_question"}

    # B.3. Extra sex chromosome (XXY, XXX, XYY, "כרומוזום X עודף") — educational
    if _detect_extra_chromosome(text):
        return {"intent": "extra_chromosome_education", "gene_symbol": None,
                "reason": "extra_chromosome_signals_in_text"}

    # B.5. Trisomy 21 / Down syndrome — educational; fires before personal block
    if _detect_trisomy21(text):
        return {"intent": "trisomy21_education", "gene_symbol": None,
                "reason": "trisomy21_signals_in_text"}

    # B.7. General chromosomal / cytogenetic question (ambiguous chromosome finding,
    #   deletion, duplication, translocation, mosaicism, karyotype, microarray, aneuploidy).
    #   Fires after the more-specific trisomy21 / extra_chromosome steps above.
    _chr_sub_intent = _detect_chromosome_education(text)
    if _chr_sub_intent:
        return {"intent": "chromosome_education", "gene_symbol": None,
                "reason": "chromosome_education_detected",
                "sub_intent": _chr_sub_intent}

    # C. Specific named variant (HGVS / rsID) — evidence summary, not refused
    if safety.contains_specific_variant(text):
        return {"intent": "specific_variant", "gene_symbol": None,
                "reason": "specific_variant_in_text"}

    # C.3. Surgery/major-procedure decision — fires before educational personal context
    # so that "האם לעשות כריתת שד?" is never misrouted to educational content.
    if _is_surgery_decision_question(text):
        return {"intent": "surgery_decision", "gene_symbol": _detect_known_gene(text),
                "reason": "surgery_decision_question"}

    # C.5. Educational personal context — personal phrasing but seeking explanation.
    # Fires BEFORE personal_high_stakes so that questions like "הרופא אמר שיש לי VUS,
    # מה האפשרויות?" are routed to education, not blocked.
    if _is_educational_personal_context(text):
        return {"intent": "educational_personal_context", "gene_symbol": None,
                "reason": "educational_personal_phrasing"}

    # C.7. Carrier screening / ancestry-related genetic testing (27.9.1 Part D).
    # Fires before personal_high_stakes so "אם אני אשכנזייה איזה בדיקות כדאי לי לעשות"
    # does not fall into the KB and misroute to penetrance.
    _cs_intent = _detect_carrier_screening_intent(text)
    if _cs_intent:
        return {"intent": _cs_intent, "gene_symbol": None,
                "reason": "carrier_screening_detected"}

    # D. Personal medical interpretation / action request
    if safety.is_personal_interpretation_request(text):
        return {"intent": "personal_high_stakes", "gene_symbol": None,
                "reason": "personal_interpretation_request"}

    # E. Explicit gene symbol in the CURRENT question text
    if not topic:
        gene_in_text: Optional[str] = None
        corrected_from_ci: Optional[str] = None
        if gene_index._GENE_INDEX_AVAILABLE:
            gene_in_text, corrected_from_ci = _extract_gene_with_correction(text)
        if gene_in_text is None:
            gene_in_text = _detect_known_gene(text)
        # Extended fallback: extract from "בגן X" / "הגן X" Hebrew phrases.
        # Supports mixed-case HGNC symbols (C12orf57, KIAA2022) that the
        # index-based extractor misses because _GENE_SYMBOL_CANDIDATE_RE
        # requires all-uppercase tokens.
        if gene_in_text is None:
            gene_in_text = _extract_gene_from_explicit_phrase(text)
        if gene_in_text and (
            _is_gene_level_question(text)
            or _is_standalone_gene_query(text, gene_in_text)
            or _mentions_vus(text)
        ):
            return {"intent": "explicit_gene_question", "gene_symbol": gene_in_text,
                    "reason": "gene_symbol_in_text",
                    "_corrected_from": corrected_from_ci}

    # F. Gene follow-up — REQUIRES explicit pronoun/possessive signal.
    #    "מה זה המוגלובין?" and "מה זה כרומוזום?" must NEVER reach here.
    if last_gene_symbol and not topic and _has_gene_followup_signal(text):
        return {"intent": "gene_followup", "gene_symbol": last_gene_symbol,
                "reason": "pronoun_followup_with_prior_gene"}

    # G. No strong signal — fall through to KB / AI fallback
    return {"intent": "unclear", "gene_symbol": None, "reason": "no_specific_match"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _answer_question_impl(
    question: str,
    topic: Optional[str] = None,
    conversation_context: Optional[list] = None,
    last_topic: Optional[str] = None,
    include_unverified_gene_draft: bool = False,
    session_context: Optional[dict] = None,
) -> dict:
    """
    Build the full response for POST /ask.

    conversation_context and last_topic are supplied fresh by the caller on
    every request (the frontend's in-memory session) — nothing here reads
    from or writes to any server-side store.

    Returns a dict with keys: answer, safety_level, needs_genetic_counselor,
    matched_topic, suggested_questions.
    """
    text = (question or "").strip()
    safe_context = _sanitize_context(conversation_context)

    # ── Single intent classification pass ─────────────────────────────────
    # classify_question_intent() runs ALL safety and routing checks once.
    # Its result drives routing for steps A–F below.
    # Steps 5–7 (KB follow-up, KB lookup, fallback) run after intent routing.
    # last_gene_symbol is intentionally NOT passed — context bleed prevention.
    # Gene follow-up routing via prior context is disabled; every question is
    # classified on its own text only.
    intent_info = classify_question_intent(text, last_gene_symbol=None, topic=topic)
    intent = intent_info["intent"]

    # A. Privacy identifiers — block, never reach LLM/KB/ClinVar.
    if intent == "privacy_identifier":
        return {
            "answer": PRIVACY_WARNING_HE,
            "safety_level": "contains_identifying_info",
            "needs_genetic_counselor": False,
            "matched_topic": None,
            "suggested_questions": [],
            "llm_used": False,
            "fallback_used": False,
        }

    # A.5. Out-of-domain — short "not my domain" message; bypasses KB lookup.
    if intent == "out_of_domain":
        return {
            "answer": _OUT_OF_DOMAIN_HE,
            "safety_level": "out_of_scope",
            "needs_genetic_counselor": False,
            "matched_topic": None,
            "suggested_questions": [],
            "llm_used": False,
            "fallback_used": True,
        }

    # B. Reproductive / abortion decision — irreversible; always fires even when
    #    the question also contains a gene name or specific variant.
    if intent == "reproductive_block":
        return {
            "answer": REPRODUCTIVE_DECISION_HE,
            "safety_level": "requires_genetic_counselor",
            "needs_genetic_counselor": True,
            "matched_topic": None,
            "suggested_questions": list(_GENERIC_SUGGESTED_QUESTIONS),
            "llm_used": False,
            "fallback_used": False,
        }

    # B.3. Extra sex chromosome educational answer.
    if intent == "extra_chromosome_education":
        return _build_extra_chromosome_answer()

    # B.5. Trisomy 21 / Down syndrome educational answer.
    if intent == "trisomy21_education":
        return _build_trisomy21_answer()

    # B.7. General chromosomal / cytogenetic educational answer.
    # Pass chromosome_number from session context so that follow-up turns
    # like "אמרו שזה mosaic" (after a chr21 first turn) include the number.
    if intent == "chromosome_education":
        _b7_chr_num = (session_context or {}).get("chromosome_number")
        return _build_chromosome_education_answer(
            text, intent_info.get("sub_intent", "chromosome_finding_general"),
            chromosome_number=_b7_chr_num,
        )

    # C. Specific named variant (HGVS / rsID) — educational evidence summary,
    #    not refused outright; runs before personal-interpretation check.
    if intent == "specific_variant":
        return _build_variant_evidence_answer(text)

    # C.7. Carrier screening / ancestry-related testing (27.9.1 Part D).
    if intent == "carrier_screening_general":
        return _build_carrier_screening_answer(text, is_personal=False)
    if intent == "carrier_screening_personal":
        return _build_carrier_screening_answer(text, is_personal=True)

    # C.5. Educational personal context — personal phrasing but seeking education.
    # Extract gene + route exactly like E, then fall through to KB/AI if no gene.
    if intent == "educational_personal_context":
        _edu_gene: Optional[str] = None
        _edu_corrected: Optional[str] = None
        if gene_index._GENE_INDEX_AVAILABLE:
            _edu_gene, _edu_corrected = _extract_gene_with_correction(text)
        if _edu_gene is None:
            _edu_gene = _detect_known_gene(text)
        if _edu_gene:
            if _mentions_vus(text):
                if _is_vus_options_request(text):
                    return _build_vus_options_answer(_edu_gene)
                return _build_known_gene_answer(
                    _edu_gene, question=text, include_unverified_gene_draft=True)
            if gene_index._GENE_INDEX_AVAILABLE:
                _edu_result = _build_gene_clinvar_answer(
                    text, _edu_gene, include_unverified_gene_draft=True,
                    corrected_from=_edu_corrected)
                if _edu_result is not None:
                    return _edu_result
            else:
                return _build_gene_education_fallback(_edu_gene)
        # No gene found — fall through to KB / AI / fallback below.

    # C.3. Surgery/major-procedure decision — VUS-aware contextual answer.
    if intent == "surgery_decision":
        return _build_surgery_decision_answer(text, gene_symbol=intent_info.get("gene_symbol"))

    # D. Personal medical interpretation / action request — redirect to counselor.
    if intent == "personal_high_stakes":
        return {
            "answer": PERSONAL_REDIRECT_HE,
            "safety_level": "requires_genetic_counselor",
            "needs_genetic_counselor": True,
            "matched_topic": None,
            "suggested_questions": list(_GENERIC_SUGGESTED_QUESTIONS),
            "llm_used": False,
            "fallback_used": False,
        }

    # E/F. Gene routing — explicit gene in text, or confirmed follow-up with pronoun.
    #
    # CRITICAL: last_gene_symbol is used ONLY when intent == "gene_followup",
    # which requires _has_gene_followup_signal() to be True (explicit pronoun/
    # possessive present). Standalone concept questions ("מה זה המוגלובין?",
    # "מה זה כרומוזום?") never reach here even when last_gene_symbol is set.
    if intent in ("explicit_gene_question", "gene_followup"):
        gene_for_routing = intent_info["gene_symbol"]
        corrected_from = intent_info.get("_corrected_from")

        # VUS + explicit gene + options request → practical options answer
        if intent == "explicit_gene_question" and _mentions_vus(text) and _is_vus_options_request(text):
            return _build_vus_options_answer(gene_for_routing)

        # VUS + explicit gene → enriched VUS answer
        if intent == "explicit_gene_question" and _mentions_vus(text):
            return _build_known_gene_answer(
                gene_for_routing, question=text,
                include_unverified_gene_draft=True,
            )

        # Gene-level question → ClinVar / curated / function-first answer
        if gene_index._GENE_INDEX_AVAILABLE:
            result = _build_gene_clinvar_answer(
                text, gene_for_routing,
                include_unverified_gene_draft=True,
                corrected_from=corrected_from,
            )
            if result is not None:
                return result
        elif intent == "explicit_gene_question":
            # Gene index unavailable: use pattern-based educational fallback.
            known = _detect_known_gene(text)
            if known:
                return _build_gene_education_fallback(known)

    # 4.5. Chromosome follow-up routing (Session 27.8 Part F).
    # Fires when session_context.active_topic is a chromosome education sub-intent
    # and the message contains a finding-type keyword (e.g. 'מחיקה', 'duplication').
    # Must be AFTER gene routing (E/F) and BEFORE general follow-up detection (step 5).
    _chr_followup = _resolve_chromosome_followup(text, session_context)
    if _chr_followup:
        return _build_chromosome_education_answer(
            text,
            _chr_followup["sub_intent"],
            chromosome_number=_chr_followup.get("chromosome_number"),
        )

    # 5. Follow-up handling — vague continuation phrases resolved via
    #    last_topic / sanitized conversation context, not KB keyword scoring.
    if not topic and _is_followup_question(text):
        followup_topic, followup_gene = _resolve_followup_context(safe_context, last_topic)
        if followup_topic:
            result = _build_followup_answer(followup_topic, followup_gene)
            if result is not None:
                return result
        # Couldn't resolve any prior context — fall through to normal
        # handling below (most likely ends in the helpful fallback).

    # 5.5. Pre-KB general education bypass (Session 27.9.1 Part C).
    # _classify_general_question correctly identifies questions like
    # "כמה גנים יש לצב ים?" as safe_general_education — but without this
    # step the KB fires first and returns a false-positive entry (e.g.,
    # what_is_gene fuzzy-matching any question that contains 'גנים').
    # Running the classifier here, before kb.match_question, lets the AI
    # route fire first when the flag is enabled and the LLM is available.
    if not topic and intent == "unclear" and _ai_general_education_fallback_enabled():
        _pre_route = _classify_general_question(text)
        if _pre_route == "safe_general_education":
            _pre_result, _pre_debug = _build_general_education_answer(text)
            if _pre_result is not None:
                return _pre_result
            # LLM unavailable — fall through to KB. The KB may return a false
            # positive (e.g., human-gene definition for an animal question), but
            # that is preferable to a hard failure when AI is not configured.

    # 6. Knowledge-base lookup (exact + fuzzy fallback tier inside kb.py).
    entry = kb.match_question(text, topic_hint=topic)
    # Guard: reject x_linked KB match when the question has no X-chromosome signal.
    # "מה זה כרומוזום?" fuzzy-scores 0.600 against x_linked via "תלוי כרומוזום x".
    if entry is not None and entry.get("id") == "x_linked":
        lower_q = text.lower()
        _x_signals = (
            "x-linked", "x linked", "x_linked",
            "תאחיזה", "תלויית x", "תלוי x",
            "תלוי-x", "תלויה ב-x", "כרומוזום x", "chromosome x",
        )
        if not any(sig in lower_q for sig in _x_signals):
            entry = None
    # Guard (27.9.1 Part C): reject what_is_gene for non-human biology questions
    # only (e.g. sea turtle, dog chromosomes).  Human questions like "מה זה גן?"
    # or "כמה גנים יש בבן אדם?" must still reach the KB answer.
    # Extended (27.9.8): also reject when an explicit gene symbol is present —
    # "איך הגן BRCA1 פועל?" must route to gene-specific pipeline, not generic definition.
    if entry is not None and entry.get("id") == "what_is_gene":
        _lower_q = text.lower()
        if any(sig in _lower_q for sig in _NON_HUMAN_BIOLOGY_ORGANISMS):
            entry = None
        elif _detect_known_gene(text) or _extract_gene_from_explicit_phrase(text):
            entry = None  # specific gene present → route to gene-specific pipeline
    # Guard (27.9.8): reject human_genome_size for history/discovery questions.
    # "מתי גילו את הגנום?" is NOT asking how many genes there are; fuzzy difflib
    # matches "גנום" to this entry's keywords ("גנום אנושי").
    if entry is not None and entry.get("id") == "human_genome_size":
        _lower_q = text.lower()
        _count_signals = ("כמה", "מספר", "how many", "number of")
        if not any(sig in _lower_q for sig in _count_signals):
            entry = None
    # Guard (27.9.1 Part D): reject penetrance when the question is about
    # carrier screening / ancestry-related testing — fuzzy difflib incorrectly
    # matches these queries against penetrance KB keywords.
    if entry is not None and entry.get("id") == "penetrance":
        if _detect_carrier_screening_intent(text) is not None:
            entry = None
    if entry is None:
        # 6.4. Out-of-domain — clearly non-genetics/medicine — return short message
        # regardless of AI settings; fires before AI fallback.
        if not topic and _detect_out_of_domain(text):
            return {
                "answer": _OUT_OF_DOMAIN_HE,
                "safety_level": "out_of_scope",
                "needs_genetic_counselor": False,
                "matched_topic": None,
                "suggested_questions": [],
                "llm_used": False,
                "fallback_used": True,
            }

        # 6.5. General education AI fallback — fires only in staging/development
        # with AI_GENERAL_EDUCATION_FALLBACK_ENABLED=true, and only for safe
        # concept questions that are not personal or high-stakes.
        _gen_debug: Optional[dict] = None
        _app_env_val = os.environ.get("APP_ENV", "production").strip().lower()
        _in_staging = _app_env_val in ("staging", "development")
        if not topic and _ai_general_education_fallback_enabled():
            _route = _classify_general_question(text)
            if _route == "safe_general_education":
                _gen_result, _gen_debug = _build_general_education_answer(text)
                if _gen_result is not None:
                    return _gen_result
                # Draft failed — _gen_debug carries LLM error info.
            else:
                _gen_debug = {
                    "attempted": False,
                    "enabled": True,
                    "safety_route": _route,
                    "reason": f"classifier_route_{_route}",
                }
        elif _in_staging:
            _flag_val = os.environ.get("AI_GENERAL_EDUCATION_FALLBACK_ENABLED", "").strip().lower()
            _gen_debug = {
                "attempted": False,
                "enabled": _flag_val in ("1", "true", "yes"),
                "reason": "topic_set" if topic else "flag_disabled_or_env_mismatch",
            }
        # Non-human biology questions (sea turtle, dog chromosomes, etc.) must not
        # receive a personal-medical referral — they are comparative genomics questions
        # that have nothing to do with the user's own results.
        # Only fire when the question has an explicit non-human organism signal —
        # avoid catching generic questions that match "מה ה" (educational intent phrase).
        if not topic:
            _lower_text_fb = text.lower()
            if any(sig in _lower_text_fb for sig in _NON_HUMAN_BIOLOGY_ORGANISMS):
                _edu_fallback: dict = {
                    "answer": (
                        "שאלה ביולוגית מעניינת. אין לי כרגע נתונים ספציפיים מאושרים לגבי "
                        "שאלה זו במאגר. מידע על גנומיקה השוואתית ניתן למצוא במקורות כגון "
                        "NCBI Gene ו-Ensembl."
                    ),
                    "safety_level": "general_information",
                    "needs_genetic_counselor": False,
                    "matched_topic": None,
                    "suggested_questions": [],
                    "llm_used": False,
                    "fallback_used": True,
                }
                if _gen_debug is not None:
                    _edu_fallback["ai_general_debug"] = _gen_debug
                return _edu_fallback
        _fallback = _build_helpful_fallback(text)
        if _gen_debug is not None:
            _fallback["ai_general_debug"] = _gen_debug
        return _fallback

    # 7. Intro-only LLM (optional).  The LLM may add ONE validated Hebrew opening
    #    sentence; the deterministic KB answer is always used as the substantive
    #    content.  Full KB rewriting is disabled — the LLM never generates medical
    #    facts for user-facing answers.
    # Curated KB/FAQ answers are always deterministic — no LLM intro is prepended.
    # This prevents poor-quality Hebrew filler sentences from appearing before
    # the authoritative curated answer.
    deterministic_kb = entry["approved_answer_he"]

    requires_gc = bool(entry.get("requires_genetic_counselor", False))
    return {
        "answer": deterministic_kb,
        "safety_level": "requires_genetic_counselor" if requires_gc else "general_information",
        "needs_genetic_counselor": requires_gc,
        "matched_topic": entry["id"],
        "suggested_questions": list(entry.get("suggested_questions", []))[:3],
        "llm_used": False,
        "fallback_used": True,
        "llm_mode": "none",
    }


# ---------------------------------------------------------------------------
# Session context output (Session 27.8.1 Part B/E)
# ---------------------------------------------------------------------------

def _build_session_context_out(
    result: dict,
    question: str,
    session_context: "Optional[dict]",
) -> dict:
    """
    Build the next-turn safe session context from the current answer.

    The returned dict is sent back to the frontend as `conversation_context`
    in the response.  The frontend stores it and sends it as `context` on the
    next request.

    Only whitelisted topic values are included.  Never contains PII, HGVS/ISCN
    notation, raw answer text, or medical free text.
    """
    matched = result.get("matched_topic")
    prev_turn = int((session_context or {}).get("turn_count") or 0)
    ctx_out: dict = {"turn_count": min(prev_turn + 1, 200)}

    # Active topic — whitelist-validated.
    if matched and matched in _SAFE_CONTEXT_ALLOWED_ACTIVE_TOPICS:
        ctx_out["active_topic"] = matched

    # Chromosome number — from chromosome answer metadata or carried forward.
    chr_meta = result.get("chromosome_draft_metadata") or {}
    chr_num = chr_meta.get("chromosome_number_detected")
    if chr_num:
        ctx_out["chromosome_number"] = chr_num
    elif matched in _CHROMOSOME_TOPIC_INTENTS and session_context:
        prev_chr = session_context.get("chromosome_number")
        if prev_chr:
            ctx_out["chromosome_number"] = prev_chr

    # Normalized sub-intent from chromosome follow-up.
    sub_intent = chr_meta.get("sub_intent")
    if sub_intent and sub_intent in _SAFE_CONTEXT_ALLOWED_ACTIVE_TOPICS:
        ctx_out["normalized_intent"] = sub_intent

    # finding_type — specific chromosome finding type, persisted across turns (Part I).
    # "chromosome_finding_general" is the base/unresolved category; finding_type is only
    # set when a specific finding (deletion, duplication, etc.) is identified.
    if sub_intent and sub_intent in _CHROMOSOME_TOPIC_INTENTS and sub_intent != "chromosome_finding_general":
        ctx_out["finding_type"] = sub_intent
    elif matched in _CHROMOSOME_TOPIC_INTENTS and session_context:
        prev_finding = session_context.get("finding_type")
        if prev_finding and prev_finding in _CHROMOSOME_TOPIC_INTENTS:
            ctx_out["finding_type"] = prev_finding

    # Gene symbol from gene metadata — alphanumeric only.
    gene_meta = result.get("gene_metadata") or {}
    gene_sym = gene_meta.get("gene_symbol")
    if gene_sym and re.match(r"^[A-Za-z0-9]{1,20}$", gene_sym):
        ctx_out["gene_symbol"] = gene_sym.upper()

    return ctx_out


def answer_question(
    question: str,
    topic: Optional[str] = None,
    conversation_context: Optional[list] = None,
    last_topic: Optional[str] = None,
    include_unverified_gene_draft: bool = False,
    session_context: Optional[dict] = None,
) -> dict:
    """
    Public entry point for POST /ask.  Delegates to _answer_question_impl()
    and appends session_context_out for the frontend to store and echo back.
    """
    result = _answer_question_impl(
        question=question,
        topic=topic,
        conversation_context=conversation_context,
        last_topic=last_topic,
        include_unverified_gene_draft=include_unverified_gene_draft,
        session_context=session_context,
    )
    result["session_context_out"] = _build_session_context_out(
        result, (question or "").strip(), session_context
    )
    return result


# # gene card LLM framing removed
