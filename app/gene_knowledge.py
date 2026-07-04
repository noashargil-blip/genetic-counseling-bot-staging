"""
app/gene_knowledge.py

Gene Knowledge Base loader.

Loads structured gene records from data/gene_knowledge_base.json.
Only records with approved=True are served as patient-facing answers.
Records with approved=False are available for draft generation only.

Schema (gene_knowledge_base.json):
  gene_symbol          str   — canonical uppercase symbol (BRCA1, POLE, …)
  gene_name            str   — full English gene name
  clinical_area        str   — cancer_predisposition | lynch_polyposis |
                               cardiology | neuromuscular | neurocutaneous |
                               recessive_carrier | hematology | generic
  patient_summary_he   str   — patient-friendly Hebrew educational text
  vus_note_type        str   — key for the VUS note template category
  vus_note_he          str   — pre-written Hebrew VUS note for this gene
  source_1_name        str   — primary source name (MedlinePlus, NCBI, …)
  source_1_url_or_id   str   — primary source URL or identifier
  source_2_name        str   — secondary source name (may be null)
  source_2_url_or_id   str   — secondary source URL (may be null)
  source_status        str   — "verified" | "source_missing" | "needs_review"
  review_status        str   — "draft" | "approved"
  approved             bool  — ONLY True after explicit manual review
  reviewed_by          str   — reviewer name/ID (null until approved)
  reviewed_at          str   — ISO-8601 datetime of approval (null until approved)
  reviewer_notes       str   — free-text notes (null until approved)
  last_updated              str   — ISO-8601 date of last content edit
  approved_context_summary_he str  — approved ClinVar-context summary (from
                                     clinvar_context_summary draft type).
                                     Separate from patient_summary_he so a
                                     context-only draft is never silently used
                                     as a full biological gene summary.

Safety invariant: approved is NEVER set to True automatically.
It is set only by scripts/review_drafts.py after explicit human action.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent / "data" / "gene_knowledge_base.json"
_RECORDS: dict[str, dict] = {}
_GENE_KNOWLEDGE_AVAILABLE: bool = False


def _load() -> None:
    global _GENE_KNOWLEDGE_AVAILABLE
    try:
        raw = _KB_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        loaded = 0
        for record in data:
            sym = (record.get("gene_symbol") or "").strip().upper()
            if sym:
                _RECORDS[sym] = record
                loaded += 1
        _GENE_KNOWLEDGE_AVAILABLE = bool(_RECORDS)
        approved_count = sum(1 for r in _RECORDS.values() if r.get("approved") is True)
        logger.info(
            "gene_knowledge: loaded %d record(s) (%d approved) from %s",
            loaded, approved_count, _KB_PATH,
        )
    except FileNotFoundError:
        logger.info(
            "gene_knowledge: %s not found — gene knowledge base unavailable", _KB_PATH
        )
    except Exception as exc:
        logger.warning(
            "gene_knowledge: failed to parse %s (%s) — gene knowledge base unavailable",
            _KB_PATH, exc,
        )


_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gene_knowledge() -> list[dict]:
    """Return all gene knowledge records (including unapproved drafts)."""
    return list(_RECORDS.values())


def get_gene_knowledge(gene_symbol: str) -> Optional[dict]:
    """Return the full gene knowledge record for the given symbol, or None."""
    return _RECORDS.get(gene_symbol.upper())


def has_approved_gene_knowledge(gene_symbol: str) -> bool:
    """
    Return True only if the gene has a record with approved=True.
    Never returns True for drafts, source_missing, or needs_review records.
    """
    record = _RECORDS.get(gene_symbol.upper())
    return bool(record and record.get("approved") is True)


def get_gene_patient_summary(gene_symbol: str) -> Optional[str]:
    """
    Return the approved patient-friendly Hebrew summary, or None.
    Returns None if the record exists but is not yet approved.
    """
    record = _RECORDS.get(gene_symbol.upper())
    if record and record.get("approved") is True:
        return record.get("patient_summary_he") or None
    return None


def get_gene_vus_note(gene_symbol: str) -> Optional[str]:
    """
    Return the approved VUS note for this gene's clinical area, or None.
    Returns None if the record exists but is not yet approved.
    """
    record = _RECORDS.get(gene_symbol.upper())
    if record and record.get("approved") is True:
        return record.get("vus_note_he") or None
    return None


def get_gene_context_summary(gene_symbol: str) -> Optional[str]:
    """
    Return the approved ClinVar-context summary (approved_context_summary_he),
    or None.  This field is written when a clinvar_context_summary draft type is
    approved via the review CLI.  It is distinct from patient_summary_he and is
    not currently served in patient-facing answers — it exists for future use
    once a physician has reviewed and confirmed its suitability.
    """
    record = _RECORDS.get(gene_symbol.upper())
    if record and record.get("approved") is True:
        return record.get("approved_context_summary_he") or None
    return None


def list_approved_genes() -> list[str]:
    """Return sorted list of all genes with approved=True records."""
    return sorted(sym for sym, r in _RECORDS.items() if r.get("approved") is True)


def list_all_genes() -> list[str]:
    """Return sorted list of all genes in the knowledge base (including drafts)."""
    return sorted(_RECORDS.keys())
