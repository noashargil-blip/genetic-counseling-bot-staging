"""
app/report_text_parser.py

Parse plain-text clinical genetic report snippets into the same normalized
variant format used by report_parser.py and the upload/analyze flow.

Supported input formats
-----------------------
Single variant — any lines in "Label: Value" format:
    Gene: COL4A2
    cDNA: c.2552G>T
    Clinical significance: Uncertain significance
    ...

Multiple variants — separated by "Variant N:" / "Finding N:" header lines:
    Variant 1:
    Gene: BRCA1
    cDNA: c.5266dup
    ...

    Variant 2:
    Gene: TP53
    ...

Multiple variants — separated by two or more consecutive blank lines (fields
within a single variant may be separated by at most one blank line):
    Gene: BRCA1
    cDNA: c.5266dup

    [blank line here separates variants]
    [blank line]

    Gene: TP53
    ...

Recognized labels (case-insensitive)
--------------------------------------
Gene, Symbol, Transcript, Variant, cDNA, HGVS, HGVSc, Protein, HGVSp,
Genomic, Genomic change, Chromosome, Position, Ref, Alt, Zygosity, Inheritance,
Classification, Clinical significance, Interpretation, ClinVar, ClinVar accession,
VCV, rsID, dbSNP, Condition, Associated condition, ACMG, ACMG criteria,
ACMG classification, HPO, Variant type, and more.

Output
------
Same structure returned by parse_structured_report_json():
    {
        "file_type":        "plain_text_report",
        "detected_columns": [...],
        "variants":         [...],
        "report_metadata":  {},
        "warnings":         [...],
    }
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REPORT_TEXT_PARSER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Label → canonical key mapping
# Keys are normalised (lowercase, alphanumeric + spaces only).
# Sorted longest-first so more specific labels take priority.
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, str] = {
    # ── Gene ──────────────────────────────────────────────────────
    "gene":                         "gene",
    "symbol":                       "gene",
    "gene symbol":                  "gene",
    "gene name":                    "gene",
    # ── Transcript ────────────────────────────────────────────────
    "transcript":                   "transcript",
    "transcript id":                "transcript",
    "nm":                           "transcript",
    # ── cDNA variant (aliases → cdna_variant; _extract_variant_fields reads it) ──
    "variant":                      "cdna_variant",
    "cdna":                         "cdna_variant",
    "cdna variant":                 "cdna_variant",
    "c variant":                    "cdna_variant",
    "hgvs":                         "cdna_variant",
    "hgvsc":                        "cdna_variant",
    "hgvs c":                       "cdna_variant",
    "hgvs cdna":                    "cdna_variant",
    "dna change":                   "cdna_variant",
    # ── Protein change ────────────────────────────────────────────
    "protein":                      "protein_change",
    "protein change":               "protein_change",
    "protein effect":               "protein_change",
    "amino acid change":            "protein_change",
    "hgvsp":                        "protein_change",
    "hgvs p":                       "protein_change",
    "hgvs protein":                 "protein_change",
    # ── Genomic coordinates ───────────────────────────────────────
    "genomic change":               "genomic_change",
    "genomic position":             "genomic_change",
    "genomic variant":              "genomic_change",
    "genomic":                      "genomic_change",
    "chromosome":                   "chromosome",
    "chrom":                        "chromosome",
    "chr":                          "chromosome",
    "position":                     "position",
    "pos":                          "position",
    "start":                        "position",
    "ref":                          "ref",
    "reference allele":             "ref",
    "reference":                    "ref",
    "alt":                          "alt",
    "alternate allele":             "alt",
    "alternate":                    "alt",
    "alternative":                  "alt",
    # ── Zygosity / inheritance ────────────────────────────────────
    "zygosity":                     "zygosity",
    "zygosity proband":             "zygosity_proband",
    "genotype":                     "zygosity",
    "inheritance":                  "inheritance",
    "inheritance pattern":          "inheritance",
    "inheritance mode":             "inheritance",
    # ── Variant type ──────────────────────────────────────────────
    "variant type":                 "variant_type",
    "mutation type":                "variant_type",
    "type":                         "variant_type",
    # ── Clinical significance (longest matches first) ─────────────
    "clinical significance":        "clinical_significance",
    "clinical classification":      "clinical_significance",
    "clinical interpretation":      "clinical_significance",
    "acmg classification":          "clinical_significance",  # VUS/Pathogenic not criteria codes
    "classification":               "clinical_significance",
    "interpretation":               "clinical_significance",
    "significance":                 "clinical_significance",
    # ── ClinVar accession ─────────────────────────────────────────
    "clinvar accession id":         "clinvar_accession",
    "clinvar accession":            "clinvar_accession",
    "clinvar id":                   "clinvar_accession",
    "clinvar":                      "clinvar_accession",
    "vcv accession":                "clinvar_accession",
    "vcv":                          "clinvar_accession",
    # ── rsID ──────────────────────────────────────────────────────
    "dbsnp id":                     "rsid",
    "dbsnp rs":                     "rsid",
    "dbsnp":                        "rsid",
    "rs id":                        "rsid",
    "rsid":                         "rsid",
    "rs":                           "rsid",
    # ── Condition / phenotype ─────────────────────────────────────
    "associated condition":         "associated_condition",
    "condition":                    "associated_condition",
    "disease":                      "associated_condition",
    "disorder":                     "associated_condition",
    # ── ACMG criteria codes (not classification label) ────────────
    "acmg criteria":                "acmg_criteria",
    "acmg criterion":               "acmg_criteria",
    "acmg evidence":                "acmg_criteria",
    "acmg codes":                   "acmg_criteria",
    "acmg":                         "acmg_criteria",
    # ── HPO ───────────────────────────────────────────────────────
    "phenotype":                    "phenotypes_hpo",
    "hpo terms":                    "phenotypes_hpo",
    "hpo ids":                      "phenotypes_hpo",
    "hpo":                          "phenotypes_hpo",
}

# Sort by label length descending so more specific keys are tried first
_SORTED_LABELS: list[tuple[str, str]] = sorted(
    _LABEL_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Matches "Label: Value" — label starts with a letter, colon separator
_LABEL_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9 /().#_\-]{0,50}?)\s*:\s*(?P<value>.+)$"
)

# Matches explicit variant/finding block headers like "Variant 1:", "Finding 2"
_BLOCK_HEADER_RE = re.compile(
    r"^(?:variant|finding|result|gene finding)\s+\d+\s*:?\s*$",
    re.IGNORECASE,
)

# ACMG criteria codes: PP3, PM2, PS1, BA1, BP1, etc.
_ACMG_CODE_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}[a-z]?$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_label(label: str) -> str:
    """Lowercase, remove non-alphanumeric (except spaces), collapse whitespace."""
    s = re.sub(r"[^a-z0-9 ]", " ", label.lower())
    return re.sub(r"\s+", " ", s).strip()


def _parse_acmg_criteria(value: str) -> list[str]:
    """
    Parse an ACMG criteria string into a list of individual code strings.

    "PP3, PM2"  → ["PP3", "PM2"]
    "PP3/PM2"   → ["PP3", "PM2"]
    "PP3 PM2"   → ["PP3", "PM2"]
    If no valid codes are found, returns the whole string in a one-item list.
    """
    parts = re.split(r"[,;/\s]+", value.strip())
    codes = [p.strip() for p in parts if _ACMG_CODE_RE.match(p.strip())]
    return codes if codes else [value.strip()]


def _parse_hpo(value: str) -> list[str]:
    """
    Parse an HPO terms string into a list.

    "Hypoplastic right heart HP:0010954; Stroke HP:0001297"
    → ["Hypoplastic right heart HP:0010954", "Stroke HP:0001297"]
    """
    parts = re.split(r"[;|]+", value)
    return [p.strip() for p in parts if p.strip()]


def _split_into_blocks(text: str) -> list[str]:
    """
    Split a plain-text report into one or more variant blocks.

    Priority:
    1. Explicit block headers ("Variant 1:", "Finding 2", etc.) — most reliable.
    2. Two or more consecutive blank lines — block boundary.
    3. Single block fallback.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Strategy 1: explicit headers
    header_idxs = [
        i for i, line in enumerate(lines)
        if _BLOCK_HEADER_RE.match(line.strip())
    ]
    if header_idxs:
        blocks: list[str] = []
        for k, pos in enumerate(header_idxs):
            start = pos + 1
            end = header_idxs[k + 1] if k + 1 < len(header_idxs) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if block:
                blocks.append(block)
        return blocks or [text.strip()]

    # Strategy 2: two or more consecutive blank lines (≥ 3 newlines in a row)
    blocks = [b.strip() for b in re.split(r"\n[ \t]*\n[ \t]*\n", text)]
    blocks = [b for b in blocks if b]
    if len(blocks) > 1:
        return blocks

    # Strategy 3: single block
    return [text.strip()]


def _extract_fields_from_block(block: str) -> dict:
    """
    Extract label→value pairs from a single text block.

    Returns a dict with canonical field names (e.g. "gene", "cdna_variant",
    "clinical_significance") ready to pass to _extract_variant_fields.
    First match for each canonical key wins (earlier lines take priority).
    """
    result: dict = {}

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        m = _LABEL_RE.match(line)
        if not m:
            continue

        label_raw = m.group("label").strip()
        value_raw = m.group("value").strip()
        if not value_raw:
            continue

        normalised = _normalise_label(label_raw)

        # Find the first (longest) matching canonical key
        canonical: Optional[str] = None
        for known, canon in _SORTED_LABELS:
            if normalised == known:
                canonical = canon
                break

        if canonical is None:
            continue  # unrecognised label — skip

        if canonical in result:
            continue  # first match wins

        # Light cleanup for gene values (strip parenthetical notes)
        if canonical == "gene":
            value_raw = re.sub(r"\s*\([^)]*\)", "", value_raw).strip()

        result[canonical] = value_raw

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_report_text(text: str) -> dict:
    """
    Parse a plain-text clinical genetic report into the normalized variant format.

    Reuses _extract_variant_fields from report_parser so all existing
    normalization (zygosity, genomic_change parsing, assembly detection, etc.)
    is applied consistently.

    Returns:
        {
            "file_type":        "plain_text_report",
            "detected_columns": list of non-empty canonical field names,
            "variants":         list of normalized variant dicts,
            "report_metadata":  {},
            "warnings":         list of warning strings,
        }
    """
    # Import here to avoid module-level circular dependency
    from app.report_parser import _extract_variant_fields, _compute_detected_columns

    all_warnings: list[str] = []

    if not text or not text.strip():
        return {
            "file_type":        "plain_text_report",
            "detected_columns": [],
            "variants":         [],
            "report_metadata":  {},
            "warnings":         ["Input text is empty."],
        }

    blocks = _split_into_blocks(text)
    logger.info("report_text_parser: split into %d block(s)", len(blocks))

    variants_out: list[dict] = []
    multi = len(blocks) > 1

    for i, block in enumerate(blocks, 1):
        prefix = f"Variant {i}: " if multi else ""

        raw = _extract_fields_from_block(block)

        if not raw:
            all_warnings.append(
                f"{prefix}No recognised label:value pairs found in this block. "
                "Expected lines like 'Gene: BRCA1', 'cDNA: c.5266dup', "
                "'Clinical significance: Pathogenic'."
            )
            continue

        # Pre-process special fields before handing to _extract_variant_fields
        if "acmg_criteria" in raw:
            raw["acmg_criteria"] = _parse_acmg_criteria(raw["acmg_criteria"])

        if "phenotypes_hpo" in raw:
            raw["phenotypes_hpo"] = _parse_hpo(raw["phenotypes_hpo"])

        try:
            v_warnings: list[str] = []
            v_obj = _extract_variant_fields(raw, v_warnings)
            for w in v_warnings:
                all_warnings.append(f"{prefix}{w}")
            variants_out.append(v_obj)
        except Exception as exc:
            all_warnings.append(f"{prefix}Could not parse variant fields: {exc}")

    if not variants_out:
        all_warnings.append(
            "No variants could be extracted. "
            "Expected 'Label: Value' lines. Recognised labels include: "
            "Gene, cDNA (or Variant / HGVS), Protein, Genomic change, "
            "Clinical significance (or Classification / Interpretation), "
            "ClinVar accession (or VCV), rsID (or dbSNP), Zygosity, "
            "ACMG criteria, Condition."
        )

    detected = _compute_detected_columns(variants_out)

    return {
        "file_type":        "plain_text_report",
        "detected_columns": detected,
        "variants":         variants_out,
        "report_metadata":  {},
        "warnings":         all_warnings,
    }
