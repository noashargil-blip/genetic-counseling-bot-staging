"""
app/report_parser.py

Parse a structured clinical genetic report JSON into the normalized variant
format used by the upload/analyze flow.

Supported top-level shapes (after JS auto-wrapping via /analyze-report-json):
  {"variant": {...}}                      single variant
  {"variants": [{...}, {...}, ...]}       multiple variants
  {"findings": [{...}, {...}, ...]}       multiple variants (alias key)

The "report" wrapper produced by the UI client is unwrapped by the endpoint
before this function is called.

Output:
    {
        "file_type":       "structured_report_json",
        "detected_columns": [...],     # union of non-empty fields across variants
        "variants":        [...],      # list of normalized variant dicts
        "report_metadata": {...},      # provenance, predictors, family zyg, gene summary
        "warnings":        [...],
    }

Field aliases recognised (all map to canonical names):
  cdna_variant, hgvsc, hgvs_c, variant  → variant
  protein_change, hgvsp, hgvs_p         → protein_change
  clinvar_accession, vcv, variation_id  → clinvar_accession
  rsid, dbsnp_id, rs_id                 → rsid
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

REPORT_PARSER_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# Fields that belong on the variant object (identity + ClinVar-linkage)
# ---------------------------------------------------------------------------

_VARIANT_FIELDS = (
    "gene",
    "transcript",
    "variant",            # cDNA / HGVS c. notation
    "protein_change",
    "genomic_change",     # raw g. string, preserved verbatim
    "assembly",
    "chromosome",
    "position",
    "ref",
    "alt",
    "rsid",               # rsID / dbSNP identifier
    "zygosity",
    "inheritance",
    "variant_type",
    "clinical_significance",
    "clinvar_accession",
    "acmg_criteria",
    "phenotypes_hpo",     # list of {phenotype, hpo_id} or plain strings
    "associated_condition",
)

# Keys that identify a list of variants at the top level
_VARIANT_LIST_KEYS = ("variants", "findings")

# ---------------------------------------------------------------------------
# Fields that belong in report_metadata (annotations, provenance, family)
# ---------------------------------------------------------------------------

_METADATA_KEYS = (
    "source",
    "extraction_status",
    "notes",
    "gene_summary",
    "condition_sources",
    "clinical_evidence",
    "pmids",
    "maf_gnomad",
    "internal_frequency",
    "pli_score",
    "gene_coverage",
    "zygosity_mother",
    "zygosity_father",
    "prediction_revel",
    "prediction_mt",
    "prediction_sift",
    "prediction_polyphen2",
    "prediction_gerp",
    "prediction_aggregated",
    "prediction_revel_text",
    "prediction_alphamissense",
    "prediction_conservation_gerp",
    "prediction_spliceai",
)

# ---------------------------------------------------------------------------
# Zygosity normalisation
# ---------------------------------------------------------------------------

_ZYGOSITY_SHORT_MAP: dict[str, str] = {
    "het":              "heterozygous",
    "heterozygous":     "heterozygous",
    "hemi":             "hemizygous",
    "hemizygous":       "hemizygous",
    "hom":              "homozygous",
    "homozygous":       "homozygous",
    "wt":               "homozygous_ref",
    "homozygous_ref":   "homozygous_ref",
    "homref":           "homozygous_ref",
    "unknown":          "unknown",
}

# g.111130476G>T  or  g.111130476_111130477del  (only simple SNV form extracted)
_GENOMIC_CHANGE_RE = re.compile(
    r"(?:chr)?(\w+):g\.(\d+)([ACGTN]+)>([ACGTN]+)",
    re.IGNORECASE,
)

# Loose assembly detection in a string ("hg19", "hg38", "grch37", "grch38")
_ASSEMBLY_RE = re.compile(r"\b(hg(?:19|38)|grch(?:37|38))\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_first(d: dict, *keys: str) -> str:
    """
    Return the first non-empty string value found in d for any of the given keys.
    Supports field aliases — earlier keys take priority.
    Returns "" when no key is present or all values are empty.
    """
    for k in keys:
        v = d.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _normalise_zygosity(raw: str) -> str:
    """Normalise short clinical labels to our canonical zygosity strings."""
    return _ZYGOSITY_SHORT_MAP.get(raw.strip().lower(), raw.strip())


def _parse_genomic_change(gc: str) -> Optional[dict]:
    """
    Extract chromosome / position / ref / alt from a g. notation string
    such as "chr13:g.111130476G>T" or "13:g.111130476G>T".

    Returns a dict of the extracted values, or None if the pattern doesn't match
    (e.g. indel notation, missing fields).
    """
    m = _GENOMIC_CHANGE_RE.search(gc)
    if not m:
        return None
    chrom, pos, ref, alt = m.group(1), m.group(2), m.group(3), m.group(4)
    chrom = chrom.lstrip("chrCHR").lstrip("0") or chrom
    if chrom.isdigit():
        chrom = str(int(chrom))
    return {"chromosome": chrom, "position": pos, "ref": ref, "alt": alt}


def _detect_assembly(report: dict, variant: dict) -> str:
    """
    Best-effort assembly detection. Priority:
      1. variant["assembly"] field
      2. any "hg19/hg38/grch37/grch38" substring in variant["genomic_change"]
      3. top-level "assembly" key
    Returns empty string if not determinable.
    """
    asm = str(variant.get("assembly", "")).strip()
    if asm:
        return asm
    gc = str(variant.get("genomic_change", ""))
    m = _ASSEMBLY_RE.search(gc)
    if m:
        return m.group(1).lower()
    return str(report.get("assembly", "")).strip()


def _compute_detected_columns(variants_out: list[dict]) -> list[str]:
    """
    Return the ordered list of _VARIANT_FIELDS that have a non-empty value
    in at least one variant.  Preserves the canonical field ordering.
    """
    detected: list[str] = []
    for field in _VARIANT_FIELDS:
        for v in variants_out:
            val = v.get(field)
            if val:  # non-empty string or non-empty list
                detected.append(field)
                break
    return detected


# ---------------------------------------------------------------------------
# Core field extractor (single variant dict → normalized output)
# ---------------------------------------------------------------------------

def _extract_variant_fields(v: dict, warnings: list[str]) -> dict:
    """
    Map all fields from a variant sub-dict into the normalized variant object.

    Supports field aliases (earlier keys take priority):
      cdna_variant | hgvsc | hgvs_c | variant  → canonical "variant"
      protein_change | hgvsp | hgvs_p           → canonical "protein_change"
      clinvar_accession | vcv | variation_id    → canonical "clinvar_accession"
      rsid | dbsnp_id | rs_id                   → canonical "rsid"

    Explicit coordinate fields take priority over values parsed from genomic_change.
    Emits per-field warnings for missing critical identity fields.
    """
    out: dict = {}

    # ── Core identity ─────────────────────────────────────────────────────
    out["gene"]           = _get_first(v, "gene")
    out["transcript"]     = _get_first(v, "transcript")

    # cDNA variant — aliases: cdna_variant, hgvsc, hgvs_c, variant
    out["variant"]        = _get_first(v, "cdna_variant", "hgvsc", "hgvs_c", "variant")

    # Protein change — aliases: protein_change, hgvsp, hgvs_p
    out["protein_change"] = _get_first(v, "protein_change", "hgvsp", "hgvs_p")

    # rsID — aliases: rsid, dbsnp_id, rs_id
    out["rsid"]           = _get_first(v, "rsid", "dbsnp_id", "rs_id")

    # ── Genomic coordinates ───────────────────────────────────────────────
    out["genomic_change"] = _get_first(v, "genomic_change")
    out["chromosome"]     = _get_first(v, "chromosome")
    out["position"]       = _get_first(v, "position")
    out["ref"]            = _get_first(v, "ref")
    out["alt"]            = _get_first(v, "alt")

    # Fall back to parsing genomic_change when explicit coords are absent
    if out["genomic_change"] and not (out["chromosome"] and out["position"]):
        parsed = _parse_genomic_change(out["genomic_change"])
        if parsed:
            for k in ("chromosome", "position", "ref", "alt"):
                if not out[k]:
                    out[k] = parsed[k]
        else:
            warnings.append(
                f"genomic_change '{out['genomic_change']}' is present but could not be "
                "parsed into chromosome/position/ref/alt. Only simple SNV notation "
                "(e.g. chr13:g.111130476G>T) is currently supported."
            )

    # ── Assembly ──────────────────────────────────────────────────────────
    out["assembly"] = _detect_assembly({}, v)

    # ── Clinical / report-specific ────────────────────────────────────────
    raw_zyg = _get_first(v, "zygosity_proband", "zygosity")
    out["zygosity"] = _normalise_zygosity(raw_zyg) if raw_zyg else ""

    out["inheritance"]           = _get_first(v, "inheritance")
    out["variant_type"]          = _get_first(v, "variant_type")
    out["clinical_significance"] = _get_first(v, "clinical_significance")

    # ClinVar accession — aliases: clinvar_accession, vcv, variation_id
    out["clinvar_accession"] = _get_first(v, "clinvar_accession", "vcv", "variation_id")

    # ACMG criteria — preserve list or string
    raw_acmg = v.get("acmg_criteria", "")
    if isinstance(raw_acmg, list):
        out["acmg_criteria"] = [str(x).strip() for x in raw_acmg if x is not None]
    else:
        out["acmg_criteria"] = str(raw_acmg).strip() if raw_acmg else ""

    out["associated_condition"] = _get_first(v, "associated_condition")

    # ── Phenotypes ────────────────────────────────────────────────────────
    raw_hpo = v.get("phenotypes_hpo")
    if isinstance(raw_hpo, list):
        normalized_hpo: list = []
        for e in raw_hpo:
            if isinstance(e, dict):
                normalized_hpo.append({
                    "phenotype": str(e.get("phenotype", "")).strip(),
                    "hpo_id":    str(e.get("hpo_id",    "")).strip(),
                })
            elif isinstance(e, str) and e.strip():
                normalized_hpo.append(e.strip())
        out["phenotypes_hpo"] = normalized_hpo
    else:
        out["phenotypes_hpo"] = []

    # ── Warnings for missing critical fields ──────────────────────────────
    if not out["gene"]:
        warnings.append("Missing field: 'gene'. Variant may not match any ClinVar records.")
    if not out["chromosome"] and not out["variant"] and not out["rsid"] and not out["clinvar_accession"]:
        warnings.append(
            "No usable identifier found (chromosome/position, cDNA variant, rsid, or "
            "clinvar_accession). ClinVar matching will likely fail."
        )
    if not out["clinical_significance"]:
        warnings.append(
            "Field 'clinical_significance' is absent. "
            "The report may not include a lab classification."
        )
    acc = out["clinvar_accession"]
    if acc and not acc.upper().startswith(("VCV", "RCV")) and not acc.isdigit():
        warnings.append(
            f"clinvar_accession '{acc}' does not look like a standard "
            "VCV/RCV accession or numeric variation_id. Verify the value."
        )

    return out


# ---------------------------------------------------------------------------
# Report-level metadata extractor
# ---------------------------------------------------------------------------

def _extract_report_metadata(report: dict, variant: dict) -> dict:
    """
    Collect provenance, computational predictors, family zygosity, and gene
    summary into a flat metadata dict.  These fields do not drive ClinVar matching.
    """
    meta: dict = {}

    for key in ("source", "extraction_status", "notes", "gene_summary"):
        val = report.get(key)
        if val is not None:
            meta[key] = str(val).strip()

    for key in (
        "condition_sources", "clinical_evidence", "pmids",
        "maf_gnomad", "internal_frequency", "pli_score", "gene_coverage",
        "zygosity_mother", "zygosity_father",
        "prediction_revel", "prediction_mt", "prediction_sift",
        "prediction_polyphen2", "prediction_gerp", "prediction_aggregated",
        "prediction_revel_text", "prediction_alphamissense",
        "prediction_conservation_gerp", "prediction_spliceai",
    ):
        val = variant.get(key)
        if val is not None:
            meta[key] = str(val).strip()

    return meta


# ---------------------------------------------------------------------------
# Multi-variant list parser
# ---------------------------------------------------------------------------

def _parse_variant_list(
    items: list,
    outer_dict: dict,
    all_warnings: list[str],
) -> list[dict]:
    """
    Parse a list of raw variant dicts into normalized variant objects.

    Malformed items (non-dict or raised exception) are skipped and a warning
    is appended to all_warnings instead of aborting the whole parse.
    Per-variant warnings are prefixed with "Variant N: " for traceability.
    """
    variants_out: list[dict] = []

    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            all_warnings.append(
                f"Variant {i}: item is not a JSON object (got {type(item).__name__}); skipped."
            )
            continue

        # Inherit top-level phenotypes_hpo if not already on the item
        if "phenotypes_hpo" not in item and "phenotypes_hpo" in outer_dict:
            item = {**item, "phenotypes_hpo": outer_dict["phenotypes_hpo"]}

        try:
            v_warnings: list[str] = []
            v_obj = _extract_variant_fields(item, v_warnings)
            for w in v_warnings:
                all_warnings.append(f"Variant {i}: {w}")
            variants_out.append(v_obj)
        except Exception as exc:
            all_warnings.append(
                f"Variant {i}: could not be parsed ({exc}); skipped."
            )

    return variants_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_structured_report_json(report: dict) -> dict:
    """
    Parse a structured clinical genetic report dict into the normalized variant format.

    Supported shapes (after JS wrapping):
      {"variant": {...}}                 → single variant
      {"variants": [{...}, {...}]}       → multiple variants
      {"findings": [{...}, {...}]}       → multiple variants (alias key)

    The caller is responsible for JSON decoding.  Use parse_report_json_bytes
    for raw bytes.

    Returns:
        {
            "file_type":        "structured_report_json",
            "detected_columns": list of non-empty field names (union across variants),
            "variants":         list of normalized variant dicts,
            "report_metadata":  dict of provenance + annotation fields,
            "warnings":         list of warning strings,
        }
    """
    warnings: list[str] = []

    if not isinstance(report, dict):
        return {
            "file_type":        "structured_report_json",
            "detected_columns": [],
            "variants":         [],
            "report_metadata":  {},
            "warnings":         ["Input is not a JSON object. Expected a dict at the top level."],
        }

    # ── Detect format ────────────────────────────────────────────────────────

    # Priority 1: single-variant key "variant"
    if "variant" in report:
        v_raw = report["variant"]
        if not isinstance(v_raw, dict):
            warnings.append(
                "Top-level 'variant' key is not a JSON object. "
                "No variant fields could be extracted."
            )
            return {
                "file_type":        "structured_report_json",
                "detected_columns": [],
                "variants":         [],
                "report_metadata":  _extract_report_metadata(report, {}),
                "warnings":         warnings,
            }

        # Inject phenotypes_hpo from top level if not already on the variant
        if "phenotypes_hpo" not in v_raw and "phenotypes_hpo" in report:
            v_raw = {**v_raw, "phenotypes_hpo": report["phenotypes_hpo"]}

        variant_obj = _extract_variant_fields(v_raw, warnings)
        report_meta = _extract_report_metadata(report, v_raw)
        detected    = _compute_detected_columns([variant_obj])

        return {
            "file_type":        "structured_report_json",
            "detected_columns": detected,
            "variants":         [variant_obj],
            "report_metadata":  report_meta,
            "warnings":         warnings,
        }

    # Priority 2: multi-variant keys "variants" or "findings"
    for list_key in _VARIANT_LIST_KEYS:
        if list_key in report:
            raw_list = report[list_key]
            if not isinstance(raw_list, list):
                warnings.append(
                    f"Top-level '{list_key}' key is not a JSON array; "
                    "no variants could be extracted."
                )
                break
            if not raw_list:
                warnings.append(f"Top-level '{list_key}' array is empty.")
                break

            variants_out = _parse_variant_list(raw_list, report, warnings)
            report_meta  = _extract_report_metadata(report, {})
            detected     = _compute_detected_columns(variants_out)

            if not variants_out:
                warnings.append(
                    f"'{list_key}' array contained {len(raw_list)} item(s) but none "
                    "could be parsed as variant objects."
                )

            return {
                "file_type":        "structured_report_json",
                "detected_columns": detected,
                "variants":         variants_out,
                "report_metadata":  report_meta,
                "warnings":         warnings,
            }

    # ── Nothing recognised ───────────────────────────────────────────────────
    recognised = ("'variant'",) + tuple(f"'{k}'" for k in _VARIANT_LIST_KEYS)
    warnings.append(
        f"No recognised variant key found. Expected one of: {', '.join(recognised)}. "
        "No variants could be extracted."
    )
    return {
        "file_type":        "structured_report_json",
        "detected_columns": [],
        "variants":         [],
        "report_metadata":  _extract_report_metadata(report, {}),
        "warnings":         warnings,
    }


def parse_report_json_bytes(file_bytes: bytes) -> dict:
    """
    Decode raw bytes, parse JSON, then call parse_structured_report_json.
    Handles UnicodeDecodeError and json.JSONDecodeError gracefully.
    """
    _error_base = {
        "file_type":        "structured_report_json",
        "detected_columns": [],
        "variants":         [],
        "report_metadata":  {},
    }

    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return {**_error_base, "warnings": ["Could not decode file bytes as UTF-8 or latin-1."]}

    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        return {**_error_base, "warnings": [f"Invalid JSON: {exc}"]}

    return parse_structured_report_json(report)


# ---------------------------------------------------------------------------
# Self-test  (python -m app.report_parser)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib, pprint

    fixture = pathlib.Path(__file__).parent.parent / "examples" / "extracted_genetic_report.json"
    if not fixture.exists():
        print(f"Fixture not found: {fixture}")
        raise SystemExit(1)

    raw = fixture.read_bytes()
    result = parse_report_json_bytes(raw)

    print(f"\n=== file_type ===\n{result['file_type']}")
    print(f"\n=== detected_columns ({len(result['detected_columns'])}) ===")
    print(result["detected_columns"])
    print(f"\n=== warnings ({len(result['warnings'])}) ===")
    for w in result["warnings"]:
        print(" •", w)
    print(f"\n=== variants ({len(result['variants'])}) ===")
    for i, v in enumerate(result["variants"], 1):
        print(f"\n--- Variant {i} ---")
        pprint.pprint(v)
    print(f"\n=== report_metadata keys ===")
    print(list(result["report_metadata"].keys()))
