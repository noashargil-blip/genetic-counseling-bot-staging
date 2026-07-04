import duckdb
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/clinvar.duckdb")

# ---------------------------------------------------------------------------
# Schema compatibility layer
# ---------------------------------------------------------------------------
# The local dev build and the university server build use different column
# names in the `clinvar` table.  This block detects the actual schema once at
# module load and builds a canonical → actual-column-name mapping.
#
# Downstream code always receives canonical field names because every SELECT
# uses AS aliases.  WHERE / GROUP BY clauses call _c("canonical") to get the
# actual column name before embedding it in a query string.
#
# Known schema variants:
#   Local dev:          gene_symbol, clinical_significance, review_status,
#                       last_evaluated, dbsnp_id
#   University server:  genes,       germline_classification, germline_review_status,
#                       germline_last_eval,  allele_ids (or dbsnp_id)

_COL_CANDIDATES: dict[str, list[str]] = {
    "variation_id":          ["variation_id"],
    "gene_symbol":           ["gene_symbol", "genes"],
    "clinical_significance": ["clinical_significance", "germline_classification"],
    "review_status":         ["review_status", "germline_review_status"],
    "phenotype_list":        ["phenotype_list", "phenotypes", "conditions"],
    "variant_type":          ["variant_type", "type"],
    "chromosome":            ["chromosome", "chrom"],
    "start_pos":             ["start_pos", "start"],
    "stop_pos":              ["stop_pos", "stop"],
    "dbsnp_id":              ["dbsnp_id", "rs_ids", "allele_ids"],
    "last_evaluated":        ["last_evaluated", "germline_last_eval"],
}

_COL: dict[str, Optional[str]] = {}   # canonical → actual column name (None if missing)
_SELECT: str = ""                       # "    SELECT ... FROM clinvar\n" with AS aliases
_HGVS_SELECT: str = ""                 # SELECT for gene + clinvar_raw HGVS join

# Whether the ClinVar database was successfully opened at import time. The
# app must be able to start (and the counseling assistant must work fully)
# even when data/clinvar.duckdb is missing — this is a legacy data source
# used only by the not-exposed-in-the-UI ClinVar endpoints and by the
# specific-variant evidence-summary feature (which degrades gracefully to
# its "no evidence found" educational answer when this is False).
_DB_AVAILABLE: bool = False


def _c(canonical: str) -> str:
    """
    Return the actual column name for WHERE / GROUP BY usage.

    Raises RuntimeError when no matching column was found so callers can catch
    it and fall back to a different matching strategy rather than executing a
    broken query.
    """
    actual = _COL.get(canonical)
    if actual is None:
        raise RuntimeError(
            f"Column '{canonical}' has no mapping in the current clinvar schema "
            f"(mapping: {_COL})"
        )
    return actual


def _has_cols(*canonical_names: str) -> bool:
    """Return True only when every requested canonical column has a DB mapping."""
    return all(_COL.get(n) is not None for n in canonical_names)


def _missing_cols(*canonical_names: str) -> list[str]:
    """Return the subset of requested canonical names that have no DB mapping."""
    return [n for n in canonical_names if _COL.get(n) is None]


def _init_schema() -> None:
    """Detect actual column names once and build _COL, _SELECT, _HGVS_SELECT."""
    global _COL, _SELECT, _HGVS_SELECT

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='clinvar' AND table_schema='main'"
        ).fetchall()

        actual_cols = {row[0].lower() for row in rows}

        _COL = {}
        for canonical, candidates in _COL_CANDIDATES.items():
            for cand in candidates:
                if cand.lower() in actual_cols:
                    _COL[canonical] = cand
                    break
            else:
                _COL[canonical] = None

        missing = [k for k, v in _COL.items() if v is None]
        if missing:
            logger.warning("clinvar schema: canonical fields not found in DB: %s", missing)

        remapped = {k: v for k, v in _COL.items() if v and v != k}
        if remapped:
            logger.info("clinvar schema remapped fields: %s", remapped)
        else:
            logger.info("clinvar schema: all canonical column names match DB columns")

        # Build SELECT clause (all queries share this)
        parts = []
        for canonical in _COL_CANDIDATES:
            actual = _COL[canonical]
            if actual is None:
                parts.append(f"NULL AS {canonical}")
            elif actual == canonical:
                parts.append(actual)
            else:
                parts.append(f"{actual} AS {canonical}")
        cols_str = ",\n        ".join(parts)
        _SELECT = f"    SELECT\n        {cols_str}\n    FROM clinvar\n"

        # Build HGVS SELECT (JOINs clinvar_raw on variation_id)
        # Only built when clinvar_raw table is present; otherwise left empty so
        # match_uploaded_variant skips Strategy 2.5 gracefully.
        raw_tables = {
            row[0].lower()
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
    finally:
        con.close()

    if "clinvar_raw" in raw_tables and _COL.get("variation_id") and _COL.get("gene_symbol"):
        hgvs_parts = []
        for canonical in _COL_CANDIDATES:
            actual = _COL[canonical]
            if actual is None:
                hgvs_parts.append(f"NULL AS {canonical}")
            elif actual == canonical:
                hgvs_parts.append(f"c.{actual}")
            else:
                hgvs_parts.append(f"c.{actual} AS {canonical}")
        hgvs_cols = ",\n        ".join(hgvs_parts)
        vid_col  = _COL["variation_id"]
        gsym_col = _COL["gene_symbol"]
        _HGVS_SELECT = (
            f"    SELECT DISTINCT\n        {hgvs_cols}\n"
            f"    FROM clinvar c\n"
            f"    JOIN clinvar_raw r ON c.{vid_col} = r.VariationID\n"
            f"    WHERE c.{gsym_col} = ?\n"
            f"      AND r.Name LIKE ?\n"
            f"    LIMIT ?\n"
        )
        logger.info("clinvar_raw table found — HGVS partial matching enabled")
    else:
        _HGVS_SELECT = ""
        logger.info("clinvar_raw table absent or variation_id/gene_symbol missing — HGVS partial matching disabled")


try:
    _init_schema()
    _DB_AVAILABLE = True
except Exception as exc:
    logger.warning(
        "ClinVar database unavailable at %s (%s). The app will still start; "
        "legacy ClinVar endpoints and the variant-evidence-summary feature "
        "will report 'no evidence found' instead of crashing.",
        DB_PATH, exc,
    )
    _DB_AVAILABLE = False
    _COL = {}
    _SELECT = ""
    _HGVS_SELECT = ""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def _con():
    if not _DB_AVAILABLE:
        raise RuntimeError("ClinVar database is not available on this server.")
    return duckdb.connect(str(DB_PATH), read_only=True)


# ---------------------------------------------------------------------------
# Legacy module-level functions (kept for backward compatibility)
# ---------------------------------------------------------------------------

def get_variants_by_gene(gene: str, limit: int = 20, clin_sig: Optional[str] = None) -> List[Dict]:
    gene = gene.strip().upper()
    query = _SELECT + f"WHERE {_c('gene_symbol')} = ?\n"
    params: list = [gene]

    if clin_sig:
        query += f"  AND {_c('clinical_significance')} LIKE ?\n"
        params.append(f"%{clin_sig}%")

    query += "LIMIT ?"
    params.append(int(limit))

    con = _con()
    try:
        return con.execute(query, params).fetchdf().to_dict(orient="records")
    finally:
        con.close()


def search_variants(text: str, limit: int = 20) -> List[Dict]:
    text = text.strip()
    like = f"%{text}%"
    query = (
        _SELECT +
        f"WHERE {_c('phenotype_list')} LIKE ?\n"
        f"   OR {_c('clinical_significance')} LIKE ?\n"
        f"LIMIT ?"
    )
    con = _con()
    try:
        return con.execute(query, [like, like, int(limit)]).fetchdf().to_dict(orient="records")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# ClinVarRetriever class (backward-compatible)
# ---------------------------------------------------------------------------

class ClinVarRetriever:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path

    def _con(self):
        return duckdb.connect(self.db_path, read_only=True)

    def by_gene(self, gene: str, limit: int = 20, clin_sig: str | None = None):
        gene = gene.strip().upper()
        query = _SELECT + f"WHERE {_c('gene_symbol')} = ?\n"
        params: list = [gene]

        if clin_sig:
            query += f"  AND {_c('clinical_significance')} LIKE ?\n"
            params.append(f"%{clin_sig}%")

        query += "LIMIT ?"
        params.append(int(limit))

        con = self._con()
        try:
            return con.execute(query, params).fetchdf().to_dict(orient="records")
        finally:
            con.close()

    def search(self, text: str, limit: int = 20):
        text = text.strip()
        like = f"%{text}%"
        query = (
            _SELECT +
            f"WHERE {_c('phenotype_list')} LIKE ?\n"
            f"   OR {_c('clinical_significance')} LIKE ?\n"
            f"LIMIT ?"
        )
        con = self._con()
        try:
            return con.execute(query, [like, like, int(limit)]).fetchdf().to_dict(orient="records")
        finally:
            con.close()

    # Backward-compatible aliases
    def search_by_gene(self, gene: str, limit: int = 20):
        return self.by_gene(gene=gene, limit=limit)

    def search_by_text(self, text: str, limit: int = 20):
        return self.search(text=text, limit=limit)

    def search_text(self, text: str, limit: int = 20):
        return self.search(text=text, limit=limit)


# ---------------------------------------------------------------------------
# Module-level API expected by main.py
# ---------------------------------------------------------------------------

_retriever = ClinVarRetriever()


def retrieve_by_gene(gene: str, limit: int = 20) -> List[Dict]:
    return _retriever.by_gene(gene, limit=limit)


def retrieve_by_variant(variant: str, limit: int = 20) -> List[Dict]:
    variant = variant.strip()
    like = f"%{variant}%"
    con = _con()
    try:
        df = con.execute(
            _SELECT +
            f"WHERE {_c('dbsnp_id')} = ?\n"
            f"   OR CAST({_c('variation_id')} AS VARCHAR) = ?\n"
            f"   OR {_c('phenotype_list')} LIKE ?\n"
            f"LIMIT ?",
            [variant, variant, like, int(limit)],
        ).fetchdf()
        return df.to_dict(orient="records")
    finally:
        con.close()


def retrieve_by_condition(condition: str, limit: int = 20) -> List[Dict]:
    like = f"%{condition.strip()}%"
    con = _con()
    try:
        df = con.execute(
            _SELECT + f"WHERE {_c('phenotype_list')} LIKE ? LIMIT ?",
            [like, int(limit)],
        ).fetchdf()
        return df.to_dict(orient="records")
    finally:
        con.close()


def search(
    q: str,
    gene: Optional[str] = None,
    significance: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    like = f"%{q.strip()}%"
    params: list = [like, like]
    query = (
        _SELECT +
        f"WHERE ({_c('phenotype_list')} LIKE ? OR {_c('clinical_significance')} LIKE ?)"
    )
    if gene:
        query += f" AND {_c('gene_symbol')} = ?"
        params.append(gene.strip().upper())
    if significance:
        query += f" AND {_c('clinical_significance')} LIKE ?"
        params.append(f"%{significance}%")
    query += " LIMIT ?"
    params.append(int(limit))
    con = _con()
    try:
        return con.execute(query, params).fetchdf().to_dict(orient="records")
    finally:
        con.close()


def get_summary(gene: str) -> Dict:
    gene = gene.strip().upper()
    clinsig_col = _c("clinical_significance")
    gsym_col    = _c("gene_symbol")
    con = _con()
    try:
        df = con.execute(
            f"""
            SELECT {clinsig_col} AS clinical_significance, COUNT(*) AS count
            FROM clinvar
            WHERE {gsym_col} = ?
            GROUP BY {clinsig_col}
            ORDER BY count DESC
            """,
            [gene],
        ).fetchdf()
        if df.empty:
            return {"total": 0}
        by_sig = {
            (str(row["clinical_significance"]) if row["clinical_significance"] else "Unknown"): int(row["count"])
            for _, row in df.iterrows()
        }
        return {"total": sum(by_sig.values()), "by_significance": by_sig}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Uploaded-variant matching
# ---------------------------------------------------------------------------

_RS_RE  = re.compile(r"^rs(\d+)$", re.IGNORECASE)
_VCV_RE = re.compile(r"^(?:VCV)?0*(\d+)$", re.IGNORECASE)


def _parse_clinvar_accession(raw: str) -> Optional[int]:
    """
    Parse a ClinVar VCV accession or bare numeric string into a variation_id integer.

    Accepts:  "VCV2858375", "VCV000002858375", "2858375"
    Rejects:  RCV accessions, empty strings, zero → returns None
    """
    raw = raw.strip()
    m = _VCV_RE.match(raw)
    if m:
        v = int(m.group(1))
        return v if v > 0 else None
    return None


def _parse_rsid(raw: str) -> Optional[int]:
    """
    Parse an rsID string into a positive integer.

    Accepts:  "rs80358538", "RS80358538", "80358538"
    Rejects:  empty, non-numeric, zero or negative → returns None
    """
    raw = raw.strip()
    m = _RS_RE.match(raw)
    if m:
        return int(m.group(1))
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _normalise_chrom(raw: str) -> str:
    """Strip leading 'chr'/'Chr' prefix; preserve X, Y, MT as-is."""
    return re.sub(r"^chr", "", raw.strip(), flags=re.IGNORECASE).upper()


def _check_gene_consistency(
    uploaded_gene: str, matches: list[dict]
) -> tuple[str, list[str]]:
    """
    Compare the uploaded gene name against gene_symbol values in ClinVar matches.

    Returns (consistency_label, new_warnings).
      "match"       — all match rows carry the same gene as uploaded
      "mixed"       — uploaded gene is among matches but other genes also present
      "mismatch"    — uploaded gene is absent from all match rows
      "not_checked" — uploaded gene was empty or there were no matches to compare
    """
    if not uploaded_gene or not matches:
        return "not_checked", []

    uploaded_upper = uploaded_gene.strip().upper()
    matched_upper = {
        str(m.get("gene_symbol", "")).strip().upper()
        for m in matches
        if m.get("gene_symbol")
    }

    if not matched_upper:
        return "not_checked", []

    if uploaded_upper in matched_upper:
        if len(matched_upper) == 1:
            return "match", []
        return "mixed", [
            "ClinVar returned matches from multiple genes; review gene consistency."
        ]

    matched_display = ", ".join(sorted({
        str(m.get("gene_symbol", "")).strip()
        for m in matches
        if m.get("gene_symbol")
    }))
    return "mismatch", [
        f"Uploaded gene '{uploaded_gene.strip()}' does not match ClinVar matched "
        f"gene(s): {matched_display}. "
        "The rsID may correspond to a different gene than the uploaded row."
    ]


def _normalise_hgvs_token(raw: str) -> Optional[str]:
    """
    Derive a safe LIKE-search token from an HGVS-like variant string.

    c.68_69delAG  → 'c.68_69del'   strips trailing allele seq after del/ins/dup
    c.5946delT    → 'c.5946del'
    c.2836G>C     → 'c.2836G>C'    no stripping needed, used as-is
    p.Glu23Valfs  → 'p.Glu23'      3-letter AA + position number; change suffix dropped
    p.Arg175His   → 'p.Arg175'
    Returns None when no safe search token can be derived (too short or unrecognized).
    """
    raw = raw.strip()
    if not raw:
        return None

    # c. or n. notation — strip trailing allele sequence from del/ins/dup suffix
    if re.match(r"^[cn]\.", raw, re.IGNORECASE):
        token = re.sub(r"((?:del|ins|dup))[ACGTN]+$", r"\1", raw, flags=re.IGNORECASE)
        return token if len(token) >= 5 else None

    # p. notation — keep only the 3-letter AA code + position number
    p_match = re.match(r"^(p\.[A-Z][a-z]{2}\d+)", raw)
    if p_match:
        return p_match.group(1)

    return None


def match_uploaded_variant(variant_record: dict, limit: int = 10) -> dict:
    """
    Match a single normalized variant record against ClinVar.

    Match priority (first hit wins):
      0. ClinVar accession → exact variation_id match           (exact_clinvar_accession)
      1. rsID              → exact dbsnp_id integer match       (exact_rsid)
      2a. chr + exact pos  → start_pos = pos AND stop_pos = pos (position_exact)
         Preferred for SNVs; avoids returning large CNV records that merely
         overlap the coordinate.
      2b. chr + range      → start_pos <= pos AND stop_pos >= pos — only if 2a fails.
         Returns large structural/CNV matches; always adds a region-overlap warning.
         (region_overlap)
      3. gene + HGVS       → gene-scoped JOIN on clinvar_raw.Name LIKE token
         (gene_hgvs_partial)
      4. gene symbol only  → gene-level fallback                (gene_only)
      5. nothing usable    → no_match

    match_confidence values:
        "exact_clinvar_accession" | "exact_rsid" | "position_exact" | "region_overlap" |
        "gene_hgvs_partial" | "gene_only" | "no_match"
    """
    if not _DB_AVAILABLE:
        return {
            "query_used": "unavailable",
            "match_confidence": "no_match",
            "gene_consistency": "not_checked",
            "matches": [],
            "warnings": ["ClinVar database is not available on this server."],
        }

    warnings: list[str] = []
    uploaded_gene = str(variant_record.get("gene", "")).strip()

    # --- Strategy 0: ClinVar accession (VCV / bare numeric) ---
    raw_accession = str(variant_record.get("clinvar_accession", "")).strip()
    if raw_accession:
        vcv_id = _parse_clinvar_accession(raw_accession)
        if vcv_id:
            con = _con()
            try:
                df = con.execute(
                    _SELECT + f"WHERE {_c('variation_id')} = ? LIMIT ?",
                    [vcv_id, int(limit)],
                ).fetchdf()
                matches = df.to_dict(orient="records")
            finally:
                con.close()
            if matches:
                consistency, cons_warnings = _check_gene_consistency(uploaded_gene, matches)
                warnings.extend(cons_warnings)
                return {
                    "query_used": f"variation_id = {vcv_id}",
                    "match_confidence": "exact_clinvar_accession",
                    "gene_consistency": consistency,
                    "matches": matches,
                    "warnings": warnings,
                }
            warnings.append(
                f"ClinVar accession '{raw_accession}' (variation_id={vcv_id}) was not found "
                "in the local ClinVar database snapshot. "
                "The database may predate this record."
            )
        else:
            warnings.append(
                f"ClinVar accession '{raw_accession}' is not a recognized VCV/numeric ID "
                "and was skipped. RCV accessions are not currently supported."
            )

    # --- Strategy 1: rsID ---
    raw_rsid = str(variant_record.get("rsid", "")).strip()
    if raw_rsid and not _has_cols("dbsnp_id"):
        warnings.append("rsID matching skipped: dbsnp_id column not present in this ClinVar schema.")
        raw_rsid = ""
    if raw_rsid:
        rsid_int = _parse_rsid(raw_rsid)
        if rsid_int:
            con = _con()
            try:
                df = con.execute(
                    _SELECT + f"WHERE {_c('dbsnp_id')} = ? AND {_c('dbsnp_id')} > 0 LIMIT ?",
                    [rsid_int, int(limit)],
                ).fetchdf()
                matches = df.to_dict(orient="records")
            finally:
                con.close()
            if matches:
                consistency, cons_warnings = _check_gene_consistency(uploaded_gene, matches)
                warnings.extend(cons_warnings)
                return {
                    "query_used": f"dbsnp_id = {rsid_int}",
                    "match_confidence": "exact_rsid",
                    "gene_consistency": consistency,
                    "matches": matches,
                    "warnings": warnings,
                }
            warnings.append(f"rsID rs{rsid_int} was not found in ClinVar.")
        else:
            warnings.append(f"Could not parse rsID value: '{raw_rsid}'.")

    # --- Strategy 2a: Exact coordinate (point match) ---
    _COORD_COLS = ("chromosome", "start_pos", "stop_pos")
    _coord_missing = _missing_cols(*_COORD_COLS)
    if _coord_missing:
        warnings.append(
            "Coordinate matching skipped because this ClinVar database schema does not "
            f"include chromosome/start_pos/stop_pos columns (missing: {_coord_missing})."
        )

    raw_chrom = str(variant_record.get("chromosome", "")).strip()
    raw_pos   = str(variant_record.get("position", "")).strip()
    chrom: Optional[str] = None
    pos: Optional[int] = None

    if not _coord_missing and raw_chrom and raw_pos:
        chrom = _normalise_chrom(raw_chrom)
        try:
            pos = int(raw_pos)
        except ValueError:
            warnings.append(f"Could not parse position value: '{raw_pos}'.")

    if chrom and pos is not None:
        chrom_col = _c("chromosome")
        sp_col    = _c("start_pos")
        ep_col    = _c("stop_pos")

        con = _con()
        try:
            df = con.execute(
                _SELECT +
                f"WHERE {chrom_col} = ? AND {sp_col} = ? AND {ep_col} = ? LIMIT ?",
                [chrom, pos, pos, int(limit)],
            ).fetchdf()
            matches = df.to_dict(orient="records")
        finally:
            con.close()
        if matches:
            consistency, cons_warnings = _check_gene_consistency(uploaded_gene, matches)
            warnings.extend(cons_warnings)
            return {
                "query_used": (
                    f"chromosome = '{chrom}' AND start_pos = {pos} AND stop_pos = {pos}"
                ),
                "match_confidence": "position_exact",
                "gene_consistency": consistency,
                "matches": matches,
                "warnings": warnings,
            }
        warnings.append(
            f"No exact-position ClinVar records found at {chrom}:{pos}. "
            "Trying broader regional overlap."
        )

        # --- Strategy 2b: Region overlap (demoted — CNVs / structural variants) ---
        con = _con()
        try:
            df = con.execute(
                _SELECT +
                f"WHERE {chrom_col} = ? AND {sp_col} <= ? AND {ep_col} >= ? LIMIT ?",
                [chrom, pos, pos, int(limit)],
            ).fetchdf()
            matches = df.to_dict(orient="records")
        finally:
            con.close()
        if matches:
            consistency, cons_warnings = _check_gene_consistency(uploaded_gene, matches)
            warnings.extend(cons_warnings)
            warnings.append(
                "Only broad regional/CNV overlaps were found at this coordinate; "
                "these records likely represent large structural variants, not the same "
                "nucleotide-level variant. "
                "Provide a ClinVar accession, rsID, or gene+HGVS notation for a precise match."
            )
            return {
                "query_used": (
                    f"chromosome = '{chrom}' AND start_pos <= {pos} AND stop_pos >= {pos} "
                    "(region overlap)"
                ),
                "match_confidence": "region_overlap",
                "gene_consistency": consistency,
                "matches": matches,
                "warnings": warnings,
            }
        warnings.append(f"No ClinVar records (exact or regional) found at {chrom}:{pos}.")

    # --- Strategy 2.5: gene + HGVS partial (c./n./p. notation) ---
    # Requires clinvar_raw table (detected at init time) and variation_id column.
    raw_variant = str(variant_record.get("variant", "")).strip()
    raw_protein = str(variant_record.get("protein_change", "")).strip()
    _hgvs_available = _has_cols("variation_id", "gene_symbol") and bool(_HGVS_SELECT)
    if uploaded_gene and (raw_variant or raw_protein) and _hgvs_available:
        gene_upper = uploaded_gene.upper()
        tokens_tried: list[str] = []
        for src in (raw_variant, raw_protein):
            tok = _normalise_hgvs_token(src)
            if tok and tok not in tokens_tried:
                tokens_tried.append(tok)

        for token in tokens_tried:
            like_pattern = f"%{token}%"
            con = _con()
            try:
                df = con.execute(
                    _HGVS_SELECT, [gene_upper, like_pattern, int(limit)]
                ).fetchdf()
                matches = df.to_dict(orient="records")
            finally:
                con.close()
            if matches:
                consistency, cons_warnings = _check_gene_consistency(uploaded_gene, matches)
                warnings.extend(cons_warnings)
                warnings.append(
                    f"Matched by gene ({uploaded_gene.strip()}) + partial HGVS notation "
                    f"('{token}'). Verify the exact variant against the ClinVar record."
                )
                return {
                    "query_used": (
                        f"gene_symbol = '{gene_upper}' AND "
                        f"clinvar_raw.Name LIKE '{like_pattern}'"
                    ),
                    "match_confidence": "gene_hgvs_partial",
                    "gene_consistency": consistency,
                    "matches": matches,
                    "warnings": warnings,
                }

    # --- Strategy 3: gene symbol ---
    raw_gene = str(variant_record.get("gene", "")).strip()
    if raw_gene:
        gene_upper = raw_gene.upper()
        con = _con()
        try:
            df = con.execute(
                _SELECT + f"WHERE {_c('gene_symbol')} = ? LIMIT ?",
                [gene_upper, int(limit)],
            ).fetchdf()
            matches = df.to_dict(orient="records")
        finally:
            con.close()
        warnings.append(
            "Only gene-level matching was possible; "
            "this does not identify a specific variant."
        )
        if not matches:
            warnings.append(f"Gene '{gene_upper}' was not found in ClinVar.")
        consistency, cons_warnings = _check_gene_consistency(uploaded_gene, matches)
        warnings.extend(cons_warnings)
        return {
            "query_used": f"gene_symbol = '{gene_upper}'",
            "match_confidence": "gene_only",
            "gene_consistency": consistency,
            "matches": matches,
            "warnings": warnings,
        }

    # --- Strategy 4: nothing usable ---
    return {
        "query_used": "none",
        "match_confidence": "no_match",
        "gene_consistency": "not_checked",
        "matches": [],
        "warnings": warnings + [
            "Could not match this variant: no ClinVar accession, rsID, "
            "chromosome+position, or gene symbol was provided."
        ],
    }
