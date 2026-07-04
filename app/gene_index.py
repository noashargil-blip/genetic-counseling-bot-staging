"""
Gene-level ClinVar index.

Derives a `clinvar_gene_stats` table from the main ClinVar DuckDB snapshot
and persists it in a separate file (data/clinvar_gene_stats.duckdb) so that
gene-level summaries can be served without repeatedly scanning the full
variant table.

The index is built once at module import time (idempotent — skips rebuild
when the table already exists).  Each query opens and closes its own DuckDB
connection; no connection is held between calls.

Public API
----------
    list_genes(limit, offset)                       → list of summary dicts
    count_genes()                                   → int
    get_gene_summary(gene)                          → stats dict or None
    get_gene_variants(gene, limit, offset, sig)     → list of variant dicts
    _GENE_INDEX_AVAILABLE                           → bool
    METADATA                                        → safety disclaimer dict
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

STATS_DB_PATH = Path("data/clinvar_gene_stats.duckdb")

# Safety disclaimer attached to every gene endpoint response
_DISCLAIMER = (
    "This information is for educational purposes only and does not constitute "
    "medical advice. ClinVar statistics reflect aggregated public database "
    "submissions and may not be current or complete. Consult a certified genetics "
    "professional or genetic counselor for clinical interpretation of any genetic finding."
)
_DATA_NOTE = (
    "Source: ClinVar (NCBI). Data is aggregated at index-build time; "
    "variant counts may not reflect the most recent ClinVar release."
)
METADATA: Dict = {
    "source": "ClinVar",
    "disclaimer": _DISCLAIMER,
    "data_note": _DATA_NOTE,
}

_EXCLUDED_PHENOS = frozenset({"not provided", "not specified", "not applicable", ""})
_MAX_PHENOTYPES = 20   # top-N phenotypes stored per gene

# Module-level availability flag — True once the stats table is confirmed usable
_GENE_INDEX_AVAILABLE: bool = False


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _stats_con_rw():
    """Writable connection to the stats DB (used only during build)."""
    STATS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(STATS_DB_PATH))


def _stats_con():
    """Read-only connection to the stats DB (used for queries)."""
    if not _GENE_INDEX_AVAILABLE:
        raise RuntimeError("Gene index is not available.")
    return duckdb.connect(str(STATS_DB_PATH), read_only=True)


def _stats_table_exists() -> bool:
    con = _stats_con_rw()
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main'"
            ).fetchall()
        }
        return "clinvar_gene_stats" in tables
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_stats_table() -> None:
    """
    Aggregate the main ClinVar table into clinvar_gene_stats.

    Reads all rows from the main ClinVar DuckDB (read-only), performs
    per-gene aggregations in DuckDB + Python, then writes the result to
    the stats DuckDB (writable, separate file).  Idempotent — exits early
    when the table already exists.

    Stored fields per gene
    ~~~~~~~~~~~~~~~~~~~~~~
    gene_symbol       VARCHAR PRIMARY KEY
    total_variants    INTEGER
    by_significance   VARCHAR  JSON: {"Pathogenic": N, ...}
    by_review_status  VARCHAR  JSON: {"criteria provided, ...": N, ...}
    phenotypes        VARCHAR  JSON: ["Breast cancer", ...]  top-20 by frequency
    variant_types     VARCHAR  JSON: {"single nucleotide variant": N, ...}
    date_earliest     VARCHAR  earliest last_evaluated across all rows
    date_latest       VARCHAR  latest  last_evaluated across all rows
    built_at          VARCHAR  ISO-8601 UTC timestamp of when this row was built
    """
    from app.retriever import _COL, _c, DB_PATH as CLINVAR_DB_PATH  # noqa: PLC0415

    if _stats_table_exists():
        logger.info("clinvar_gene_stats already exists — skipping rebuild")
        return

    t0 = time.monotonic()
    logger.info("Building clinvar_gene_stats from %s …", CLINVAR_DB_PATH)

    gsym_col    = _c("gene_symbol")
    clinsig_col = _c("clinical_significance")
    review_col  = _c("review_status")
    pheno_col   = _COL.get("phenotype_list")   # may be None
    vtype_col   = _COL.get("variant_type")      # may be None
    date_col    = _COL.get("last_evaluated")    # may be None

    # ------------------------------------------------------------------ #
    # Phase 1 — read aggregates from main ClinVar DB (read-only)
    # ------------------------------------------------------------------ #
    ccon = duckdb.connect(str(CLINVAR_DB_PATH), read_only=True)
    try:
        # Totals + optional date range
        if date_col:
            date_sql = (
                f", CAST(MIN(CAST({date_col} AS VARCHAR)) AS VARCHAR) AS date_earliest"
                f", CAST(MAX(CAST({date_col} AS VARCHAR)) AS VARCHAR) AS date_latest"
            )
        else:
            date_sql = ", NULL AS date_earliest, NULL AS date_latest"

        totals_df = ccon.execute(f"""
            SELECT
                CAST({gsym_col} AS VARCHAR)  AS gene_symbol,
                COUNT(*)                      AS total_variants
                {date_sql}
            FROM clinvar
            WHERE {gsym_col} IS NOT NULL
              AND CAST({gsym_col} AS VARCHAR) != ''
            GROUP BY {gsym_col}
            ORDER BY total_variants DESC
        """).fetchdf()

        # By clinical significance
        sig_df = ccon.execute(f"""
            SELECT
                CAST({gsym_col}    AS VARCHAR) AS gene_symbol,
                COALESCE(CAST({clinsig_col} AS VARCHAR), 'Unknown') AS significance,
                COUNT(*) AS cnt
            FROM clinvar
            WHERE {gsym_col} IS NOT NULL
              AND CAST({gsym_col} AS VARCHAR) != ''
            GROUP BY {gsym_col}, {clinsig_col}
        """).fetchdf()

        # By review status
        review_df = ccon.execute(f"""
            SELECT
                CAST({gsym_col}  AS VARCHAR) AS gene_symbol,
                COALESCE(CAST({review_col} AS VARCHAR), 'Unknown') AS review_status,
                COUNT(*) AS cnt
            FROM clinvar
            WHERE {gsym_col} IS NOT NULL
              AND CAST({gsym_col} AS VARCHAR) != ''
            GROUP BY {gsym_col}, {review_col}
        """).fetchdf()

        # Variant types (optional column)
        if vtype_col:
            vtype_df = ccon.execute(f"""
                SELECT
                    CAST({gsym_col}  AS VARCHAR) AS gene_symbol,
                    CAST({vtype_col} AS VARCHAR) AS variant_type,
                    COUNT(*) AS cnt
                FROM clinvar
                WHERE {gsym_col} IS NOT NULL
                  AND CAST({gsym_col} AS VARCHAR) != ''
                  AND {vtype_col} IS NOT NULL
                  AND CAST({vtype_col} AS VARCHAR) != ''
                GROUP BY {gsym_col}, {vtype_col}
            """).fetchdf()
        else:
            vtype_df = None

        # Phenotype lists (pipe-separated values, optional column)
        if pheno_col:
            pheno_raw_df = ccon.execute(f"""
                SELECT
                    CAST({gsym_col}  AS VARCHAR) AS gene_symbol,
                    CAST({pheno_col} AS VARCHAR) AS phenotype_list
                FROM clinvar
                WHERE {gsym_col} IS NOT NULL
                  AND CAST({gsym_col} AS VARCHAR) != ''
                  AND {pheno_col} IS NOT NULL
                  AND CAST({pheno_col} AS VARCHAR) NOT IN ('', 'not provided')
            """).fetchdf()
        else:
            pheno_raw_df = None

    finally:
        ccon.close()

    # ------------------------------------------------------------------ #
    # Phase 2 — Python-side aggregation
    # ------------------------------------------------------------------ #

    gene_sig: Dict[str, Dict[str, int]] = defaultdict(dict)
    for _, row in sig_df.iterrows():
        gene_sig[str(row["gene_symbol"])][str(row["significance"])] = int(row["cnt"])

    gene_review: Dict[str, Dict[str, int]] = defaultdict(dict)
    for _, row in review_df.iterrows():
        gene_review[str(row["gene_symbol"])][str(row["review_status"])] = int(row["cnt"])

    gene_vtypes: Dict[str, Dict[str, int]] = defaultdict(dict)
    if vtype_df is not None:
        for _, row in vtype_df.iterrows():
            gene_vtypes[str(row["gene_symbol"])][str(row["variant_type"])] = int(row["cnt"])

    # Split pipe-separated phenotype lists and count per gene
    gene_pheno_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if pheno_raw_df is not None:
        for _, row in pheno_raw_df.iterrows():
            g = str(row["gene_symbol"])
            for p in str(row["phenotype_list"]).split("|"):
                p = p.strip()
                if p.lower() not in _EXCLUDED_PHENOS:
                    gene_pheno_counts[g][p] += 1

    gene_phenos_top: Dict[str, List[str]] = {
        g: [p for p, _ in sorted(cnt.items(), key=lambda x: -x[1])[:_MAX_PHENOTYPES]]
        for g, cnt in gene_pheno_counts.items()
    }

    # ------------------------------------------------------------------ #
    # Phase 3 — write to stats DB
    # ------------------------------------------------------------------ #
    import pandas as pd  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415

    now_str = datetime.utcnow().isoformat()
    rows = []
    for _, row in totals_df.iterrows():
        g = str(row["gene_symbol"])
        rows.append({
            "gene_symbol":     g,
            "total_variants":  int(row["total_variants"]),
            "by_significance": json.dumps(gene_sig.get(g, {}),     ensure_ascii=False),
            "by_review_status": json.dumps(gene_review.get(g, {}), ensure_ascii=False),
            "phenotypes":      json.dumps(gene_phenos_top.get(g, []), ensure_ascii=False),
            "variant_types":   json.dumps(gene_vtypes.get(g, {}),  ensure_ascii=False),
            "date_earliest":   str(row.get("date_earliest") or ""),
            "date_latest":     str(row.get("date_latest") or ""),
            "built_at":        now_str,
        })

    final_df = pd.DataFrame(rows, columns=[
        "gene_symbol", "total_variants", "by_significance", "by_review_status",
        "phenotypes", "variant_types", "date_earliest", "date_latest", "built_at",
    ])

    scon = _stats_con_rw()
    try:
        scon.register("_build_tmp", final_df)
        scon.execute("""
            CREATE TABLE clinvar_gene_stats (
                gene_symbol      VARCHAR PRIMARY KEY,
                total_variants   INTEGER  NOT NULL,
                by_significance  VARCHAR  NOT NULL DEFAULT '{}',
                by_review_status VARCHAR  NOT NULL DEFAULT '{}',
                phenotypes       VARCHAR  NOT NULL DEFAULT '[]',
                variant_types    VARCHAR  NOT NULL DEFAULT '{}',
                date_earliest    VARCHAR,
                date_latest      VARCHAR,
                built_at         VARCHAR  NOT NULL
            )
        """)
        scon.execute("INSERT INTO clinvar_gene_stats SELECT * FROM _build_tmp")
        # Fast lookup by gene symbol (PRIMARY KEY creates an index automatically in DuckDB
        # but we add an explicit one for clarity and cross-version safety)
        try:
            scon.execute(
                "CREATE INDEX IF NOT EXISTS idx_cgs_gene ON clinvar_gene_stats(gene_symbol)"
            )
        except Exception:
            pass  # older DuckDB versions may not support CREATE INDEX on DuckDB files
    finally:
        scon.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "clinvar_gene_stats built: %d genes indexed in %.1fs",
        len(rows),
        elapsed,
    )


# ---------------------------------------------------------------------------
# Module initialisation
# ---------------------------------------------------------------------------

def _init() -> None:
    """
    Check whether a pre-built gene index is available.

    This function intentionally does NOT build the index.  Building the index
    requires scanning the entire ClinVar database and takes significant time;
    it must never run automatically during web server startup.

    To build the index, run:
        python scripts/build_gene_index.py
    """
    global _GENE_INDEX_AVAILABLE

    if not STATS_DB_PATH.exists():
        logger.warning(
            "Gene index not found at %s — "
            "GET /genes and GET /gene/* will return HTTP 503. "
            "Run `python scripts/build_gene_index.py` to build it.",
            STATS_DB_PATH,
        )
        return

    try:
        if not _stats_table_exists():
            logger.warning(
                "Gene index file exists at %s but does not contain the expected table. "
                "Run `python scripts/build_gene_index.py --rebuild` to rebuild it.",
                STATS_DB_PATH,
            )
            return
    except Exception as exc:
        logger.warning("Gene index check failed: %s", exc)
        return

    _GENE_INDEX_AVAILABLE = True
    logger.info("Gene index ready (%s)", STATS_DB_PATH)


_init()


def rebuild_index() -> None:
    """
    Public entry-point for scripts/build_gene_index.py.

    Builds (or rebuilds) the clinvar_gene_stats table from the main ClinVar DB
    and updates the module-level availability flag.
    """
    global _GENE_INDEX_AVAILABLE

    # Drop existing table so _build_stats_table() is not skipped
    if STATS_DB_PATH.exists() and _stats_table_exists():
        con = _stats_con_rw()
        try:
            con.execute("DROP TABLE IF EXISTS clinvar_gene_stats")
        finally:
            con.close()

    _build_stats_table()
    _GENE_INDEX_AVAILABLE = True


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def list_genes(limit: int = 200, offset: int = 0) -> List[Dict]:
    """
    Return gene symbols with total variant counts, sorted by count descending.

    Each dict: {"gene_symbol": str, "total_variants": int}
    """
    con = _stats_con()
    try:
        rows = con.execute(
            "SELECT gene_symbol, total_variants "
            "FROM clinvar_gene_stats "
            "ORDER BY total_variants DESC "
            "LIMIT ? OFFSET ?",
            [int(limit), int(offset)],
        ).fetchall()
        return [{"gene_symbol": r[0], "total_variants": r[1]} for r in rows]
    finally:
        con.close()


def count_genes() -> int:
    """Return the total number of distinct genes in the index."""
    con = _stats_con()
    try:
        return int(con.execute("SELECT COUNT(*) FROM clinvar_gene_stats").fetchone()[0])
    finally:
        con.close()


def get_gene_summary(gene: str) -> Optional[Dict]:
    """
    Return full aggregated statistics for one gene, or None if not found.

    JSON fields (by_significance, by_review_status, phenotypes, variant_types)
    are decoded to Python objects before returning.
    """
    gene = gene.strip().upper()
    con = _stats_con()
    try:
        row = con.execute(
            "SELECT gene_symbol, total_variants, by_significance, by_review_status, "
            "       phenotypes, variant_types, date_earliest, date_latest, built_at "
            "FROM clinvar_gene_stats "
            "WHERE gene_symbol = ?",
            [gene],
        ).fetchone()
    finally:
        con.close()

    if not row:
        return None

    return {
        "gene_symbol":     row[0],
        "total_variants":  row[1],
        "by_significance": json.loads(row[2] or "{}"),
        "by_review_status": json.loads(row[3] or "{}"),
        "phenotypes":      json.loads(row[4] or "[]"),
        "variant_types":   json.loads(row[5] or "{}"),
        "date_range": {
            "earliest": row[6] or None,
            "latest":   row[7] or None,
        },
        "index_built_at": row[8],
    }


def get_gene_variants(
    gene: str,
    limit: int = 20,
    offset: int = 0,
    significance: Optional[str] = None,
) -> List[Dict]:
    """
    Return individual ClinVar variant records for a gene.

    Always queries the main ClinVar DB directly (not the cached stats) so
    returned records reflect the current snapshot, not the index build time.

    Parameters
    ----------
    gene         : gene symbol (case-insensitive; normalised to UPPER)
    limit        : max records to return (1–200)
    offset       : zero-based row offset for pagination
    significance : optional filter substring, e.g. "Pathogenic"
    """
    from app.retriever import _SELECT, _c, DB_PATH as CLINVAR_DB_PATH  # noqa: PLC0415

    gene = gene.strip().upper()
    query = _SELECT + f"WHERE {_c('gene_symbol')} = ?\n"
    params: list = [gene]

    if significance:
        query += f"  AND {_c('clinical_significance')} LIKE ?\n"
        params.append(f"%{significance}%")

    query += "LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])

    con = duckdb.connect(str(CLINVAR_DB_PATH), read_only=True)
    try:
        df = con.execute(query, params).fetchdf()
        records = df.to_dict(orient="records")
        # Normalise numpy scalars to native Python types
        clean: List[Dict] = []
        for rec in records:
            clean.append({
                k: (int(v) if hasattr(v, "item") else v)
                for k, v in rec.items()
            })
        return clean
    finally:
        con.close()
