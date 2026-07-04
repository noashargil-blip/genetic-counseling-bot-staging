#!/usr/bin/env python
"""
Build (or rebuild) the ClinVar gene-level statistics index.

Usage
-----
    # First-time build:
    python scripts/build_gene_index.py

    # Force a full rebuild (drops and recreates the table):
    python scripts/build_gene_index.py --rebuild

    # Also write data_version.json with build metadata:
    python scripts/build_gene_index.py --write-version

    # Full production build — rebuild + version file:
    python scripts/build_gene_index.py --rebuild --write-version

Why this is a separate script (not automatic at startup)
---------------------------------------------------------
Building the gene index scans the entire ClinVar database (~40 k+ rows per
gene, millions of rows total).  On a standard server this takes 30–120 seconds
and uses significant memory.  Running it automatically every time the web server
starts would:

  1. Delay the first request by up to two minutes.
  2. Risk partial builds if the server is killed mid-build.
  3. Prevent hot-reload without long delays.

The web server only READS the pre-built index (data/clinvar_gene_stats.duckdb).
Run this script once after each ClinVar data update, on a Slurm compute node
if running on the university cluster.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make sure we can import app modules when running from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(*, rebuild: bool, write_version: bool) -> None:
    # -----------------------------------------------------------------
    # 1. Validate environment
    # -----------------------------------------------------------------
    from app.retriever import _DB_AVAILABLE, DB_PATH  # noqa: PLC0415
    if not _DB_AVAILABLE:
        print(f"ERROR: ClinVar database not found at {DB_PATH}.", file=sys.stderr)
        print("       Download it with scripts/download_clinvar.sh first.", file=sys.stderr)
        sys.exit(1)

    from app import gene_index  # noqa: PLC0415

    # -----------------------------------------------------------------
    # 2. Build / rebuild
    # -----------------------------------------------------------------
    print(f"[{_now()}] Starting gene index build …")
    t0 = time.monotonic()

    if rebuild:
        print(f"  --rebuild: dropping existing table in {gene_index.STATS_DB_PATH}")
    gene_index.rebuild_index()

    elapsed = time.monotonic() - t0
    total_genes = gene_index.count_genes()
    print(f"[{_now()}] Done.  {total_genes:,} genes indexed in {elapsed:.1f}s.")

    # -----------------------------------------------------------------
    # 3. Optionally write data_version.json
    # -----------------------------------------------------------------
    if write_version:
        import duckdb  # noqa: PLC0415

        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            total_records = int(con.execute("SELECT COUNT(*) FROM clinvar").fetchone()[0])
            con.close()
        except Exception as exc:
            total_records = None
            print(f"  WARNING: could not count ClinVar records: {exc}", file=sys.stderr)

        # Pull build timestamp from the first row of the stats table
        try:
            con = gene_index._stats_con()
            row = con.execute(
                "SELECT built_at FROM clinvar_gene_stats LIMIT 1"
            ).fetchone()
            con.close()
            index_built_at = row[0] if row else _now()
        except Exception:
            index_built_at = _now()

        version_data = {
            "app_version":           "2.1.0",
            "clinvar_source":        "ClinVar (NCBI)",
            "clinvar_build_date":    _now(),
            "clinvar_total_records": total_records,
            "gene_index_total_genes": total_genes,
            "gene_index_built_at":   index_built_at,
            "db_path":               str(DB_PATH),
            "index_db_path":         str(gene_index.STATS_DB_PATH),
        }

        out_path = Path("data/data_version.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(version_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{_now()}] Wrote {out_path}")
        for k, v in version_data.items():
            print(f"  {k}: {v}")

    print(f"\nGene index is ready at: {gene_index.STATS_DB_PATH}")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Drop and recreate the index even if it already exists.",
    )
    parser.add_argument(
        "--write-version", action="store_true",
        help="Write data/data_version.json after building.",
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild, write_version=args.write_version)
