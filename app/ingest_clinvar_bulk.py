import duckdb
from pathlib import Path

DB_PATH = Path("data/clinvar.duckdb")
GZ_PATH = Path("data/bulk/variant_summary.txt.gz")

def main():
    if not GZ_PATH.exists():
        raise FileNotFoundError(f"Missing file: {GZ_PATH}. Download it first.")

    con = duckdb.connect(str(DB_PATH))

    # 1) Create a raw table from the ClinVar bulk file (DuckDB can read .gz directly)
    con.execute("DROP TABLE IF EXISTS clinvar_raw;")
    con.execute(
        """
        CREATE TABLE clinvar_raw AS
        SELECT *
        FROM read_csv_auto(
            ?,
            delim='\t',
            header=true,
            quote='"',
            escape='"'
        );
        """,
        [str(GZ_PATH)],
    )

    # 2) Keep only columns we care about (edit names if needed after inspecting your file)
    con.execute("DROP TABLE IF EXISTS clinvar;")
    con.execute(
        """
        CREATE TABLE clinvar AS
        SELECT
            "VariationID"            AS variation_id,
            "GeneSymbol"             AS gene_symbol,
            "ClinicalSignificance"   AS clinical_significance,
            "ReviewStatus"           AS review_status,
            "PhenotypeIDS"           AS phenotype_ids,
            "PhenotypeList"          AS phenotype_list,
            "Type"                   AS variant_type,
            "Assembly"               AS assembly,
            "Chromosome"             AS chromosome,
            "Start"                  AS start_pos,
            "Stop"                   AS stop_pos,
            "RS# (dbSNP)"            AS dbsnp_id
        FROM clinvar_raw;
        """
    )

    # 3) Optional: basic sanity counts
    n = con.execute("SELECT COUNT(*) FROM clinvar;").fetchone()[0]
    print(f"Loaded rows into clinvar: {n}")

    con.close()
    print(f"Saved DuckDB to: {DB_PATH}")

if __name__ == "__main__":
    main()
