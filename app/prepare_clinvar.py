import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_raw_clinvar_tsv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    # txt אבל זה בעצם TSV (tab-separated)
    df = pd.read_csv(path, sep="\t", dtype=str)
    return df

def clean_clinvar_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant columns from the ClinVar export
    and rename them to simpler names.
    """

    # מפה: שם עמודה בקובץ → שם עמודה נקי שנשתמש בו בקוד
    col_map = {
        "Name": "name",
        "Gene(s)": "genes",
        "Condition(s)": "conditions",
        "VariationID": "variation_id",
        "AlleleID(s)": "allele_ids",
        "dbSNP ID": "dbsnp_id",
        "Variant type": "variant_type",
        "Molecular consequence": "molecular_consequence",
        "Germline classification": "germline_classification",
        "Germline date last evaluated": "germline_last_eval",
        "Germline review status": "germline_review_status",
    }

    # נשמור רק עמודות שבאמת קיימות בקובץ
    existing = {raw: new for raw, new in col_map.items() if raw in df.columns}

    cleaned = df[list(existing.keys())].rename(columns=existing)

    # נחליף NaN במחרוזות ריקות כדי שלא יתפוצץ בחיפושים
    cleaned = cleaned.fillna("")

    # נבנה לינק ישיר ל-ClinVar לכל וריאנט
    if "variation_id" in cleaned.columns:
        cleaned["clinvar_url"] = (
            "https://www.ncbi.nlm.nih.gov/clinvar/variation/"
            + cleaned["variation_id"].astype(str)
        )

    return cleaned

def main():
    # שימי לב לשם הקובץ – כמו ששמרת אותו
    df_raw = load_raw_clinvar_tsv("clinvar_nf1_vus.txt")
    print("Raw columns:", list(df_raw.columns))

    df_clean = clean_clinvar_df(df_raw)

    out_path = DATA_DIR / "clinvar_nf1_vus_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"Saved cleaned file to: {out_path}")
    print(df_clean.head(5))

if __name__ == "__main__":
    main()
