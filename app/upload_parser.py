"""
upload_parser.py

Parse uploaded genetic test result files into normalized variant dictionaries.

Supported: CSV, TSV, TXT (tab- or comma-delimited), VCF, XLSX/XLS.
Not supported: PDF (detected, returned with a clear warning).

Input:  raw bytes — no file path, nothing is persisted to disk.
Output: {"file_type", "detected_columns", "variants", "warnings", "parser_version"}
"""

import csv
import io
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# Canonical field names + all recognized aliases
# ---------------------------------------------------------------------------

_COLUMN_MAP: dict[str, str] = {
    # gene
    "gene": "gene",
    "gene_symbol": "gene",
    "genesymbol": "gene",
    "symbol": "gene",
    "gene symbol": "gene",
    # variant / nucleotide change
    "variant": "variant",
    "hgvs": "variant",
    "hgvs_c": "variant",
    "hgvsc": "variant",
    "nucleotide_change": "variant",
    "nucleotide change": "variant",
    "cdna_change": "variant",
    "cdna change": "variant",
    "dna change": "variant",
    # protein change
    "protein_change": "protein_change",
    "protein change": "protein_change",
    "hgvs_p": "protein_change",
    "hgvsp": "protein_change",
    "aa_change": "protein_change",
    "amino acid change": "protein_change",
    # chromosome
    "chromosome": "chromosome",
    "chrom": "chromosome",
    "chr": "chromosome",
    # position
    "position": "position",
    "pos": "position",
    "start": "position",
    "start_pos": "position",
    "genomic_position": "position",
    # ref allele
    "ref": "ref",
    "reference": "ref",
    "ref_allele": "ref",
    "reference_allele": "ref",
    # alt allele
    "alt": "alt",
    "alternate": "alt",
    "alt_allele": "alt",
    "alternate_allele": "alt",
    "allele": "alt",
    # zygosity
    "zygosity": "zygosity",
    "genotype": "zygosity",
    "gt": "zygosity",
    "zyg": "zygosity",
    # clinical significance
    "clinical_significance": "clinical_significance",
    "clinicalsignificance": "clinical_significance",
    "classification": "clinical_significance",
    "interpretation": "clinical_significance",
    "clinvar_classification": "clinical_significance",
    "clinvar classification": "clinical_significance",
    "pathogenicity": "clinical_significance",
    # rsid / dbSNP
    "rsid": "rsid",
    "rs_id": "rsid",
    "rs": "rsid",
    "dbsnp_id": "rsid",
    "dbsnp": "rsid",
    "snp_id": "rsid",
}

_CANONICAL_FIELDS: frozenset[str] = frozenset(_COLUMN_MAP.values())

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_bytes(file_bytes: bytes) -> str:
    """Decode bytes to str. Try UTF-8-sig first (handles BOM), then UTF-8, then latin-1."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1", errors="replace")


def _is_pdf_content(file_bytes: bytes) -> bool:
    """Detect PDF by magic bytes (%PDF-)."""
    return file_bytes[:5] == b"%PDF-"


def _is_vcf_content(text: str) -> bool:
    """True if the content looks like a VCF file (##fileformat=VCF or #CHROM header)."""
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("##fileformat=VCF"):
            return True
        if stripped.upper().startswith("#CHROM\tPOS"):
            return True
    return False


def _detect_delimiter(text: str) -> str:
    """
    Infer column delimiter from the first non-comment, non-empty line.
    Returns '\t' or ','. Falls back to ',' when ambiguous.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tab_count = stripped.count("\t")
        comma_count = stripped.count(",")
        if tab_count > comma_count:
            return "\t"
        if comma_count > tab_count:
            return ","
        # equal counts — try csv.Sniffer on the first 4 KB
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t")
            return dialect.delimiter
        except csv.Error:
            return ","
    return ","


def _map_column(name: str) -> Optional[str]:
    """Return canonical field name for a raw column header, or None if unrecognized."""
    return _COLUMN_MAP.get(name.strip().lower().replace("-", "_"))


def _parse_delimited(
    text: str, delimiter: str
) -> tuple[list[dict], list[str], list[str]]:
    """
    Parse delimiter-separated text into raw row dicts.

    Skips lines starting with '#'.
    Returns (raw_rows, detected_canonical_columns, warnings).
    """
    warnings: list[str] = []
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return [], [], ["File contains no data rows (all lines are empty or comments)."]

    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    raw_rows: list[dict] = [dict(row) for row in reader]

    if not raw_rows:
        return [], [], ["No rows could be parsed from the file."]

    # Identify which headers map to recognized canonical fields
    headers = list(raw_rows[0].keys())
    detected: list[str] = []
    for h in headers:
        canon = _map_column(h)
        if canon and canon not in detected:
            detected.append(canon)

    if not detected:
        warnings.append(
            "No recognized genetic column headers were found. "
            "Expected headers such as: gene, variant, chromosome, position, "
            "rsid, clinical_significance. "
            "Unrecognized columns are preserved under the 'extra' key."
        )

    return raw_rows, detected, warnings


def _gt_to_zygosity(gt: str) -> str:
    """Convert a VCF GT field (e.g. '0/1', '1/1', '0|1') to a plain label."""
    alleles = re.split(r"[/|]", gt)
    if len(alleles) < 2:
        return gt
    a, b = alleles[0], alleles[1]
    if a == "." or b == ".":
        return "unknown"
    if a == b:
        return "homozygous" if a != "0" else "homozygous_ref"
    return "heterozygous"


def _extract_geneinfo(info: str) -> str:
    """Extract gene symbol from GENEINFO=GENE:id or GENE=symbol in VCF INFO field."""
    m = re.search(r"GENEINFO=([^:;,\s]+)", info)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|;)GENE=([^;,\s]+)", info)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Public: normalize_variant_record
# ---------------------------------------------------------------------------

def normalize_variant_record(raw_row: dict) -> dict:
    """
    Map raw column names to canonical field names.

    - Recognized columns are mapped to their canonical key.
    - If two raw columns map to the same canonical field, the first non-empty
      value wins.
    - Unrecognized columns are collected under 'extra' — nothing is dropped.
    - All canonical fields are always present (empty string if absent).
    - No medical interpretation is performed.
    """
    canonical: dict = {}
    extra: dict = {}

    for key, value in raw_row.items():
        mapped = _map_column(key)
        val_str = str(value).strip() if value is not None else ""
        if mapped:
            if mapped not in canonical or not canonical[mapped]:
                canonical[mapped] = val_str
        else:
            extra[key] = val_str

    for field in _CANONICAL_FIELDS:
        canonical.setdefault(field, "")

    if extra:
        canonical["extra"] = extra

    return canonical


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def parse_vcf_file(file_bytes: bytes) -> dict:
    """
    Parse a VCF (Variant Call Format) file from raw bytes.

    Extracts: CHROM→chromosome, POS→position, ID→rsid (if rs…), REF, ALT,
    zygosity from GT field, gene from GENEINFO= in INFO.
    """
    warnings: list[str] = []
    text = _decode_bytes(file_bytes)

    header_cols: list[str] = []
    variants: list[dict] = []
    format_col_index: Optional[int] = None
    sample_col_index: Optional[int] = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##"):
            continue
        if stripped.upper().startswith("#CHROM"):
            # VCF header line — parse column names
            header_cols = stripped.lstrip("#").split("\t")
            # Locate FORMAT and first sample column
            for i, col in enumerate(header_cols):
                if col.upper() == "FORMAT":
                    format_col_index = i
                    if i + 1 < len(header_cols):
                        sample_col_index = i + 1
                    break
            continue

        if not header_cols:
            warnings.append("VCF data row found before header line; row skipped.")
            continue

        cols = stripped.split("\t")
        row: dict[str, str] = {}

        for i, col_name in enumerate(header_cols):
            row[col_name] = cols[i] if i < len(cols) else ""

        chrom = row.get("CHROM", "").lstrip("#")
        pos = row.get("POS", "")
        vcf_id = row.get("ID", "")
        ref = row.get("REF", "")
        alt = row.get("ALT", "")
        info = row.get("INFO", "")

        rsid = vcf_id if re.match(r"^rs\d+$", vcf_id, re.IGNORECASE) else ""
        gene = _extract_geneinfo(info)

        zygosity = ""
        if format_col_index is not None and sample_col_index is not None:
            fmt_str = cols[format_col_index] if format_col_index < len(cols) else ""
            smp_str = cols[sample_col_index] if sample_col_index < len(cols) else ""
            fmt_fields = fmt_str.split(":")
            smp_fields = smp_str.split(":")
            if "GT" in fmt_fields:
                gt_idx = fmt_fields.index("GT")
                gt_val = smp_fields[gt_idx] if gt_idx < len(smp_fields) else ""
                zygosity = _gt_to_zygosity(gt_val) if gt_val else ""

        normalized = {
            "gene": gene,
            "variant": "",
            "protein_change": "",
            "chromosome": chrom,
            "position": pos,
            "ref": ref,
            "alt": alt,
            "zygosity": zygosity,
            "clinical_significance": "",
            "rsid": rsid,
        }

        variants.append(normalized)

    if not variants:
        warnings.append("No variant rows were extracted from the VCF file.")

    detected_columns: list[str] = []
    if variants:
        for field in ("chromosome", "position", "ref", "alt", "rsid", "gene", "zygosity"):
            if any(v.get(field) for v in variants):
                detected_columns.append(field)

    return {
        "file_type": "vcf",
        "detected_columns": detected_columns,
        "variants": variants,
        "warnings": warnings,
        "parser_version": PARSER_VERSION,
    }


def parse_excel_file(file_bytes: bytes) -> dict:
    """
    Parse an Excel .xlsx (or .xls) file from raw bytes using openpyxl.

    Reads the first worksheet. First row is treated as column headers.
    Applies the same normalize_variant_record column mapping as CSV/TSV.
    """
    warnings: list[str] = []

    try:
        import openpyxl
    except ImportError:
        return {
            "file_type": "xlsx",
            "detected_columns": [],
            "variants": [],
            "warnings": [
                "Excel parsing requires the 'openpyxl' library, which is not installed. "
                "Install it with: pip install openpyxl"
            ],
            "parser_version": PARSER_VERSION,
        }

    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception as exc:
        return {
            "file_type": "xlsx",
            "detected_columns": [],
            "variants": [],
            "warnings": [f"Could not open Excel file: {exc}"],
            "parser_version": PARSER_VERSION,
        }

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return {
            "file_type": "xlsx",
            "detected_columns": [],
            "variants": [],
            "warnings": ["Excel file is empty."],
            "parser_version": PARSER_VERSION,
        }

    # First row → headers
    headers = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
    data_rows = rows[1:]

    if not data_rows:
        return {
            "file_type": "xlsx",
            "detected_columns": [],
            "variants": [],
            "warnings": ["Excel file has a header row but no data rows."],
            "parser_version": PARSER_VERSION,
        }

    raw_rows: list[dict] = []
    for row in data_rows:
        raw_row = {}
        for col_name, cell_val in zip(headers, row):
            raw_row[col_name] = str(cell_val).strip() if cell_val is not None else ""
        raw_rows.append(raw_row)

    detected: list[str] = []
    for h in headers:
        canon = _map_column(h)
        if canon and canon not in detected:
            detected.append(canon)

    if not detected:
        warnings.append(
            "No recognized genetic column headers were found in the Excel file. "
            "Expected headers such as: gene, variant, chromosome, position, "
            "rsid, clinical_significance."
        )

    variants = [normalize_variant_record(row) for row in raw_rows]

    if not variants:
        warnings.append("No variant rows were extracted from the Excel file.")

    return {
        "file_type": "xlsx",
        "detected_columns": detected,
        "variants": variants,
        "warnings": warnings,
        "parser_version": PARSER_VERSION,
    }


def parse_csv_file(file_bytes: bytes) -> dict:
    """
    Parse a CSV (or auto-detected TSV) file from raw bytes.

    If the content is detected as VCF, delegates to parse_vcf_file.
    If the content is detected as PDF, returns a graceful warning.
    """
    warnings: list[str] = []

    if _is_pdf_content(file_bytes):
        return {
            "file_type": "pdf",
            "detected_columns": [],
            "variants": [],
            "warnings": ["PDF parsing is not supported. Please upload a CSV or TSV file."],
            "parser_version": PARSER_VERSION,
        }

    text = _decode_bytes(file_bytes)

    if _is_vcf_content(text):
        return parse_vcf_file(file_bytes)

    delimiter = _detect_delimiter(text)
    file_type = "tsv" if delimiter == "\t" else "csv"

    raw_rows, detected_columns, parse_warnings = _parse_delimited(text, delimiter)
    warnings.extend(parse_warnings)

    variants = [normalize_variant_record(row) for row in raw_rows]

    if not variants:
        warnings.append("No variant rows were extracted from the file.")

    return {
        "file_type": file_type,
        "detected_columns": detected_columns,
        "variants": variants,
        "warnings": warnings,
        "parser_version": PARSER_VERSION,
    }


def parse_text_file(file_bytes: bytes) -> dict:
    """
    Parse a plain TXT or TSV file from raw bytes.

    Auto-detects tab vs comma delimiter. Handles VCF and PDF gracefully.
    """
    warnings: list[str] = []

    if _is_pdf_content(file_bytes):
        return {
            "file_type": "pdf",
            "detected_columns": [],
            "variants": [],
            "warnings": ["PDF parsing is not supported. Please upload a CSV or TSV file."],
            "parser_version": PARSER_VERSION,
        }

    text = _decode_bytes(file_bytes)

    if _is_vcf_content(text):
        return parse_vcf_file(file_bytes)

    delimiter = _detect_delimiter(text)
    file_type = "tsv" if delimiter == "\t" else "txt"

    raw_rows, detected_columns, parse_warnings = _parse_delimited(text, delimiter)
    warnings.extend(parse_warnings)

    variants = [normalize_variant_record(row) for row in raw_rows]

    if not variants:
        warnings.append("No variant rows were extracted from the file.")

    return {
        "file_type": file_type,
        "detected_columns": detected_columns,
        "variants": variants,
        "warnings": warnings,
        "parser_version": PARSER_VERSION,
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def parse_uploaded_file(filename: str, file_bytes: bytes) -> dict:
    """
    Dispatch to the correct parser based on file extension (and magic bytes).

    Handles empty files and unsupported formats gracefully — never raises.
    """
    if not file_bytes:
        return {
            "file_type": "unknown",
            "detected_columns": [],
            "variants": [],
            "warnings": ["Empty file received. Please upload a non-empty file."],
            "parser_version": PARSER_VERSION,
        }

    # PDF detection via magic bytes before extension check
    if _is_pdf_content(file_bytes):
        return {
            "file_type": "pdf",
            "detected_columns": [],
            "variants": [],
            "warnings": ["PDF parsing is not supported. Please upload a CSV, TSV, VCF, or XLSX file."],
            "parser_version": PARSER_VERSION,
        }

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "vcf":
        return parse_vcf_file(file_bytes)
    if ext in ("xlsx", "xls"):
        return parse_excel_file(file_bytes)
    if ext == "csv":
        return parse_csv_file(file_bytes)
    if ext in ("txt", "tsv"):
        return parse_text_file(file_bytes)
    if ext == "pdf":
        return {
            "file_type": "pdf",
            "detected_columns": [],
            "variants": [],
            "warnings": ["PDF parsing is not supported. Please upload a CSV, TSV, VCF, or XLSX file."],
            "parser_version": PARSER_VERSION,
        }

    # Unknown extension — still try to parse as delimited text
    text = _decode_bytes(file_bytes)
    if _is_vcf_content(text):
        return parse_vcf_file(file_bytes)

    delimiter = _detect_delimiter(text)
    file_type = ext or "unknown"
    raw_rows, detected_columns, parse_warnings = _parse_delimited(text, delimiter)
    warnings: list[str] = parse_warnings[:]
    if not raw_rows:
        warnings.append(
            f"Unsupported or unrecognized file type: '.{ext}'. "
            "Supported formats: CSV, TSV, TXT, VCF, XLSX."
        )
    variants = [normalize_variant_record(row) for row in raw_rows]
    return {
        "file_type": file_type,
        "detected_columns": detected_columns,
        "variants": variants,
        "warnings": warnings,
        "parser_version": PARSER_VERSION,
    }
