#!/usr/bin/env python
"""
scripts/generate_gene_knowledge_drafts.py

Offline draft-generation tool for the Gene Knowledge Base.

Usage:
    python scripts/generate_gene_knowledge_drafts.py [--gene SYMBOL] [--dry-run]

For each gene record in data/gene_knowledge_base.json that has:
  - approved = false
  - patient_summary_he is empty or missing

This script asks a local LLM (if available) to write a short draft Hebrew summary
and writes it back to the JSON file.  The record is ALWAYS kept as:
  - approved = false
  - source_status = "source_missing"  (unchanged — no sources were validated)
  - review_status = "draft"

A human reviewer must run scripts/approve_gene_knowledge.py to promote any
record to approved=true.  This script NEVER auto-approves anything.

Requirements:
  - LOCAL_LLM_URL environment variable must point to a running OpenAI-compatible
    inference endpoint (e.g. http://localhost:11434/v1).
  - The script is designed to run on the server as an offline utility, not during
    request serving.  Never import this script from app/ code.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_KB_PATH = _REPO_ROOT / "app" / "data" / "gene_knowledge_base.json"  # app/data/ — same as kb.py convention

# ── LLM system prompt ────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are writing a short, patient-friendly Hebrew paragraph about a gene's "
    "general biological role.\n\n"
    "AUDIENCE: A patient in Israel who just had genetic counseling. "
    "Use plain, clear Hebrew.\n\n"
    "You MAY write about:\n"
    "  - The gene's role in DNA replication, DNA repair, cell division, "
    "or another well-known cellular process.\n"
    "  - The protein product of the gene (e.g., DNA polymerase epsilon for POLE).\n\n"
    "You MUST NOT:\n"
    "  - Mention RNA, mRNA, transcription, or gene expression.\n"
    "  - List diseases or medical conditions.\n"
    "  - Recommend any action (surgery, testing, surveillance, medication).\n"
    "  - Estimate personal risk or interpret test results.\n"
    "  - Diagnose any condition or imply the patient has a disease.\n"
    "  - Transliterate gene symbols into Hebrew phonetics "
    "(e.g., do NOT write 'פול-א' instead of POLE).\n"
    "  - Mix English letters inside Hebrew words.\n"
    "  - Use invented, truncated, vague, or uncertain Hebrew terminology.\n"
    "  - Include question marks, emoji, or disclaimers.\n"
    "  - If you are not confident about this gene's role, "
    "output only a dash: -\n\n"
    "STRICT FORMAT:\n"
    "  - Hebrew ONLY. Gene symbols (BRCA1, POLE, etc.) and DNA are kept as-is.\n"
    "  - 2-3 sentences maximum. Maximum 400 characters.\n"
    "  - Output ONLY the sentences. No labels, no quotes, no preamble."
)

_REJECTION_PATTERNS = [
    "mrna", "mRNA", "rna", "RNA",
    "הקניית", "אסימטריה",
    "פול-א", "פול א",
]


def _call_llm(gene_symbol: str, gene_name: str) -> str | None:
    url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if not url:
        print(f"  [SKIP] LOCAL_LLM_URL not set — cannot generate draft for {gene_symbol}")
        return None
    try:
        import urllib.request
        import json as _json

        user_msg = (
            f"Gene symbol: {gene_symbol}\n"
            f"Gene name: {gene_name}\n\n"
            "Write 2-3 short Hebrew sentences about the general biological role "
            f"of the {gene_symbol} gene.  Use the official symbol '{gene_symbol}', "
            "not a Hebrew transliteration."
        )
        payload = _json.dumps({
            "model": "local",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 200,
            "temperature": 0.3,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{url.rstrip('/')}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        if not text or text == "-":
            print(f"  [SKIP] LLM returned empty/dash for {gene_symbol}")
            return None
        # Basic rejection check
        lower = text.lower()
        for pat in _REJECTION_PATTERNS:
            if pat.lower() in lower:
                print(f"  [REJECT] Draft for {gene_symbol} contains forbidden term '{pat}'")
                return None
        return text
    except Exception as exc:
        print(f"  [ERROR] LLM call failed for {gene_symbol}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate draft Hebrew summaries for gene KB.")
    parser.add_argument("--gene", metavar="SYMBOL", help="Process only this gene symbol.")
    parser.add_argument("--dry-run", action="store_true", help="Print drafts but do not write.")
    args = parser.parse_args()

    if not _KB_PATH.exists():
        print(f"ERROR: {_KB_PATH} not found.")
        sys.exit(1)

    records: list[dict] = json.loads(_KB_PATH.read_text(encoding="utf-8"))

    updated = 0
    for rec in records:
        sym = (rec.get("gene_symbol") or "").strip().upper()
        if not sym:
            continue
        if args.gene and sym != args.gene.upper():
            continue
        if rec.get("approved") is True:
            print(f"  [SKIP] {sym} is already approved — not overwriting.")
            continue
        existing = (rec.get("patient_summary_he") or "").strip()
        if existing:
            print(f"  [SKIP] {sym} already has a draft summary.")
            continue

        print(f"  Generating draft for {sym} ({rec.get('gene_name', '')})…")
        draft = _call_llm(sym, rec.get("gene_name") or sym)
        if draft is None:
            continue

        if args.dry_run:
            print(f"  [DRY-RUN] Would write for {sym}:\n    {draft}")
        else:
            rec["patient_summary_he"] = draft
            rec["review_status"] = "draft"
            rec["approved"] = False
            rec["source_status"] = rec.get("source_status") or "source_missing"
            updated += 1
            print(f"  [OK] Draft written for {sym}.")

    if not args.dry_run and updated > 0:
        _KB_PATH.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {updated} draft(s) to {_KB_PATH}.")
        print("IMPORTANT: No record was approved. Run scripts/approve_gene_knowledge.py")
        print("           after human review to promote a record to approved=true.")
    elif args.dry_run:
        print("\n[DRY-RUN] No files were modified.")
    else:
        print("\nNo records updated.")


if __name__ == "__main__":
    main()
