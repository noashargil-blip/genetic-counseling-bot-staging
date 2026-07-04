#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/smoke_test_app.py — Lightweight smoke test for the genetic-counseling assistant.

Tests the running app via FastAPI TestClient (no uvicorn required).
Prints concise PASS / FAIL lines and exits with code 0 (all pass) or 1 (any fail).

Works without LOCAL_LLM_URL — deterministic fallback is sufficient for all checks.

Usage (server or local):
    PYTHONUTF8=1 /powerapps/share/rocky9/anaconda3-2024.10/bin/python \
        scripts/smoke_test_app.py

Or locally:
    python scripts/smoke_test_app.py
"""

import sys
import os
import re
from pathlib import Path

# Ensure project root is on sys.path so `app` package is importable
# regardless of which directory the script is invoked from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # make relative data/ and app/ paths resolve correctly

# Ensure UTF-8 output on Windows / server
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from app.main import app
except ImportError as exc:
    print(f"FAIL  [bootstrap] Cannot import app: {exc}", flush=True)
    sys.exit(1)

client = TestClient(app)

# Remove LLM env vars so smoke test is fully deterministic
for _var in ("LOCAL_LLM_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_var, None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_results: list[tuple[bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}", flush=True)
    _results.append((passed, name))


def ask(question: str, **kw) -> dict:
    payload = {"question": question}
    payload.update(kw)
    resp = client.post("/ask", json=payload)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 1. GET /version
# ---------------------------------------------------------------------------
print("\n── /version ─────────────────────────────────────────────────────────")

try:
    resp = client.get("/version")
    data = resp.json()
    check("/version returns 200", resp.status_code == 200)
    check("/version has app_version", "app_version" in data)
    check("/version has safety_policy_version", "safety_policy_version" in data)
    check("/version has llm_policy_version", "llm_policy_version" in data)
    check("/version gene_cards_available is True", data.get("gene_cards_available") is True)
    check("/version answer_tiers_available is True", data.get("answer_tiers_available") is True)
    check("/version deterministic_fallback_available is True",
          data.get("deterministic_fallback_available") is True)
    ver = data.get("app_version", "")
    parts = ver.split(".")
    check("/version format is X.Y.Z",
          len(parts) == 3 and all(p.isdigit() for p in parts),
          ver)
except Exception as exc:
    check("/version (exception)", False, str(exc))

# ---------------------------------------------------------------------------
# 2. GET /health/llm
# ---------------------------------------------------------------------------
print("\n── /health/llm ───────────────────────────────────────────────────────")

try:
    resp = client.get("/health/llm")
    data = resp.json()
    check("/health/llm returns 200", resp.status_code == 200)
    check("/health/llm has deterministic_fallback_available",
          data.get("deterministic_fallback_available") is True)
    # When no LLM is configured, ok must be False but that is expected
    check("/health/llm ok field is bool", isinstance(data.get("ok"), bool))
    # No LLM env vars are set — ok should be False
    check("/health/llm ok is False (no LLM configured)", data.get("ok") is False)
except Exception as exc:
    check("/health/llm (exception)", False, str(exc))

# ---------------------------------------------------------------------------
# 3. POST /ask — core questions
# ---------------------------------------------------------------------------
print("\n── POST /ask ─────────────────────────────────────────────────────────")

# ---- 3a. VUS general -------------------------------------------------------
try:
    data = ask("מה זה VUS?")
    check("VUS general — HTTP 200 + answer present",
          bool(data.get("answer")))
    check("VUS general — safety_level general_information",
          data.get("safety_level") == "general_information")
    check("VUS general — Hebrew in answer",
          bool(re.search(r"[א-ת]", data.get("answer", ""))))
    check("VUS general — llm_used False (deterministic mode)",
          data.get("llm_used") is False)
    check("VUS general — fallback_used True",
          data.get("fallback_used") is True)
    check("VUS general — 7-field schema",
          all(k in data for k in ("answer", "safety_level", "needs_genetic_counselor",
                                   "matched_topic", "suggested_questions",
                                   "llm_used", "fallback_used")))
except Exception as exc:
    check("VUS general (exception)", False, str(exc))

# ---- 3b. VUS + gene (step 4 path, vus_known_gene) -------------------------
try:
    data = ask("אמרו לי שיש לי VUS בגן APC, מה זה אומר?")
    check("VUS+APC — HTTP 200 + answer present",
          bool(data.get("answer")))
    check("VUS+APC — safety_level general_information",
          data.get("safety_level") == "general_information")
    check("VUS+APC — answer contains APC",
          "APC" in data.get("answer", ""))
    check("VUS+APC — matched_topic is vus_known_gene",
          data.get("matched_topic") == "vus_known_gene")
    check("VUS+APC — no surgery recommendation",
          "ניתוח" not in data.get("answer", ""))
    check("VUS+APC — no personal risk statement",
          not re.search(r"הסיכון\s+שלך|את\s+חולה|אתה\s+חולה", data.get("answer", "")))
except Exception as exc:
    check("VUS+APC (exception)", False, str(exc))

# ---- 3c-gene. Direct gene question — APC Tier 1 (gene_clinvar_summary) ----
try:
    data = ask("מה ידוע על APC?")
    check("APC direct — HTTP 200 + answer present",
          bool(data.get("answer")))
    check("APC direct — matched_topic gene_clinvar_summary",
          data.get("matched_topic") == "gene_clinvar_summary")
    check("APC direct — gene_metadata present",
          isinstance(data.get("gene_metadata"), dict))
    meta = data.get("gene_metadata") or {}
    check("APC direct — answer_tier is tier1",
          meta.get("answer_tier") == "tier1")
    check("APC direct — gene_symbol is APC",
          meta.get("gene_symbol") == "APC")
    check("APC direct — llm_used False",
          data.get("llm_used") is False)
    check("APC direct — curated biology in answer",
          any(w in data.get("answer", "") for w in ("גדילה", "תאים", "בקרה")))
except Exception as exc:
    check("APC direct (exception)", False, str(exc))

# ---- 3c. Non-curated gene — HBB (Tier 2 if in index, else fallback) --------
try:
    data = ask("מה זה HBB?")
    check("HBB — HTTP 200 + answer present",
          bool(data.get("answer")))
    check("HBB — safety_level general_information",
          data.get("safety_level") == "general_information")
    check("HBB — answer contains Hebrew",
          bool(re.search(r"[א-ת]", data.get("answer", ""))))
    check("HBB — no personal medical recommendation",
          not re.search(r"ניתוח|הסיכון\s+שלך|יש\s+לך\s+סרטן", data.get("answer", "")))
    if data.get("gene_metadata"):
        tier = data["gene_metadata"].get("answer_tier", "")
        check("HBB — tier is tier2 or tier3",
              tier in ("tier2", "tier3"), tier)
        if tier == "tier2":
            check("HBB Tier-2 — transparency note present",
                  "אין לי כרגע" in data["answer"] or "ClinVar" in data["answer"])
except Exception as exc:
    check("HBB (exception)", False, str(exc))

# ---- 3d. Unknown gene (Tier 3) --------------------------------------------
try:
    data = ask("מה ידוע על XXXXXXXXUNKNOWNGENE?")
    check("Unknown gene — HTTP 200",
          bool(data.get("answer")))
    check("Unknown gene — no personal recommendation",
          not re.search(r"ניתוח|כריתה|הסיכון\s+שלך", data.get("answer", "")))
    check("Unknown gene — answer contains Hebrew",
          bool(re.search(r"[א-ת]", data.get("answer", ""))))
    # Must point toward genetics team or local DB
    check("Unknown gene — refers to genetics team or local DB",
          any(w in data.get("answer", "") for w in ("גנטי", "מאגר", "צוות", "פנ")))
except Exception as exc:
    check("Unknown gene (exception)", False, str(exc))

# ---- 3e. Carrier status ---------------------------------------------------
try:
    data = ask("אמרו לי שאני נשאית, מה זה?")
    check("Carrier — HTTP 200 + answer",
          bool(data.get("answer")))
    check("Carrier — safety_level general_information",
          data.get("safety_level") == "general_information")
    check("Carrier — answer contains Hebrew",
          bool(re.search(r"[א-ת]", data.get("answer", ""))))
except Exception as exc:
    check("Carrier (exception)", False, str(exc))

# ---- 3f. PII block --------------------------------------------------------
try:
    data = ask("קוראים לי יוסי כהן, יש לי VUS ב-BRCA1")
    check("PII block — safety_level contains_identifying_info",
          data.get("safety_level") == "contains_identifying_info")
    check("PII block — needs_genetic_counselor True or safety block",
          data.get("needs_genetic_counselor") is True
          or data.get("safety_level") == "contains_identifying_info")
except Exception as exc:
    check("PII block (exception)", False, str(exc))

# ---- 3g. Personal medical-action redirect ----------------------------------
try:
    # A variant-specific surgery question is reliably caught by step 3
    data = ask("האם אני צריכה ניתוח בגלל הווריאנט?")
    check("Medical action blocked — safety_level requires_genetic_counselor",
          data.get("safety_level") == "requires_genetic_counselor")
    check("Medical action blocked — needs_genetic_counselor True",
          data.get("needs_genetic_counselor") is True)
    check("Medical action blocked — answer contains Hebrew",
          bool(re.search(r"[א-ת]", data.get("answer", ""))))
except Exception as exc:
    check("Medical action blocked (exception)", False, str(exc))

# ---- 3h. BRCA1 Tier 1 with VUS framing ------------------------------------
try:
    data = ask("יש לי VUS ב-BRCA1, מה זה?")
    check("BRCA1 VUS — HTTP 200",
          bool(data.get("answer")))
    check("BRCA1 VUS — contains BRCA1",
          "BRCA1" in data.get("answer", ""))
    check("BRCA1 VUS — no surgery recommendation",
          "ניתוח" not in data.get("answer", ""))
    check("BRCA1 VUS — no personal risk statement",
          not re.search(r"הסיכון\s+שלך|את\s+חולה|אתה\s+חולה", data.get("answer", "")))
except Exception as exc:
    check("BRCA1 VUS (exception)", False, str(exc))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total  = len(_results)
passed = sum(1 for ok, _ in _results if ok)
failed = total - passed

print(f"\n── Summary ───────────────────────────────────────────────────────────")
print(f"  {passed}/{total} checks passed", flush=True)

if failed:
    print(f"  {failed} FAILED:", flush=True)
    for ok, name in _results:
        if not ok:
            print(f"    ✗ {name}", flush=True)
    sys.exit(1)
else:
    print("  All checks passed.", flush=True)
    sys.exit(0)
