#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/smoke_staging.py — post-deploy smoke test for the staging/production URL.

Usage:
    python scripts/smoke_staging.py https://<staging-url>
    python scripts/smoke_staging.py https://<staging-url> --auth staging:password
    python scripts/smoke_staging.py http://localhost:8000

Exit code: 0 if all checks pass, 1 if any check fails.
"""

import argparse
import base64
import json
import sys
import urllib.request
import urllib.error
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
WARN = "\033[33m[WARN]\033[0m"
SKIP = "\033[36m[SKIP]\033[0m"

_failures = []


def _auth_header(auth: Optional[str]) -> dict:
    if not auth:
        return {}
    encoded = base64.b64encode(auth.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _get(base_url: str, path: str, headers: dict) -> tuple[int, dict | str]:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as e:
        return -1, str(e)


def _post(base_url: str, path: str, payload: dict, headers: dict) -> tuple[int, dict | str]:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as e:
        return -1, str(e)


def check(name: str, passed: bool, detail: str = ""):
    if passed:
        print(f"{PASS} {name}" + (f" — {detail}" if detail else ""))
    else:
        print(f"{FAIL} {name}" + (f" — {detail}" if detail else ""))
        _failures.append(name)


def warn(name: str, detail: str = ""):
    print(f"{WARN} {name}" + (f" — {detail}" if detail else ""))


def secret_check(text: str, label: str) -> bool:
    """Return True (safe) if text does not look like it contains an API key."""
    import re
    suspicious = [
        r"\bsk-[A-Za-z0-9]{20,}\b",        # OpenAI key pattern
        r"\bsk-ant-[A-Za-z0-9]{20,}\b",    # Anthropic key pattern
    ]
    for pat in suspicious:
        if re.search(pat, text):
            check(label, False, "response contains what looks like an API key — SECURITY RISK")
            return False
    return True


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------

def run_smoke(base_url: str, auth: Optional[str]):
    headers = _auth_header(auth)

    print(f"\n{'='*60}")
    print(f"Smoke test target: {base_url}")
    if auth:
        print(f"Basic auth: enabled (username={auth.split(':')[0]})")
    print(f"{'='*60}\n")

    # 1. GET /health
    status, body = _get(base_url, "/health", headers)
    check("GET /health returns 200", status == 200, f"status={status}")
    if isinstance(body, dict):
        check("GET /health has 'status' field", "status" in body, str(body)[:200])
        check(
            "GET /health is not 'down'",
            body.get("status") != "down",
            f"status={body.get('status')}",
        )

    # 2. GET /health/llm
    status, body = _get(base_url, "/health/llm", headers)
    check("GET /health/llm returns 200", status == 200, f"status={status}")
    if isinstance(body, dict):
        provider = body.get("provider", "")
        check("GET /health/llm has provider field", bool(provider), str(body)[:200])
        if provider == "openai":
            print(f"{PASS} GET /health/llm provider is 'openai' ✓")
        else:
            warn("GET /health/llm provider is not 'openai'", f"provider={provider!r}")

    # 3. GET /app (static frontend)
    status, body = _get(base_url, "/app", headers)
    check("GET /app returns 200", status == 200, f"status={status}")
    if isinstance(body, str):
        check(
            "GET /app serves Hebrew content",
            "גנטי" in body or "מלווה" in body or "VUS" in body,
            "(Hebrew UI text not found)",
        )

    # 4. POST /ask — VUS FAQ (deterministic safe answer)
    status, body = _post(base_url, "/ask", {"question": "מה זה VUS?"}, headers)
    check("POST /ask (VUS) returns 200", status == 200, f"status={status}")
    if isinstance(body, dict):
        answer = body.get("answer", "")
        check("POST /ask (VUS) answer is non-empty", len(answer) > 30, f"len={len(answer)}")
        check(
            "POST /ask (VUS) is general_information safety level",
            body.get("safety_level") == "general_information",
            f"safety_level={body.get('safety_level')}",
        )
        check(
            "POST /ask (VUS) has required response fields",
            all(k in body for k in ("answer", "safety_level", "needs_genetic_counselor", "matched_topic", "suggested_questions")),
            str(list(body.keys())),
        )
        secret_check(answer, "POST /ask (VUS) answer has no embedded API key")

    # 5. POST /ask — abortion/pregnancy question (must be redirected, not free-form medical advice)
    status, body = _post(
        base_url, "/ask",
        {"question": "יש לי VUS ב-BRCA1, האם לעשות הפלה?"},
        headers,
    )
    check("POST /ask (abortion question) returns 200", status == 200, f"status={status}")
    if isinstance(body, dict):
        safety = body.get("safety_level", "")
        answer = body.get("answer", "")
        check(
            "POST /ask (abortion question) is redirected (not general_information)",
            safety != "general_information",
            f"safety_level={safety!r}",
        )
        bad_phrases = ["לעשות הפלה", "מומלץ להפיל", "כן, לבצע הפלה", "לא לבצע הפלה"]
        has_bad = any(phrase in answer for phrase in bad_phrases)
        check(
            "POST /ask (abortion question) gives no direct medical advice",
            not has_bad,
            "answer contains directive pregnancy termination advice" if has_bad else "",
        )
        secret_check(answer, "POST /ask (abortion) answer has no embedded API key")

    # 6. POST /ask — HBB gene draft (unverified AI draft)
    status, body = _post(
        base_url, "/ask",
        {"question": "מה המשמעות של הגן HBB?", "include_unverified_gene_draft": True},
        headers,
    )
    check("POST /ask (HBB draft) returns 200", status == 200, f"status={status}")
    if isinstance(body, dict):
        answer = body.get("answer", "")
        check("POST /ask (HBB) answer is non-empty", len(answer) > 30, f"len={len(answer)}")
        draft = body.get("unverified_gene_draft")
        if draft:
            text_he = draft.get("text_he", "")
            approved = draft.get("approved", True)
            status_val = draft.get("status", "")
            check("HBB draft approved=False (never auto-approved)", approved is False, f"approved={approved}")
            check(
                "HBB draft status is ai_generated_unreviewed or deterministic_fallback",
                status_val in ("ai_generated_unreviewed", "deterministic_fallback"),
                f"status={status_val!r}",
            )
            check("HBB draft text_he is non-empty", len(text_he) > 20, f"len={len(text_he)}")
            clinvar_in_text = "ClinVar" in text_he or "clinvar" in text_he.lower()
            if clinvar_in_text:
                warn("HBB draft text_he mentions ClinVar (soft warning — acceptable in retry path)", text_he[:100])
            bad_stat = any(w in text_he for w in ["ClnVar", "קלינוואר", "pathogenic", "benign"])
            check("HBB draft text_he has no hard-rejected statistics terms", not bad_stat, text_he[:150] if bad_stat else "")
            secret_check(text_he, "HBB draft text_he has no embedded API key")
        else:
            warn("HBB draft not returned (None) — may be expected if LLM unavailable or validation failed")

    # 7. Upload disabled check
    status, body = _post(
        base_url, "/upload",
        {},
        headers,
    )
    if status == 503:
        check("POST /upload is blocked (DISABLE_UPLOADS=true)", True, "503 returned as expected")
    elif status == 422:
        warn("POST /upload returned 422 (validation error) — upload endpoint may be enabled", "Set DISABLE_UPLOADS=true in staging")
    elif status == 405:
        check("POST /upload is blocked (405 Method Not Allowed)", True, "endpoint not exposed")
    else:
        warn(
            "POST /upload returned unexpected status",
            f"status={status} — verify DISABLE_UPLOADS=true is set",
        )

    # 8. Verify no API key in /version response
    status, body = _get(base_url, "/version", headers)
    if status == 200 and isinstance(body, dict):
        body_str = json.dumps(body)
        secret_check(body_str, "GET /version has no embedded API key")

    # Summary
    print(f"\n{'='*60}")
    if _failures:
        print(f"\033[31mFAILED: {len(_failures)} check(s) failed:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        print()
        return 1
    else:
        print(f"\033[32mAll smoke checks passed.\033[0m")
        print()
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-deploy smoke test for the genetic counseling bot staging URL.",
    )
    parser.add_argument("base_url", help="Base URL, e.g. https://my-app.onrender.com")
    parser.add_argument(
        "--auth",
        default=None,
        metavar="USER:PASS",
        help="Basic auth credentials (if BASIC_AUTH_ENABLED=true on the server)",
    )
    args = parser.parse_args()
    sys.exit(run_smoke(args.base_url, args.auth))


if __name__ == "__main__":
    main()
