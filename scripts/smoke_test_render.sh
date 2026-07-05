#!/usr/bin/env bash
# smoke_test_render.sh — Session 18 AI draft health check
# Usage: BASE_URL=https://your-app.onrender.com bash scripts/smoke_test_render.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PASS=0; FAIL=0
pass() { echo "[PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }

echo "=== Smoke test: AI draft (Session 18) ==="
echo "BASE_URL: ${BASE_URL}"

# 1. /health/llm
HEALTH=$(curl -sf "${BASE_URL}/health/llm" 2>/dev/null || echo "ERROR")
if echo "${HEALTH}" | grep -q '"configured":true'; then
    pass "/health/llm configured=true"; LLM_OK=true
elif echo "${HEALTH}" | grep -q '"configured":false'; then
    pass "/health/llm configured=false (no LLM)"; LLM_OK=false
else
    fail "/health/llm failed"; LLM_OK=false
fi

# 2. POST /ask APOE
APOE=$(curl -sf -X POST "${BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d '{"question":"APOE"}' 2>/dev/null || echo "ERROR")
if echo "${APOE}" | grep -q "gene_clinvar_summary"; then
    pass "matched_topic=gene_clinvar_summary"
else
    fail "matched_topic != gene_clinvar_summary"
fi

# 3. ai_draft_debug.attempted
if echo "${APOE}" | grep -q '"attempted":true'; then
    pass "ai_draft_debug.attempted=true"
elif echo "${APOE}" | grep -q '"attempted":false'; then
    pass "ai_draft_debug.attempted=false (no LLM)"
else
    fail "ai_draft_debug.attempted not found"
fi

# 4. unverified_gene_draft
if [ "${LLM_OK}" = "true" ]; then
    if echo "${APOE}" | grep -q "unverified_gene_draft"; then
        pass "unverified_gene_draft present"
    else
        fail "unverified_gene_draft absent (LLM configured but no draft)"
    fi
else
    if echo "${APOE}" | grep -q "unverified_gene_draft"; then
        fail "unverified_gene_draft present but LLM not configured"
    else
        pass "unverified_gene_draft absent (expected)"
    fi
fi

# 5. High-stakes safety
HS=$(curl -sf -X POST "${BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d '{"question":"APOE risk"}' 2>/dev/null || echo "ERROR")
if echo "${HS}" | grep -q "unverified_gene_draft"; then
    fail "AI draft shown for risk question - SAFETY CHECK FAILED"
else
    pass "No AI draft for risk/safety question"
fi

echo ""
echo "=== ${PASS} passed, ${FAIL} failed ==="
[ "${FAIL}" -eq 0 ] || exit 1
