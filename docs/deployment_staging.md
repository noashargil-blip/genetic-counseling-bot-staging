# Staging / Beta Deployment Guide

Hebrew post-genetic-counseling assistant — low-cost public staging on Render or Railway.

---

## Recommended stack

| Layer | Choice | Estimated cost |
|-------|--------|---------------|
| Hosting | Render (Web Service) or Railway | $0–7/month on starter tier |
| LLM | OpenAI API (`gpt-4o-mini`) | ~$0.15/1M input tokens; negligible for small beta |
| Database | None required (core KB is JSON; ClinVar optional) | $0 |

**Expected total: a few dollars/month** for a small private beta with light usage.

---

## Build and start commands

```bash
# Build (install dependencies)
pip install -r requirements.txt

# Start
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

For local dev (port 8000):

```bash
uvicorn app.main:app --reload --port 8000
```

---

## Environment variables

### Required for staging (OpenAI)

| Variable | Value | Notes |
|----------|-------|-------|
| `LLM_PROVIDER` | `openai` | Activates OpenAI backend |
| `OPENAI_API_KEY` | `sk-...` | Set as a secret in Render/Railway — never commit |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cheapest capable model; change to `gpt-4o` for higher quality |
| `APP_ENV` | `staging` | Informational; used in future monitoring |

### Recommended for staging safety

| Variable | Value | Notes |
|----------|-------|-------|
| `DISABLE_UPLOADS` | `true` | Disables /upload and /analyze-upload endpoints |
| `LOG_PERSONAL_DATA` | `false` | (default) Never log user messages |

### Optional tuning

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM response |
| `LLM_TEMPERATURE` | `0.3` | Low for consistent medical answers |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout for LLM API calls |

### University / local dev (Slurm)

| Variable | Value | Notes |
|----------|-------|-------|
| `LLM_PROVIDER` | `local` | Activates local HTTP endpoint |
| `LOCAL_LLM_URL` | `http://compute-node:8765/generate` | Slurm compute node endpoint |
| `LLM_TIMEOUT_SECONDS` | `60` | Local models may be slower |

### Deterministic-only mode (no LLM)

```bash
LLM_PROVIDER=none
```

All answers use the curated KB only. No LLM API calls. Fully free.

---

## How to set LLM_PROVIDER on Render

1. Go to **Dashboard → your service → Environment**.
2. Add `LLM_PROVIDER = openai`.
3. Add `OPENAI_API_KEY = sk-...` (mark as **Secret**).
4. Add `OPENAI_MODEL = gpt-4o-mini`.
5. Add `DISABLE_UPLOADS = true`.
6. Click **Save Changes** → service redeploys automatically.

---

## How to test after deploy

```bash
# Health check
curl https://your-app.onrender.com/health

# LLM provider check (no key exposed)
curl https://your-app.onrender.com/health/llm

# Ask a question
curl -X POST https://your-app.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "מה זה VUS?"}'

# Open the UI
open https://your-app.onrender.com/app
```

---

## Data files required at runtime

| File | Required | Notes |
|------|----------|-------|
| `app/data/genetic_counseling_kb.json` | **Yes** | Core FAQ/KB answers |
| `app/data/gene_knowledge_base.json` | **Yes** | Gene knowledge (all `approved=false` in beta) |
| `app/data/draft_review_queue.json` | **Yes** | Review queue (5 pending drafts) |
| `app/static/` | **Yes** | Frontend HTML/CSS/JS |
| `app/data/clinvar.duckdb` | No | ClinVar variant DB — omit for staging; gene endpoints return 503 gracefully |
| `app/data/clinvar_gene_stats.duckdb` | No | Gene index — optional; gene stats degrade gracefully |

> The app never crashes on missing optional data files. Gene-level stats and variant lookup
> endpoints return 503 if the database files are absent.

---

## Known limitations (beta)

- **Not clinically approved.** No gene draft has been reviewed by a physician yet.
- **Gene summaries may be incomplete.** Tier 1b gene knowledge requires human approval of review drafts.
- **No personal report interpretation.** The bot does not read or interpret uploaded genetic reports in staging (`DISABLE_UPLOADS=true`).
- **Unverified AI draft clearly labeled.** Opt-in gene draft generation is available behind a button but always marked as unreviewed.
- **No file uploads in staging.** Set `DISABLE_UPLOADS=false` only after clinical review.
- **gpt-4o-mini is not gpt-4o.** For higher quality, change `OPENAI_MODEL=gpt-4o` (higher cost).

---

## Render deployment checklist

- [ ] `requirements.txt` committed (no torch/transformers)
- [ ] Start command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- [ ] `LLM_PROVIDER=openai` set as env var
- [ ] `OPENAI_API_KEY` set as **secret** env var
- [ ] `DISABLE_UPLOADS=true` set
- [ ] Verify `/health` returns `{"status": "ok" | "degraded"}` (degraded is fine — ClinVar DB is absent)
- [ ] Verify `/health/llm` returns `{"ok": true, "provider": "openai", "configured": true}`
- [ ] Verify `/app` loads the Hebrew chat UI
- [ ] Verify privacy notice appears before first interaction
- [ ] Verify `POST /ask` with `{"question": "מה זה VUS?"}` returns a Hebrew answer
