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

---

## Physician review portal (Session 27)

The portal lets an authorized physician review and approve AI-generated gene education drafts
before they are shown to patients.  It is **disabled by default** — the patient chat works
without it.

### One-time setup (run locally before first deploy)

```bash
# 1. Generate a PBKDF2-SHA256 password hash
python -c "from app.physician_auth import generate_hash; print(generate_hash('your-password'))"
# → pbkdf2sha256$260000$<hex-salt>$<hex-digest>

# 2. Generate a signing secret for the session cookie
python -c "import secrets; print(secrets.token_hex(32))"
# → <64 hex chars>
```

### Render environment variables (physician portal)

| Variable | Value | Required | Notes |
|----------|-------|----------|-------|
| `REVIEW_PORTAL_ENABLED` | `true` | Yes | Enables the portal |
| `REVIEWER_USERNAME` | `dr_lastname` | Yes | Login username |
| `REVIEWER_PASSWORD_HASH` | `pbkdf2sha256$260000$...` | Yes | Output of `generate_hash()` |
| `REVIEW_SESSION_SECRET` | 64 hex chars | Yes | Output of `secrets.token_hex(32)`; must be ≥ 32 chars |
| `HTTPS_ONLY_COOKIES` | `true` | Yes (prod) | Set when behind TLS termination (Render) |
| `DATABASE_URL` | `postgresql://user:pass@host/db` | Yes (prod) | Persistent storage; Render PostgreSQL recommended |
| `AI_DRAFT_VISIBILITY_MODE` | `approved_only` | Recommended | Withhold unreviewed drafts from patients |

> **Why PostgreSQL?** Render's local filesystem is ephemeral — SQLite data is lost on
> restart or deploy.  Provision a Render PostgreSQL database and set `DATABASE_URL`.

### Provision PostgreSQL on Render

1. In the Render dashboard, click **New → PostgreSQL**.
2. Choose the region nearest your web service.
3. After creation, copy the **External Database URL** (starts with `postgresql://`).
4. Add it as `DATABASE_URL` in your web service's environment.

### Post-deployment smoke tests (portal)

```bash
BASE=https://your-app.onrender.com

# Portal redirects unauthenticated users
curl -I "$BASE/physician"
# → HTTP 303 → /physician?next=...

# Login
curl -c cookies.txt -X POST "$BASE/api/physician/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"dr_lastname","password":"your-password"}'
# → {"ok":true,"identity":"dr_lastname"}

# List drafts (empty on fresh deploy)
curl -b cookies.txt "$BASE/api/physician/drafts"
# → {"drafts":[],"total":0}

# Patient chat still works with portal enabled
curl -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"מה זה VUS?"}'
# → 200 with standard 5-key response schema
```

### Rollback procedure

If the portal causes a production incident:

1. Set `REVIEW_PORTAL_ENABLED=` (empty string or remove the variable) in Render env vars.
   The portal endpoints return `503 Service Unavailable`; the patient chat is unaffected.
2. Optionally set `AI_DRAFT_VISIBILITY_MODE=approved_only` to stop showing unreviewed AI drafts.
3. If the issue is in `review_db.py` or `physician_auth.py`, deploy a hotfix branch.
   The patient chat (`/ask`) does not depend on those modules at request time.

### Security notes

- **Brute-force protection**: 5 failed logins within 5 minutes lock out the username for 10 minutes.
- **CSRF**: mitigated by `SameSite=lax` cookie + JSON `Content-Type` requirement.
  Cross-origin POST requests cannot attach the session cookie in modern browsers.
- **Session secret**: must be at least 32 characters.  The app logs a warning if it is shorter.
- **Cookie flags**: `HttpOnly`, `SameSite=lax`, and `Secure` (when `HTTPS_ONLY_COOKIES=true`).

### Render deployment checklist (portal additions)

- [ ] Render PostgreSQL database provisioned and `DATABASE_URL` set
- [ ] `REVIEW_PORTAL_ENABLED=true` set
- [ ] `REVIEWER_USERNAME` set (no spaces)
- [ ] `REVIEWER_PASSWORD_HASH` set (generated with `generate_hash()`, not a plaintext password)
- [ ] `REVIEW_SESSION_SECRET` set (≥ 32 chars, from `secrets.token_hex(32)`)
- [ ] `HTTPS_ONLY_COOKIES=true` set
- [ ] `AI_DRAFT_VISIBILITY_MODE=approved_only` set (recommended for production)
- [ ] Smoke-test login at `/physician`
- [ ] Verify `/api/physician/drafts` returns `{"drafts":[],"total":0}` on fresh DB
- [ ] Verify patient chat `/ask` still returns 5-key JSON schema
- [ ] Verify `POST /ask` with `{"question": "מה זה VUS?"}` returns a Hebrew answer
