# Final Launch Checklist — Genetic Counseling Bot (Staging)

## Platform options

### Render (preferred)
Free tier available. Persistent disk not required (data is embedded).

1. Create a new **Web Service** at [render.com](https://render.com).
2. Connect your GitHub repository (or upload the source tarball).
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Select a **Free** or **Starter** instance (512 MB RAM is sufficient).
6. Set all required environment variables (see below).
7. Deploy. Render assigns a `https://<your-app>.onrender.com` URL automatically.

### Railway
1. Create a new project at [railway.app](https://railway.app).
2. Select **Deploy from GitHub** or **Deploy from template**.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set all required environment variables (see below).
6. Click Deploy. Railway assigns a public HTTPS domain.

---

## Required environment variables

Set these in the Render/Railway dashboard under **Environment → Variables**.
Never paste these into any file that is committed to git.

| Variable | Value | Notes |
|---|---|---|
| `APP_ENV` | `staging` | Marks this as a staging instance |
| `LLM_PROVIDER` | `openai` | Must match the key you provide |
| `OPENAI_API_KEY` | `sk-...` | **Secret** — set in the dashboard only |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cost-effective for staging |
| `LLM_TIMEOUT_SECONDS` | `20` | |
| `LLM_MAX_TOKENS` | `500` | |
| `LLM_TEMPERATURE` | `0.2` | |
| `DISABLE_UPLOADS` | `true` | Required for staging |
| `LOG_PERSONAL_DATA` | `false` | Required for privacy |
| `ADMIN_REVIEW_ENABLED` | `false` | |
| `AI_EXPANDED_ANSWERS_ENABLED` | `false` | |
| `SOURCE_GROUNDING_ENABLED` | `false` | |

## Optional: basic auth (recommended for first private link)

| Variable | Value |
|---|---|
| `BASIC_AUTH_ENABLED` | `true` |
| `BASIC_AUTH_USERNAME` | `staging` (or choose your own) |
| `BASIC_AUTH_PASSWORD` | A strong random password |

Basic auth protects `/app` and `/ask` but never blocks `/health`.

---

## How to set the OpenAI key safely

1. **Never** write the key in `.env`, source code, or any committed file.
2. In Render: Dashboard → your service → **Environment** → **Add Environment Variable** → `OPENAI_API_KEY` = `sk-...`
3. In Railway: Project → Variables → `OPENAI_API_KEY` = `sk-...`
4. The key is stored encrypted by the platform and injected at runtime only.

---

## Verification steps after deploy

### 1. Health check
```
GET https://<staging-url>/health
```
Expected: `{"status": "ok", ...}` or `{"status": "degraded", ...}` (degraded is fine if ClinVar DB is absent)

### 2. LLM check
```
GET https://<staging-url>/health/llm
```
Expected: `{"provider": "openai", "configured": true, ...}`

### 3. Open the app
```
GET https://<staging-url>/app
```
Expected: Hebrew RTL chat interface with privacy notice visible.

### 4. Smoke test script
```bash
pip install requests  # if not already installed
python scripts/smoke_staging.py https://<staging-url>
# With basic auth:
python scripts/smoke_staging.py https://<staging-url> --auth staging:yourpassword
```
Expected: `All smoke checks passed.`

---

## Data files included (no separate download needed)

| File | Status |
|---|---|
| `app/data/genetic_counseling_kb.json` | Included — core KB |
| `app/data/gene_knowledge_base.json` | Included — approved gene cards |
| `app/data/gene_aliases.json` | Included — gene alias map |
| `app/data/draft_review_queue.json` | Included — empty at start |
| `app/data/clinvar.duckdb` | **Not included** — large file; app degrades gracefully |
| `app/data/clinvar_gene_stats.duckdb` | **Not included** — app degrades gracefully |

The app runs without ClinVar DBs. Only the gene index endpoints (`/genes`, `/gene/{symbol}/summary`) return 503 when the DB is absent. The main `/ask` endpoint always works.

---

## Build verification (local, before deploy)

```bash
# Clean install
pip install -r requirements.txt

# Import check
PYTHONUTF8=1 python -c "from app.main import app; print('OK')"

# Full test suite (must pass before any deploy)
PYTHONUTF8=1 python -m pytest tests/ -q
# Expected: 1325 passed, 0 failed
```

---

## What NOT to claim in the demo

- Do **not** claim the system diagnoses or interprets specific genetic variants.
- Do **not** claim it replaces a genetic counselor.
- Do **not** claim the AI-generated draft is clinically verified.
- Do **not** claim the ClinVar data is real-time or fully comprehensive.
- The system provides **general educational information** to patients who have already seen a counselor.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `500` on `/ask` | Import error at startup | Check build logs; run import check above |
| `503` on `/genes` | ClinVar DB missing | Expected — not needed for staging |
| `401` on `/app` | Basic auth enabled | Pass `--auth user:pass` to smoke test |
| `health/llm` shows `configured: false` | Missing `OPENAI_API_KEY` | Add key in platform dashboard |
| Draft returns `None` | LLM call failed or validation rejected | Check app logs for `Unverified gene draft` lines |
| App shows blank page | Static file missing | Confirm `app/static/index.html` exists |
