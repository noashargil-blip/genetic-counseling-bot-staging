# Handoff Document — Genetic Counseling Assistant (Beta)

**For:** Academic supervisors, department review committee, testing group  
**Purpose:** Non-technical overview of what the system does, what it does not do, and how to continue operating it after handoff.

---

## What this system does

This is a Hebrew-language chatbot designed for patients who have **already met with a genetic counselor**. It answers general educational questions about:

- What is a VUS (Variant of Uncertain Significance)?
- What is carrier status?
- What does it mean for a genetic variant to be pathogenic vs. benign?
- What inheritance patterns exist?
- What is known about a specific gene (e.g. BRCA1, HBB, POLE)?
- What questions should I ask my genetics team?

All **core answers are pre-written, reviewed, and deterministic**. They do not change between users and cannot be altered by the user or the AI.

---

## What this system does NOT do

| Not allowed | Why |
|---|---|
| Diagnose any condition | Safety boundary — explicitly blocked |
| Interpret a specific personal variant | Safety boundary — explicitly blocked |
| Recommend surgery, medication, or surveillance | Safety boundary — explicitly blocked |
| Estimate personal risk | Safety boundary — explicitly blocked |
| Recommend pregnancy termination or continuation | Safety boundary — explicitly blocked |
| Store user conversations | Privacy — no server-side persistence |
| Accept uploaded genetic reports in staging | Disabled (`DISABLE_UPLOADS=true`) |
| Accept name, ID number, phone, or email | Blocked by privacy classifier in every request |

If a user tries to enter identifying information, the system blocks the response and asks them not to share it.

---

## Current safety boundaries

The system has a multi-layer safety pipeline:

1. **Identifying information check** — blocks if the message contains a name phrase, Israeli ID number, phone, or email.
2. **Personal interpretation check** — redirects to genetic counselor if the user asks what a specific result means for them personally.
3. **Medical action check** — refuses to recommend surgery, medication, termination, or screening.
4. **LLM phrasing layer** — if used, the AI is constrained to rephrase only the pre-approved KB answer. The raw approved text is always returned if the AI fails validation.

---

## Why file uploads are disabled in staging

Parsing and interpreting uploaded genetic report files requires:
- A real ClinVar database snapshot (large file, not shipped with the app)
- Additional safety validation for arbitrary file formats
- Clinical review of the parsed output before showing to patients

These are all available in the university server environment but are not appropriate for an unchecked public staging deployment. Upload features will remain disabled (`DISABLE_UPLOADS=true`) until a clinical review process is in place.

---

## What AI is used for

The AI (OpenAI `gpt-4o-mini`) is used **only** as a phrasing layer for two purposes:

1. **Rephrasing pre-approved KB answers** — the approved text is sent to the AI with instructions to rephrase it in natural conversational Hebrew. If the rephrasing fails quality or safety checks, the original approved text is shown instead.

2. **Generating an unverified gene background note** for "Tier 2" genes (genes not yet in the curated gene card list). This appears in a clearly labeled "experimental, unreviewed" section and is never mixed with the main approved answer. It is always labeled as AI-generated and unreviewed.

**The AI does not access the internet, does not search PubMed, and does not invent clinical facts.**

---

## What content is curated vs. AI-assisted

| Content type | Curated? | AI-assisted? |
|---|---|---|
| General genetic concept answers (VUS, carrier, inheritance, etc.) | ✓ Curated KB | Optional phrasing layer only |
| Tier 1 gene cards (BRCA1, BRCA2, NF1, etc.) | ✓ Curated KB | Optional phrasing layer only |
| Tier 2 gene background note | ✗ Not curated | ✓ AI-generated, labeled as unreviewed |
| Suggested follow-up questions | ✓ Curated | No AI |
| Safety redirects | ✓ Curated | No AI |

---

## What still requires physician or genetic counselor approval

- **Tier 2 gene background notes**: The AI generates a short general biological explanation. Each note is stored in a review queue (`app/data/draft_review_queue.json`) and can be approved by a genetic counselor using the CLI tool (`scripts/review_drafts.py`). Approved notes move to `app/data/gene_knowledge_base.json` and become deterministic (no more AI for that gene).

- **Any new KB content**: All additions to `app/data/genetic_counseling_kb.json` require manual review. The file format is documented in `app/data/`.

---

## How to add future approved gene summaries

```bash
# List drafts pending review
python scripts/review_drafts.py --list

# Approve a draft (moves it to gene_knowledge_base.json)
python scripts/review_drafts.py --approve <draft-id> --reviewer "Dr. Name" --confirm

# Reject a draft
python scripts/review_drafts.py --reject <draft-id> --reason "Reason"
```

Approved summaries are then served deterministically — no AI call needed for that gene.

---

## How to continue funding / hosting

### Render (free tier)
- Free tier spins down after 15 min of inactivity and has cold-start latency (~30 s).
- Upgrading to Starter ($7/month) keeps the service always-on.
- No other infrastructure costs are required for the app itself.

### OpenAI API costs
- At `gpt-4o-mini` pricing (approximately $0.15/1M input tokens, $0.60/1M output tokens), a conversation generating one AI draft costs roughly $0.001–$0.003.
- The main KB answers do not use the AI at all.
- Set a monthly spending limit in the OpenAI dashboard to cap costs.

### To completely disable OpenAI
Set `LLM_PROVIDER=none` (or remove `OPENAI_API_KEY`). The app falls back entirely to deterministic answers and does not generate AI gene drafts. The core functionality is unaffected.

---

## How to use local university LLM mode

If the university Slurm compute node is available:

1. Start the local LLM wrapper (see `run_local_llm_server_envfixed.sbatch`).
2. Set `LOCAL_LLM_URL=http://localhost:<port>` on the server.
3. Remove `OPENAI_API_KEY` from the environment.
4. The app auto-selects the local LLM.

This avoids OpenAI API costs entirely. Performance depends on the local model and GPU allocation.

---

## How to verify the system is running correctly (for non-technical supervisors)

1. Open the staging URL in a browser.
2. Accept the privacy notice.
3. Type: **"מה זה VUS?"** — should get a clear Hebrew explanation without claiming what your specific result means.
4. Type: **"האם לעשות ניתוח בגלל המוטציה?"** — should be redirected to the genetics team, not give a recommendation.
5. Type: **"קוראים לי שרה"** — should block the message with a privacy notice.
6. Type: **"מה ידוע על הגן HBB?"** — should get a general biological explanation.

If all four respond as described, the system is working correctly.

---

## Known limitations

- The curated knowledge base covers general genetic concepts and a small number of well-known genes (BRCA1, BRCA2, NF1, TP53, ATM, CHEK2, MLH1, MSH2, MSH6, PMS2, APC, PTEN, VHL, RB1). For other genes, the system falls back to a general explanation.
- The system is in Hebrew only; English questions receive a redirect to speak in Hebrew.
- ClinVar data is from the university server snapshot and is not updated in real time in the cloud staging environment.
- The AI-generated gene draft is labeled as unreviewed and should not be cited as clinical information.
- The system is in beta and has not been clinically validated as a standalone patient tool.

---

## Repository structure (for technical handoff)

```
app/
  main.py              — FastAPI app, endpoints, auth middleware
  counseling_engine.py — 8-step answer pipeline, all chatbot logic
  kb.py                — KB loading and fuzzy matching
  safety.py            — Privacy and safety classifiers
  llm_client.py        — LLM provider abstraction (OpenAI, local, Anthropic)
  data/
    genetic_counseling_kb.json   — Core curated KB
    gene_knowledge_base.json     — Approved gene cards
    draft_review_queue.json      — AI draft review queue
  static/              — Hebrew RTL frontend (index.html, app.js, styles.css)
scripts/
  review_drafts.py     — Human review CLI for AI gene drafts
  smoke_staging.py     — Post-deploy smoke test
docs/
  final_launch_checklist.md   — Deployment steps and verification
  handoff_for_supervisors.md  — This file
tests/                 — 1325 automated tests (all passing)
requirements.txt       — Cloud runtime dependencies only
requirements-llm.txt   — Local/HuggingFace/Anthropic extras
.env.example           — Template for environment variables (no secrets)
```
