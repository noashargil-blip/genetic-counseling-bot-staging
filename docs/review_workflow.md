# Physician / Genetic Counselor Review Workflow

This document describes how gene-level drafts are reviewed and promoted to Tier 1b gene knowledge.

---

## What is the review queue?

The review queue (`app/data/draft_review_queue.json`) holds gene-level drafts that need
professional review before they can be used as patient-facing content.

Drafts reach the queue either from:
- `scripts/seed_first_gene_review_drafts.py` — curated drafts seeded by the development team
- `enqueue_gene_draft_for_review()` — called programmatically when the chatbot generates an unverified summary

Only **reusable gene-level knowledge records** are enqueued — never one-off patient chat answers.

Three draft types are accepted:

| `draft_type` | Written to KB field when approved |
|---|---|
| `gene_summary` | `patient_summary_he` |
| `vus_note` | `patient_summary_he` |
| `clinvar_context_summary` | `approved_context_summary_he` |

`clinvar_context_summary` is kept in a separate field so it is never silently treated as a full biological gene summary.

---

## Safety rules (enforced automatically)

- **No auto-approval.** `approved` starts as `False` and is set to `True` only by explicit human action via this CLI.
- **No patient-identifying data.** The enqueue helper rejects text containing Israeli ID numbers, email addresses, or name phrases.
- **No raw patient questions.** The queue record never stores the user's original question.
- **Rejected drafts stay rejected.** They are never promoted to KB content.
- **Already-approved drafts require `--force` to re-approve or edit.** This is always announced with a visible warning.
- **`--force` always prints a warning** — it is never silent.
- **Archived drafts are preserved** for audit but removed from the default pending list.

---

## CLI usage

Run from the project root (requires the project virtualenv):

```bash
python scripts/review_drafts.py --help
```

### List drafts

```bash
# Default: shows only pending (needs_review, draft)
python scripts/review_drafts.py --list

# All statuses including approved, rejected, archived
python scripts/review_drafts.py --list --status all

# Specific status filters
python scripts/review_drafts.py --list --status approved
python scripts/review_drafts.py --list --status rejected
python scripts/review_drafts.py --list --status archived
```

### Preview a draft

```bash
python scripts/review_drafts.py --preview <DRAFT_ID>
```

Shows the full draft text, metadata, source information, and review history.

### Approve as-is

```bash
python scripts/review_drafts.py --approve <DRAFT_ID> \
  --reviewer "Dr. Cohen" \
  --confirm \
  [--notes "Optional notes"]
```

Both `--reviewer` and `--confirm` are required. The approved text is written to `gene_knowledge_base.json` immediately.

### Edit, then approve

1. Copy the draft text from `--preview` to a plain text file.
2. Edit the file as needed.
3. Run:

```bash
python scripts/review_drafts.py --edit <DRAFT_ID> \
  --from-file /path/to/edited_draft.txt \
  --reviewer "Dr. Cohen" \
  --confirm \
  [--notes "Corrected phrasing for patient audience"]
```

The original AI-generated text is preserved in `original_text_he` for audit purposes.
The approved text written to the KB is the edited version.

**Editing an already-approved draft requires `--force`** — this prints a visible warning.

### Reject a draft

```bash
python scripts/review_drafts.py --reject <DRAFT_ID> \
  --reason "Phenotype list is inaccurate" \
  [--reviewer "Dr. Cohen"]
```

`--reason` is required. Rejected drafts remain in the queue and are never written to `gene_knowledge_base.json`.

### Archive a draft (technical / test artifacts)

Use `--archive` to remove a draft from the pending list without deleting it.
This is appropriate for:
- Technical workflow test records
- Superseded drafts replaced by a newer version
- Any draft that should not be reviewed but must be kept for audit

```bash
python scripts/review_drafts.py --archive <DRAFT_ID> \
  --reason "Technical workflow test — not real content" \
  [--reviewer "Your name"]
```

`--reason` is required. Archived drafts:
- Are removed from the default `--list` (pending) view
- Remain visible with `--list --status archived` and `--list --status all`
- Are kept permanently in the queue file for audit
- Have `approved=false` — archived content is never served as patient-facing knowledge

---

## Seeding the first gene review batch

The seed script enqueues pre-written drafts for the first batch of genes.
These are **drafts only** — `approved=false` for all.

```bash
# Dry run — see what would be enqueued
python scripts/seed_first_gene_review_drafts.py --dry-run

# Enqueue (idempotent — safe to run multiple times)
python scripts/seed_first_gene_review_drafts.py
```

Genes in the first batch: BRCA1, BRCA2, APC, POLE, HBB

After seeding, verify with:
```bash
python scripts/review_drafts.py --list
```

---

## What gets written to Gene Knowledge

Approved content is written to `app/data/gene_knowledge_base.json`. The gene record is updated (or created) with:

- The approved text in the correct field (`patient_summary_he` or `approved_context_summary_he`)
- `approved: true`
- `reviewed_by`, `reviewed_at`, `reviewer_notes`
- Any existing fields on the record are **preserved** — approving a `clinvar_context_summary` draft does not overwrite an existing `patient_summary_he`

Records with `approved: false` are never served as patient-facing answers, even if they have content.

---

## Warning about seeded drafts

**Seeded drafts are not approved medical content.**

Even though the draft text is patient-friendly and based on established sources
(MedlinePlus, NCBI Gene), it has not been reviewed by a physician or genetic
counselor. Every draft must be explicitly approved via `--approve` or `--edit`
before it becomes Tier 1b Gene Knowledge.

Running the seed script twice does not create duplicates (deduplicated by
gene_symbol + draft_type + content_hash).

---

## Workflow summary

```
scripts/seed_first_gene_review_drafts.py
       │                     └── or: enqueue_gene_draft_for_review()
       ▼
draft_review_queue.json  (review_status: "needs_review", approved: false)
       │
       ├── --approve  → gene_knowledge_base.json (approved: true)
       ├── --edit     → gene_knowledge_base.json (approved: true, original preserved)
       ├── --reject   → stays in queue (approved: false, never promoted)
       └── --archive  → stays in queue (approved: false, hidden from pending list)
```

Human approval is the **only** path from draft to Tier 1b gene knowledge.
