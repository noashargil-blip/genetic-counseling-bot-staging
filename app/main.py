"""
Genetic Counseling Bot — FastAPI application.

Primary product — Hebrew post-genetic-counseling assistant:
  GET  /                            → health + product description
  GET  /topics                      → available knowledge-base topics
  GET  /faq                         → FAQ-style knowledge-base entries
  POST /ask                         → KB-grounded Q&A with a safety classifier
                                       (identifying-info / personal-interpretation
                                       guardrails) and optional local-LLM phrasing

Legacy ClinVar variant lookup (kept in code, not exposed in the current UI):
  GET  /gene?symbol=BRCA1           → top variants for a gene
  GET  /search?q=...                → free-text search across variants
  GET  /summary?gene=BRCA1          → aggregate stats for a gene
  POST /clinvar/ask                 → LLM-powered Q&A with evidence + safety guardrails
  POST /analyze-report-json         → parse structured clinical report JSON + ClinVar matching
"""

import base64
import logging
import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field, model_serializer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from app import retriever, responder, policy, session_store, kb, counseling_engine, gene_index
from app import health as _health_module
from app import feedback as _feedback_module
from app.upload_parser import parse_uploaded_file
from app.report_parser import parse_structured_report_json
from app.llm_client import (
    BaseLLMClient,
    LLMClientError,
    LLMRateLimitError,
    create_llm_client,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="מלווה לאחר ייעוץ גנטי — Post-Genetic-Counseling Assistant",
    description=(
        "Hebrew assistant that helps patients understand general genetic concepts "
        "after they have already met with a genetic counselor. It does not interpret "
        "personal genetic results, does not diagnose, does not calculate personal risk, "
        "and does not replace a genetic counselor."
    ),
    version=_health_module.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional basic auth middleware for staging
# Set BASIC_AUTH_ENABLED=true to activate. /health/* is always exempt.
# Credentials are read from env vars only — never hardcoded or logged.
# ---------------------------------------------------------------------------
_BASIC_AUTH_ENABLED = os.environ.get("BASIC_AUTH_ENABLED", "").strip().lower() in ("true", "1", "yes")

if _BASIC_AUTH_ENABLED:
    _AUTH_USERNAME = os.environ.get("BASIC_AUTH_USERNAME", "staging")
    _AUTH_PASSWORD = os.environ.get("BASIC_AUTH_PASSWORD", "")
    _AUTH_REALM = "Genetic Counseling Bot — Staging"

    class _BasicAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            if request.url.path.startswith("/health"):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Basic "):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": f'Basic realm="{_AUTH_REALM}"'},
                    content="Authentication required.",
                )
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                username, _, password = decoded.partition(":")
            except Exception:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": f'Basic realm="{_AUTH_REALM}"'},
                    content="Authentication required.",
                )
            if username != _AUTH_USERNAME or password != _AUTH_PASSWORD:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": f'Basic realm="{_AUTH_REALM}"'},
                    content="Authentication required.",
                )
            return await call_next(request)

    app.add_middleware(_BasicAuthMiddleware)
    logger.info("Basic auth middleware enabled for staging.")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Gate upload endpoints with DISABLE_UPLOADS=true (recommended for staging)
_UPLOADS_DISABLED = os.environ.get("DISABLE_UPLOADS", "").strip().lower() in ("true", "1", "yes")


@app.get("/demo", include_in_schema=False)
def demo():
    """Serve the static demo UI (legacy alias for /app)."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/app")
def serve_app():
    """Serve the Hebrew RTL post-genetic-counseling assistant frontend."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# LLM client (optional — falls back to deterministic responder if absent)
# ---------------------------------------------------------------------------
_llm: Optional[BaseLLMClient] = None

try:
    _llm = create_llm_client()
except ValueError:
    logger.warning(
        "No LLM configured (LOCAL_LLM_URL, ANTHROPIC_API_KEY, or OPENAI_API_KEY not set). "
        "LLM layer disabled — deterministic responder will be used for all /ask endpoints."
    )



# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ClinVarAskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="User question")
    gene: Optional[str] = Field(None, description="Optional gene symbol filter (e.g. BRCA1)")
    variant: Optional[str] = Field(None, description="Optional variant name filter")
    condition: Optional[str] = Field(None, description="Optional condition filter")


class ClinVarAskResponse(BaseModel):
    answer: str
    evidence: list[str]
    limitations: list[str]
    safety_disclaimer: str
    llm_used: bool = Field(description="Whether the LLM layer was used for this response")
    policy_flags: list[str] = Field(
        default_factory=list,
        description="Safety policy flags raised for this question",
    )
    redirect_message: Optional[str] = Field(
        None, description="Policy redirect message when clinical guidance was requested"
    )


# ---------------------------------------------------------------------------
# New product: post-genetic-counseling assistant schemas
# ---------------------------------------------------------------------------

class ConversationContextMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message text")
    matched_topic: Optional[str] = Field(
        None, description="For assistant messages, the matched_topic of that turn, if any"
    )


class CounselingAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User question, in Hebrew")
    topic: Optional[str] = Field(
        None, description="Optional knowledge-base topic id (see GET /topics) to scope the answer"
    )
    conversation_context: Optional[List[ConversationContextMessage]] = Field(
        None,
        description=(
            "Optional recent conversation turns (e.g. last 6 messages) kept in the "
            "frontend's in-memory session only — never persisted server-side. "
            "Used to resolve vague follow-up questions ('can you elaborate?')."
        ),
    )
    last_topic: Optional[str] = Field(
        None, description="Optional matched_topic of the previous assistant turn, for follow-up resolution"
    )
    include_unverified_gene_draft: bool = Field(
        False,
        description=(
            "Opt-in: when True, for Tier 2 genes (no approved Gene Card) the response may "
            "include an unverified AI-generated gene background draft in `unverified_gene_draft`. "
            "Default is False — no draft is generated unless explicitly requested."
        ),
    )


class CounselingAskResponse(BaseModel):
    answer: str
    safety_level: str = Field(
        description="general_information | contains_identifying_info | requires_genetic_counselor | out_of_scope"
    )
    needs_genetic_counselor: bool
    matched_topic: Optional[str] = None
    suggested_questions: list[str] = Field(default_factory=list)
    llm_used: bool = False
    fallback_used: bool = True
    llm_mode: str = "none"
    llm_attempted: Optional[bool] = None
    llm_rejected_reason: Optional[str] = None
    llm_repaired: Optional[bool] = None
    llm_repair_reason: Optional[str] = None
    llm_retry_used: Optional[bool] = None
    gene_metadata: Optional[dict] = Field(
        None,
        description=(
            "Present only for gene-level ClinVar summary responses "
            "(matched_topic == 'gene_clinvar_summary'). "
            "Contains gene_symbol, data_source, llm_used, fallback_used, total_variants, "
            "answer_tier, gene_knowledge_status, unverified_gene_draft_available."
        ),
    )
    unverified_gene_draft: Optional[dict] = Field(
        None,
        description=(
            "Present only when include_unverified_gene_draft=True was sent, the gene is Tier 2, "
            "and the LLM successfully generated a valid draft. "
            "Contains visible, status, gene_symbol, warning_he, text_he, generated_by_model, "
            "review_status, approved, generated_at. "
            "approved is always False. review_status is always 'unreviewed'. "
            "This field is absent (not null) when no draft was generated."
        ),
    )
    ai_draft_debug: Optional[dict] = Field(
        None,
        description=(
            "Present only when a tier2 gene draft was attempted but failed. "
            "Contains safe diagnostic keys: attempted, provider, generated, "
            "validation_passed, rejection_code. Never contains API keys or prompts."
        ),
    )

    @model_serializer
    def _serialize(self) -> dict:
        """
        Serialize the response. gene_metadata and debug LLM fields are
        included only when present. llm_used, fallback_used, and llm_mode
        are always included.
        """
        out: dict = {
            "answer": self.answer,
            "safety_level": self.safety_level,
            "needs_genetic_counselor": self.needs_genetic_counselor,
            "matched_topic": self.matched_topic,
            "suggested_questions": self.suggested_questions,
            "llm_used": self.llm_used,
            "fallback_used": self.fallback_used,
            "llm_mode": self.llm_mode,
        }
        if self.llm_attempted is not None:
            out["llm_attempted"] = self.llm_attempted
        if self.llm_rejected_reason is not None:
            out["llm_rejected_reason"] = self.llm_rejected_reason
        if self.llm_retry_used is not None:
            out["llm_retry_used"] = self.llm_retry_used
        if self.llm_repaired is not None:
            out["llm_repaired"] = self.llm_repaired
        if self.llm_repair_reason is not None:
            out["llm_repair_reason"] = self.llm_repair_reason
        if self.gene_metadata is not None:
            out["gene_metadata"] = self.gene_metadata
        if self.unverified_gene_draft is not None:
            out["unverified_gene_draft"] = self.unverified_gene_draft
        if self.ai_draft_debug is not None:
            out["ai_draft_debug"] = self.ai_draft_debug
        return out


class FeedbackRequest(BaseModel):
    helpful: bool = Field(..., description="Whether the answer was helpful")
    reason: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional short reason for the rating",
    )
    matched_topic: Optional[str] = Field(
        None, description="matched_topic from the /ask response being rated"
    )
    safety_level: Optional[str] = Field(
        None, description="safety_level from the /ask response being rated"
    )
    question_length: Optional[int] = Field(
        None,
        ge=0,
        description="Character length of the question (NOT the question text itself)",
    )


class FeedbackResponse(BaseModel):
    feedback_id: str
    recorded: bool


class AskUploadRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000,
        description="Educational question about the uploaded variant",
    )
    uploaded_variant: dict = Field(
        ..., description="Normalized variant record from /upload or /analyze-upload"
    )
    clinvar_result: dict = Field(
        ..., description="ClinVar match result from /analyze-upload"
    )
    use_llm: bool = Field(
        True,
        description=(
            "Set to false to force the deterministic responder even when an LLM is configured. "
            "When true (default), the LLM is used if an API key is available."
        ),
    )


class AskSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /analyze-upload")
    variant_index: int = Field(
        ..., ge=0, description="Zero-based index into the session's variant list"
    )
    question: str = Field(
        ..., min_length=3, max_length=1000,
        description="Educational question about the selected variant",
    )
    use_llm: bool = Field(
        True,
        description=(
            "Set to false to force the deterministic responder even when an LLM is configured."
        ),
    )


class AnalyzeReportJsonRequest(BaseModel):
    report: dict = Field(
        ...,
        description=(
            "Structured clinical genetic report as a JSON object. "
            "Expected top-level keys: 'variant' (object), optional 'phenotypes_hpo' "
            "(list of {phenotype, hpo_id}), optional 'gene_summary' (string)."
        ),
    )


class AskReportSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /analyze-upload or /analyze-report-json")
    question: str   = Field(..., min_length=3, max_length=1000,
                            description="Educational question about the whole report")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """Redirect browser visitors to the Hebrew chat UI."""
    return RedirectResponse(url="/app", status_code=302)


@app.get("/topics")
def get_topics():
    """Return the available knowledge-base topics."""
    return {"topics": kb.list_topics()}


@app.get("/faq")
def get_faq():
    """Return FAQ-style entries (approved answers) from the knowledge base."""
    return {"faq": kb.list_faq()}


@app.post("/ask", response_model=CounselingAskResponse)
def ask(request: CounselingAskRequest):
    """
    Answer a general post-genetic-counseling question.

    1. Runs the safety classifier (identifying info / personal interpretation
       / specific-variant detection).
    2. Resolves vague follow-up questions ("can you elaborate?") using
       last_topic / conversation_context, if provided.
    3. Looks up the best matching approved knowledge-base topic (exact
       keyword match, with a fuzzy fallback tier).
    4. Optionally phrases the answer via a local LLM, constrained to the
       matched KB content (falls back to deterministic KB text if no LLM
       is configured or the call fails).

    conversation_context and last_topic are supplied fresh by the caller on
    every request (the frontend's in-memory session) — nothing is stored or
    persisted server-side. No identifying information is ever forwarded to
    the LLM, even if it appears in conversation_context.
    """
    context = (
        [m.model_dump() for m in request.conversation_context]
        if request.conversation_context
        else None
    )
    result = counseling_engine.answer_question(
        request.question,
        topic=request.topic,
        conversation_context=context,
        last_topic=request.last_topic,
        include_unverified_gene_draft=request.include_unverified_gene_draft,
    )
    return CounselingAskResponse(**result)


# ---------------------------------------------------------------------------
# Health and version endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health():
    """
    Overall system health — returns 200 even when components are degraded.

    `status` values:
    - "ok"       — all core components are available
    - "degraded" — one or more non-critical components are unavailable
                   (the assistant still works using deterministic KB answers)
    - "down"     — critical failure (currently not used; assistant degrades gracefully)
    """
    return _health_module.get_overall_health()


@app.get("/health/clinvar", tags=["health"])
def health_clinvar():
    """Check ClinVar database availability and record count."""
    return _health_module.check_clinvar()


@app.get("/health/gene-index", tags=["health"])
def health_gene_index():
    """Check gene-index availability and indexed gene count."""
    return _health_module.check_gene_index()


@app.get("/health/llm", tags=["health"])
def health_llm():
    """
    Check LLM provider configuration and — for local endpoints — probe reachability.

    Never exposes API keys or secrets.
    For OpenAI/Anthropic/HuggingFace, reachability is not probed (expensive / rate-limited).
    """
    result = dict(_health_module.check_llm())
    # Only probe reachability for local university endpoint (cheap HEAD request)
    provider = result.get("provider")
    if provider == "local" and result.get("configured"):
        local_url = os.environ.get("LOCAL_LLM_URL", "").strip()
        if local_url:
            result["reachable"] = _health_module.probe_llm_live(local_url)
    return result


@app.get("/version", tags=["health"])
def version():
    """
    Return full version and capability status for the running instance.

    Fields
    ------
    app_version                  : semantic version of this codebase
    safety_policy_version        : version of SAFETY_POLICY.md / safety.py rules
    llm_policy_version           : version of LLM_POLICY.md — governs how the LLM may be used
    data_version                 : metadata from data/data_version.json (ClinVar build date, counts, etc.)
    clinvar_available            : True when data/clinvar.duckdb is readable
    gene_index_available         : True when data/clinvar_gene_stats.duckdb is built and readable
    gene_cards_available         : True when approved gene cards are loaded (data/gene_cards.json or built-in)
    llm_available                : True when an LLM is configured (LOCAL_LLM_URL / API key)
    llm_role                     : human-readable description of the LLM's permitted role
    answer_tiers_available       : True — three-tier gene answer system is active
    deterministic_fallback_available : always True — KB-grounded answers never require an LLM
    """
    from app import gene_cards as _gc  # local import to avoid circular dep at module level
    clinvar  = _health_module.check_clinvar()
    gene_idx = _health_module.check_gene_index()
    llm      = _health_module.check_llm()
    return {
        "app_version":                    _health_module.APP_VERSION,
        "safety_policy_version":          _health_module.SAFETY_POLICY_VERSION,
        "llm_policy_version":             _health_module.LLM_POLICY_VERSION,
        "data_version":                   _health_module.load_data_version(),
        "clinvar_available":              clinvar["ok"],
        "gene_index_available":           gene_idx["ok"],
        "gene_cards_available":           _gc.CARDS_AVAILABLE,
        "llm_available":                  llm["ok"],
        "llm_role":                       "intro-only tone layer — validated Hebrew sentence prepended to deterministic content",
        "answer_tiers_available":         True,
        "deterministic_fallback_available": True,
    }


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------

@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def submit_feedback(request: FeedbackRequest):
    """
    Record a helpful / not-helpful signal for a previous /ask response.

    PRIVACY: No question text, answer text, or user-identifiable information
    is stored.  Only the boolean signal, optional short reason, topic category,
    safety level, and character length of the question are recorded.

    Feedback is appended to logs/feedback.jsonl on the server.
    """
    feedback_id = _feedback_module.record(
        helpful=request.helpful,
        reason=request.reason,
        matched_topic=request.matched_topic,
        safety_level=request.safety_level,
        question_length=request.question_length,
    )
    return FeedbackResponse(feedback_id=feedback_id, recorded=True)


# ---------------------------------------------------------------------------
# Gene-level ClinVar index endpoints
# ---------------------------------------------------------------------------

def _gene_unavailable():
    """Raise 503 when the gene index has not been built."""
    raise HTTPException(
        status_code=503,
        detail={
            "error": "Gene index not available.",
            "reason": (
                "The ClinVar gene index (data/clinvar_gene_stats.duckdb) has not "
                "been built — either data/clinvar.duckdb is missing or the index "
                "build failed at startup.  Check server logs for details."
            ),
        },
    )


@app.get("/genes")
def get_genes(
    limit: int = Query(200, ge=1, le=2000, description="Max genes to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Return all unique gene symbols indexed from ClinVar, with total variant counts.

    Genes are sorted by total variant count (descending).  Use `limit` and
    `offset` for pagination across the full gene list.
    """
    if not gene_index._GENE_INDEX_AVAILABLE:
        _gene_unavailable()

    genes = gene_index.list_genes(limit=limit, offset=offset)
    total = gene_index.count_genes()
    return {
        "total_genes": total,
        "returned": len(genes),
        "offset": offset,
        "limit": limit,
        "genes": genes,
        "metadata": gene_index.METADATA,
    }


@app.get("/gene/{gene_symbol}/summary")
def get_gene_summary(gene_symbol: str):
    """
    Return aggregated ClinVar statistics for a single gene.

    Statistics include:
    - total_variants: count of all ClinVar records for this gene
    - by_significance: counts grouped by clinical significance
    - by_review_status: counts grouped by review status
    - phenotypes: top associated conditions (from phenotype_list)
    - variant_types: counts grouped by variant type
    - date_range: earliest and latest last_evaluated dates

    All medical-safety disclaimers are included in the `metadata` field.
    """
    if not gene_index._GENE_INDEX_AVAILABLE:
        _gene_unavailable()

    summary = gene_index.get_gene_summary(gene_symbol)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Gene '{gene_symbol.upper()}' was not found in the ClinVar index.",
        )
    return {
        "gene_symbol": summary["gene_symbol"],
        "statistics":  {
            "total_variants":   summary["total_variants"],
            "by_significance":  summary["by_significance"],
            "by_review_status": summary["by_review_status"],
            "phenotypes":       summary["phenotypes"],
            "variant_types":    summary["variant_types"],
            "date_range":       summary["date_range"],
        },
        "index_built_at": summary["index_built_at"],
        "metadata":       gene_index.METADATA,
    }


@app.get("/gene/{gene_symbol}/variants")
def get_gene_variants(
    gene_symbol: str,
    limit: int = Query(20, ge=1, le=200, description="Max variants to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    significance: Optional[str] = Query(
        None,
        description=(
            "Filter by clinical significance substring, e.g. 'Pathogenic', 'Benign', 'Uncertain'."
        ),
    ),
):
    """
    Return individual ClinVar variant records for a gene.

    Records are fetched from the live ClinVar snapshot (not the cached stats
    index) so they always reflect the current database state.

    Use `significance` to filter by clinical significance (substring match).
    Use `limit` and `offset` for pagination.  The `total_variants` for the
    unfiltered gene is available from GET /gene/{gene_symbol}/summary.

    All medical-safety disclaimers are included in the `metadata` field.
    """
    if not gene_index._GENE_INDEX_AVAILABLE:
        _gene_unavailable()

    gene_upper = gene_symbol.strip().upper()

    # Confirm the gene exists in the index before hitting the main DB
    if gene_index.get_gene_summary(gene_upper) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Gene '{gene_upper}' was not found in the ClinVar index.",
        )

    variants = gene_index.get_gene_variants(
        gene_upper,
        limit=limit,
        offset=offset,
        significance=significance,
    )
    return {
        "gene_symbol": gene_upper,
        "returned":    len(variants),
        "offset":      offset,
        "limit":       limit,
        "significance_filter": significance,
        "variants":    variants,
        "metadata":    gene_index.METADATA,
    }


@app.get("/gene", include_in_schema=False)
def get_gene(
    symbol: str = Query(..., description="Gene symbol, e.g. BRCA1"),
    limit: int = Query(10, ge=1, le=50),
):
    """Return top ClinVar variants for a gene, ordered by clinical significance."""
    records = retriever.retrieve_by_gene(symbol, limit=limit)
    if not records:
        raise HTTPException(status_code=404, detail=f"No variants found for gene: {symbol}")
    return {"gene": symbol.upper(), "count": len(records), "variants": records}


@app.get("/search", include_in_schema=False)
def search_variants(
    q: str = Query(..., description="Free-text search term"),
    gene: Optional[str] = Query(None, description="Optional gene symbol filter"),
    significance: Optional[str] = Query(None, description="Optional significance filter"),
    limit: int = Query(10, ge=1, le=50),
):
    """Search ClinVar variants by name, gene, or condition."""
    records = retriever.search(q, gene=gene, significance=significance, limit=limit)
    return {"query": q, "count": len(records), "variants": records}


@app.get("/summary", include_in_schema=False)
def get_summary(gene: str = Query(..., description="Gene symbol")):
    """Return aggregate variant statistics for a gene."""
    stats = retriever.get_summary(gene)
    if not stats or stats.get("total", 0) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for gene: {gene}")
    return {"gene": gene.upper(), "statistics": stats}


@app.post("/clinvar/ask", response_model=ClinVarAskResponse, include_in_schema=False)
def clinvar_ask(request: ClinVarAskRequest):
    """
    [Legacy — not exposed in the current UI] Answer a genetic variant
    question using ClinVar evidence.

    1. Sanitises and policy-checks the question.
    2. Retrieves relevant ClinVar records.
    3. Passes evidence to the LLM for structured response generation.
    4. Falls back to deterministic responder if LLM is unavailable.
    5. Enforces safety disclaimer unconditionally.
    """
    # 1. Sanitise
    question = policy.sanitise_question(request.question)

    # 2. Policy check
    pol = policy.check_question(question)
    # (We always proceed — policy.allowed is True unless future rules block)

    # 3. Retrieve evidence
    evidence_records: list[dict] = []

    if request.variant:
        evidence_records = retriever.retrieve_by_variant(request.variant)
    elif request.gene:
        evidence_records = retriever.retrieve_by_gene(request.gene)
    elif request.condition:
        evidence_records = retriever.retrieve_by_condition(request.condition)
    else:
        # Fall back to free-text search across all fields
        evidence_records = retriever.search(question)

    # 4. Generate response
    llm_used = False
    if _llm is not None:
        try:
            result = _llm.ask(question, evidence_records)
            llm_used = True
            logger.info("LLM response generated for question: %s", question[:80])
        except LLMRateLimitError as exc:
            logger.warning("LLM rate limited (%s); falling back to deterministic responder.", exc)
            result = responder.build_response(question, evidence_records)
        except LLMClientError as exc:
            logger.warning("LLM failed (%s); falling back to deterministic responder.", exc)
            result = responder.build_response(question, evidence_records)
    else:
        result = responder.build_response(question, evidence_records)

    # 5. Enforce safety disclaimer (unconditional override)
    result = policy.enforce_disclaimer(result)

    return ClinVarAskResponse(
        **result,
        llm_used=llm_used,
        policy_flags=pol.flags,
        redirect_message=pol.redirect_message or None,
    )


@app.post("/upload", include_in_schema=False)
async def upload_file(file: UploadFile = File(...)):
    """
    Parse an uploaded genetic test file (CSV, TSV, or TXT).

    Returns normalized variant records extracted from the file.
    - Does not save the file to disk.
    - Does not create a session.
    - Does not match variants to ClinVar yet.
    """
    if _UPLOADS_DISABLED:
        raise HTTPException(
            status_code=503,
            detail="File uploads are disabled in this environment (DISABLE_UPLOADS=true).",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was provided.")

    raw: bytes = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        result = parse_uploaded_file(file.filename, raw)
    except Exception as exc:
        logger.warning("upload_parser failed for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the uploaded file: {exc}",
        )

    return {"filename": file.filename, **result}


_ANALYZE_DISCLAIMER = (
    "This tool provides educational genetic information only and does not provide "
    "diagnosis, treatment recommendations, or replace consultation with a qualified "
    "genetics professional."
)

_EXACT_CONFIDENCES = frozenset(("exact_clinvar_accession", "exact_rsid", "position_exact"))


def _classify_variant_review(uploaded_variant: dict, clinvar_result: dict) -> dict:
    """
    Assign a non-diagnostic review category to a matched variant.

    Returns {"label": str, "level": "info"|"caution"|"review", "explanation": str}.

    Rules:
    - No risk language, no urgency, no diagnosis, no treatment recommendations.
    - All levels prompt professional consultation.
    - "review" means a specific ClinVar record exists with a significant classification.
    - "caution" means uncertain, conflicting, or low-confidence evidence.
    - "info"    means no match, benign, or low-information state.
    """
    conf    = clinvar_result.get("match_confidence", "no_match")
    matches = clinvar_result.get("matches", [])

    # Collect unique ClinVar significance values (lowercase, deduplicated)
    clinvar_sigs: list[str] = list({
        str(m.get("clinical_significance") or "").strip().lower()
        for m in matches
        if m.get("clinical_significance")
    })
    # Fallback to uploaded significance when no ClinVar records were found
    if not clinvar_sigs:
        uv_sig = str(uploaded_variant.get("clinical_significance") or "").strip().lower()
        if uv_sig:
            clinvar_sigs = [uv_sig]

    if conf in _EXACT_CONFIDENCES:
        has_path     = any(
            "pathogenic" in s and "benign" not in s
            and "uncertain" not in s and "conflicting" not in s
            for s in clinvar_sigs
        )
        has_benign   = any(
            "benign" in s and "pathogenic" not in s
            and "uncertain" not in s and "conflicting" not in s
            for s in clinvar_sigs
        )
        has_vus      = any("uncertain" in s for s in clinvar_sigs)
        has_conflict = (
            any("conflicting" in s for s in clinvar_sigs)
            or (has_path and has_benign)
        )

        if has_conflict:
            return {
                "label": "Variant-specific ClinVar match with conflicting classifications",
                "level": "caution",
                "explanation": (
                    "ClinVar submissions are not fully consistent for this variant. "
                    "A genetics professional should review these conflicting interpretations "
                    "in the context of the full clinical picture."
                ),
            }
        if has_path:
            return {
                "label": "Variant-specific ClinVar match with clinically significant classification",
                "level": "review",
                "explanation": (
                    "This variant has a specific ClinVar match with a pathogenic or likely pathogenic "
                    "classification. This should be reviewed with a genetics professional in the "
                    "context of the complete clinical and family history."
                ),
            }
        if has_vus:
            return {
                "label": "Variant-specific ClinVar match with uncertain significance",
                "level": "caution",
                "explanation": (
                    "Current evidence is insufficient to classify this variant as benign or pathogenic. "
                    "VUS findings should not be used alone for clinical decision-making, "
                    "and the classification may change as more evidence becomes available."
                ),
            }
        if has_benign:
            return {
                "label": "Variant-specific ClinVar match with benign classification",
                "level": "info",
                "explanation": (
                    "ClinVar classifies this variant as benign or likely benign, "
                    "but interpretation still depends on the full clinical context."
                ),
            }
        return {
            "label": "Variant-specific ClinVar match",
            "level": "info",
            "explanation": (
                "A specific ClinVar record was found for this variant. "
                "Review with a genetics professional for clinical context."
            ),
        }

    if conf == "gene_hgvs_partial":
        return {
            "label": "Approximate variant match",
            "level": "caution",
            "explanation": (
                "An approximate match was found based on gene name and partial HGVS notation. "
                "Verify the exact variant against the ClinVar record before drawing conclusions."
            ),
        }

    if conf == "region_overlap":
        return {
            "label": "Broad chromosomal region overlap only",
            "level": "caution",
            "explanation": (
                "Only large structural variants overlap this coordinate. "
                "These records may not represent this specific variant."
            ),
        }

    if conf == "gene_only":
        return {
            "label": "Gene-level match only",
            "level": "caution",
            "explanation": (
                "Records were found for this gene, but no specific variant match was identified. "
                "Gene-level results cannot be used to interpret this particular variant."
            ),
        }

    # no_match or error
    return {
        "label": "No ClinVar match found",
        "level": "info",
        "explanation": (
            "No matching ClinVar record was found using the available identifiers "
            "(accession, rsID, position, or gene symbol). "
            "The variant may not yet be catalogued in ClinVar."
        ),
    }

_SAFETY_DISCLAIMER = (
    "This information is for educational purposes only and does not constitute medical "
    "advice. Please consult a certified genetics professional or genetic counselor for "
    "clinical interpretation and personalized recommendations."
)


def _build_report_summary(variants_out: list[dict]) -> dict:
    """
    Compute aggregate statistics from a list of matched variant entries.
    Classifications are counted once per variant using the best available
    significance value (ClinVar match preferred, uploaded_variant fallback).
    Always sets needs_professional_review=True.
    """
    total = len(variants_out)
    exact = gene_only = no_match_count = 0
    vus = path = benign = conflict = warn_count = 0

    for entry in variants_out:
        uv  = entry.get("uploaded_variant", {})
        cr  = entry.get("clinvar_result",   {})
        conf = cr.get("match_confidence", "no_match")

        if conf in ("exact_clinvar_accession", "exact_rsid", "position_exact"):
            exact += 1
        elif conf == "gene_only":
            gene_only += 1
        elif conf in ("no_match", "error"):
            no_match_count += 1
        # gene_hgvs_partial / region_overlap count neither as exact nor gene_only

        warn_count += len(cr.get("warnings", []))

        # Collect all significance strings for this variant (ClinVar first)
        sigs: list[str] = []
        for m in cr.get("matches", []):
            s = str(m.get("clinical_significance") or "").strip().lower()
            if s:
                sigs.append(s)
        uv_sig = str(uv.get("clinical_significance") or "").strip().lower()
        if uv_sig:
            sigs.append(uv_sig)

        if any("uncertain" in s for s in sigs):
            vus += 1
        if any("pathogenic" in s and "benign" not in s and "uncertain" not in s for s in sigs):
            path += 1
        if any("benign" in s and "pathogenic" not in s and "uncertain" not in s for s in sigs):
            benign += 1
        if any("conflicting" in s for s in sigs):
            conflict += 1

    # review_category level counts
    rc_review  = sum(1 for e in variants_out if e.get("review_category", {}).get("level") == "review")
    rc_caution = sum(1 for e in variants_out if e.get("review_category", {}).get("level") == "caution")
    rc_info    = sum(1 for e in variants_out if e.get("review_category", {}).get("level") == "info")

    return {
        "total_variants":                        total,
        "matched_variants":                      total - no_match_count,
        "exact_matches":                         exact,
        "gene_only_matches":                     gene_only,
        "no_matches":                            no_match_count,
        "vus_count":                             vus,
        "pathogenic_or_likely_pathogenic_count": path,
        "benign_or_likely_benign_count":         benign,
        "conflicting_count":                     conflict,
        "warnings_count":                        warn_count,
        "needs_professional_review":             True,
        "review_category_counts": {
            "review":  rc_review,
            "caution": rc_caution,
            "info":    rc_info,
        },
    }


def _build_report_answer(question: str, session_variants: list[dict]) -> dict:
    """
    Deterministic report-level answer covering all session variants.
    Used as fallback when LLM is unavailable or fails validation.
    """
    lines: list[str] = [f"Question: {question}", ""]
    lines.append(f"This report contains {len(session_variants)} variant(s).")
    lines.append("")

    evidence: list[str] = []

    for i, entry in enumerate(session_variants, 1):
        uv  = entry.get("uploaded_variant", {})
        cr  = entry.get("clinvar_result",   {})
        gene    = str(uv.get("gene")    or "Unknown").strip()
        variant = str(uv.get("variant") or "").strip()
        conf    = cr.get("match_confidence", "no_match")

        lines.append(f"Variant {i}: {gene}" + (f" ({variant})" if variant else ""))
        lines.append(f"  ClinVar match: {conf}")

        matches = cr.get("matches", [])
        if matches:
            sigs = list({str(m.get("clinical_significance", "")).strip()
                         for m in matches if m.get("clinical_significance")})
            statuses = list({str(m.get("review_status", "")).strip()
                             for m in matches if m.get("review_status")})
            if sigs:
                lines.append(f"  ClinVar significance: {', '.join(sigs)}")
                if any("uncertain" in s.lower() for s in sigs):
                    lines.append(
                        "  VUS note: Current evidence is insufficient to classify "
                        "this variant as benign or pathogenic. "
                        "VUS findings must not be used alone for clinical decision-making."
                    )
            if statuses:
                lines.append(f"  Review status: {', '.join(statuses)}")
            ev_parts = []
            if gene:
                ev_parts.append(f"Gene: {gene}")
            if sigs:
                ev_parts.append(f"Significance: {', '.join(sigs)}")
            if ev_parts:
                evidence.append(" | ".join(ev_parts))
        else:
            uv_sig = str(uv.get("clinical_significance") or "").strip()
            if uv_sig:
                lines.append(f"  Reported significance (not verified in ClinVar): {uv_sig}")

        for w in cr.get("warnings", [])[:2]:
            lines.append(f"  Notice: {w}")
        lines.append("")

    lines += [
        "Limitations:",
        "- This summary is based on ClinVar data and may not reflect the most recent evidence.",
        "- Variant interpretation requires full clinical context and expert review.",
        "- Classifications may differ between submitters and may be updated over time.",
        "",
        "Please consult a certified genetics professional or genetic counselor for "
        "interpretation of these findings in the context of your personal and family "
        "medical history.",
    ]

    return {
        "answer": "\n".join(lines),
        "evidence": evidence,
        "limitations": [
            "Based on ClinVar data only; may not reflect the most current evidence.",
            "Variant interpretation requires full clinical context and expert review.",
        ],
        "safety_disclaimer": _SAFETY_DISCLAIMER,
    }


@app.post("/analyze-upload", include_in_schema=False)
async def analyze_upload(file: UploadFile = File(...)):
    """
    Parse an uploaded genetic test file and match each detected variant to ClinVar.

    Flow:
      1. Read file bytes into memory (nothing saved to disk).
      2. Parse with upload_parser → normalized variant dicts.
      3. For each variant, call retriever.match_uploaded_variant().
         A failure on a single variant does not abort the whole request.
      4. Return structured per-variant ClinVar results + safety disclaimer.
    """
    if _UPLOADS_DISABLED:
        raise HTTPException(
            status_code=503,
            detail="File uploads are disabled in this environment (DISABLE_UPLOADS=true).",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was provided.")

    raw: bytes = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        parsed = parse_uploaded_file(file.filename, raw)
    except Exception as exc:
        logger.warning("upload_parser failed for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the uploaded file: {exc}",
        )

    global_warnings: list[str] = []
    parsed_variants: list[dict] = parsed.get("variants", [])

    if not parsed_variants:
        global_warnings.append(
            "No variants were detected in the uploaded file. "
            "Check that the file contains recognized column headers "
            "(e.g. gene, rsid, chromosome, position)."
        )

    variants_out: list[dict] = []
    for variant_record in parsed_variants:
        try:
            clinvar_result = retriever.match_uploaded_variant(variant_record, limit=10)
        except Exception as exc:
            logger.warning(
                "match_uploaded_variant failed for record %s: %s", variant_record, exc
            )
            clinvar_result = {
                "query_used": "error",
                "match_confidence": "no_match",
                "matches": [],
                "warnings": [f"ClinVar matching failed for this variant: {exc}"],
            }
        variants_out.append({
            "uploaded_variant": variant_record,
            "clinvar_result":   clinvar_result,
            "review_category":  _classify_variant_review(variant_record, clinvar_result),
        })

    session_id = session_store.generate_session_id()
    session_store.save_session(session_id, file.filename, variants_out)

    return {
        "filename": file.filename,
        "session_id": session_id,
        "file_summary": {
            "file_type": parsed.get("file_type", "unknown"),
            "number_of_variants_detected": len(parsed_variants),
            "detected_columns": parsed.get("detected_columns", []),
            "warnings": parsed.get("warnings", []),
        },
        "report_summary": _build_report_summary(variants_out),
        "variants": variants_out,
        "global_warnings": global_warnings,
        "safety_disclaimer": _ANALYZE_DISCLAIMER,
    }


@app.post("/analyze-report-json", include_in_schema=False)
def analyze_report_json(request: AnalyzeReportJsonRequest):
    """
    Parse a structured clinical genetic report JSON object and match each
    variant to ClinVar.

    Accepts the report structure produced by manual extraction or OCR pipelines
    (see examples/extracted_genetic_report.json for the expected shape).
    Stores the analyzed result in the session store so /ask-session can be
    used for follow-up questions without re-sending the full report.

    Flow:
      1. Call parse_structured_report_json() on the 'report' field of the body.
      2. For each parsed variant, call retriever.match_uploaded_variant().
         A failure on a single variant does not abort the entire request.
      3. Save variants to the session store.
      4. Return session_id, file_summary, report_metadata, per-variant results,
         and a safety disclaimer.
    """
    try:
        parsed = parse_structured_report_json(request.report)
    except Exception as exc:
        logger.warning("parse_structured_report_json failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the report JSON: {exc}",
        )

    global_warnings: list[str] = []
    parsed_variants: list[dict] = parsed.get("variants", [])

    if not parsed_variants:
        global_warnings.append(
            "No variants were detected in the report JSON. "
            "Ensure the payload contains a top-level 'variant' object with at minimum "
            "a 'gene', 'chromosome'/'position', or 'cdna_variant' field."
        )

    variants_out: list[dict] = []
    for variant_record in parsed_variants:
        try:
            clinvar_result = retriever.match_uploaded_variant(variant_record, limit=10)
        except Exception as exc:
            logger.warning(
                "match_uploaded_variant failed for record %s: %s", variant_record, exc
            )
            clinvar_result = {
                "query_used": "error",
                "match_confidence": "no_match",
                "matches": [],
                "warnings": [f"ClinVar matching failed for this variant: {exc}"],
            }
        variants_out.append({
            "uploaded_variant": variant_record,
            "clinvar_result":   clinvar_result,
            "review_category":  _classify_variant_review(variant_record, clinvar_result),
        })

    session_id = session_store.generate_session_id()
    session_store.save_session(session_id, "structured_report_json", variants_out)

    return {
        "session_id": session_id,
        "file_summary": {
            "file_type": parsed.get("file_type", "structured_report_json"),
            "number_of_variants_detected": len(parsed_variants),
            "detected_columns": parsed.get("detected_columns", []),
            "warnings": parsed.get("warnings", []),
        },
        "report_metadata": parsed.get("report_metadata", {}),
        "report_summary": _build_report_summary(variants_out),
        "variants": variants_out,
        "global_warnings": global_warnings,
        "safety_disclaimer": _ANALYZE_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# /ask-upload helpers
# ---------------------------------------------------------------------------

def _dedup_ordered(seq: list[str]) -> list[str]:
    """Return seq with duplicates removed, preserving first-occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_upload_answer(
    question: str,
    uploaded_variant: dict,
    clinvar_result: dict,
) -> dict:
    """
    Build a deterministic, safe educational answer from an uploaded variant
    and its ClinVar match result. No LLM, no invented facts.

    Returns {"answer": str, "evidence": list[str],
             "limitations": list[str], "warnings": list[str]}.
    """
    matches: list[dict] = clinvar_result.get("matches", [])
    match_confidence: str = clinvar_result.get("match_confidence", "no_match")
    gene_consistency: str = clinvar_result.get("gene_consistency", "not_checked")
    clinvar_warnings: list[str] = list(clinvar_result.get("warnings", []))

    uv_gene = str(uploaded_variant.get("gene", "") or "").strip()
    uv_variant = str(uploaded_variant.get("variant", "") or "").strip()
    uv_rsid = str(uploaded_variant.get("rsid", "") or "").strip()
    uv_chrom = str(uploaded_variant.get("chromosome", "") or "").strip()
    uv_pos = str(uploaded_variant.get("position", "") or "").strip()
    uv_sig = str(uploaded_variant.get("clinical_significance", "") or "").strip()

    lines: list[str] = []
    lines.append(f"Question: {question}")
    lines.append("")

    # --- Uploaded row summary ---
    lines.append("Uploaded variant information:")
    if uv_gene:
        lines.append(f"  Gene: {uv_gene}")
    if uv_variant:
        lines.append(f"  Variant: {uv_variant}")
    if uv_rsid:
        lines.append(f"  rsID: {uv_rsid}")
    if uv_chrom and uv_pos:
        lines.append(f"  Genomic position: chr{uv_chrom}:{uv_pos}")
    if uv_sig:
        lines.append(f"  Significance reported in uploaded file: {uv_sig}")
    lines.append("")

    # --- Gene consistency block ---
    if gene_consistency == "mismatch":
        lines.append(
            "IMPORTANT — Gene mismatch detected: The gene in your uploaded file "
            "does not match the gene associated with the ClinVar record found using "
            "the provided rsID or position. These may not refer to the same variant. "
            "Please verify your data carefully before drawing any conclusions."
        )
        lines.append("")
    elif gene_consistency == "mixed":
        lines.append(
            "Note: ClinVar returned matches spanning multiple genes. "
            "Please review gene consistency before drawing any conclusions."
        )
        lines.append("")

    # --- ClinVar match section ---
    if not matches:
        lines.append(
            "ClinVar result: No matching ClinVar record was found using the available "
            "identifiers (rsID, genomic position, or gene symbol)."
        )
        lines.append(
            "This may mean the variant is not yet catalogued in ClinVar, "
            "or that the identifiers in the uploaded file do not correspond to a "
            "known ClinVar entry."
        )
    else:
        if match_confidence == "gene_only":
            lines.append(
                "ClinVar match type: Gene-level only. The results below represent "
                "all ClinVar variants for this gene — not a match to the specific "
                "variant in your uploaded file. They cannot be used to interpret "
                "the significance of your particular variant."
            )
        elif match_confidence == "gene_hgvs_partial":
            lines.append(
                "ClinVar match type: Gene + partial HGVS notation. Results were found "
                "by matching the gene name and a normalized variant notation "
                "(e.g. 'c.68_69del' extracted from 'c.68_69delAG'). These results are "
                "approximate and may not correspond precisely to the exact uploaded variant."
            )
        elif match_confidence == "exact_rsid":
            lines.append("ClinVar match type: Exact rsID match.")
        elif match_confidence == "position_match":
            lines.append("ClinVar match type: Genomic position match.")
        lines.append("")

        unique_sigs = sorted({
            str(m.get("clinical_significance", "")).strip()
            for m in matches
            if m.get("clinical_significance")
        })
        unique_statuses = sorted({
            str(m.get("review_status", "")).strip()
            for m in matches
            if m.get("review_status")
        })
        unique_genes = sorted({
            str(m.get("gene_symbol", "")).strip()
            for m in matches
            if m.get("gene_symbol")
        })
        unique_phenos: set[str] = set()
        for m in matches:
            for p in str(m.get("phenotype_list", "")).split("|"):
                p = p.strip()
                if p and p.lower() not in ("not provided", "not specified", ""):
                    unique_phenos.add(p)

        lines.append(f"ClinVar records found: {len(matches)}")
        if unique_genes:
            lines.append(f"  Gene(s) in ClinVar: {', '.join(unique_genes)}")
        if unique_sigs:
            lines.append(f"  Clinical significance: {', '.join(unique_sigs)}")
            if any("uncertain" in s.lower() for s in unique_sigs):
                lines.append(
                    "  VUS note: 'Variant of Uncertain Significance' means current "
                    "evidence is insufficient to classify the variant as benign or "
                    "pathogenic. VUS findings must not be used for clinical "
                    "decision-making without expert review."
                )
        if unique_statuses:
            lines.append(f"  Review status: {', '.join(unique_statuses)}")
        if unique_phenos:
            shown = sorted(unique_phenos)[:5]
            lines.append(f"  Associated conditions: {'; '.join(shown)}")
            if len(unique_phenos) > 5:
                lines.append(f"  ... and {len(unique_phenos) - 5} additional condition(s) in ClinVar.")

    # --- Limitations ---
    lines.append("")
    lines.append("Limitations:")
    lines.append(
        "- This information is sourced from ClinVar and may not reflect the most "
        "recent evidence or all available submissions."
    )
    lines.append(
        "- ClinVar aggregates submissions from multiple sources; "
        "classifications may differ between submitters."
    )
    lines.append(
        "- Variant interpretation requires full clinical context, family history, "
        "and review by a certified genetics professional."
    )
    if match_confidence == "gene_only":
        lines.append(
            "- Gene-level results include many variants and cannot be used to "
            "interpret any specific nucleotide change."
        )

    # --- Build evidence list, deduplicate (GRCh37/GRCh38 rows produce identical strings) ---
    evidence_raw: list[str] = []
    for m in matches:
        parts: list[str] = []
        if m.get("gene_symbol"):
            parts.append(f"Gene: {m['gene_symbol']}")
        if m.get("clinical_significance"):
            parts.append(f"Significance: {m['clinical_significance']}")
        if m.get("review_status"):
            parts.append(f"Review: {m['review_status']}")
        rs = m.get("dbsnp_id")
        if rs is not None and int(rs) > 0:
            parts.append(f"rsID: rs{int(rs)}")
        if parts:
            evidence_raw.append(" | ".join(parts))

    evidence_deduped = _dedup_ordered(evidence_raw)
    if len(evidence_raw) > len(evidence_deduped):
        clinvar_warnings.append(
            "Some duplicate evidence lines were collapsed for display; "
            "full ClinVar matches may include multiple genome assemblies."
        )
    evidence = evidence_deduped[:5]

    limitations: list[str] = [
        "Based on ClinVar data only; may not reflect the most current evidence.",
        "Variant interpretation requires full clinical context and expert review.",
    ]
    if match_confidence == "gene_only":
        limitations.append(
            "Gene-level results cannot identify or interpret a specific variant."
        )

    return {
        "answer": "\n".join(lines),
        "evidence": evidence,
        "limitations": limitations,
        "warnings": clinvar_warnings,
    }


@app.post("/ask-upload", include_in_schema=False)
def ask_upload(request: AskUploadRequest):
    """
    Answer a cautious educational question about an uploaded variant and its ClinVar match.

    Uses the same policy and safety guardrails as /ask.
    Works without an LLM API key (deterministic responder always runs).
    If an LLM key is configured, the LLM is used with the ClinVar matches as evidence,
    following the same fallback pattern as /ask.

    Does not invent facts beyond the provided uploaded_variant and clinvar_result.
    """
    question = policy.sanitise_question(request.question)
    pol = policy.check_question(question)

    # Deterministic answer — always built, used as fallback if LLM unavailable
    raw = _build_upload_answer(question, request.uploaded_variant, request.clinvar_result)

    _has_matches = bool(request.clinvar_result.get("matches"))
    llm_used = False
    if _llm is not None and request.use_llm and _has_matches:
        try:
            raw = _llm.ask_upload(question, request.uploaded_variant, request.clinvar_result)
            llm_used = True
            logger.info("LLM used for /ask-upload: %s", question[:80])
        except LLMRateLimitError as exc:
            logger.warning("LLM rate limited for /ask-upload (%s); using deterministic.", exc)
        except LLMClientError as exc:
            logger.warning("LLM failed for /ask-upload (%s); using deterministic.", exc)
    elif _llm is not None and request.use_llm and not _has_matches:
        logger.info("LLM skipped for /ask-upload: no ClinVar matches (match_confidence=%s)",
                    request.clinvar_result.get("match_confidence", "no_match"))

    result = policy.enforce_disclaimer(raw)

    all_warnings = list(result.get("warnings", []))
    if pol.flags:
        all_warnings.extend(pol.flags)

    return {
        "answer": result["answer"],
        "uploaded_variant": request.uploaded_variant,
        "match_confidence": request.clinvar_result.get("match_confidence", "no_match"),
        "gene_consistency": request.clinvar_result.get("gene_consistency", "not_checked"),
        "evidence": result.get("evidence", []),
        "warnings": all_warnings,
        "safety_disclaimer": result["safety_disclaimer"],
        "llm_used": llm_used,
        "policy_flags": pol.flags,
        "redirect_message": pol.redirect_message,
    }


@app.post("/ask-session", include_in_schema=False)
def ask_session(request: AskSessionRequest):
    """
    Ask an educational question about a specific variant from a previous /analyze-upload call.

    The caller provides:
      - session_id  returned by /analyze-upload
      - variant_index  (0-based) selecting which variant from that session
      - question  the educational question to answer

    Returns the same structure as /ask-upload. The uploaded_variant and clinvar_result
    are retrieved from the in-memory session store; the caller does not need to resend them.
    """
    variants: list[dict] | None = session_store.get_session(request.session_id)
    if variants is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{request.session_id}' not found. "
                "Run /analyze-upload first and use the session_id from its response."
            ),
        )

    if request.variant_index >= len(variants):
        raise HTTPException(
            status_code=400,
            detail=(
                f"variant_index {request.variant_index} is out of range. "
                f"This session contains {len(variants)} variant(s) "
                f"(valid indices: 0–{len(variants) - 1})."
            ),
        )

    entry = variants[request.variant_index]
    uploaded_variant: dict = entry["uploaded_variant"]
    clinvar_result: dict = entry["clinvar_result"]

    question = policy.sanitise_question(request.question)
    pol = policy.check_question(question)

    raw = _build_upload_answer(question, uploaded_variant, clinvar_result)

    _has_matches = bool(clinvar_result.get("matches"))
    llm_used = False
    if _llm is not None and request.use_llm and _has_matches:
        try:
            raw = _llm.ask_upload(question, uploaded_variant, clinvar_result)
            llm_used = True
            logger.info("LLM used for /ask-session: %s", question[:80])
        except LLMRateLimitError as exc:
            logger.warning("LLM rate limited for /ask-session (%s); using deterministic.", exc)
        except LLMClientError as exc:
            logger.warning("LLM failed for /ask-session (%s); using deterministic.", exc)
    elif _llm is not None and request.use_llm and not _has_matches:
        logger.info("LLM skipped for /ask-session: no ClinVar matches (match_confidence=%s)",
                    clinvar_result.get("match_confidence", "no_match"))

    result = policy.enforce_disclaimer(raw)

    all_warnings = list(result.get("warnings", []))
    if pol.flags:
        all_warnings.extend(pol.flags)

    return {
        "answer": result["answer"],
        "uploaded_variant": uploaded_variant,
        "match_confidence": clinvar_result.get("match_confidence", "no_match"),
        "gene_consistency": clinvar_result.get("gene_consistency", "not_checked"),
        "evidence": result.get("evidence", []),
        "warnings": all_warnings,
        "safety_disclaimer": result["safety_disclaimer"],
        "llm_used": llm_used,
        "policy_flags": pol.flags,
        "redirect_message": pol.redirect_message,
    }


@app.post("/ask-report-session", include_in_schema=False)
def ask_report_session(request: AskReportSessionRequest):
    """
    Ask an educational question about the entire report (all variants) from a session.

    Builds compact evidence across all variants and uses the LLM as an explanation
    layer when at least one variant has ClinVar matches.  Falls back to the
    deterministic responder when no ClinVar evidence is available or the LLM fails
    quality validation.

    The LLM must not diagnose, recommend treatment, estimate personal risk, or
    invent evidence.  The forbidden-phrase and minimum-length validators are applied
    before accepting any LLM answer.
    """
    variants: list[dict] | None = session_store.get_session(request.session_id)
    if variants is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{request.session_id}' not found. "
                "Run /analyze-upload or /analyze-report-json first."
            ),
        )

    question = policy.sanitise_question(request.question)
    pol = policy.check_question(question)

    raw = _build_report_answer(question, variants)

    has_any_matches = any(
        bool(entry.get("clinvar_result", {}).get("matches"))
        for entry in variants
    )
    llm_used = False
    if _llm is not None and has_any_matches:
        try:
            raw = _llm.ask_report(question, variants)
            llm_used = True
            logger.info("LLM used for /ask-report-session: %s", question[:80])
        except LLMRateLimitError as exc:
            logger.warning("LLM rate limited for /ask-report-session (%s); using deterministic.", exc)
        except LLMClientError as exc:
            logger.warning("LLM failed for /ask-report-session (%s); using deterministic.", exc)
    elif _llm is not None and not has_any_matches:
        logger.info("LLM skipped for /ask-report-session: no ClinVar matches in session")

    result = policy.enforce_disclaimer(raw)
    all_warnings = list(result.get("warnings", []))
    if pol.flags:
        all_warnings.extend(pol.flags)

    return {
        "answer":           result["answer"],
        "session_id":       request.session_id,
        "variant_count":    len(variants),
        "evidence":         result.get("evidence", []),
        "warnings":         all_warnings,
        "safety_disclaimer": result["safety_disclaimer"],
        "llm_used":         llm_used,
        "policy_flags":     pol.flags,
        "redirect_message": pol.redirect_message,
    }
