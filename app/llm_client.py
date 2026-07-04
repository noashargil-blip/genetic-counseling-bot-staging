"""
LLM client for Genetic Counseling Bot.

Two concrete backends are provided behind a shared interface:

  AnthropicLLMClient  — uses the Anthropic Messages API (Claude)
  OpenAILLMClient     — uses the OpenAI Chat Completions API (GPT-4o)
  LocalLLMClient      — uses a local HTTP POST endpoint (university server)

`LLMClient` remains an alias for AnthropicLLMClient so that existing imports
and tests in main.py are unaffected.

`create_llm_client()` is the recommended factory: it checks environment
variables and returns whichever backend is configured, or raises ValueError
if none is present.  Priority: LOCAL_LLM_URL > ANTHROPIC_API_KEY > OPENAI_API_KEY.

Safety contract
---------------
All backends receive only the evidence records retrieved from DuckDB.
No backend is given tools, web access, or conversation history.
The policy layer in main.py enforces the canonical disclaimer on top of
whatever the LLM returns.
After parsing the JSON response, _validate_upload_answer() checks for
forbidden causality language and the required 5-section structure; if either
check fails the LLMClientError is raised and main.py falls back to the
deterministic responder automatically.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared prompt — used by /ask (gene / condition search endpoint)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a genetic counseling support assistant that provides
evidence-based information about genetic variants using ClinVar data.

STRICT SAFETY RULES — you must follow these unconditionally:
1. Never provide a medical diagnosis.
2. Never recommend a treatment or clinical decision.
3. Only cite the evidence explicitly provided to you in the user message.
4. Always acknowledge uncertainty when evidence is limited or conflicting.
5. Always end by recommending consultation with a certified genetics professional.
6. Do not speculate beyond the retrieved evidence.

RESPONSE FORMAT — you must return valid JSON and nothing else. No preamble,
no markdown fences. Match this schema exactly:
{
  "answer": "<concise, accurate summary of the variant evidence>",
  "evidence": ["<evidence point 1>", "<evidence point 2>", ...],
  "limitations": ["<limitation 1>", "<limitation 2>", ...],
  "safety_disclaimer": "<standard safety disclaimer>"
}

Rules for each field:
- "answer": 2-4 sentences. Factual, citing significance classifications, condition
  associations, and review status from the provided data only.
- "evidence": list of specific data points extracted from the provided ClinVar
  records (e.g. classification, submitter, condition, review status).
- "limitations": honest list of what is unknown or uncertain (e.g. conflicting
  interpretations, limited submitters, VUS status, missing functional data).
- "safety_disclaimer": always include this text verbatim:
  "This information is for educational purposes only and does not constitute
   medical advice. Please consult a certified genetics professional or genetic
   counselor for clinical interpretation and personalized recommendations."
"""

USER_PROMPT_TEMPLATE = """User question: {question}

Retrieved ClinVar evidence:
{evidence_block}

Respond with valid JSON only, strictly according to the schema in the system prompt."""

# ---------------------------------------------------------------------------
# Upload-specific system prompt (used by /ask-upload and /ask-session)
# ---------------------------------------------------------------------------

UPLOAD_SYSTEM_PROMPT = """\
You are an educational assistant. Explain genetic variant data from ClinVar to a patient.
Use ONLY the data provided. Do not add, invent, or infer any facts.

RULES:
- Never diagnose. Never recommend treatment or any medical action. Never estimate personal risk.
- If no ClinVar records were found (no_match), state that clearly and do not speculate.
- If gene consistency is mismatch, note that the ClinVar gene differs from the reported gene.
- Never say the variant causes, contributes to, or is responsible for anything.
- Write in plain language. 200-350 words total.

RESPONSE FORMAT — return ONLY this JSON, no markdown, no other text:
{
  "answer": "your structured answer here",
  "evidence": ["fact from ClinVar records", "another fact"],
  "limitations": ["specific limitation 1", "specific limitation 2"],
  "safety_disclaimer": "This information is for educational purposes only and does not constitute medical advice. Please consult a certified genetics professional or genetic counselor for clinical interpretation and personalized recommendations."
}

ANSWER STRUCTURE — use these five headings exactly as written:

1. What was found in the report
[Gene name, variant notation, zygosity, ClinVar accession if given.
 Any phenotypes listed in the report were "reported in the uploaded report".]

2. What ClinVar says
[State the match confidence:
  exact_clinvar_accession = the accession in the report matched a ClinVar record directly; highest confidence.
  exact_rsid = matched by rsID; high confidence.
  gene_only = all ClinVar variants for this gene; NOT specific to the uploaded variant.
  no_match = no ClinVar record found.
 State number of records found, clinical significance, and review status.]

3. What this classification means
[Explain the ClinVar classification in plain language.
 For Uncertain significance — write these sentences:
   "Current evidence is insufficient to classify this variant as benign or pathogenic.
    VUS findings must not be used alone for clinical decision-making.
    The classification may change as more evidence becomes available."]

4. Key limitations
[At least two specific limitations, e.g.: single submitter, VUS status, gene-only match,
 limited functional data, database snapshot may not be current.]

5. Recommended next step
[Tell the user to discuss findings with a certified genetics professional or genetic
 counselor before drawing any clinical conclusions.]

FINAL CHECK — before writing your answer, verify that NONE of these words appear in it:
  cause, causes, caused by, disease-causing, contributes to, responsible for,
  explains the phenotype, personal risk, will develop, risk of developing.
If any appear, rewrite those sentences without them.
"""

# ---------------------------------------------------------------------------
# Upload-specific user prompt template
# ---------------------------------------------------------------------------

UPLOAD_USER_PROMPT_TEMPLATE = """\
User question: {question}

=== UPLOADED VARIANT ===
{uploaded_block}

=== CLINVAR MATCH ===
Match confidence: {match_confidence}
Gene consistency: {gene_consistency}
{warnings_block}

=== CLINVAR RECORDS ({n_matches} found) ===
{evidence_block}

{no_match_note}{vus_note}\
Now write your answer using ONLY the data above.
Use these five headings in order (copy them verbatim):

1. What was found in the report
2. What ClinVar says
3. What this classification means
4. Key limitations
5. Recommended next step

Return valid JSON only.
"""

# ---------------------------------------------------------------------------
# Report-level prompt (used by /ask-report-session)
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """\
You are an educational assistant summarizing a multi-variant genetic report using ClinVar data.
Use ONLY the data provided. Do not add, invent, or infer any facts.

RULES:
- Never diagnose. Never recommend treatment. Never estimate personal risk.
- Never say a variant causes, contributes to, or is responsible for anything.
- Clearly distinguish: exact ClinVar matches vs gene-only matches vs unmatched variants.
- For VUS: "Current evidence is insufficient to classify this variant as benign or pathogenic."
- Write in plain language. 150-400 words.

RESPONSE FORMAT — return ONLY this JSON, no markdown, no extra text:
{
  "answer": "your report summary",
  "evidence": ["specific ClinVar fact 1", "another fact"],
  "limitations": ["limitation 1", "limitation 2"],
  "safety_disclaimer": "This information is for educational purposes only and does not constitute medical advice. Please consult a certified genetics professional or genetic counselor for clinical interpretation and personalized recommendations."
}

ANSWER STRUCTURE — address these points in order:
1. How many variants are in the report and how many have ClinVar evidence.
2. For each variant with a ClinVar match: gene name, classification, match confidence level.
3. For any VUS: include the required phrase above.
4. For unmatched variants: state no ClinVar record was found.
5. Key limitations of this data.
6. Recommend consulting a certified genetics professional.

FINAL CHECK — verify none of these appear in your answer:
cause, causes, caused by, disease-causing, contributes to, responsible for,
explains the phenotype, personal risk, will develop, risk of developing.
"""

REPORT_USER_PROMPT_TEMPLATE = """\
User question: {question}

=== REPORT OVERVIEW ===
Total variants: {total_variants}
Variants with ClinVar evidence: {matched_count}
Exact ClinVar matches: {exact_count}
No ClinVar match found: {no_match_count}

=== VARIANT DETAILS ===
{variants_block}
{vus_note}Write a clear educational summary using only the data above. Return valid JSON only.
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_evidence_block(evidence: list[dict]) -> str:
    """Convert retrieved variant dicts into a readable block for the prompt."""
    if not evidence:
        return "No ClinVar records retrieved."
    lines = []
    for i, rec in enumerate(evidence, 1):
        lines.append(f"Record {i}:")
        for key, value in rec.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _format_uploaded_variant_block(uv: dict) -> str:
    """Render the uploaded variant dict as a compact labelled block."""
    lines = []
    for key in (
        "gene", "transcript", "variant", "protein_change", "genomic_change",
        "rsid", "chromosome", "position", "ref", "alt",
        "zygosity", "inheritance", "variant_type",
        "clinical_significance", "clinvar_accession",
        "associated_condition",
    ):
        val = uv.get(key)
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val if x)
        val = str(val or "").strip()
        if val:
            lines.append(f"  {key}: {val}")
    acmg = uv.get("acmg_criteria")
    if isinstance(acmg, list) and acmg:
        lines.append(f"  acmg_criteria: {', '.join(str(x) for x in acmg)}")
    elif isinstance(acmg, str) and acmg.strip():
        lines.append(f"  acmg_criteria: {acmg.strip()}")
    return "\n".join(lines) if lines else "  (no variant details provided)"


def _format_upload_user_prompt(
    question: str,
    uploaded_variant: dict,
    clinvar_result: dict,
) -> str:
    """
    Build the user-turn prompt for ask_upload / ask_session.

    The LLM receives the uploaded variant details, match metadata, and
    ClinVar records — nothing else.  It must only rewrite what is here.
    """
    uploaded_block = _format_uploaded_variant_block(uploaded_variant)

    match_confidence = str(clinvar_result.get("match_confidence", "no_match"))
    gene_consistency = str(clinvar_result.get("gene_consistency", "not_checked"))
    warnings = list(clinvar_result.get("warnings", []))
    matches = list(clinvar_result.get("matches", []))

    warnings_block = (
        "Warnings:\n" + "\n".join(f"  - {w}" for w in warnings)
        if warnings
        else "Warnings: none"
    )

    evidence_block = _format_evidence_block(matches)
    n_matches = len(matches)

    if n_matches == 0:
        no_match_note = (
            "IMPORTANT: No ClinVar records were found for this variant. "
            "State this clearly; do not speculate about significance.\n"
        )
    elif match_confidence in ("gene_only", "gene_hgvs_partial"):
        no_match_note = (
            f"IMPORTANT: Match confidence is '{match_confidence}' — these ClinVar records "
            "are NOT specific to the exact uploaded variant. "
            "Make this limitation explicit in section 4.\n"
        )
    else:
        no_match_note = ""

    # Detect VUS across uploaded report and ClinVar records
    all_sigs = [str(uploaded_variant.get("clinical_significance") or "")]
    all_sigs += [str(m.get("clinical_significance") or "") for m in matches]
    is_vus = any("uncertain" in s.lower() for s in all_sigs if s)
    vus_note = (
        "IMPORTANT: One or more classifications are 'Uncertain significance' (VUS). "
        "Section 3 MUST include all three required VUS sentences from the system prompt.\n"
        if is_vus else ""
    )

    return UPLOAD_USER_PROMPT_TEMPLATE.format(
        question=question,
        uploaded_block=uploaded_block,
        match_confidence=match_confidence,
        gene_consistency=gene_consistency,
        warnings_block=warnings_block,
        n_matches=n_matches,
        evidence_block=evidence_block,
        no_match_note=no_match_note,
        vus_note=vus_note,
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _strip_fences(raw: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Handles:
      ```json\n{...}\n```
      ```\n{...}\n```
      {...}   (no fences — returned unchanged)
    """
    return _FENCE_RE.sub("", raw).strip()


def _parse_response(raw: str) -> dict:
    """
    Parse and validate the JSON blob returned by any LLM backend.

    Raises LLMClientError on bad JSON or missing required fields.
    """
    clean = _strip_fences(raw)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON (first 300 chars): %s", raw[:300])
        raise LLMClientError("LLM response was not valid JSON.") from exc

    required = {"answer", "evidence", "limitations", "safety_disclaimer"}
    missing = required - data.keys()
    if missing:
        raise LLMClientError(f"LLM response missing fields: {missing}")

    return {
        "answer": str(data["answer"]),
        "evidence": list(data.get("evidence", [])),
        "limitations": list(data.get("limitations", [])),
        "safety_disclaimer": str(data["safety_disclaimer"]),
    }


# ---------------------------------------------------------------------------
# Upload answer quality validation
# ---------------------------------------------------------------------------

# Word-boundary regex for forbidden causality phrases.
# Uses \b so "because" does NOT match "cause".
_FORBIDDEN_RE = re.compile(
    r"\b(causes?\b|caused\s+by|cause\s+of|disease[- ]causing"
    r"|contributes?\s+to|responsible\s+for|explains?\s+the\s+phenotype)\b",
    re.IGNORECASE,
)

# Lowercase substrings that identify the 5 required section headers.
_REQUIRED_SECTIONS = [
    "what was found in the report",
    "what clinvar says",
    "what this classification means",
    "key limitations",
    "recommended next step",
]


def _validate_upload_answer(parsed: dict) -> list[str]:
    """
    Quality-check the LLM answer for the upload/session endpoints.

    Checks:
      1. Forbidden causality phrases (word-boundary regex)
      2. Required 5-section headers (at least 4 of 5 must be present)
      3. Minimum word count (80 words)

    Returns a list of violation strings.  Empty list means the answer is OK.
    Violations trigger LLMClientError in ask_upload(), which causes main.py
    to fall back to the deterministic responder.
    """
    violations: list[str] = []
    answer = parsed.get("answer", "")

    # Check forbidden phrases (deduplicate — report each phrase once)
    seen_phrases: set[str] = set()
    for m in _FORBIDDEN_RE.finditer(answer):
        phrase = m.group(0).lower()
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            violations.append(f"forbidden phrase: '{m.group(0)}'")

    # Check section structure (at least 4/5 required headers must be present)
    answer_lower = answer.lower()
    found = [s for s in _REQUIRED_SECTIONS if s in answer_lower]
    if len(found) < 4:
        missing = [s for s in _REQUIRED_SECTIONS if s not in answer_lower]
        violations.append(
            f"section structure incomplete: {len(found)}/5 headers found "
            f"(missing: {missing})"
        )

    # Check minimum length
    word_count = len(answer.split())
    if word_count < 80:
        violations.append(f"answer too short: {word_count} words (minimum 80)")

    return violations


def _format_report_variants_block(session_variants: list[dict]) -> str:
    """Compact per-variant evidence block for the report-level user prompt."""
    lines: list[str] = []
    for i, entry in enumerate(session_variants, 1):
        uv  = entry.get("uploaded_variant", {})
        cr  = entry.get("clinvar_result",   {})
        gene    = str(uv.get("gene")    or "Unknown").strip()
        variant = str(uv.get("variant") or "").strip()
        conf    = str(cr.get("match_confidence") or "no_match")
        consistency = str(cr.get("gene_consistency") or "not_checked")

        lines.append(f"Variant {i}: {gene}" + (f" ({variant})" if variant else ""))
        lines.append(f"  Match confidence: {conf}")
        lines.append(f"  Gene consistency: {consistency}")

        matches = cr.get("matches", [])
        if matches:
            sigs = list({str(m.get("clinical_significance") or "") for m in matches
                         if m.get("clinical_significance")})
            statuses = list({str(m.get("review_status") or "") for m in matches
                             if m.get("review_status")})
            if sigs:
                lines.append(f"  ClinVar significance: {'; '.join(sigs)}")
            if statuses:
                lines.append(f"  Review status: {'; '.join(statuses)}")
        else:
            uv_sig = str(uv.get("clinical_significance") or "").strip()
            if uv_sig:
                lines.append(f"  Reported significance (not verified in ClinVar): {uv_sig}")

        warns = cr.get("warnings", [])
        if warns:
            lines.append(f"  Notices: {'; '.join(warns[:2])}")
        lines.append("")
    return "\n".join(lines)


def _format_report_user_prompt(question: str, session_variants: list[dict]) -> str:
    """Build the user-turn prompt for ask_report."""
    total = len(session_variants)
    no_match_count = sum(
        1 for e in session_variants
        if e.get("clinvar_result", {}).get("match_confidence") in ("no_match", "error", None)
    )
    matched = total - no_match_count
    exact = sum(
        1 for e in session_variants
        if e.get("clinvar_result", {}).get("match_confidence")
        in ("exact_clinvar_accession", "exact_rsid", "position_exact")
    )

    variants_block = _format_report_variants_block(session_variants)

    all_sigs: list[str] = []
    for entry in session_variants:
        cr = entry.get("clinvar_result", {})
        for m in cr.get("matches", []):
            all_sigs.append(str(m.get("clinical_significance") or "").lower())
        uv = entry.get("uploaded_variant", {})
        all_sigs.append(str(uv.get("clinical_significance") or "").lower())
    has_vus = any("uncertain" in s for s in all_sigs if s)
    vus_note = (
        "IMPORTANT: One or more variants have Uncertain significance (VUS). "
        "Your answer MUST state: \"Current evidence is insufficient to classify this "
        "variant as benign or pathogenic. VUS findings must not be used alone for "
        "clinical decision-making.\"\n"
        if has_vus else ""
    )

    return REPORT_USER_PROMPT_TEMPLATE.format(
        question=question,
        total_variants=total,
        matched_count=matched,
        exact_count=exact,
        no_match_count=no_match_count,
        variants_block=variants_block,
        vus_note=vus_note,
    )


def _validate_report_answer(parsed: dict) -> list[str]:
    """
    Quality-check the LLM answer for the report-level endpoint.

    Checks forbidden causality phrases and minimum word count.
    No fixed section-header requirement (report answers are more open-ended).
    """
    violations: list[str] = []
    answer = parsed.get("answer", "")

    seen_phrases: set[str] = set()
    for m in _FORBIDDEN_RE.finditer(answer):
        phrase = m.group(0).lower()
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            violations.append(f"forbidden phrase: '{m.group(0)}'")

    word_count = len(answer.split())
    if word_count < 60:
        violations.append(f"answer too short: {word_count} words (minimum 60)")

    return violations


# ---------------------------------------------------------------------------
# Shared exception hierarchy
# ---------------------------------------------------------------------------

class LLMClientError(Exception):
    """Raised when the LLM client cannot produce a usable response."""


class LLMRateLimitError(LLMClientError):
    """
    Raised specifically on HTTP 429 / rate-limit responses.

    Callers can catch this separately if they want to implement retry logic,
    rather than treating it the same as an auth failure or bad request.
    """


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Shared interface for all LLM backends.

    Subclasses must implement _call_api(user_content) and return the raw
    text string from the model.  _parse_response and the prompt templates
    are shared across all backends.
    """

    @abstractmethod
    def _call_api(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Send user_content to the model with system_prompt; return raw response text."""

    def call_text_raw(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Call the model expecting plain text output (not JSON-constrained).

        Default implementation delegates to _call_api(). OpenAILLMClient overrides
        this to omit response_format=json_object, which would otherwise force JSON
        output. Use this method when the system prompt expects plain prose (e.g.,
        Hebrew patient-facing unverified gene drafts).
        """
        return self._call_api(user_content, system_prompt)

    def ask(self, question: str, evidence: list[dict]) -> dict:
        """
        Build the prompt from question + evidence, call the backend,
        parse and validate the JSON response.

        Returns:
            {"answer": str, "evidence": list[str],
             "limitations": list[str], "safety_disclaimer": str}

        Raises:
            LLMRateLimitError  — on HTTP 429 (caller may retry)
            LLMClientError     — on all other unrecoverable failures
        """
        evidence_block = _format_evidence_block(evidence)
        user_content = USER_PROMPT_TEMPLATE.format(
            question=question,
            evidence_block=evidence_block,
        )
        raw = self._call_api(user_content)
        return _parse_response(raw)

    def ask_upload(
        self,
        question: str,
        uploaded_variant: dict,
        clinvar_result: dict,
    ) -> dict:
        """
        Explain an uploaded variant's ClinVar match result in plain language.

        After parsing the JSON response, runs _validate_upload_answer() to
        catch forbidden causality language and missing section headers.
        Raises LLMClientError on validation failure so main.py falls back
        to the deterministic responder automatically.

        Returns the same schema as ask():
            {"answer": str, "evidence": list[str],
             "limitations": list[str], "safety_disclaimer": str}
        """
        user_content = _format_upload_user_prompt(question, uploaded_variant, clinvar_result)
        raw = self._call_api(user_content, system_prompt=UPLOAD_SYSTEM_PROMPT)
        result = _parse_response(raw)

        violations = _validate_upload_answer(result)
        if violations:
            logger.warning(
                "LLM answer failed quality validation (%d violation(s)) — "
                "triggering deterministic fallback. Violations: %s",
                len(violations), violations,
            )
            raise LLMClientError(
                f"LLM answer quality check failed ({len(violations)} violation(s)): "
                + "; ".join(violations)
            )

        logger.info("LLM answer passed quality validation.")
        return result

    def ask_report(self, question: str, session_variants: list[dict]) -> dict:
        """
        Summarize all variants in a session in response to a report-level question.

        Applies _validate_report_answer() (forbidden phrases + minimum length).
        Raises LLMClientError on validation failure so main.py falls back to the
        deterministic responder automatically.
        """
        user_content = _format_report_user_prompt(question, session_variants)
        raw = self._call_api(user_content, system_prompt=REPORT_SYSTEM_PROMPT)
        result = _parse_response(raw)

        violations = _validate_report_answer(result)
        if violations:
            logger.warning(
                "LLM report answer failed validation (%d violation(s)) — "
                "triggering deterministic fallback. Violations: %s",
                len(violations), violations,
            )
            raise LLMClientError(
                f"LLM report answer validation failed: {'; '.join(violations)}"
            )

        logger.info("LLM report answer passed quality validation.")
        return result


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicLLMClient(BaseLLMClient):
    """LLM backend using the Anthropic Messages API (Claude)."""

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 1024

    def __init__(self, api_key: Optional[str] = None):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicLLMClient. "
                "Install it with: pip install anthropic"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it as an environment variable or pass api_key explicitly."
            )
        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic(api_key=key)

    def _call_api(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        # Import anthropic locally so exception classes are always available,
        # even when this method is called on an instance created via __new__
        # (which bypasses __init__ and therefore never sets self._anthropic).
        import anthropic as _anth
        try:
            message = self._client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
        except _anth.RateLimitError as exc:
            logger.warning("Anthropic rate limit hit: %s", exc)
            raise LLMRateLimitError("Anthropic API rate limit reached.") from exc
        except _anth.APIStatusError as exc:
            logger.error("Anthropic API status error %s: %s", exc.status_code, exc)
            raise LLMClientError(f"API error {exc.status_code}: {exc.message}") from exc
        except _anth.APIConnectionError as exc:
            logger.error("Anthropic API connection error: %s", exc)
            raise LLMClientError("Could not reach the Anthropic API.") from exc

        return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAILLMClient(BaseLLMClient):
    """
    LLM backend using the OpenAI Chat Completions API.

    Configured via environment variables:
      OPENAI_API_KEY       (required)
      OPENAI_MODEL         (default: gpt-4o-mini — cheaper for beta staging)
      LLM_MAX_TOKENS       (default: 1024)
      LLM_TEMPERATURE      (default: 0.3)
      LLM_TIMEOUT_SECONDS  (default: 30)

    Uses `response_format={"type": "json_object"}` so the model is constrained
    to return JSON (shared _strip_fences + _parse_response handles both fenced
    and plain JSON).
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TIMEOUT = 30

    def __init__(self, api_key: Optional[str] = None):
        try:
            import openai as _openai  # local import — optional dependency
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAILLMClient. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Export it as an environment variable or pass api_key explicitly."
            )
        self._openai = _openai
        self._client = _openai.OpenAI(api_key=key)
        self._model = os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL).strip()
        self._max_tokens = int(os.environ.get("LLM_MAX_TOKENS", self.DEFAULT_MAX_TOKENS))
        self._temperature = float(os.environ.get("LLM_TEMPERATURE", self.DEFAULT_TEMPERATURE))
        self._timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", self.DEFAULT_TIMEOUT))
        logger.info(
            "OpenAI client: model=%s max_tokens=%d temperature=%.2f timeout=%ds",
            self._model, self._max_tokens, self._temperature, self._timeout,
        )

    def _call_api(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        # Use getattr so __new__-based mocks (which bypass __init__) still work
        model       = getattr(self, "_model",       self.DEFAULT_MODEL)
        max_tokens  = getattr(self, "_max_tokens",  self.DEFAULT_MAX_TOKENS)
        temperature = getattr(self, "_temperature", self.DEFAULT_TEMPERATURE)
        timeout     = getattr(self, "_timeout",     self.DEFAULT_TIMEOUT)
        try:
            response = self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except self._openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit hit: %s", exc)
            raise LLMRateLimitError("OpenAI API rate limit reached.") from exc
        except self._openai.APIStatusError as exc:
            logger.error("OpenAI API status error %s: [status omitted from logs]", exc.status_code)
            raise LLMClientError(f"OpenAI API error {exc.status_code}") from exc
        except self._openai.APIConnectionError as exc:
            # APITimeoutError is a subclass of APIConnectionError in the openai SDK
            logger.error("OpenAI API connection/timeout error: %s", exc)
            raise LLMClientError("Could not reach the OpenAI API.") from exc

        return response.choices[0].message.content.strip()

    def call_text_raw(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Call the model expecting plain text — no response_format=json_object constraint."""
        model       = getattr(self, "_model",       self.DEFAULT_MODEL)
        max_tokens  = getattr(self, "_max_tokens",  self.DEFAULT_MAX_TOKENS)
        temperature = getattr(self, "_temperature", self.DEFAULT_TEMPERATURE)
        timeout     = getattr(self, "_timeout",     self.DEFAULT_TIMEOUT)
        try:
            response = self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except self._openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit hit: %s", exc)
            raise LLMRateLimitError("OpenAI API rate limit reached.") from exc
        except self._openai.APIStatusError as exc:
            logger.error("OpenAI API status error %s: [status omitted from logs]", exc.status_code)
            raise LLMClientError(f"OpenAI API error {exc.status_code}") from exc
        except self._openai.APIConnectionError as exc:
            logger.error("OpenAI API connection/timeout error: %s", exc)
            raise LLMClientError("Could not reach the OpenAI API.") from exc
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Local HTTP backend (university / self-hosted inference server)
# ---------------------------------------------------------------------------

class LocalLLMClient(BaseLLMClient):
    """
    LLM backend using a local HTTP POST endpoint.

    Request body:  {"prompt": "<combined system + user prompt>"}
    Response body: {"response": "<model output text>"}

    Enabled by setting LOCAL_LLM_URL in the environment.
    All errors (timeout, connection, bad response) raise LLMClientError so
    main.py falls back to the deterministic responder automatically.
    """

    def __init__(self, url: str):
        self._url = url.rstrip("/")
        self.TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", 60))
        logger.info("LOCAL_LLM_URL configured — endpoint: %s timeout=%ds", self._url, self.TIMEOUT)

    def _call_api(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        import urllib.request
        import urllib.error

        prompt = f"{system_prompt}\n\n{user_content}"
        payload = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            logger.warning("Local LLM HTTP error %s — falling back.", exc.code)
            raise LLMClientError(f"Local LLM returned HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            if "timed out" in reason.lower():
                logger.warning("Local LLM timed out after %ds — falling back.", self.TIMEOUT)
                raise LLMClientError(
                    f"Local LLM request timed out after {self.TIMEOUT}s."
                ) from exc
            logger.warning("Local LLM connection error: %s — falling back.", exc.reason)
            raise LLMClientError(
                f"Could not connect to local LLM at {self._url}."
            ) from exc
        except Exception as exc:
            logger.warning("Local LLM request failed: %s — falling back.", exc)
            raise LLMClientError(f"Local LLM request failed: {exc}") from exc

        try:
            data = json.loads(raw_body)
        except Exception as exc:
            raise LLMClientError("Local LLM returned a non-JSON body.") from exc

        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMClientError(
                "Local LLM response missing 'response' field or it is empty."
            )

        logger.info("Local LLM call succeeded (%d chars).", len(text))
        return text.strip()


# ---------------------------------------------------------------------------
# Alias + factory
# ---------------------------------------------------------------------------

# Keep LLMClient pointing at the Anthropic backend so existing imports
# and tests in main.py are unaffected.
LLMClient = AnthropicLLMClient


# ---------------------------------------------------------------------------
# HuggingFace Inference Endpoint backend
# ---------------------------------------------------------------------------

class HuggingFaceEndpointLLMClient(BaseLLMClient):
    """
    LLM backend using a Hugging Face Inference Endpoint (raw HTTP, no hf_hub package).

    Request body:  {"inputs": "<prompt>", "parameters": {"max_new_tokens": N}}
    Response body: {"generated_text": "<output>"} or a list containing it.

    Enable with:
      LLM_PROVIDER=huggingface_endpoint
      HF_ENDPOINT_URL=https://...
      HF_TOKEN=hf_... (optional for public endpoints)
    """

    def __init__(self, url: str, token: str = ""):
        self._url = url.rstrip("/")
        self._token = token
        self._timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", 60))
        self._max_tokens = int(os.environ.get("LLM_MAX_TOKENS", 512))
        if token:
            logger.info("HuggingFace endpoint: %s (token present)", self._url)
        else:
            logger.info("HuggingFace endpoint: %s (no token)", self._url)

    def _call_api(self, user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        import urllib.request
        import urllib.error

        prompt = f"{system_prompt}\n\n{user_content}"
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": self._max_tokens},
        }).encode()
        headers: dict = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(
            self._url, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            raise LLMClientError(
                f"HuggingFace endpoint returned HTTP {exc.code}."
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(
                f"Could not connect to HuggingFace endpoint at {self._url}."
            ) from exc

        try:
            data = json.loads(raw_body)
        except Exception as exc:
            raise LLMClientError("HuggingFace endpoint returned non-JSON body.") from exc

        # Handle both single-object and list responses
        if isinstance(data, list) and data:
            data = data[0]
        text = data.get("generated_text") or data.get("response") or data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise LLMClientError(
                "HuggingFace endpoint response missing 'generated_text' field."
            )
        return text.strip()


def create_llm_client() -> BaseLLMClient:
    """
    Select a backend from environment variables.

    When LLM_PROVIDER is set, it takes explicit priority:
      none                 → raises ValueError (LLM disabled)
      local                → LocalLLMClient (requires LOCAL_LLM_URL)
      openai               → OpenAILLMClient (requires OPENAI_API_KEY)
      anthropic            → AnthropicLLMClient (requires ANTHROPIC_API_KEY)
      huggingface_endpoint → HuggingFaceEndpointLLMClient (requires HF_ENDPOINT_URL)

    When LLM_PROVIDER is not set, falls back to legacy auto-detect:
      LOCAL_LLM_URL set    → LocalLLMClient
      ANTHROPIC_API_KEY    → AnthropicLLMClient
      OPENAI_API_KEY       → OpenAILLMClient

    Raises ValueError if no provider is configured or LLM_PROVIDER=none.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if provider == "none":
        raise ValueError("LLM explicitly disabled via LLM_PROVIDER=none.")

    if provider == "local":
        url = os.environ.get("LOCAL_LLM_URL", "").strip()
        if not url:
            raise ValueError("LLM_PROVIDER=local requires LOCAL_LLM_URL to be set.")
        logger.info("LLM backend: local HTTP endpoint (%s)", url)
        return LocalLLMClient(url)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        logger.info("LLM backend: OpenAI (model=%s)",
                    os.environ.get("OPENAI_MODEL", OpenAILLMClient.DEFAULT_MODEL))
        return OpenAILLMClient()

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")
        logger.info("LLM backend: Anthropic (Claude)")
        return AnthropicLLMClient()

    if provider == "huggingface_endpoint":
        url = os.environ.get("HF_ENDPOINT_URL", "").strip()
        if not url:
            raise ValueError(
                "LLM_PROVIDER=huggingface_endpoint requires HF_ENDPOINT_URL to be set."
            )
        token = os.environ.get("HF_TOKEN", "").strip()
        logger.info("LLM backend: HuggingFace endpoint (%s)", url)
        return HuggingFaceEndpointLLMClient(url, token)

    if provider and provider not in ("", "local", "openai", "anthropic",
                                     "huggingface_endpoint", "none"):
        raise ValueError(
            f"Unknown LLM_PROVIDER value: '{provider}'. "
            "Valid values: local, openai, anthropic, huggingface_endpoint, none."
        )

    # Legacy auto-detect (LLM_PROVIDER not set)
    local_url = os.environ.get("LOCAL_LLM_URL", "").strip()
    if local_url:
        logger.info("LLM backend: local HTTP endpoint (%s) [auto-detected]", local_url)
        return LocalLLMClient(local_url)
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        logger.info("LLM backend: Anthropic (Claude) [auto-detected]")
        return AnthropicLLMClient()
    if os.environ.get("OPENAI_API_KEY", "").strip():
        logger.info("LLM backend: OpenAI [auto-detected]")
        return OpenAILLMClient()

    raise ValueError(
        "No LLM configured. "
        "Set LLM_PROVIDER=openai and OPENAI_API_KEY for cloud staging, "
        "or LLM_PROVIDER=local and LOCAL_LLM_URL for university development, "
        "or LLM_PROVIDER=none to run in deterministic-only mode."
    )
