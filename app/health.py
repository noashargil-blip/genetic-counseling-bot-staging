"""
Health-check and version reporting for the genetic counseling assistant.

All functions return plain dicts — the FastAPI endpoints in main.py wrap
them into HTTP responses.  Each component check is independently failable:
a broken ClinVar DB must not prevent the LLM or gene-index checks from
running.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

APP_VERSION = "2.3.0"

# Bumped when the safety policy document (SAFETY_POLICY.md) or the
# classifier rules in app/safety.py change in a substantive way.
SAFETY_POLICY_VERSION = "1.0.0"

# Bumped when the LLM usage policy changes — see LLM_POLICY.md.
# 2.0.0 = intro-only architecture: LLM may only prepend a validated short
# Hebrew sentence; all medical content comes from the deterministic KB.
LLM_POLICY_VERSION = "2.0.0"

_DATA_VERSION_PATH = Path("data/data_version.json")


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------

def check_clinvar() -> Dict[str, Any]:
    """Return ClinVar DB availability and record count."""
    try:
        from app.retriever import _DB_AVAILABLE, DB_PATH  # noqa: PLC0415
    except Exception as exc:
        return {"ok": False, "reason": f"retriever import error: {exc}"}

    if not _DB_AVAILABLE:
        return {"ok": False, "reason": "ClinVar database file is missing or unreadable"}

    try:
        import duckdb  # noqa: PLC0415
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            total = int(con.execute("SELECT COUNT(*) FROM clinvar").fetchone()[0])
        finally:
            con.close()
        return {"ok": True, "total_records": total, "db_path": str(DB_PATH)}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def check_gene_index() -> Dict[str, Any]:
    """Return gene-index availability and indexed gene count."""
    try:
        from app import gene_index  # noqa: PLC0415
    except Exception as exc:
        return {"ok": False, "reason": f"gene_index import error: {exc}"}

    if not gene_index._GENE_INDEX_AVAILABLE:
        if not gene_index.STATS_DB_PATH.exists():
            reason = (
                f"Index file not found at {gene_index.STATS_DB_PATH}. "
                "Run `python scripts/build_gene_index.py` to build it."
            )
        else:
            reason = "Index file exists but table is missing or corrupt."
        return {"ok": False, "reason": reason}

    try:
        total = gene_index.count_genes()
        return {
            "ok": True,
            "total_genes": total,
            "db_path": str(gene_index.STATS_DB_PATH),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def probe_llm_live(url: str, timeout: float = 2.0) -> bool:
    """HTTP-level reachability check for a local LLM server. Returns True if any HTTP
    response is received (even an error), False on connection failure or timeout."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # Got an HTTP response — server is alive
    except Exception:
        return False


def check_llm() -> Dict[str, Any]:
    """
    Return LLM configuration status. Never raises. Never exposes secrets.

    When LLM_PROVIDER is set explicitly, reports that provider's status.
    Otherwise falls back to legacy auto-detection (LOCAL_LLM_URL > ANTHROPIC_API_KEY > OPENAI_API_KEY).
    """
    provider_env = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if provider_env == "none":
        return {
            "ok": False,
            "provider": "none",
            "configured": True,
            "reason": "LLM explicitly disabled via LLM_PROVIDER=none.",
            "deterministic_fallback_available": True,
        }

    if provider_env == "openai":
        configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        result: Dict[str, Any] = {
            "ok": configured,
            "provider": "openai",
            "configured": configured,
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini") if configured else None,
            "deterministic_fallback_available": True,
        }
        if not configured:
            result["reason"] = "OPENAI_API_KEY not set."
        return result

    if provider_env == "anthropic":
        configured = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        result = {
            "ok": configured,
            "provider": "anthropic",
            "configured": configured,
            "deterministic_fallback_available": True,
        }
        if not configured:
            result["reason"] = "ANTHROPIC_API_KEY not set."
        return result

    if provider_env == "local":
        url = os.environ.get("LOCAL_LLM_URL", "").strip()
        configured = bool(url)
        result = {
            "ok": configured,
            "provider": "local",
            "configured": configured,
            "deterministic_fallback_available": True,
        }
        if not configured:
            result["reason"] = "LOCAL_LLM_URL not set."
        return result

    if provider_env == "huggingface_endpoint":
        configured = bool(os.environ.get("HF_ENDPOINT_URL", "").strip())
        result = {
            "ok": configured,
            "provider": "huggingface_endpoint",
            "configured": configured,
            "deterministic_fallback_available": True,
        }
        if not configured:
            result["reason"] = "HF_ENDPOINT_URL not set."
        return result

    # Legacy auto-detect (LLM_PROVIDER not set)
    has_local     = bool(os.environ.get("LOCAL_LLM_URL", "").strip())
    has_openai    = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    if has_local:
        detected = "local"
    elif has_openai:
        detected = "openai"
    elif has_anthropic:
        detected = "anthropic"
    else:
        detected = None

    if detected:
        return {
            "ok": True,
            "provider": detected,
            "configured": True,
            "deterministic_fallback_available": True,
        }
    return {
        "ok": False,
        "provider": None,
        "configured": False,
        "reason": (
            "No LLM configured. Set LLM_PROVIDER=openai and OPENAI_API_KEY for cloud staging, "
            "or LLM_PROVIDER=local and LOCAL_LLM_URL for university dev, "
            "or LLM_PROVIDER=none to run deterministic-only."
        ),
        "deterministic_fallback_available": True,
        "note": "The assistant uses deterministic KB answers when no LLM is available.",
    }


# ---------------------------------------------------------------------------
# data_version.json
# ---------------------------------------------------------------------------

def load_data_version() -> Dict[str, Any]:
    """Load data/data_version.json, returning {} if missing or unreadable."""
    if not _DATA_VERSION_PATH.exists():
        return {}
    try:
        return json.loads(_DATA_VERSION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read data_version.json: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Aggregate health
# ---------------------------------------------------------------------------

def get_overall_health() -> Dict[str, Any]:
    """
    Combine all component checks into a single health report.

    The assistant is considered *degraded* (not *down*) when the gene index
    or LLM is unavailable, because the core Q&A pipeline still works via the
    deterministic KB fallback.  Only a broken ClinVar DB (used by the core
    safety classifier) raises the severity to *degraded* overall.

    Returns a dict with:
      status: "ok" | "degraded" | "down"
      version: app version string
      components: {clinvar, gene_index, llm} sub-reports
      data_version: contents of data/data_version.json (or {})
    """
    clinvar    = check_clinvar()
    gene_idx   = check_gene_index()
    llm        = check_llm()
    data_ver   = load_data_version()

    # Core KB pipeline works without ClinVar variant DB — but without gene
    # index the gene-level answers degrade.  Without LLM only phrasing is lost.
    if not clinvar["ok"] and not gene_idx["ok"]:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "version": APP_VERSION,
        "components": {
            "clinvar":    clinvar,
            "gene_index": gene_idx,
            "llm":        llm,
        },
        "data_version": data_ver,
    }
