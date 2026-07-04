# -*- coding: utf-8 -*-
"""
Tests for the production health and version endpoints.

Coverage
--------
* GET /health         — overall aggregate health
* GET /health/clinvar — ClinVar DB component
* GET /health/gene-index — gene index component
* GET /health/llm     — LLM config component
* GET /version        — app version + data_version

All tests run regardless of whether the gene index or ClinVar DB is available;
the health endpoints always return HTTP 200 (degraded states are not errors).
"""
import json
import os
import pathlib
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import health as health_module
from app import gene_index

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_local_llm(monkeypatch):
    """Ensure tests start with no LLM configured."""
    monkeypatch.delenv("LOCAL_LLM_URL",       raising=False)
    monkeypatch.delenv("OPENAI_API_KEY",       raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY",    raising=False)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_has_required_fields(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "components" in data
        assert data["version"] == health_module.APP_VERSION

    def test_status_is_valid_value(self):
        status = client.get("/health").json()["status"]
        assert status in ("ok", "degraded", "down")

    def test_components_has_all_keys(self):
        components = client.get("/health").json()["components"]
        assert "clinvar"    in components
        assert "gene_index" in components
        assert "llm"        in components

    def test_components_have_ok_field(self):
        components = client.get("/health").json()["components"]
        for name, comp in components.items():
            assert "ok" in comp, f"component '{name}' missing 'ok' field"
            assert isinstance(comp["ok"], bool), f"component '{name}' 'ok' is not bool"

    def test_data_version_key_present(self):
        data = client.get("/health").json()
        assert "data_version" in data
        assert isinstance(data["data_version"], dict)

    def test_no_exception_when_llm_missing(self, monkeypatch):
        """Health endpoint must not raise even when no LLM is configured."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_no_exception_when_gene_index_unavailable(self, monkeypatch):
        """Simulate missing gene index — health must still return 200."""
        original = gene_index._GENE_INDEX_AVAILABLE
        try:
            gene_index._GENE_INDEX_AVAILABLE = False
            resp = client.get("/health")
            assert resp.status_code == 200
        finally:
            gene_index._GENE_INDEX_AVAILABLE = original


# ---------------------------------------------------------------------------
# GET /health/clinvar
# ---------------------------------------------------------------------------

class TestHealthClinvar:
    def test_returns_200(self):
        assert client.get("/health/clinvar").status_code == 200

    def test_has_ok_field(self):
        data = client.get("/health/clinvar").json()
        assert "ok" in data
        assert isinstance(data["ok"], bool)

    def test_ok_true_has_record_count(self):
        data = client.get("/health/clinvar").json()
        if data["ok"]:
            assert "total_records" in data
            assert isinstance(data["total_records"], int)
            assert data["total_records"] > 0

    def test_ok_false_has_reason(self):
        data = client.get("/health/clinvar").json()
        if not data["ok"]:
            assert "reason" in data
            assert isinstance(data["reason"], str)


# ---------------------------------------------------------------------------
# GET /health/gene-index
# ---------------------------------------------------------------------------

class TestHealthGeneIndex:
    def test_returns_200(self):
        assert client.get("/health/gene-index").status_code == 200

    def test_has_ok_field(self):
        data = client.get("/health/gene-index").json()
        assert "ok" in data

    def test_ok_true_has_gene_count(self):
        data = client.get("/health/gene-index").json()
        if data["ok"]:
            assert "total_genes" in data
            assert data["total_genes"] > 0

    def test_ok_false_has_reason(self):
        data = client.get("/health/gene-index").json()
        if not data["ok"]:
            assert "reason" in data

    def test_missing_index_returns_ok_false(self, monkeypatch):
        """Simulate missing gene index file."""
        original = gene_index._GENE_INDEX_AVAILABLE
        original_path = gene_index.STATS_DB_PATH
        try:
            gene_index._GENE_INDEX_AVAILABLE = False
            gene_index.STATS_DB_PATH = pathlib.Path("data/nonexistent_gene_stats.duckdb")
            data = client.get("/health/gene-index").json()
            assert data["ok"] is False
            assert "reason" in data
        finally:
            gene_index._GENE_INDEX_AVAILABLE = original
            gene_index.STATS_DB_PATH = original_path


# ---------------------------------------------------------------------------
# GET /health/llm
# ---------------------------------------------------------------------------

class TestHealthLlm:
    def test_returns_200(self):
        assert client.get("/health/llm").status_code == 200

    def test_has_ok_field(self):
        data = client.get("/health/llm").json()
        assert "ok" in data

    def test_no_env_vars_ok_false(self):
        data = client.get("/health/llm").json()
        assert data["ok"] is False

    def test_no_env_vars_has_deterministic_fallback(self):
        data = client.get("/health/llm").json()
        assert data.get("deterministic_fallback_available") is True

    def test_local_llm_url_makes_ok_true(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434")
        data = health_module.check_llm()
        assert data["ok"] is True
        assert data["provider"] == "local"

    def test_openai_key_makes_ok_true(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        data = health_module.check_llm()
        assert data["ok"] is True
        assert data["provider"] == "openai"

    def test_anthropic_key_makes_ok_true(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        data = health_module.check_llm()
        assert data["ok"] is True
        assert data["provider"] == "anthropic"

    def test_local_overrides_openai(self, monkeypatch):
        """LOCAL_LLM_URL takes priority over API keys."""
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        data = health_module.check_llm()
        assert data["provider"] == "local"


# ---------------------------------------------------------------------------
# GET /version
# ---------------------------------------------------------------------------

class TestVersionEndpoint:
    def test_returns_200(self):
        assert client.get("/version").status_code == 200

    def test_has_app_version(self):
        data = client.get("/version").json()
        assert "app_version" in data
        assert data["app_version"] == health_module.APP_VERSION

    def test_version_format(self):
        ver = client.get("/version").json()["app_version"]
        parts = ver.split(".")
        assert len(parts) == 3, f"Version should be X.Y.Z, got {ver}"
        assert all(p.isdigit() for p in parts)

    def test_has_data_version_key(self):
        data = client.get("/version").json()
        assert "data_version" in data
        assert isinstance(data["data_version"], dict)

    def test_data_version_from_file(self, tmp_path, monkeypatch):
        """When data_version.json exists, /version reflects its content."""
        dv = {
            "app_version": "2.1.0",
            "clinvar_source": "ClinVar (NCBI)",
            "gene_index_total_genes": 40344,
        }
        dv_file = tmp_path / "data_version.json"
        dv_file.write_text(json.dumps(dv), encoding="utf-8")
        monkeypatch.setattr(health_module, "_DATA_VERSION_PATH", dv_file)
        data = client.get("/version").json()
        assert data["data_version"]["gene_index_total_genes"] == 40344

    def test_data_version_empty_when_file_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing.json"
        monkeypatch.setattr(health_module, "_DATA_VERSION_PATH", missing)
        data = client.get("/version").json()
        assert data["data_version"] == {}


# ---------------------------------------------------------------------------
# check_* unit tests (direct function calls, no HTTP)
# ---------------------------------------------------------------------------

class TestHealthModuleFunctions:
    def test_check_llm_returns_dict(self):
        result = health_module.check_llm()
        assert isinstance(result, dict)
        assert "ok" in result

    def test_check_clinvar_returns_dict(self):
        result = health_module.check_clinvar()
        assert isinstance(result, dict)
        assert "ok" in result

    def test_check_gene_index_returns_dict(self):
        result = health_module.check_gene_index()
        assert isinstance(result, dict)
        assert "ok" in result

    def test_get_overall_health_returns_dict(self):
        result = health_module.get_overall_health()
        assert isinstance(result, dict)
        assert result["status"] in ("ok", "degraded", "down")

    def test_load_data_version_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_module, "_DATA_VERSION_PATH", tmp_path / "nope.json")
        assert health_module.load_data_version() == {}

    def test_load_data_version_reads_file(self, tmp_path, monkeypatch):
        payload = {"app_version": "2.1.0", "test": True}
        f = tmp_path / "data_version.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(health_module, "_DATA_VERSION_PATH", f)
        result = health_module.load_data_version()
        assert result["test"] is True
