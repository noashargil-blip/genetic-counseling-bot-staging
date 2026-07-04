# -*- coding: utf-8 -*-
"""
Tests for the gene-level ClinVar index and three new FastAPI endpoints:
  GET /genes
  GET /gene/{gene_symbol}/summary
  GET /gene/{gene_symbol}/variants

Test strategy
-------------
* When the gene index IS available (data/clinvar.duckdb present and
  data/clinvar_gene_stats.duckdb successfully built): all database-backed
  tests run and must pass for BRCA1, BRCA2, NF1, SHANK3, TP53, CFTR.

* When the gene index is NOT available (DB files absent — local dev): those
  tests are skipped via `needs_gene_index` mark.  Only the "unavailable"
  tests (TestGeneIndexUnavailable) always run, verifying that the endpoints
  return HTTP 503 with a meaningful error when the index is down.

Safety invariants verified on every non-error response
-------------------------------------------------------
1. "metadata" key is present.
2. metadata["disclaimer"] is a non-empty string.
3. metadata["source"] == "ClinVar".
4. No personal risk estimates, no treatment recommendations.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import gene_index

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared helpers and marks
# ---------------------------------------------------------------------------

#: Decorator that skips a test class / function when the gene index is absent.
needs_gene_index = pytest.mark.skipif(
    not gene_index._GENE_INDEX_AVAILABLE,
    reason="ClinVar gene index not available (data/clinvar.duckdb missing or unreadable)",
)

#: The six genes required by the task specification.
REQUIRED_GENES = ["BRCA1", "BRCA2", "NF1", "SHANK3", "TP53", "CFTR"]


def _assert_metadata(data: dict) -> None:
    """Validate that every response carries correct safety metadata."""
    assert "metadata" in data, "Response missing 'metadata' key"
    meta = data["metadata"]
    assert meta.get("source") == "ClinVar"
    assert isinstance(meta.get("disclaimer"), str) and len(meta["disclaimer"]) > 10
    assert isinstance(meta.get("data_note"), str) and len(meta["data_note"]) > 5


# ---------------------------------------------------------------------------
# 1. /genes — gene list endpoint
# ---------------------------------------------------------------------------

@needs_gene_index
class TestGenesList:
    def test_status_200(self):
        resp = client.get("/genes")
        assert resp.status_code == 200

    def test_response_keys(self):
        data = client.get("/genes").json()
        for key in ("total_genes", "returned", "offset", "limit", "genes", "metadata"):
            assert key in data, f"Missing key: {key}"

    def test_metadata_present(self):
        _assert_metadata(client.get("/genes").json())

    def test_total_genes_positive(self):
        data = client.get("/genes").json()
        assert data["total_genes"] > 0

    def test_genes_is_list(self):
        data = client.get("/genes").json()
        assert isinstance(data["genes"], list)

    def test_each_gene_has_required_fields(self):
        data = client.get("/genes?limit=10").json()
        for entry in data["genes"]:
            assert "gene_symbol" in entry
            assert "total_variants" in entry
            assert isinstance(entry["gene_symbol"], str) and entry["gene_symbol"]
            assert isinstance(entry["total_variants"], int) and entry["total_variants"] > 0

    def test_sorted_by_total_variants_descending(self):
        data = client.get("/genes?limit=50").json()
        counts = [e["total_variants"] for e in data["genes"]]
        assert counts == sorted(counts, reverse=True)

    def test_pagination_limit_respected(self):
        data = client.get("/genes?limit=5").json()
        assert data["returned"] <= 5
        assert len(data["genes"]) <= 5

    def test_pagination_offset(self):
        page0 = client.get("/genes?limit=5&offset=0").json()["genes"]
        page1 = client.get("/genes?limit=5&offset=5").json()["genes"]
        if page1:
            symbols0 = {g["gene_symbol"] for g in page0}
            symbols1 = {g["gene_symbol"] for g in page1}
            assert symbols0.isdisjoint(symbols1), "Offset pagination returned overlapping genes"

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_required_gene_in_index(self, gene):
        """Each of the 6 required genes must appear somewhere in the full gene list."""
        offset = 0
        batch = 500
        found = False
        while True:
            data = client.get(f"/genes?limit={batch}&offset={offset}").json()
            genes_batch = data["genes"]
            if not genes_batch:
                break
            if any(g["gene_symbol"] == gene for g in genes_batch):
                found = True
                break
            if data["returned"] < batch:
                break
            offset += batch
        assert found, f"Gene '{gene}' was not found in /genes response"


# ---------------------------------------------------------------------------
# 2. /gene/{symbol}/summary — per-gene statistics
# ---------------------------------------------------------------------------

@needs_gene_index
class TestGeneSummary:
    def test_brca1_status_200(self):
        assert client.get("/gene/BRCA1/summary").status_code == 200

    def test_unknown_gene_404(self):
        resp = client.get("/gene/GENE_DOES_NOT_EXIST_XYZ/summary")
        assert resp.status_code == 404

    def test_response_keys(self):
        data = client.get("/gene/BRCA1/summary").json()
        for key in ("gene_symbol", "statistics", "index_built_at", "metadata"):
            assert key in data, f"Missing key: {key}"

    def test_statistics_keys(self):
        stats = client.get("/gene/BRCA1/summary").json()["statistics"]
        for key in (
            "total_variants",
            "by_significance",
            "by_review_status",
            "phenotypes",
            "variant_types",
            "date_range",
        ):
            assert key in stats, f"Missing statistics key: {key}"

    def test_gene_symbol_normalised_uppercase(self):
        data_lower = client.get("/gene/brca1/summary").json()
        data_upper = client.get("/gene/BRCA1/summary").json()
        assert data_lower["gene_symbol"] == data_upper["gene_symbol"] == "BRCA1"

    def test_metadata_present(self):
        _assert_metadata(client.get("/gene/BRCA1/summary").json())

    def test_total_variants_positive(self):
        stats = client.get("/gene/BRCA1/summary").json()["statistics"]
        assert stats["total_variants"] > 0

    def test_by_significance_is_dict(self):
        stats = client.get("/gene/BRCA1/summary").json()["statistics"]
        assert isinstance(stats["by_significance"], dict)
        assert len(stats["by_significance"]) > 0

    def test_by_significance_values_positive(self):
        sig = client.get("/gene/BRCA1/summary").json()["statistics"]["by_significance"]
        assert all(isinstance(v, int) and v > 0 for v in sig.values())

    def test_by_review_status_is_dict(self):
        stats = client.get("/gene/BRCA1/summary").json()["statistics"]
        assert isinstance(stats["by_review_status"], dict)
        assert len(stats["by_review_status"]) > 0

    def test_phenotypes_is_list(self):
        stats = client.get("/gene/BRCA1/summary").json()["statistics"]
        assert isinstance(stats["phenotypes"], list)

    def test_phenotypes_no_excluded_values(self):
        phenos = client.get("/gene/BRCA1/summary").json()["statistics"]["phenotypes"]
        excluded = {"not provided", "not specified", "not applicable"}
        for p in phenos:
            assert p.lower() not in excluded, f"Excluded phenotype in response: {p!r}"

    def test_variant_types_is_dict(self):
        stats = client.get("/gene/BRCA1/summary").json()["statistics"]
        assert isinstance(stats["variant_types"], dict)

    def test_date_range_structure(self):
        dr = client.get("/gene/BRCA1/summary").json()["statistics"]["date_range"]
        assert "earliest" in dr and "latest" in dr

    def test_index_built_at_present(self):
        data = client.get("/gene/BRCA1/summary").json()
        assert data.get("index_built_at")

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_required_gene_summary(self, gene):
        """Each required gene must return HTTP 200 with positive total_variants."""
        resp = client.get(f"/gene/{gene}/summary")
        assert resp.status_code == 200, f"GET /gene/{gene}/summary returned {resp.status_code}"
        data = resp.json()
        stats = data["statistics"]
        assert stats["total_variants"] > 0, f"{gene}: total_variants is 0"
        assert isinstance(stats["by_significance"], dict), f"{gene}: by_significance not a dict"
        _assert_metadata(data)

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_required_gene_has_at_least_one_significance_category(self, gene):
        stats = client.get(f"/gene/{gene}/summary").json()["statistics"]
        assert len(stats["by_significance"]) >= 1, (
            f"{gene}: expected at least one significance category"
        )

    def test_brca1_significance_sum_equals_total(self):
        """Sum of by_significance counts must equal total_variants for BRCA1."""
        data = client.get("/gene/BRCA1/summary").json()
        stats = data["statistics"]
        total = stats["total_variants"]
        sig_sum = sum(stats["by_significance"].values())
        assert sig_sum == total, (
            f"BRCA1: sum of by_significance ({sig_sum}) != total_variants ({total})"
        )

    def test_brca2_significance_sum_equals_total(self):
        data = client.get("/gene/BRCA2/summary").json()
        stats = data["statistics"]
        assert sum(stats["by_significance"].values()) == stats["total_variants"]

    def test_no_personal_risk_language_in_response(self):
        """Summary JSON must not contain personal risk or treatment phrases."""
        text = str(client.get("/gene/BRCA1/summary").json())
        forbidden = [
            "your risk", "you should", "recommended surgery",
            "you have cancer", "מסוכן לך",
        ]
        for phrase in forbidden:
            assert phrase.lower() not in text.lower(), (
                f"Forbidden phrase in /gene/BRCA1/summary response: {phrase!r}"
            )


# ---------------------------------------------------------------------------
# 3. /gene/{symbol}/variants — variant records
# ---------------------------------------------------------------------------

@needs_gene_index
class TestGeneVariants:
    def test_brca1_status_200(self):
        assert client.get("/gene/BRCA1/variants").status_code == 200

    def test_unknown_gene_404(self):
        resp = client.get("/gene/GENE_DOES_NOT_EXIST_XYZ/variants")
        assert resp.status_code == 404

    def test_response_keys(self):
        data = client.get("/gene/BRCA1/variants?limit=5").json()
        for key in ("gene_symbol", "returned", "offset", "limit", "variants", "metadata"):
            assert key in data, f"Missing key: {key}"

    def test_metadata_present(self):
        _assert_metadata(client.get("/gene/BRCA1/variants?limit=5").json())

    def test_gene_symbol_uppercase(self):
        data = client.get("/gene/brca1/variants?limit=1").json()
        assert data["gene_symbol"] == "BRCA1"

    def test_variants_is_list(self):
        data = client.get("/gene/BRCA1/variants?limit=5").json()
        assert isinstance(data["variants"], list)

    def test_limit_respected(self):
        data = client.get("/gene/BRCA1/variants?limit=3").json()
        assert data["returned"] <= 3
        assert len(data["variants"]) <= 3

    def test_variant_records_have_gene_symbol(self):
        data = client.get("/gene/BRCA1/variants?limit=10").json()
        for v in data["variants"]:
            assert v.get("gene_symbol") == "BRCA1", (
                f"Variant has wrong gene_symbol: {v.get('gene_symbol')}"
            )

    def test_significance_filter(self):
        data = client.get("/gene/BRCA1/variants?limit=50&significance=Pathogenic").json()
        for v in data["variants"]:
            sig = str(v.get("clinical_significance", "")).lower()
            assert "pathogenic" in sig, (
                f"Variant with significance={sig!r} passed Pathogenic filter"
            )

    def test_significance_filter_field_in_response(self):
        data = client.get("/gene/BRCA1/variants?significance=Benign").json()
        assert data["significance_filter"] == "Benign"

    def test_significance_filter_none_when_not_provided(self):
        data = client.get("/gene/BRCA1/variants").json()
        assert data["significance_filter"] is None

    def test_pagination_offset(self):
        page0 = client.get("/gene/BRCA1/variants?limit=5&offset=0").json()["variants"]
        page1 = client.get("/gene/BRCA1/variants?limit=5&offset=5").json()["variants"]
        if page0 and page1:
            # ClinVar has one row per genome assembly per variant (GRCh37 + GRCh38),
            # so variation_id is not unique per row. Compare full row content instead.
            rows0 = {str(v) for v in page0}
            rows1 = {str(v) for v in page1}
            assert rows0.isdisjoint(rows1), "Pagination returned identical variant rows"

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_required_gene_returns_variants(self, gene):
        """Each required gene must return at least one variant record."""
        data = client.get(f"/gene/{gene}/variants?limit=5").json()
        assert data["returned"] >= 1, f"{gene}: no variants returned"
        assert len(data["variants"]) >= 1


# ---------------------------------------------------------------------------
# 4. Unavailability — always run (no skip), monkeypatch index flag
# ---------------------------------------------------------------------------

class TestGeneIndexUnavailable:
    """
    Verifies that all three endpoints return HTTP 503 with a meaningful error
    body when the gene index is not available.  These tests always run
    regardless of whether the real DB is present.
    """

    def test_get_genes_503(self, monkeypatch):
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", False)
        resp = client.get("/genes")
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data

    def test_get_gene_summary_503(self, monkeypatch):
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", False)
        resp = client.get("/gene/BRCA1/summary")
        assert resp.status_code == 503

    def test_get_gene_variants_503(self, monkeypatch):
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", False)
        resp = client.get("/gene/BRCA1/variants")
        assert resp.status_code == 503

    def test_503_detail_contains_error_key(self, monkeypatch):
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", False)
        data = client.get("/genes").json()
        assert "detail" in data
        assert "error" in data["detail"]

    def test_503_detail_contains_reason(self, monkeypatch):
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", False)
        data = client.get("/genes").json()
        assert "reason" in data["detail"]
        assert len(data["detail"]["reason"]) > 10


# ---------------------------------------------------------------------------
# 5. Module-level unit tests (no HTTP layer)
# ---------------------------------------------------------------------------

@needs_gene_index
class TestGeneIndexModuleFunctions:
    def test_count_genes_positive(self):
        assert gene_index.count_genes() > 0

    def test_list_genes_returns_list(self):
        result = gene_index.list_genes(limit=10)
        assert isinstance(result, list)
        assert len(result) <= 10

    def test_list_genes_each_entry_fields(self):
        for entry in gene_index.list_genes(limit=5):
            assert "gene_symbol" in entry
            assert "total_variants" in entry

    def test_get_gene_summary_brca1(self):
        s = gene_index.get_gene_summary("BRCA1")
        assert s is not None
        assert s["gene_symbol"] == "BRCA1"
        assert s["total_variants"] > 0

    def test_get_gene_summary_none_for_unknown(self):
        assert gene_index.get_gene_summary("GENE_XYZ_UNKNOWN_9999") is None

    def test_get_gene_summary_lowercase_input(self):
        s = gene_index.get_gene_summary("brca2")
        assert s is not None
        assert s["gene_symbol"] == "BRCA2"

    def test_get_gene_variants_brca1_nonempty(self):
        variants = gene_index.get_gene_variants("BRCA1", limit=5)
        assert len(variants) >= 1

    def test_get_gene_variants_returns_only_target_gene(self):
        variants = gene_index.get_gene_variants("TP53", limit=10)
        for v in variants:
            assert v.get("gene_symbol") == "TP53"

    def test_metadata_dict_shape(self):
        meta = gene_index.METADATA
        assert "source" in meta
        assert "disclaimer" in meta
        assert "data_note" in meta
        assert meta["source"] == "ClinVar"

    @pytest.mark.parametrize("gene", REQUIRED_GENES)
    def test_required_gene_summary_non_null(self, gene):
        s = gene_index.get_gene_summary(gene)
        assert s is not None, f"get_gene_summary({gene!r}) returned None"
        assert s["total_variants"] > 0
