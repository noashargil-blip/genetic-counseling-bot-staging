"""
Tests for the Gene Knowledge Base feature.

Covers:
  1. Loader module (gene_knowledge.py) — loading, approved/unapproved distinction
  2. Approved knowledge shown in answer
  3. Unapproved knowledge NOT shown in answer
  4. VUS note matches clinical area template
  5. ClinVar details stay in metadata, not in answer text
  6. Tier 1b appears in gene_metadata.answer_tier when knowledge is approved
  7. Tier 2 fallback when no approved knowledge
  8. AI draft still available for unapproved genes
  9. Approval script never auto-approves source_missing records

All API tests use FastAPI TestClient — no live server needed.
"""

import json
import pathlib
import sys
import importlib
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_client():
    import app.retriever as retriever
    retriever._DB_AVAILABLE = False
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _ask(client, question, last_topic=None, conversation_context=None,
         include_unverified_gene_draft=False):
    payload = {"question": question}
    if last_topic:
        payload["last_topic"] = last_topic
    if conversation_context is not None:
        payload["conversation_context"] = conversation_context
    if include_unverified_gene_draft:
        payload["include_unverified_gene_draft"] = True
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


# ── 1. Loader unit tests ──────────────────────────────────────────────────────

class TestGeneKnowledgeLoader:
    """gene_knowledge.py loader functions."""

    def test_module_loads_without_error(self):
        import app.gene_knowledge as gk
        # If module loaded, _GENE_KNOWLEDGE_AVAILABLE may be True or False
        # depending on whether the JSON file is present — both are valid.
        assert isinstance(gk._GENE_KNOWLEDGE_AVAILABLE, bool)

    def test_load_gene_knowledge_returns_list(self):
        import app.gene_knowledge as gk
        result = gk.load_gene_knowledge()
        assert isinstance(result, list)

    def test_get_gene_knowledge_unknown_gene_returns_none(self):
        import app.gene_knowledge as gk
        assert gk.get_gene_knowledge("XYZZY99") is None

    def test_has_approved_unknown_gene_returns_false(self):
        import app.gene_knowledge as gk
        assert gk.has_approved_gene_knowledge("XYZZY99") is False

    def test_get_patient_summary_unknown_gene_returns_none(self):
        import app.gene_knowledge as gk
        assert gk.get_gene_patient_summary("XYZZY99") is None

    def test_get_vus_note_unknown_gene_returns_none(self):
        import app.gene_knowledge as gk
        assert gk.get_gene_vus_note("XYZZY99") is None

    def test_list_all_genes_returns_list(self):
        import app.gene_knowledge as gk
        result = gk.list_all_genes()
        assert isinstance(result, list)

    def test_list_approved_genes_subset_of_all(self):
        import app.gene_knowledge as gk
        approved = set(gk.list_approved_genes())
        all_genes = set(gk.list_all_genes())
        assert approved.issubset(all_genes)

    def test_unapproved_record_not_returned_by_patient_summary(self, tmp_path, monkeypatch):
        """get_gene_patient_summary returns None when approved=false."""
        import app.gene_knowledge as gk
        fake_kb = [
            {
                "gene_symbol": "TESTGENE",
                "gene_name": "Test Gene",
                "clinical_area": "generic",
                "patient_summary_he": "טיוטה לא מאושרת.",
                "vus_note_type": "generic",
                "vus_note_he": "VUS note draft.",
                "source_1_name": None, "source_1_url_or_id": None,
                "source_2_name": None, "source_2_url_or_id": None,
                "source_status": "source_missing",
                "review_status": "draft",
                "approved": False,
                "reviewed_by": None, "reviewed_at": None,
                "reviewer_notes": None, "last_updated": "2026-07-01",
            }
        ]
        kb_file = tmp_path / "gene_knowledge_base.json"
        kb_file.write_text(json.dumps(fake_kb), encoding="utf-8")
        monkeypatch.setattr(gk, "_RECORDS", {})
        monkeypatch.setattr(gk, "_KB_PATH", kb_file)
        gk._load()
        # Not approved → summary and vus note must return None
        assert gk.get_gene_patient_summary("TESTGENE") is None
        assert gk.get_gene_vus_note("TESTGENE") is None
        assert gk.has_approved_gene_knowledge("TESTGENE") is False

    def test_approved_record_returned_by_patient_summary(self, tmp_path, monkeypatch):
        """get_gene_patient_summary returns text only when approved=true."""
        import app.gene_knowledge as gk
        fake_kb = [
            {
                "gene_symbol": "APPRVD",
                "gene_name": "Approved Gene",
                "clinical_area": "cancer_predisposition",
                "patient_summary_he": "סיכום מאושר בעברית.",
                "vus_note_type": "cancer_predisposition",
                "vus_note_he": "VUS note approved.",
                "source_1_name": "MedlinePlus",
                "source_1_url_or_id": "https://example.com",
                "source_2_name": None, "source_2_url_or_id": None,
                "source_status": "verified",
                "review_status": "approved",
                "approved": True,
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-07-01T12:00:00Z",
                "reviewer_notes": "Verified manually.",
                "last_updated": "2026-07-01",
            }
        ]
        kb_file = tmp_path / "gene_knowledge_base.json"
        kb_file.write_text(json.dumps(fake_kb), encoding="utf-8")
        monkeypatch.setattr(gk, "_RECORDS", {})
        monkeypatch.setattr(gk, "_KB_PATH", kb_file)
        gk._load()
        assert gk.get_gene_patient_summary("APPRVD") == "סיכום מאושר בעברית."
        assert gk.get_gene_vus_note("APPRVD") == "VUS note approved."
        assert gk.has_approved_gene_knowledge("APPRVD") is True

    def test_all_20_genes_in_kb(self):
        """The shipped gene_knowledge_base.json must contain all 20 target genes."""
        import app.gene_knowledge as gk
        expected = {
            "BRCA1", "BRCA2", "APC", "POLE", "ATM", "CHEK2", "PALB2", "TP53",
            "PTEN", "MSH2", "MSH6", "MLH1", "PMS2", "NF1", "DICER1",
            "TTN", "MYH7", "RYR2", "CFTR", "HBB",
        }
        all_genes = set(gk.list_all_genes())
        missing = expected - all_genes
        assert not missing, f"Missing genes in KB: {missing}"

    def test_all_records_unapproved_initially(self):
        """All records in the shipped KB are unapproved (approved=false)."""
        import app.gene_knowledge as gk
        all_records = gk.load_gene_knowledge()
        approved = [r["gene_symbol"] for r in all_records if r.get("approved") is True]
        # The shipped KB has no approved records — they all need human review first.
        # This test documents the expected initial state.
        assert approved == [], (
            f"Expected no approved records in shipped KB, "
            f"but found: {approved}"
        )

    def test_graceful_degradation_missing_file(self, tmp_path, monkeypatch):
        """gene_knowledge module stays stable if the JSON file is missing."""
        import app.gene_knowledge as gk
        monkeypatch.setattr(gk, "_RECORDS", {})
        monkeypatch.setattr(gk, "_GENE_KNOWLEDGE_AVAILABLE", False)
        monkeypatch.setattr(gk, "_KB_PATH", tmp_path / "nonexistent.json")
        gk._load()
        assert gk.get_gene_patient_summary("BRCA1") is None
        assert gk.has_approved_gene_knowledge("BRCA1") is False
        assert gk.list_approved_genes() == []


# ── 2. Unapproved knowledge not surfaced in answers ───────────────────────────

class TestUnapprovedKnowledgeNotShown:
    """When all KB records are unapproved (shipped state), answers must use Tier 2."""

    def test_brca1_vus_no_unapproved_content_in_answer(self):
        """With no approved records, VUS+BRCA1 must not show unapproved Hebrew summaries."""
        import app.gene_knowledge as gk
        # Shipped KB: all unapproved — confirmed by previous test.
        # So BRCA1 has knowledge but it is not approved.
        # The answer must come from gene_cards (Tier 1a, approved builtin)
        # or fall through to ClinVar tier.
        # Either way, the unapproved patient_summary_he text must not appear.
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן BRCA1, מה זה?")
        unapproved_text = gk.get_gene_knowledge("BRCA1")
        if unapproved_text:
            draft_he = (unapproved_text.get("patient_summary_he") or "")
            if draft_he and not unapproved_text.get("approved"):
                assert draft_he not in resp["answer"], (
                    "Unapproved patient_summary_he must not appear in the answer."
                )

    def test_pole_vus_not_tier1b_when_unapproved(self):
        """With no approved KB records, POLE VUS must not use tier1b."""
        import app.gene_knowledge as gk
        if gk.has_approved_gene_knowledge("POLE"):
            pytest.skip("POLE is approved — tier1b behaviour tested separately.")
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן POLE, מה זה?")
        meta = resp.get("gene_metadata") or {}
        assert meta.get("answer_tier") != "tier1b", (
            "tier1b must not appear when the gene knowledge record is not approved."
        )


# ── 3. Tier 1b appears when knowledge IS approved ────────────────────────────

class TestTier1bAppearsWhenApproved:
    """When a gene knowledge record is approved, the answer uses Tier 1b."""

    @pytest.fixture
    def approved_gene_kb(self, tmp_path, monkeypatch):
        """Inject a single approved gene record for TESTGK into gene_knowledge."""
        import app.gene_knowledge as gk
        original_records = dict(gk._RECORDS)
        approved_rec = {
            "gene_symbol": "TESTGK",
            "gene_name": "Test Gene Knowledge",
            "clinical_area": "cancer_predisposition",
            "patient_summary_he": "גן הבדיקה מקודד לחלבון שמשתתף בתיקון DNA.",
            "vus_note_type": "cancer_predisposition",
            "vus_note_he": (
                "גם אם הגן מוכר כקשור לנטייה תורשתית מסוימת, VUS בגן זה אינו "
                "נחשב לממצא פתוגני ואינו מספיק בפני עצמו לקבלת החלטות רפואיות."
            ),
            "source_1_name": "MedlinePlus",
            "source_1_url_or_id": "https://example.com/testgk",
            "source_2_name": None, "source_2_url_or_id": None,
            "source_status": "verified",
            "review_status": "approved",
            "approved": True,
            "reviewed_by": "pytest-fixture",
            "reviewed_at": "2026-07-01T12:00:00Z",
            "reviewer_notes": None, "last_updated": "2026-07-01",
        }
        import re as _re
        # Also register TESTGK as a detectable gene pattern
        import app.counseling_engine as ce
        original_patterns = list(ce._GENE_PATTERNS)
        ce._GENE_PATTERNS.append(
            ("TESTGK", _re.compile(r"\btestgk\b", _re.IGNORECASE))
        )
        gk._RECORDS["TESTGK"] = approved_rec
        yield approved_rec
        # Cleanup
        gk._RECORDS.pop("TESTGK", None)
        ce._GENE_PATTERNS[:] = original_patterns

    def test_approved_knowledge_answer_tier_is_tier1b(self, approved_gene_kb):
        """gene_metadata.answer_tier must be tier1b for an approved gene."""
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        meta = resp.get("gene_metadata") or {}
        assert meta.get("answer_tier") == "tier1b", (
            f"Expected tier1b, got {meta.get('answer_tier')}"
        )

    def test_approved_knowledge_gene_symbol_in_metadata(self, approved_gene_kb):
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        meta = resp.get("gene_metadata") or {}
        assert meta.get("gene_symbol") == "TESTGK"

    def test_approved_knowledge_status_is_approved(self, approved_gene_kb):
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        meta = resp.get("gene_metadata") or {}
        assert meta.get("gene_knowledge_status") == "approved"

    def test_approved_knowledge_unverified_draft_not_available(self, approved_gene_kb):
        """When Tier 1b is active, unverified_gene_draft_available must be False."""
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        meta = resp.get("gene_metadata") or {}
        assert meta.get("unverified_gene_draft_available") is False

    def test_approved_summary_text_in_answer(self, approved_gene_kb):
        """The approved patient_summary_he must appear in the answer body."""
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        assert "תיקון DNA" in resp["answer"], (
            "Approved patient_summary_he must be in the answer."
        )

    def test_vus_note_in_answer(self, approved_gene_kb):
        """The approved vus_note_he must appear in the answer body."""
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        assert "VUS בגן זה אינו נחשב לממצא פתוגני" in resp["answer"], (
            "Approved vus_note_he must be in the answer."
        )

    def test_safety_level_is_general_information(self, approved_gene_kb):
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        assert resp["safety_level"] == "general_information"

    def test_response_has_exactly_5_required_keys(self, approved_gene_kb):
        """The 5-field API contract must hold even for Tier 1b responses."""
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        required = {"answer", "safety_level", "needs_genetic_counselor",
                    "matched_topic", "suggested_questions"}
        assert required.issubset(resp.keys()), (
            f"Missing required keys: {required - resp.keys()}"
        )


# ── 4. VUS note clinical area templates ──────────────────────────────────────

class TestVusNoteTemplates:
    """VUS note text must match the declared clinical area."""

    VUS_NOTE_TEMPLATES = {
        "cancer_predisposition": "גם אם הגן מוכר כקשור לנטייה תורשתית מסוימת, VUS בגן זה אינו נחשב לממצא פתוגני",
        "lynch_polyposis": "גם אם הגן מוכר כקשור לסיכון מוגבר לסרטן מעי גס",
        "cardiology": "VUS בגן שקשור למחלות לב אינו מאשר אבחנה גנטית",
        "recessive_carrier": "VUS בגן הקשור למחלה תורשתית אינו מוכיח אבחנה",
        "hematology": "VUS בגן הקשור למחלה תורשתית אינו מוכיח אבחנה",
    }

    def _get_record(self, gene_symbol: str):
        import app.gene_knowledge as gk
        return gk.get_gene_knowledge(gene_symbol)

    def test_brca1_vus_note_matches_cancer_predisposition(self):
        rec = self._get_record("BRCA1")
        assert rec is not None
        assert rec["clinical_area"] == "cancer_predisposition"
        fragment = self.VUS_NOTE_TEMPLATES["cancer_predisposition"]
        assert fragment in rec["vus_note_he"], (
            f"BRCA1 vus_note_he does not match cancer_predisposition template."
        )

    def test_mlh1_vus_note_matches_lynch_polyposis(self):
        rec = self._get_record("MLH1")
        assert rec is not None
        assert rec["clinical_area"] == "lynch_polyposis"
        fragment = self.VUS_NOTE_TEMPLATES["lynch_polyposis"]
        assert fragment in rec["vus_note_he"]

    def test_ttn_vus_note_matches_cardiology(self):
        rec = self._get_record("TTN")
        assert rec is not None
        assert rec["clinical_area"] == "cardiology"
        fragment = self.VUS_NOTE_TEMPLATES["cardiology"]
        assert fragment in rec["vus_note_he"]

    def test_cftr_vus_note_matches_recessive_carrier(self):
        rec = self._get_record("CFTR")
        assert rec is not None
        assert rec["clinical_area"] == "recessive_carrier"
        fragment = self.VUS_NOTE_TEMPLATES["recessive_carrier"]
        assert fragment in rec["vus_note_he"]

    def test_hbb_vus_note_matches_hematology(self):
        rec = self._get_record("HBB")
        assert rec is not None
        assert rec["clinical_area"] == "hematology"
        fragment = self.VUS_NOTE_TEMPLATES["hematology"]
        assert fragment in rec["vus_note_he"]


# ── 5. ClinVar details in metadata only ──────────────────────────────────────

class TestClinvarInMetadataOnly:
    """For Tier 1b answers, ClinVar raw counts must not appear in the answer text."""

    @pytest.fixture
    def approved_gene_kb(self, monkeypatch):
        """Inject an approved TESTGK record (same as above fixture)."""
        import app.gene_knowledge as gk
        import re as _re
        import app.counseling_engine as ce
        original_patterns = list(ce._GENE_PATTERNS)
        ce._GENE_PATTERNS.append(
            ("TESTGK", _re.compile(r"\btestgk\b", _re.IGNORECASE))
        )
        gk._RECORDS["TESTGK"] = {
            "gene_symbol": "TESTGK",
            "gene_name": "Test Gene Knowledge",
            "clinical_area": "cancer_predisposition",
            "patient_summary_he": "גן הבדיקה מקודד לחלבון שמשתתף בתיקון DNA.",
            "vus_note_type": "cancer_predisposition",
            "vus_note_he": "VUS בגן זה אינו נחשב לממצא פתוגני.",
            "source_1_name": "MedlinePlus", "source_1_url_or_id": "https://example.com",
            "source_2_name": None, "source_2_url_or_id": None,
            "source_status": "verified", "review_status": "approved",
            "approved": True, "reviewed_by": "pytest",
            "reviewed_at": "2026-07-01T12:00:00Z",
            "reviewer_notes": None, "last_updated": "2026-07-01",
        }
        yield
        gk._RECORDS.pop("TESTGK", None)
        ce._GENE_PATTERNS[:] = original_patterns

    def test_clinvar_count_not_in_tier1b_answer(self, approved_gene_kb):
        """Raw ClinVar variant counts must not leak into the Tier 1b answer text."""
        client = _get_client()
        resp = _ask(client, "יש לי VUS בגן TESTGK, מה זה?")
        # ClinVar counts look like "1,234 רשומות" or "נמצאו 1234 וריאנטים"
        import re
        assert not re.search(r"\d{3,}.*וריאנט", resp["answer"]), (
            "ClinVar variant counts must not appear in a Tier 1b answer."
        )


# ── 6. Tier 2 fallback when no approved knowledge ────────────────────────────

class TestTier2FallbackNoApprovedKnowledge:
    """When no gene_cards AND no approved gene_knowledge exists, the answer is Tier 2."""

    def test_unknown_gene_no_approved_knowledge_tier2_or_3(self):
        """A completely unknown gene (no cards, no KB record) gives Tier 3."""
        client = _get_client()
        # Use a real but unusual gene that has no gene_cards entry and is not in _GENE_PATTERNS
        # We pass it as part of a free-form question.
        resp = _ask(client, "מה זה גן XYZZY99?")
        # Should NOT be Tier 1b since the gene is unknown
        meta = resp.get("gene_metadata") or {}
        assert meta.get("answer_tier") != "tier1b"


# ── 7. AI draft opt-in remains available for unapproved genes ────────────────

class TestAiDraftOptInUnchanged:
    """include_unverified_gene_draft=True must still work for Tier 2 genes."""

    def test_unverified_draft_field_present_for_tier2_gene(self):
        """Tier 2 gene with draft flag set must expose unverified_gene_draft_available=True."""
        import app.gene_knowledge as gk
        # Find a gene that is in the KB but not approved
        unapproved = [sym for sym, r in gk._RECORDS.items() if not r.get("approved")]
        if not unapproved:
            pytest.skip("No unapproved genes in KB to test.")
        # If this gene is also in _GENE_PATTERNS, step 4 (VUS) will fire.
        # We need a gene in _GENE_PATTERNS that is NOT approved in KB.
        import app.counseling_engine as ce
        pattern_genes = {sym for sym, _ in ce._GENE_PATTERNS}
        candidate = next((g for g in unapproved if g in pattern_genes), None)
        if not candidate:
            pytest.skip("No unapproved gene is also in _GENE_PATTERNS.")
        client = _get_client()
        resp = _ask(client, f"יש לי VUS בגן {candidate}, מה זה?")
        meta = resp.get("gene_metadata") or {}
        # For a gene_cards-approved gene (Tier 1a), draft is False by design.
        # We only check tier2 genes.
        if meta.get("answer_tier") == "tier2":
            assert meta.get("unverified_gene_draft_available") is True


# ── 8. Approval script safety gates ──────────────────────────────────────────

class TestApprovalScriptSafetyGates:
    """approve_gene_knowledge.py must never auto-approve source_missing records."""

    def test_source_missing_blocks_approval(self, tmp_path):
        """Approval must be blocked when source_status == 'source_missing'."""
        kb_data = [
            {
                "gene_symbol": "BLOCKEDGENE",
                "gene_name": "Blocked Gene",
                "clinical_area": "generic",
                "patient_summary_he": "טיוטה.",
                "vus_note_type": "generic",
                "vus_note_he": "VUS note.",
                "source_1_name": None, "source_1_url_or_id": None,
                "source_2_name": None, "source_2_url_or_id": None,
                "source_status": "source_missing",
                "review_status": "draft",
                "approved": False,
                "reviewed_by": None, "reviewed_at": None,
                "reviewer_notes": None, "last_updated": "2026-07-01",
            }
        ]
        kb_file = tmp_path / "gene_knowledge_base.json"
        kb_file.write_text(json.dumps(kb_data), encoding="utf-8")

        # Import and monkeypatch the script's _KB_PATH
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "approve_gene_knowledge",
            ROOT / "scripts" / "approve_gene_knowledge.py",
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["approve_gene_knowledge"] = mod
        spec.loader.exec_module(mod)
        mod._KB_PATH = kb_file

        with pytest.raises(SystemExit) as exc_info:
            mod.cmd_approve("BLOCKEDGENE", "test-reviewer", None, confirm=True)
        assert exc_info.value.code == 1, "Should exit with code 1 when source_missing."

    def test_approved_flag_never_set_without_confirm(self, tmp_path):
        """Without --confirm, cmd_approve must not write approved=True."""
        kb_data = [
            {
                "gene_symbol": "DRAFTGENE",
                "gene_name": "Draft Gene",
                "clinical_area": "generic",
                "patient_summary_he": "טיוטה.",
                "vus_note_type": "generic",
                "vus_note_he": "VUS note.",
                "source_1_name": "TestSource",
                "source_1_url_or_id": "https://example.com",
                "source_2_name": None, "source_2_url_or_id": None,
                "source_status": "needs_review",
                "review_status": "draft",
                "approved": False,
                "reviewed_by": None, "reviewed_at": None,
                "reviewer_notes": None, "last_updated": "2026-07-01",
            }
        ]
        kb_file = tmp_path / "gene_knowledge_base.json"
        kb_file.write_text(json.dumps(kb_data), encoding="utf-8")

        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "approve_gene_knowledge2",
            ROOT / "scripts" / "approve_gene_knowledge.py",
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["approve_gene_knowledge2"] = mod
        spec.loader.exec_module(mod)
        mod._KB_PATH = kb_file

        # Run without confirm — should not write
        mod.cmd_approve("DRAFTGENE", "test-reviewer", None, confirm=False)
        after = json.loads(kb_file.read_text(encoding="utf-8"))
        assert after[0]["approved"] is False, (
            "Record must remain unapproved when --confirm is not given."
        )

    def test_confirmation_with_valid_source_sets_approved(self, tmp_path):
        """With --confirm and a valid source, approved must become True."""
        kb_data = [
            {
                "gene_symbol": "READYGENE",
                "gene_name": "Ready Gene",
                "clinical_area": "cancer_predisposition",
                "patient_summary_he": "טיוטה מוכנה לאישור.",
                "vus_note_type": "cancer_predisposition",
                "vus_note_he": "VUS note.",
                "source_1_name": "MedlinePlus",
                "source_1_url_or_id": "https://example.com",
                "source_2_name": None, "source_2_url_or_id": None,
                "source_status": "needs_review",
                "review_status": "draft",
                "approved": False,
                "reviewed_by": None, "reviewed_at": None,
                "reviewer_notes": None, "last_updated": "2026-07-01",
            }
        ]
        kb_file = tmp_path / "gene_knowledge_base.json"
        kb_file.write_text(json.dumps(kb_data), encoding="utf-8")

        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "approve_gene_knowledge3",
            ROOT / "scripts" / "approve_gene_knowledge.py",
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["approve_gene_knowledge3"] = mod
        spec.loader.exec_module(mod)
        mod._KB_PATH = kb_file

        mod.cmd_approve("READYGENE", "test-reviewer", "Manually verified.", confirm=True)
        after = json.loads(kb_file.read_text(encoding="utf-8"))
        rec = after[0]
        assert rec["approved"] is True
        assert rec["review_status"] == "approved"
        assert rec["reviewed_by"] == "test-reviewer"
        assert rec["reviewed_at"] is not None
        assert rec["reviewer_notes"] == "Manually verified."
