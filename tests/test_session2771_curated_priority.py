"""
Session 27.7.1 tests — curated gene priority over approved drafts, and
chromosome education AI draft lifecycle.

Run via:
  PYTHONUTF8=1 python -m pytest tests/test_session2771_curated_priority.py -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

_REQUIRED_KEYS = {"answer", "safety_level", "needs_genetic_counselor",
                  "matched_topic", "suggested_questions"}


def _ask(question: str, **kwargs) -> dict:
    payload = {"question": question, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


class _FakeSummary:
    def get(self, key, default=None):
        return {"total_variants": 20, "by_significance": {}, "phenotypes": []}.get(key, default)


# ── Part 1: Curated gene priority over physician-approved drafts ───────────

class TestCuratedPriorityOverApprovedDraft:
    """
    Curated (Tier 1a / Tier 1b) answers must always outrank approved physician
    drafts.  The approved-draft lookup must NOT be reached when curated content
    exists; it fires only in the Tier 2 gap (no curated card, no KB record).
    """

    APPROVED_DRAFT_TEXT = "WRONG: approved draft should not appear when curated exists."

    def _setup_tier1a(self, monkeypatch, gene: str, curated_text: str):
        """Mock: gene in ClinVar index + curated Tier 1a card + approved draft."""
        monkeypatch.setattr("app.gene_index._GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(
            "app.gene_index.get_gene_summary",
            lambda g: _FakeSummary() if g == gene else None,
        )
        monkeypatch.setattr(
            "app.gene_cards.get_approved_summary",
            lambda g: curated_text if g == gene else None,
        )
        import app.review_db as rdb
        monkeypatch.setattr(
            rdb, "get_approved_draft",
            lambda *a, **kw: {
                "effective_text": self.APPROVED_DRAFT_TEXT,
                "status": "approved",
            },
        )

    def _setup_tier1b(self, monkeypatch, gene: str, curated_text: str):
        """Mock: gene in ClinVar index + Tier 1b KB record + approved draft."""
        monkeypatch.setattr("app.gene_index._GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(
            "app.gene_index.get_gene_summary",
            lambda g: _FakeSummary() if g == gene else None,
        )
        monkeypatch.setattr("app.gene_cards.get_approved_summary", lambda g: None)
        monkeypatch.setattr(
            "app.gene_knowledge.get_gene_patient_summary",
            lambda g: curated_text if g == gene else None,
        )
        monkeypatch.setattr(
            "app.gene_knowledge.get_gene_vus_note", lambda g: None
        )
        import app.review_db as rdb
        monkeypatch.setattr(
            rdb, "get_approved_draft",
            lambda *a, **kw: {
                "effective_text": self.APPROVED_DRAFT_TEXT,
                "status": "approved",
            },
        )

    def test_hbb_curated_outranks_approved_draft(self, monkeypatch):
        curated = "HBB קודד לשרשרת הביתא של המוגלובין."
        self._setup_tier1a(monkeypatch, "HBB", curated)
        data = _ask("מה הגן HBB?")
        assert curated in data["answer"], f"Curated text missing from answer"
        assert self.APPROVED_DRAFT_TEXT not in data["answer"], "Approved draft overrode curated answer"

    def test_apoe_curated_outranks_approved_draft(self, monkeypatch):
        curated = "APOE מקודד לאפוליפופרוטאין E המעורב בחילוף חומרים של שומנים."
        self._setup_tier1a(monkeypatch, "APOE", curated)
        data = _ask("מה הגן APOE?")
        assert curated in data["answer"]
        assert self.APPROVED_DRAFT_TEXT not in data["answer"]

    def test_tnf_curated_outranks_approved_draft(self, monkeypatch):
        curated = "TNF קודד לגורם נמק גידולי המעורב בתגובה דלקתית."
        self._setup_tier1b(monkeypatch, "TNF", curated)
        data = _ask("מה הגן TNF?")
        assert curated in data["answer"]
        assert self.APPROVED_DRAFT_TEXT not in data["answer"]

    def test_tyr_curated_outranks_approved_draft(self, monkeypatch):
        curated = "TYR קודד לאנזים טירוזינאז המעורב בייצור מלנין."
        self._setup_tier1b(monkeypatch, "TYR", curated)
        data = _ask("מה הגן TYR?")
        assert curated in data["answer"]
        assert self.APPROVED_DRAFT_TEXT not in data["answer"]

    def test_answer_tier_is_tier1_not_tier2(self, monkeypatch):
        """Gene with curated Tier 1a answer must have answer_tier=tier1."""
        curated = "BRCA1 מקודד לחלבון תיקון DNA."
        self._setup_tier1a(monkeypatch, "BRCA1", curated)
        data = _ask("מה הגן BRCA1?")
        meta = data.get("gene_metadata", {})
        assert meta.get("answer_tier") == "tier1"

    def test_draft_promoted_false_for_tier1(self, monkeypatch):
        """Tier 1 answers must always have draft_promoted_to_answer=False (or absent)."""
        curated = "TP53 מקודד לחלבון p53."
        self._setup_tier1a(monkeypatch, "TP53", curated)
        data = _ask("מה הגן TP53?")
        meta = data.get("gene_metadata", {})
        assert not meta.get("draft_promoted_to_answer", False)


# ── Part 1b: CCR5 gap-filling ──────────────────────────────────────────────

class TestCCR5ApprovedDraftGapFill:
    """
    CCR5 (Tier 2): no curated card, no KB record.
    Approved draft must fill the gap and become the main answer.
    """

    def _setup_ccr5_tier2(self, monkeypatch, approved_text: str,
                           physician_edited: str = None):
        monkeypatch.setattr("app.gene_index._GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(
            "app.gene_index.get_gene_summary",
            lambda g: _FakeSummary() if g == "CCR5" else None,
        )
        monkeypatch.setattr("app.gene_cards.get_approved_summary", lambda g: None)
        monkeypatch.setattr("app.gene_knowledge.get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr("app.gene_knowledge.get_gene_vus_note", lambda g: None)

        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_generate_unverified_gene_draft", lambda *a, **kw: None)

        import app.review_db as rdb
        effective = physician_edited or approved_text
        monkeypatch.setattr(
            rdb, "get_approved_draft",
            lambda gene_symbol, draft_type="gene_summary": {
                "effective_text": effective,
                "original_ai_text": approved_text,
                "physician_edited_text": physician_edited,
                "status": "approved",
            } if gene_symbol == "CCR5" else None,
        )

    def test_approved_text_replaces_fallback(self, monkeypatch):
        approved = "CCR5 מקודד לקולטן כמוקין CCR5."
        self._setup_ccr5_tier2(monkeypatch, approved)
        data = _ask("מה הגן CCR5?")
        assert approved in data["answer"]
        assert "אין לי סיכום ביולוגי" not in data["answer"]

    def test_physician_edited_text_used(self, monkeypatch):
        original = "CCR5 original AI text."
        edited = "CCR5 ערוך על ידי רופא: הגן מקודד לקולטן."
        self._setup_ccr5_tier2(monkeypatch, original, physician_edited=edited)
        data = _ask("מה הגן CCR5?")
        assert edited in data["answer"], "physician_edited_text not used"
        assert original not in data["answer"], "original AI text shown instead of edited"

    def test_draft_promoted_true_for_approved_ccr5(self, monkeypatch):
        approved = "CCR5 אושר על ידי רופא."
        self._setup_ccr5_tier2(monkeypatch, approved)
        data = _ask("מה הגן CCR5?")
        meta = data.get("gene_metadata", {})
        assert meta.get("draft_promoted_to_answer") is True

    def test_gene_knowledge_status_approved_when_promoted(self, monkeypatch):
        approved = "CCR5 אושר על ידי רופא."
        self._setup_ccr5_tier2(monkeypatch, approved)
        data = _ask("מה הגן CCR5?")
        meta = data.get("gene_metadata", {})
        assert meta.get("gene_knowledge_status") == "approved"


class TestRejectedDraftIgnored:
    """Rejected and needs_revision records must not appear as the main answer."""

    def _setup_tier2_no_approved(self, monkeypatch, gene: str):
        """Gene in ClinVar index, no curated content, no approved draft (simulates rejected/NR)."""
        import app.counseling_engine as ce
        import app.review_db as rdb

        monkeypatch.setattr("app.gene_index._GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(
            "app.gene_index.get_gene_summary",
            lambda g: _FakeSummary() if g == gene else None,
        )
        monkeypatch.setattr("app.gene_cards.get_approved_summary", lambda g: None)
        monkeypatch.setattr("app.gene_knowledge.get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr("app.gene_knowledge.get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(ce, "_generate_unverified_gene_draft", lambda *a, **kw: None)
        # get_approved_draft returns None — simulates rejected/needs_revision filtered at SQL level
        monkeypatch.setattr(rdb, "get_approved_draft", lambda *a, **kw: None)
        # Also mock gene detection so the routing sees this as a known gene
        monkeypatch.setattr(ce, "_detect_known_gene", lambda text: gene if gene in text else None)
        monkeypatch.setattr(ce, "_extract_gene_with_correction", lambda text: (gene, None) if gene in text else (None, None))

    def test_rejected_draft_means_fallback_shown(self, monkeypatch):
        self._setup_tier2_no_approved(monkeypatch, "CCR5")
        data = _ask("מה הגן CCR5?")
        assert "אין לי סיכום ביולוגי" in data["answer"], f"Expected fallback, got: {data['answer'][:200]}"

    def test_needs_revision_means_fallback_shown(self, monkeypatch):
        self._setup_tier2_no_approved(monkeypatch, "APOE")
        data = _ask("מה הגן APOE?")
        assert "אין לי סיכום ביולוגי" in data["answer"], f"Expected fallback, got: {data['answer'][:200]}"

    def test_draft_promoted_false_when_no_approved(self, monkeypatch):
        self._setup_tier2_no_approved(monkeypatch, "TNF")
        data = _ask("מה הגן TNF?")
        meta = data.get("gene_metadata", {})
        assert not meta.get("draft_promoted_to_answer", False)


# ── Part 2: Chromosome AI draft lifecycle ─────────────────────────────────

class TestChromosomeDraftLifecycle:
    """
    Chromosome education answers:
    - deterministic KB answer is always the main answer
    - AI draft is supplemental (immediate mode only)
    - approved chromosome draft stays supplemental (never replaces KB answer)
    """

    def _mock_chromosome_draft(self, monkeypatch, draft_text: str):
        """Mock LLM to return a chromosome draft."""
        import app.counseling_engine as ce
        monkeypatch.setattr(
            ce, "_generate_chromosome_education_draft",
            lambda question, sub_intent, _debug=None: {
                "text_he": draft_text,
                "sub_intent": sub_intent,
                "generated_by_model": "mock",
                "review_status": "pending",
                "approved": False,
                "warning_he": "טיוטת AI לא מבוקרת.",
            },
        )

    def _mock_no_chromosome_draft(self, monkeypatch):
        import app.counseling_engine as ce
        monkeypatch.setattr(
            ce, "_generate_chromosome_education_draft",
            lambda *a, **kw: None,
        )

    def test_main_answer_is_always_kb_not_draft(self, monkeypatch):
        """Main answer must be the KB text even when AI draft is available."""
        from app.counseling_engine import _CYTOGENETIC_KB
        kb_text = _CYTOGENETIC_KB["chromosome_finding_general"]["answer_he"]
        draft_text = "זוהי טיוטת AI לא מבוקרת שאמורה להיות כרטיס משלים בלבד."
        self._mock_chromosome_draft(monkeypatch, draft_text)

        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        answer = data["answer"]
        assert kb_text[:50] in answer, "KB main answer not present"
        assert draft_text not in answer, "AI draft leaked into main answer"

    def test_draft_displayable_immediate_mode(self, monkeypatch):
        """In immediate mode, a meaningful AI draft must be marked displayable."""
        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")
        draft_text = (
            "הרחבה על ממצאים כרומוזומליים: כל ממצא מחייב ניתוח מדויק של "
            "המיקום ומספר הכרומוזום המעורב. הצוות הגנטי יסביר את המשמעות."
        )
        self._mock_chromosome_draft(monkeypatch, draft_text)
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a, **kw: None)
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)

        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        meta = data.get("chromosome_draft_metadata", {})
        assert meta.get("draft_available") is True
        assert meta.get("draft_displayable") is True

    def test_draft_hidden_approved_only_mode(self, monkeypatch):
        """In approved_only mode, pending chromosome draft must not be shown."""
        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "approved_only")
        self._mock_chromosome_draft(monkeypatch, "טיוטה שלא אמורה להיות גלויה.")
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a, **kw: None)
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)

        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        meta = data.get("chromosome_draft_metadata", {})
        assert meta.get("draft_displayable") is False
        assert meta.get("draft_hidden_reason") == "approved_only_mode"

    def test_approved_chromosome_draft_not_promoted_to_main(self, monkeypatch):
        """An approved chromosome draft must remain supplemental, not replace the KB answer."""
        from app.counseling_engine import _CYTOGENETIC_KB
        kb_text = _CYTOGENETIC_KB["chromosome_finding_general"]["answer_he"]
        self._mock_no_chromosome_draft(monkeypatch)

        import app.review_db as rdb
        approved_text = "הרחבה מאושרת על ממצאים כרומוזומליים."
        monkeypatch.setattr(
            rdb, "get_approved_chromosome_draft",
            lambda sub_intent: {
                "effective_text": approved_text,
                "status": "approved",
                "physician_approved": True,
            },
        )
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)

        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        answer = data["answer"]
        assert kb_text[:50] in answer, "KB answer missing"
        assert approved_text not in answer, "Approved chromosome draft replaced KB main answer"
        meta = data.get("chromosome_draft_metadata", {})
        assert meta.get("approved_draft_promoted") is False

    def test_approved_draft_available_flag(self, monkeypatch):
        """When an approved chromosome draft exists, approved_draft_available must be True."""
        self._mock_no_chromosome_draft(monkeypatch)
        import app.review_db as rdb
        monkeypatch.setattr(
            rdb, "get_approved_chromosome_draft",
            lambda *a: {"effective_text": "approved", "physician_approved": True},
        )
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)

        data = _ask("מה זה טרנסלוקציה?")
        meta = data.get("chromosome_draft_metadata", {})
        assert meta.get("approved_draft_available") is True

    def test_no_duplicate_text_when_draft_similar_to_main(self, monkeypatch):
        """Draft is hidden when its text is identical/highly similar to the KB answer."""
        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")
        from app.counseling_engine import _CYTOGENETIC_KB
        # Make the draft text identical to the KB answer — should be suppressed
        kb_text = _CYTOGENETIC_KB["chromosome_finding_general"]["answer_he"]
        monkeypatch.setattr(
            ce, "_generate_chromosome_education_draft",
            lambda *a, **kw: {"text_he": kb_text, "sub_intent": "chromosome_finding_general",
                               "generated_by_model": "mock", "review_status": "pending",
                               "approved": False, "warning_he": ""},
        )
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a: None)
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)

        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        meta = data.get("chromosome_draft_metadata", {})
        assert meta.get("draft_displayable") is False
        assert meta.get("draft_hidden_reason") == "identical_to_main_answer"

    def test_draft_stored_in_review_db(self, monkeypatch):
        """When a chromosome draft is generated, create_draft must be called."""
        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")
        self._mock_chromosome_draft(monkeypatch, "הרחבה על ממצאים כרומוזומליים לצורך סקירת רופא.")
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a: None)

        create_calls = []
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: create_calls.append(kw) or None)

        _ask("מה זה מחיקה בכרומוזום?")
        assert len(create_calls) == 1, f"Expected 1 create_draft call, got {len(create_calls)}"
        assert create_calls[0]["draft_type"] == "chromosome_education"
        assert create_calls[0]["normalized_intent"] == "chromosome_deletion_general"
        assert create_calls[0]["gene_symbol"] is None

    def test_main_answer_invariant_to_draft(self, monkeypatch):
        """Main answer must be the same whether or not a draft was generated."""
        import app.counseling_engine as ce
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a: None)

        monkeypatch.setattr(ce, "_generate_chromosome_education_draft", lambda *a, **kw: None)
        data_no_draft = _ask("מה זה קריוטיפ?")

        self._mock_chromosome_draft(monkeypatch, "הרחבה על קריוטיפ.")
        data_with_draft = _ask("מה זה קריוטיפ?")

        assert data_no_draft["answer"] == data_with_draft["answer"], (
            "Main answer changed depending on draft availability"
        )


# ── Part 3: Safety blocks before chromosome AI draft generation ───────────

class TestChromosomeDraftSafetyBlocks:
    """
    Safety routing must block chromosome AI draft generation for:
    - PII-containing input (handled by pipeline step A)
    - ISCN strings
    - Termination/reproductive decisions (pipeline step B)
    """

    def test_pii_blocked_before_chromosome_route(self):
        """Israeli ID in question must be blocked before any chromosome routing."""
        data = _ask("יש לי בעיה בכרומוזום 21, ת.ז. 123456789")
        assert data["safety_level"] == "contains_identifying_info"
        assert data["matched_topic"] is None

    def test_iscn_suppresses_chromosome_draft(self, monkeypatch):
        """ISCN notation must prevent AI draft generation."""
        import app.counseling_engine as ce
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a: None)

        create_calls = []
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: create_calls.append(kw) or None)

        draft_calls = []

        def _spy_draft(question, sub_intent, _debug=None):
            draft_calls.append(question)
            return None

        monkeypatch.setattr(ce, "_generate_chromosome_education_draft", _spy_draft)

        # ISCN string: del(5)(q13q33) — a specific cytogenetic notation
        _ask("יש לי del(5)(q13) — מה זה אומר?")
        assert len(draft_calls) == 0, "Chromosome draft generated despite ISCN notation"

    def test_termination_question_not_chromosome_route(self):
        """Pregnancy termination decision must be blocked by the reproductive_block pipeline step."""
        data = _ask("יש לי בעיה בכרומוזום, האם להפסיק את ההריון?")
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True

    def test_schema_compliance_chromosome_draft_response(self, monkeypatch):
        """Chromosome education responses must satisfy the 5-field API contract."""
        import app.review_db as rdb
        monkeypatch.setattr(rdb, "get_approved_chromosome_draft", lambda *a: None)
        monkeypatch.setattr(rdb, "create_draft", lambda **kw: None)

        questions = [
            "מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?",
            "מה זה מחיקה בכרומוזום?",
            "מה זה פסיפס?",
        ]
        for q in questions:
            data = _ask(q)
            assert _REQUIRED_KEYS.issubset(data.keys()), (
                f"Missing required keys for {q!r}: {_REQUIRED_KEYS - data.keys()}"
            )
