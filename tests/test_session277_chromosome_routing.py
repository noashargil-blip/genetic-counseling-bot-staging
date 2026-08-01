"""
Session 27.7 tests — chromosome education routing and approved-draft reuse.

Parts covered:
  C-G  General chromosomal / cytogenetic question routing
  B    Physician-approved gene education drafts reused in both visibility modes

Run via:
  PYTHONUTF8=1 python -m pytest tests/test_session277_chromosome_routing.py -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_REQUIRED_KEYS = {"answer", "safety_level", "needs_genetic_counselor",
                  "matched_topic", "suggested_questions"}


def _ask(question: str, **kwargs) -> dict:
    payload = {"question": question, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


# ── Part C/D — chromosome_finding_general ──────────────────────────────────

class TestChromosomeFindingGeneral:
    """'בעיה בכרומוזום X' must NOT assume a specific syndrome."""

    AMBIGUOUS_QUESTIONS = [
        "מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?",
        "נמצאה בעיה בכרומוזום 13, מה זה אומר?",
        "יש לי ממצא כרומוזומי, מה זה?",
        "אמרו לי שיש שינוי בכרומוזום 5",
        "מה פירוש ממצא כרומוזומי?",
    ]

    @pytest.mark.parametrize("q", AMBIGUOUS_QUESTIONS)
    def test_response_schema(self, q):
        data = _ask(q)
        assert _REQUIRED_KEYS.issubset(data.keys()), f"missing keys: {_REQUIRED_KEYS - data.keys()}"

    @pytest.mark.parametrize("q", AMBIGUOUS_QUESTIONS)
    def test_safety_level_general_information(self, q):
        data = _ask(q)
        assert data["safety_level"] == "general_information"

    @pytest.mark.parametrize("q", AMBIGUOUS_QUESTIONS)
    def test_no_counselor_referral(self, q):
        data = _ask(q)
        assert data["needs_genetic_counselor"] is False

    def test_does_not_assume_down_syndrome_chr21(self):
        """'בעיה בכרומוזום 21' must NOT mention Down syndrome as the assumed diagnosis."""
        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        answer = data.get("answer", "")
        # Answer must acknowledge multiple possibilities, not diagnose
        assert "בעיה" in answer or "ממצא" in answer or "כרומוזום" in answer
        # Must not assert Down syndrome as the conclusion
        # (educational mention as a possibility is OK, asserting it for THIS patient is not)
        forbidden = ["האדם הזה יש לו תסמונת דאון", "יש לך תסמונת דאון",
                     "הממצא שלך הוא טריזומיה 21"]
        for phrase in forbidden:
            assert phrase not in answer, f"Found forbidden phrase: {phrase!r}"

    def test_matched_topic_chromosome(self):
        data = _ask("יש לי ממצא כרומוזומי, מה זה?")
        assert data["matched_topic"] in (
            "chromosome_finding_general",
            "chromosomal_finding",
            "chromosome_education",
        )

    def test_answer_not_empty(self):
        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        assert len(data.get("answer", "")) > 50

    def test_suggested_questions_non_empty(self):
        data = _ask("יש לי ממצא כרומוזומי, מה זה?")
        assert len(data.get("suggested_questions", [])) >= 1

    def test_not_routed_to_vus_answer(self):
        """Must not give the generic VUS/carrier fallback for a chromosome question."""
        data = _ask("מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?")
        answer = data.get("answer", "")
        vus_fallback_phrases = ["VUS", "variant of uncertain significance", "נשא"]
        # answer should NOT be dominated by VUS content for a chromosome question
        assert not all(p in answer for p in ["VUS", "נשאות"]), (
            f"Answer appears to be the VUS/carrier fallback: {answer[:200]}"
        )


# ── Part E — deterministic KB answers for cytogenetic concepts ─────────────

class TestCytogeneticConceptAnswers:
    """Each cytogenetic sub-intent must return an appropriate educational answer."""

    def test_deletion_routing(self):
        data = _ask("מה זה מחיקה בכרומוזום?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        # Should mention deletion concept
        assert any(w in answer for w in ["מחיקה", "חסר", "deletion"])

    def test_duplication_routing(self):
        data = _ask("מה זה כפילות כרומוזומית?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        assert any(w in answer for w in ["כפילות", "עודף", "duplication"])

    def test_translocation_routing(self):
        data = _ask("מה זה טרנסלוקציה?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        assert any(w in answer for w in ["טרנסלוקציה", "translocation", "מאוזנת", "מאחר"])

    def test_mosaicism_routing(self):
        data = _ask("מה זה מוזאיקה בגנטיקה?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        assert any(w in answer for w in ["פסיפס", "מוזאיק", "mosaic", "תאים"])

    def test_karyotype_routing(self):
        data = _ask("מה זה בדיקת קריוטיפ?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        assert any(w in answer for w in ["קריוטיפ", "karyotype", "כרומוזומים"])

    def test_microarray_routing(self):
        data = _ask("מה זה מיקרואריי כרומוזומלי?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        assert any(w in answer for w in ["מיקרואריי", "microarray", "CMA", "מחיקות"])

    def test_aneuploidy_routing(self):
        data = _ask("מה זה אנאופלואידיה?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50
        assert any(w in answer for w in ["אנאופלואידיה", "aneuploidy", "כרומוזומים"])

    def test_monosomy_routing(self):
        data = _ask("מה זה מונוזומיה?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert len(answer) > 50

    def test_mosaicism_alternate_term(self):
        data = _ask("מה זה פסיפס גנטי?")
        assert data["safety_level"] == "general_information"

    def test_translocation_balanced_question(self):
        data = _ask("מה ההבדל בין טרנסלוקציה מאוזנת ללא מאוזנת?")
        assert data["safety_level"] == "general_information"
        answer = data["answer"]
        assert any(w in answer for w in ["מאוזנת", "balanced", "translocation", "טרנסלוקציה"])


# ── Trisomy 21 still works (existing handler preserved) ────────────────────

class TestTrisomy21Preserved:
    """The existing trisomy21_education handler must still fire for explicit Down syndrome queries."""

    TRISOMY21_QUESTIONS = [
        "מה זה טריזומיה 21?",
        "מה זה תסמונת דאון?",
        "יש לי כרומוזום 21 עודף, מה זה?",
    ]

    @pytest.mark.parametrize("q", TRISOMY21_QUESTIONS)
    def test_schema(self, q):
        data = _ask(q)
        assert _REQUIRED_KEYS.issubset(data.keys())

    @pytest.mark.parametrize("q", TRISOMY21_QUESTIONS)
    def test_safety_level(self, q):
        data = _ask(q)
        assert data["safety_level"] == "general_information"

    def test_trisomy21_answer_mentions_trisomy(self):
        data = _ask("מה זה טריזומיה 21?")
        answer = data["answer"]
        assert any(w in answer for w in ["טריזומיה", "trisomy", "דאון", "Down"])


# ── Sex-chromosome aneuploidy preserved ────────────────────────────────────

class TestExtraChromosomePreserved:
    def test_xxy_still_routed(self):
        data = _ask("מה זה תסמונת קליינפלטר?")
        assert data["safety_level"] == "general_information"

    def test_xxx_still_routed(self):
        data = _ask("מה זה XXX?")
        assert data["safety_level"] == "general_information"


# ── Part B — physician-approved draft reuse in immediate mode ──────────────

class TestApprovedDraftReuseImmediate:
    """
    When an approved draft exists in the review DB, it must be used as the
    main answer even in immediate (default) mode — not just in approved_only mode.
    The fallback 'אין לי סיכום ביולוגי' must NOT appear when there is an approved draft.
    """

    def _mock_tier2_env(self, monkeypatch, gene: str, approved_text: str):
        """Set up: gene in ClinVar index, no curated card, approved draft available."""
        import app.counseling_engine as ce

        class _FakeSummary:
            def get(self, key, default=None):
                return {"total_variants": 10, "by_significance": {}, "phenotypes": []}.get(key, default)

        monkeypatch.setattr(
            "app.gene_index.get_gene_summary", lambda g: _FakeSummary() if g == gene else None
        )
        monkeypatch.setattr(
            "app.gene_index._GENE_INDEX_AVAILABLE", True
        )
        monkeypatch.setattr(
            "app.gene_cards.get_approved_summary", lambda g: None
        )
        monkeypatch.setattr(
            "app.gene_knowledge.get_gene_patient_summary", lambda g: None
        )
        monkeypatch.setattr(
            "app.gene_knowledge.get_gene_vus_note", lambda g: None
        )
        # Approved draft available
        monkeypatch.setattr(
            ce, "_generate_unverified_gene_draft",
            lambda *a, **kw: None,  # no new draft generation
        )

        import app.review_db as rdb
        monkeypatch.setattr(
            rdb, "get_approved_draft",
            lambda gene_symbol, draft_type="gene_summary": {
                "effective_text": approved_text,
                "status": "approved",
            } if gene_symbol == gene else None
        )

    def test_approved_draft_used_as_main_answer_immediate_mode(self, monkeypatch):
        gene = "TESTGENE1"
        approved = "הגן TESTGENE1 קודד לחלבון מסוים. מידע זה אושר על ידי רופא."
        self._mock_tier2_env(monkeypatch, gene, approved)

        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")

        data = _ask(f"מה הגן {gene}?")
        answer = data.get("answer", "")
        assert approved[:30] in answer, (
            f"Approved text not in main answer.\nAnswer: {answer[:300]}"
        )

    def test_fallback_not_shown_when_approved_draft_exists(self, monkeypatch):
        gene = "TESTGENE2"
        approved = "הגן TESTGENE2 אושר על ידי רופא."
        self._mock_tier2_env(monkeypatch, gene, approved)

        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")

        data = _ask(f"מה התפקיד הביולוגי של הגן {gene}?")
        answer = data.get("answer", "")
        assert "אין לי סיכום ביולוגי" not in answer, (
            f"Fallback shown even though approved draft exists.\nAnswer: {answer[:300]}"
        )

    def test_draft_promoted_flag_true_when_approved(self, monkeypatch):
        gene = "TESTGENE3"
        approved = "הגן TESTGENE3 אושר על ידי רופא."
        self._mock_tier2_env(monkeypatch, gene, approved)

        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")

        data = _ask(f"מה הגן {gene}?")
        meta = data.get("gene_metadata", {})
        assert meta.get("draft_promoted_to_answer") is True

    def test_no_approved_draft_uses_fallback_immediate(self, monkeypatch):
        """Without an approved draft, immediate mode still shows the fallback."""
        gene = "TESTGENE4"
        import app.counseling_engine as ce
        import app.review_db as rdb

        class _FakeSummary:
            def get(self, key, default=None):
                return {"total_variants": 5, "by_significance": {}, "phenotypes": []}.get(key, default)

        monkeypatch.setattr("app.gene_index.get_gene_summary", lambda g: _FakeSummary() if g == gene else None)
        monkeypatch.setattr("app.gene_index._GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr("app.gene_cards.get_approved_summary", lambda g: None)
        monkeypatch.setattr("app.gene_knowledge.get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr("app.gene_knowledge.get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(ce, "_generate_unverified_gene_draft", lambda *a, **kw: None)
        monkeypatch.setattr(rdb, "get_approved_draft", lambda *a, **kw: None)
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")

        data = _ask(f"מה הגן {gene}?")
        answer = data.get("answer", "")
        assert "אין לי סיכום ביולוגי" in answer, (
            f"Expected fallback when no approved draft.\nAnswer: {answer[:300]}"
        )

    def test_gene_knowledge_status_approved_when_promoted(self, monkeypatch):
        gene = "TESTGENE5"
        approved = "הגן TESTGENE5 אושר על ידי רופא."
        self._mock_tier2_env(monkeypatch, gene, approved)

        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")

        data = _ask(f"מה הגן {gene}?")
        meta = data.get("gene_metadata", {})
        assert meta.get("gene_knowledge_status") == "approved"

    def test_draft_card_hidden_when_approved_is_main(self, monkeypatch):
        """When approved draft is the main answer, the supplemental card is suppressed."""
        gene = "TESTGENE6"
        approved = "הגן TESTGENE6 אושר על ידי רופא."
        self._mock_tier2_env(monkeypatch, gene, approved)

        import app.counseling_engine as ce
        monkeypatch.setattr(ce, "_AI_DRAFT_VISIBILITY_MODE", "immediate")

        data = _ask(f"מה הגן {gene}?")
        meta = data.get("gene_metadata", {})
        assert meta.get("unverified_gene_draft_displayable") is False
        assert meta.get("draft_hidden_reason") == "approved_text_is_main_answer"


# ── Schema compliance for chromosome routes ────────────────────────────────

class TestChromosomeRouteSchemaCompliance:
    CHROMOSOME_QUESTIONS = [
        "מה לעשות כשאומרים לי שיש בעיה בכרומוזום 21?",
        "מה זה מחיקה בכרומוזום?",
        "מה זה טרנסלוקציה?",
        "מה זה מוזאיקה?",
        "מה זה קריוטיפ?",
        "מה זה אנאופלואידיה?",
    ]

    @pytest.mark.parametrize("q", CHROMOSOME_QUESTIONS)
    def test_five_required_keys_present(self, q):
        data = _ask(q)
        assert _REQUIRED_KEYS.issubset(data.keys()), (
            f"Missing keys {_REQUIRED_KEYS - data.keys()} for question: {q!r}"
        )

    @pytest.mark.parametrize("q", CHROMOSOME_QUESTIONS)
    def test_suggested_questions_is_list(self, q):
        data = _ask(q)
        assert isinstance(data["suggested_questions"], list)

    @pytest.mark.parametrize("q", CHROMOSOME_QUESTIONS)
    def test_answer_non_empty(self, q):
        data = _ask(q)
        assert data["answer"].strip(), f"Empty answer for: {q!r}"
