# -*- coding: utf-8 -*-
"""
Session 26 — Gene answers: biology-first, no duplicate AI card, clean fallback.

Covers:
  A. Tier-2 AI prompt input does NOT contain ClinVar counts or "Do NOT describe biology"
  B. Explicit gene questions return biological main answer (mocked OpenAI)
  C. No duplicate: draft_promoted_to_answer=True suppresses frontend card
  D. Tier-2 fallback when OpenAI unavailable is helpful, not "אין עדיין סיכום"
  E. Validation rejects ClinVar-count-only drafts; accepts biological text
  F. Safety regression: surgery/VUS still blocked; out-of-scope still rejected
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, call
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng
import app.gene_index as gene_index
import app.gene_cards as gene_cards
import app.gene_knowledge as gene_knowledge

client = TestClient(app)

_BANNED_IN_GENE_ANSWER = [
    "לתשובה אישית",
    "המשמעות האישית נקבעת",
    "המשמעות של כל ממצא ספציפי נקבעת",
    "לפרשנות אישית",
    "המידע כללי ואינו מחליף ייעוץ רפואי אישי",
]

_BIOLOGY_GENES = ["APOE", "TNF", "CCR5", "MTHFR", "FOXP2"]


# ── shared mock factory ────────────────────────────────────────────────────────

def _mock_tier2_env(monkeypatch, gene: str, ai_text: str, total: int = 40):
    """Set up a Tier-2 environment: gene in index, no approved card, mocked AI."""
    summary = {
        "gene_symbol": gene,
        "total_variants": total,
        "by_significance": {"pathogenic": 2, "vus": 20, "benign": 18},
        "phenotypes": ["some condition"],
    }
    monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
    monkeypatch.setattr(gene_index, "get_gene_summary", lambda g: summary if g == gene else None)
    monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
    monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
    monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
    monkeypatch.setattr(
        ceng, "_extract_gene_with_correction",
        lambda text: (gene, None) if gene in text.upper() else (None, None),
    )
    mc = MagicMock()
    mc.call_text_raw.return_value = ai_text
    monkeypatch.setattr(ceng, "create_llm_client", lambda: mc)
    return mc


# ── A. Prompt input check ──────────────────────────────────────────────────────

class TestTier2PromptInput:
    """The user_content passed to the LLM for Tier-2 biology must be biology-oriented."""

    def _capture_user_content(self, monkeypatch, gene: str) -> str:
        """Return the user_content string that was passed to call_text_raw."""
        captured = {}
        summary = {
            "gene_symbol": gene,
            "total_variants": 50,
            "by_significance": {"pathogenic": 3, "vus": 30, "benign": 17},
            "phenotypes": ["condition A"],
        }
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(gene_index, "get_gene_summary", lambda g: summary if g == gene else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            ceng, "_extract_gene_with_correction",
            lambda text: (gene, None) if gene in text.upper() else (None, None),
        )

        def mock_create_llm():
            mc = MagicMock()
            def _capture(user_content, system_prompt=None, **kw):
                captured["user_content"] = user_content
                return f"{gene} הוא גן שמקודד לחלבון המעורב בתהליכים ביולוגיים."
            mc.call_text_raw.side_effect = _capture
            return mc

        monkeypatch.setattr(ceng, "create_llm_client", mock_create_llm)
        client.post("/ask", json={"question": f"מה זה הגן {gene}?"})
        return captured.get("user_content", "")

    def test_apoe_prompt_contains_gene(self, monkeypatch):
        uc = self._capture_user_content(monkeypatch, "APOE")
        assert "APOE" in uc, f"user_content must contain gene symbol. Got: {uc!r}"

    def test_apoe_prompt_no_total_variants(self, monkeypatch):
        uc = self._capture_user_content(monkeypatch, "APOE")
        assert "Total variants" not in uc, f"user_content must not contain ClinVar counts. Got: {uc!r}"

    def test_apoe_prompt_no_clinvar(self, monkeypatch):
        uc = self._capture_user_content(monkeypatch, "APOE")
        assert "ClinVar" not in uc, f"user_content must not mention ClinVar. Got: {uc!r}"

    def test_apoe_prompt_no_biology_prohibition(self, monkeypatch):
        uc = self._capture_user_content(monkeypatch, "APOE")
        assert "Do NOT describe" not in uc, (
            f"user_content must not contain 'Do NOT describe'. Got: {uc!r}"
        )

    def test_apoe_prompt_no_pathogenic_count(self, monkeypatch):
        uc = self._capture_user_content(monkeypatch, "APOE")
        assert "Pathogenic" not in uc and "pathogenic" not in uc, (
            f"user_content must not contain pathogenic counts. Got: {uc!r}"
        )

    def test_apoe_prompt_asks_biological_role(self, monkeypatch):
        uc = self._capture_user_content(monkeypatch, "APOE")
        has_biology_hint = "biological" in uc.lower() or "biology" in uc.lower() or "Task" in uc
        assert has_biology_hint, (
            f"user_content should ask for biological role. Got: {uc!r}"
        )


# ── B. Biology in main answer ──────────────────────────────────────────────────

class TestGeneBiologyAnswer:
    """Explicit gene questions return the mocked biological AI text as the main answer."""

    @pytest.mark.parametrize("gene,bio_text", [
        ("APOE", "APOE הוא גן שמקודד לחלבון המעורב בהובלה ובפירוק של שומנים בגוף ובמוח."),
        ("TNF", "TNF הוא גן שמקודד לציטוקין בשם tumor necrosis factor, המעורב בתגובה דלקתית."),
        ("CCR5", "CCR5 הוא גן שמקודד לקולטן על פני תאים מסוימים של מערכת החיסון."),
        ("MTHFR", "MTHFR הוא גן שמקודד לאנזים המעורב במסלול הפולאט ובמטבוליזם של הומוציסטאין."),
        ("FOXP2", "FOXP2 הוא גן שמקודד לפקטור שעתוק המעורב בהתפתחות מערכת העצבים."),
    ])
    def test_gene_answer_contains_biology_text(self, monkeypatch, gene, bio_text):
        _mock_tier2_env(monkeypatch, gene, bio_text)
        data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
        answer = data.get("answer", "")
        assert gene in answer, f"{gene} must appear in answer. Answer: {answer[:200]!r}"
        assert len(answer) > 30, f"Answer too short for {gene}: {answer!r}"
        # The mocked biology text should be the answer content
        assert bio_text[:40] in answer, (
            f"Biology text not in answer for {gene}.\nExpected prefix: {bio_text[:40]!r}\n"
            f"Answer: {answer[:300]!r}"
        )

    @pytest.mark.parametrize("gene", _BIOLOGY_GENES)
    def test_gene_answer_no_banned_phrases(self, monkeypatch, gene):
        bio_text = f"{gene} הוא גן שמקודד לחלבון המעורב בתהליכים ביולוגיים."
        _mock_tier2_env(monkeypatch, gene, bio_text)
        data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
        answer = data.get("answer", "")
        for phrase in _BANNED_IN_GENE_ANSWER:
            assert phrase not in answer, (
                f"Banned phrase {phrase!r} in {gene} answer:\n{answer[:300]!r}"
            )

    @pytest.mark.parametrize("gene", _BIOLOGY_GENES)
    def test_gene_answer_no_clinvar_counts_in_answer(self, monkeypatch, gene):
        """Main answer must not be ClinVar-count text."""
        bio_text = f"{gene} הוא גן שמקודד לחלבון ביולוגי."
        _mock_tier2_env(monkeypatch, gene, bio_text)
        data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
        answer = data.get("answer", "")
        assert "וריאנטים פתוגניים" not in answer, (
            f"Answer for {gene} must not contain ClinVar count text:\n{answer[:300]!r}"
        )
        assert "Total variants" not in answer, (
            f"Answer for {gene} must not contain English ClinVar stats:\n{answer[:300]!r}"
        )

    @pytest.mark.parametrize("gene", _BIOLOGY_GENES)
    def test_gene_suggested_questions_no_geneticist_referral(self, monkeypatch, gene):
        """Suggested questions for gene answers must not send to genetic counselor."""
        bio_text = f"{gene} הוא גן שמקודד לחלבון ביולוגי."
        _mock_tier2_env(monkeypatch, gene, bio_text)
        data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
        sq = data.get("suggested_questions", [])
        for q in sq:
            assert "גנטיקאי" not in q, (
                f"Suggested question must not mention גנטיקאי for {gene}:\n{q!r}"
            )


# ── C. No duplicate display ────────────────────────────────────────────────────

class TestNoDuplicateDisplay:
    """When AI draft is promoted to main answer, the draft card must be suppressed."""

    def test_draft_promoted_flag_set(self, monkeypatch):
        """gene_metadata.draft_promoted_to_answer must be True when draft is the main answer."""
        bio_text = "APOE הוא גן שמקודד לחלבון המעורב בהובלה של שומנים."
        _mock_tier2_env(monkeypatch, "APOE", bio_text)
        data = client.post("/ask", json={"question": "מה זה הגן APOE?"}).json()
        meta = data.get("gene_metadata", {})
        assert meta.get("draft_promoted_to_answer") is True, (
            f"draft_promoted_to_answer must be True when draft is main answer. meta={meta}"
        )

    def test_unverified_gene_draft_still_in_response(self, monkeypatch):
        """unverified_gene_draft still present in response (for API consumers/tests)."""
        bio_text = "APOE הוא גן שמקודד לחלבון המעורב בהובלה של שומנים."
        _mock_tier2_env(monkeypatch, "APOE", bio_text)
        data = client.post("/ask", json={"question": "מה זה הגן APOE?"}).json()
        assert data.get("unverified_gene_draft") is not None, (
            "unverified_gene_draft must still be present in response for API compatibility"
        )

    def test_draft_promoted_suppresses_frontend_card(self, monkeypatch):
        """draft_promoted_to_answer=True must prevent the frontend from rendering the card."""
        bio_text = "MTHFR הוא גן שמקודד לאנזים המעורב במסלול הפולאט."
        _mock_tier2_env(monkeypatch, "MTHFR", bio_text)
        data = client.post("/ask", json={"question": "מה זה הגן MTHFR?"}).json()
        meta = data.get("gene_metadata", {})
        # The frontend checks: !meta.draft_promoted_to_answer — if True, no card rendered.
        # Here we verify the flag is present and True so frontend can act on it.
        assert meta.get("draft_promoted_to_answer") is True
        assert meta.get("answer_tier") == "tier2"

    def test_no_draft_flag_when_no_draft(self, monkeypatch):
        """When OpenAI is unavailable (no draft), draft_promoted_to_answer must be False."""
        gene = "APOE"
        summary = {
            "gene_symbol": gene, "total_variants": 40,
            "by_significance": {"pathogenic": 2, "vus": 20, "benign": 18},
            "phenotypes": [],
        }
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(gene_index, "get_gene_summary", lambda g: summary if g == gene else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            ceng, "_extract_gene_with_correction",
            lambda text: (gene, None) if gene in text.upper() else (None, None),
        )
        # No LLM configured
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
        meta = data.get("gene_metadata", {})
        assert meta.get("draft_promoted_to_answer") is False, (
            f"draft_promoted_to_answer must be False when no draft. meta={meta}"
        )


# ── D. CCR5 / no-OpenAI fallback ──────────────────────────────────────────────

class TestTier2FallbackNoOpenAI:
    """When OpenAI is unavailable, Tier-2 fallback must be helpful, not 'אין עדיין סיכום'."""

    def _setup_no_llm(self, monkeypatch, gene: str):
        summary = {
            "gene_symbol": gene, "total_variants": 35,
            "by_significance": {"pathogenic": 1, "vus": 20, "benign": 14},
            "phenotypes": [],
        }
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        monkeypatch.setattr(gene_index, "get_gene_summary", lambda g: summary if g == gene else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            ceng, "_extract_gene_with_correction",
            lambda text: (gene, None) if gene in text.upper() else (None, None),
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_ccr5_fallback_not_old_empty_message(self, monkeypatch):
        self._setup_no_llm(monkeypatch, "CCR5")
        data = client.post("/ask", json={"question": "מה זה הגן CCR5?"}).json()
        answer = data.get("answer", "")
        assert "אין עדיין סיכום עברי מאושר" not in answer, (
            f"Old fallback message must not appear. Answer: {answer!r}"
        )

    def test_ccr5_fallback_graceful(self, monkeypatch):
        self._setup_no_llm(monkeypatch, "CCR5")
        data = client.post("/ask", json={"question": "מה זה הגן CCR5?"}).json()
        answer = data.get("answer", "")
        assert len(answer) > 20, f"Fallback answer too short: {answer!r}"
        assert "CCR5" in answer, f"Fallback must mention gene name. Answer: {answer!r}"

    def test_fallback_no_banned_referral(self, monkeypatch):
        self._setup_no_llm(monkeypatch, "CCR5")
        data = client.post("/ask", json={"question": "מה זה הגן CCR5?"}).json()
        answer = data.get("answer", "")
        for phrase in _BANNED_IN_GENE_ANSWER:
            assert phrase not in answer, (
                f"Banned phrase {phrase!r} in fallback answer:\n{answer!r}"
            )

    def test_fallback_not_blocked(self, monkeypatch):
        self._setup_no_llm(monkeypatch, "MTHFR")
        data = client.post("/ask", json={"question": "מה זה הגן MTHFR?"}).json()
        assert data.get("safety_level") == "general_information", (
            f"Fallback must be general_information. Got: {data.get('safety_level')!r}"
        )

    def test_fallback_gene_metadata_present(self, monkeypatch):
        self._setup_no_llm(monkeypatch, "CCR5")
        data = client.post("/ask", json={"question": "מה זה הגן CCR5?"}).json()
        meta = data.get("gene_metadata", {})
        assert meta.get("answer_tier") == "tier2", (
            f"No-LLM Tier-2 must still report tier2. meta={meta}"
        )
        assert meta.get("found_in_index") is True


# ── E. Validation regression ───────────────────────────────────────────────────

class TestDraftValidationS26:
    """_validate_gene_education_draft rejects ClinVar-count drafts and unsafe text."""

    @pytest.mark.parametrize("text,description", [
        # ClinVar-count drafts (new rejection)
        ("הגן APOE נמצא בהקשר של מצבים קליניים שונים. יש לו 40 וריאנטים פתוגניים ב-ClinVar.",
         "clinvar count in main answer"),
        ("יש לו 40 וריאנטים פתוגניים ב-ClinVar.", "simple clinvar count"),
        ("נמצאו 208 וריאנטים במאגר ClinVar.", "found N variants in ClinVar"),
        # Existing unsafe patterns must still be rejected
        ("את צריכה ניתוח לכריתת שד.", "surgery recommendation"),
        ("הגן DMD גורם למחלה קשה.", "definitive causation - DMD"),
        ("הגן TP53 גורמת לסרטן המעי הגס.", "definitive causation"),
        ("כדאי לעשות הפלה בגלל הממצא.", "abortion recommendation"),
        ("הגן מקושר לסוגי סרטן שונים.", "vague cancer list"),
        ("הסיכון שלך לחלות הוא גבוה.", "personal risk"),
    ])
    def test_draft_rejected(self, text, description):
        result = ceng._validate_gene_education_draft(text)
        assert result is not None, (
            f"Expected rejection for [{description}]: {text!r}\nGot: None (accepted)"
        )

    @pytest.mark.parametrize("text,description", [
        ("APOE הוא גן שמקודד לחלבון המעורב בהובלה ובפירוק של שומנים בגוף ובמוח.",
         "APOE biology"),
        ("TNF הוא גן שמקודד לציטוקין בשם tumor necrosis factor, המעורב בתגובה דלקתית.",
         "TNF biology"),
        ("CCR5 הוא גן שמקודד לקולטן על פני תאים מסוימים של מערכת החיסון.",
         "CCR5 biology"),
        ("MTHFR הוא גן שמקודד לאנזים המעורב במסלול הפולאט ובמטבוליזם של הומוציסטאין.",
         "MTHFR biology"),
        ("FOXP2 הוא גן שמקודד לפקטור שעתוק המעורב בהתפתחות מערכת העצבים.",
         "FOXP2 biology"),
        ("הגן BRCA1 קשור לסרטן שד ושחלה.", "BRCA1 association (allowed: specific named)"),
        ("הגן APC מוכר בהקשר של פוליפוזיס אדנומטוטית משפחתית.", "APC specific context"),
    ])
    def test_draft_accepted(self, text, description):
        result = ceng._validate_gene_education_draft(text)
        assert result is None, (
            f"Expected acceptance for [{description}]: {text!r}\nGot rejection: {result!r}"
        )


# ── F. Safety regression ───────────────────────────────────────────────────────

class TestSafetyRegressionS26:
    """Safety-critical routing must be unaffected by gene-biology fixes."""

    def test_vus_surgery_still_blocked(self):
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS לBRCA1, האם לעשות כריתת שד?"},
        ).json()
        assert data.get("safety_level") == "requires_genetic_counselor", (
            f"VUS+surgery must be blocked. Got: {data.get('safety_level')!r}"
        )

    def test_out_of_scope_still_short(self):
        data = client.post("/ask", json={"question": "מה זה מגדל אייפל?"}).json()
        assert data.get("safety_level") == "out_of_scope", (
            f"Out-of-domain must return out_of_scope. Got: {data.get('safety_level')!r}"
        )
        assert len(data.get("answer", "")) < 200

    def test_vus_options_not_blocked(self):
        data = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS בגן APC, מהן האפשרויות שעומדות מולי?"},
        ).json()
        assert data.get("safety_level") == "general_information", (
            f"VUS options must not be blocked. Got: {data.get('safety_level')!r}"
        )

    def test_tier1a_gene_unaffected(self):
        """Tier-1a genes (approved card: BRCA1, BRCA2, NF1) must still return curated answer."""
        data = client.post("/ask", json={"question": "מה זה הגן BRCA1?"}).json()
        meta = data.get("gene_metadata", {})
        assert meta.get("answer_tier") in ("tier1", "tier1a", None), (
            f"BRCA1 (approved card) must use tier1. Got: {meta.get('answer_tier')!r}"
        )
        assert data.get("safety_level") == "general_information"
