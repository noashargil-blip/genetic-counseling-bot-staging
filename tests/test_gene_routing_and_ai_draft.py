# -*- coding: utf-8 -*-
"""Session 16 hotfix — gene routing and AI draft tests."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_local_llm(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    engine._known_gene_set_cache = None
    yield
    engine._known_gene_set_cache = None


class TestStandaloneGeneDetection:
    def test_bare(self):
        assert engine._is_standalone_gene_query("CFTR", "CFTR") is True
    def test_question_mark(self):
        assert engine._is_standalone_gene_query("CFTR?", "CFTR") is True
    def test_hebrew_prefix(self):
        assert engine._is_standalone_gene_query("בCFTR", "CFTR") is True
    def test_hebrew_prefix_hyphen(self):
        assert engine._is_standalone_gene_query("ב-CFTR", "CFTR") is True
    def test_lamed_prefix(self):
        assert engine._is_standalone_gene_query("לPTEN", "PTEN") is True
    def test_full_phrase_not_standalone(self):
        assert engine._is_standalone_gene_query("מה זה הגן CFTR?", "CFTR") is False
    def test_cftr_returns_200(self):
        assert client.post("/ask", json={"question": "CFTR"}).status_code == 200
    def test_cftr_not_generic_kb(self):
        data = client.post("/ask", json={"question": "CFTR"}).json()
        assert data.get("matched_topic") != "gene"
    def test_cftr_answer_has_cftr(self):
        assert "CFTR" in client.post("/ask", json={"question": "CFTR"}).json()["answer"]


class TestFuzzyGeneMatching:
    def test_xftr_to_cftr(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR", "BRCA1", "NF1"}))
        assert engine._fuzzy_match_gene_symbol("XFTR") == "CFTR"
    def test_socks1_to_sox1(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"SOX1", "SOX2", "BRCA1"}))
        assert engine._fuzzy_match_gene_symbol("SOCKS1") == "SOX1"
    def test_vus_not_matched(self):
        assert engine._fuzzy_match_gene_symbol("VUS") is None
    def test_dna_not_matched(self):
        assert engine._fuzzy_match_gene_symbol("DNA") is None
    def test_exact_returns_none(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR"}))
        assert engine._fuzzy_match_gene_symbol("CFTR") is None
    def test_extract_exact_no_correction(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR"}))
        gene, corrected = engine._extract_gene_with_correction("מה זה הגן CFTR?")
        assert gene == "CFTR" and corrected is None
    def test_extract_typo_corrected(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR", "BRCA1"}))
        gene, corrected = engine._extract_gene_with_correction("מה זה הגן XFTR?")
        assert gene == "CFTR" and corrected == "XFTR"
    def test_xftr_returns_200(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR", "BRCA1", "NF1"}))
        assert client.post("/ask", json={"question": "מה זה הגן XFTR?"}).status_code == 200
    def test_xftr_not_generic_kb(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR", "BRCA1", "NF1"}))
        data = client.post("/ask", json={"question": "מה זה הגן XFTR?"}).json()
        assert data.get("matched_topic") != "gene"


class TestGeneEducationDraftValidation:
    def test_allows_bio_function(self):
        txt = "גן CFTR מכיל הוראות לייצור חלבון. המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי."
        assert engine._validate_gene_education_draft(txt) is None
    def test_allows_disease_association(self):
        txt = "גן MSH2 קשור לתיקון שגיאות ב-DNA. שינויים פתוגניים בגן זה קשורים לתסמונת לינץ'. המשמעות האישית נקבעת על ידי הצוות הגנטי."
        assert engine._validate_gene_education_draft(txt) is None
    def test_allows_english_bio_terms(self):
        txt = "גן MSH2 קשור לתהליך mismatch repair. המשמעות האישית נקבעת על ידי הצוות הגנטי."
        assert engine._validate_gene_education_draft(txt) is None
    def test_allows_vus_general(self):
        txt = "גן PTEN עשוי לכלול VUS. המשמעות נקבעת על ידי הצוות הגנטי."
        assert engine._validate_gene_education_draft(txt) is None
    def test_blocks_personal_risk(self):
        assert engine._validate_gene_education_draft("הסיכון שלך גבוה.") is not None
    def test_blocks_surgery(self):
        assert engine._validate_gene_education_draft("עליך לעבור ניתוח.") is not None
    def test_blocks_empty(self):
        assert engine._validate_gene_education_draft("") is not None
    def test_blocks_dash(self):
        assert engine._validate_gene_education_draft("-") is not None
    def test_blocks_hype(self):
        assert engine._validate_gene_education_draft("גן CFTR הוא מדהים.") is not None
    def test_blocks_no_hebrew(self):
        assert engine._validate_gene_education_draft("CFTR is a gene.") is not None
    def test_allows_pathogenic_general(self):
        # "גורמים לסרטן" (causation) is correctly blocked — use "קשור ל" instead.
        txt = "גן MSH2 מקודד לחלבון המשתתף בתיקון שגיאות שכפול DNA. הגן קשור לסרטן המעי הגס. המשמעות האישית נקבעת על ידי הצוות הגנטי."
        assert engine._validate_gene_education_draft(txt) is None


class TestDraftFailureSilence:
    FAILURE_MSG = "לא הצלחנו ליצור"
    BROKEN_PROMISE = "אפשר לבחור לראות טיוטת מידע"
    def test_no_failure_msg_cftr(self):
        assert self.FAILURE_MSG not in client.post("/ask", json={"question": "מה זה הגן CFTR?"}).json()["answer"]
    def test_no_broken_promise_cftr(self):
        assert self.BROKEN_PROMISE not in client.post("/ask", json={"question": "מה זה הגן CFTR?"}).json()["answer"]
    def test_no_failure_msg_vus_brca1(self):
        assert self.FAILURE_MSG not in client.post("/ask", json={"question": "מה זה VUS בBRCA1?"}).json()["answer"]


class TestVusPtenRouting:
    def test_returns_200(self):
        assert client.post("/ask", json={"question": "מה זה VUS בPTEN?"}).status_code == 200
    def test_safety_level(self):
        assert client.post("/ask", json={"question": "מה זה VUS בPTEN?"}).json()["safety_level"] == "general_information"
    def test_includes_vus(self):
        assert "VUS" in client.post("/ask", json={"question": "מה זה VUS בPTEN?"}).json()["answer"]
    def test_includes_pten(self):
        assert "PTEN" in client.post("/ask", json={"question": "מה זה VUS בPTEN?"}).json()["answer"]


class TestPlainGeneNoVus:
    def test_cftr_plain_no_vus(self):
        data = client.post("/ask", json={"question": "מה זה הגן CFTR?"}).json()
        if data.get("matched_topic") in ("gene_clinvar_summary", "gene_info"):
            assert "VUS" not in data["answer"]
    def test_brca1_plain_no_vus(self):
        assert "VUS" not in client.post("/ask", json={"question": "מה זה הגן BRCA1?"}).json()["answer"]


class TestHighStakesPersonalQuestion:
    def test_cancer_question_blocked(self):
        data = client.post("/ask", json={"question": "יש לי שינוי בMSH2 האם יש לי סרטן?"}).json()
        assert data["safety_level"] in ("requires_genetic_counselor", "general_information")
    def test_no_personal_risk_in_answer(self):
        answer = client.post("/ask", json={"question": "יש לי שינוי בMSH2 האם יש לי סרטן?"}).json()["answer"]
        for phrase in ("הסיכון שלך", "יש לך סרטן", "אובחנת"):
            assert phrase not in answer


class TestResponseSchema:
    def test_cftr_five_fields(self):
        data = client.post("/ask", json={"question": "CFTR"}).json()
        for key in ("answer", "safety_level", "needs_genetic_counselor", "matched_topic", "suggested_questions"):
            assert key in data
    def test_xftr_five_fields(self, monkeypatch):
        monkeypatch.setattr(engine, "_known_gene_set_cache", frozenset({"CFTR"}))
        data = client.post("/ask", json={"question": "XFTR"}).json()
        for key in ("answer", "safety_level", "needs_genetic_counselor", "matched_topic", "suggested_questions"):
            assert key in data


# ── Session 25: Validator relaxation for cautious associations ─────────────────

class TestDraftValidationS25CautiousAssociations:
    """Session 25: cautious gene-disease associations must PASS _validate_gene_education_draft."""

    @pytest.mark.parametrize("text", [
        # MTHFR — folate pathway (the key case from the bug report)
        "הגן MTHFR מקודד לאנזים המעורב במטבוליזם של פולאט. "
        "MTHFR קשור למסלול פולאט ולמטבוליזם של הומוציסטאין.",

        # APOE — neurological association (mחלות נוירולוגיות כמו — specific adjective)
        "הגן APOE מקודד לחלבון אפוליפופרוטאין E, המעורב בהובלת שומנים. "
        "הוא מוכר בהקשרים של מחלות נוירולוגיות כמו אלצהיימר.",

        # DMD — specific named disease using "מוכר בהקשר של"
        "הגן DMD מכיל הוראות ליצירת חלבון הדיסטרופין, שנמצא בשריר השלד. "
        "הגן מוכר בהקשר של Duchenne ו-Becker muscular dystrophy.",

        # SRY — sex development
        "הגן SRY קשור להתפתחות מערכת המין הזכרית בעובר.",

        # MSH2 — specific named syndrome
        "הגן MSH2 מעורב במנגנון תיקון שגיאות שכפול DNA. "
        "הוא קשור למצבים של תסמונת לינץ׳.",

        # "נמצא בהקשר של" phrasing
        "הגן BRCA2 מעורב בתיקון שברים כפולי-גדיל ב-DNA. "
        "הגן נמצא בהקשר של נטייה תורשתית לסרטן שד ושחלה.",

        # "associated with" (English allowed in mixed text)
        "הגן CFTR מקודד לחלבון CFTR. It is associated with cystic fibrosis.",

        # "קשור למחלת" — specific disease name
        "הגן APC קשור למחלת פוליפוזיס אדנומטוטית משפחתית.",
    ])
    def test_cautious_association_passes(self, text):
        result = engine._validate_gene_education_draft(text)
        assert result is None, (
            f"Cautious association text was incorrectly rejected ({result!r}):\n{text!r}"
        )


class TestDraftValidationS25UnsafeBlocked:
    """Session 25: truly unsafe text must still be rejected."""

    @pytest.mark.parametrize("text,description", [
        ("הסיכון שלך לחלות הוא גבוה.", "personal risk"),
        ("את צריכה ניתוח לכריתת שד.", "surgery recommendation"),
        ("כדאי לעשות הפלה בגלל הממצא.", "abortion recommendation"),
        ("הגן TP53 גורמת לסרטן המעי הגס.", "definitive causation"),
        ("הגן DMD גורם למחלה קשה.", "definitive causation - DMD"),
        ("הגן מקושר לסוגי סרטן שונים.", "vague cancer list"),
        ("הגן גורם לנטייה לסרטן.", "cancer predisposition"),
        ("מגוון רחב של מחלות נגרמות מגן זה.", "broad disease list"),
        ("הגן קשור למצבים רבים כמו סרטן, מחלות לב ואלצהיימר.", "many conditions list"),
        ("הגן מוכר בהקשר של מצבים כמו סרטן, השמנה וסוכרת.", "vague conditions like"),
        ("הגן FOXP2 קשור למחלות שונות של מערכת העצבים.", "various diseases - vague"),
        ("הגן קשור למחלות רבות במערכות שונות.", "many diseases - vague"),
    ])
    def test_unsafe_text_rejected(self, text, description):
        result = engine._validate_gene_education_draft(text)
        assert result is not None, (
            f"Unsafe text ({description}) was NOT rejected:\n{text!r}"
        )


class TestDraftWarningCompactS25:
    """Session 25: _UNVERIFIED_DRAFT_WARNING_HE must be a compact badge, not a verbose paragraph."""

    BANNED = [
        "המידע הבא נוצר אוטומטית",
        "מודל שפה",
        "המשמעות האישית נקבעת",
        "לפירוש המשמעות האישית",
        "לקבלת החלטות רפואיות",
    ]

    def test_warning_short(self):
        assert len(engine._UNVERIFIED_DRAFT_WARNING_HE) < 100, (
            f"Warning must be compact. Got: {engine._UNVERIFIED_DRAFT_WARNING_HE!r}"
        )

    def test_warning_no_banned_phrases(self):
        w = engine._UNVERIFIED_DRAFT_WARNING_HE
        for phrase in self.BANNED:
            assert phrase not in w, f"Banned phrase in warning: {phrase!r}"


class TestMockedOpenAIDraftGenerationS25:
    """Session 25: mocked OpenAI must produce gene_metadata.ai_draft_generated=true."""

    BANNED_REFERRAL = [
        "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי",
        "המשמעות האישית נקבעת על ידי הצוות הגנטי",
        "לפרשנות אישית יש לפנות לצוות הגנטי",
        "לתשובה אישית",
    ]

    def _mock_gene_env(self, monkeypatch, gene, ai_text):
        import app.gene_index as gene_index
        import app.gene_cards as gene_cards
        import app.gene_knowledge as gene_knowledge

        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        summary = {
            "gene_symbol": gene,
            "total_variants": 80,
            "by_significance": {"pathogenic": 2, "vus": 50, "benign": 28},
            "phenotypes": [],
        }
        monkeypatch.setattr(gene_index, "get_gene_summary",
                            lambda g: summary if g == gene else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            engine, "_extract_gene_with_correction",
            lambda text: (gene, None) if gene in text.upper() else (None, None),
        )
        mc = engine.MagicMock() if hasattr(engine, "MagicMock") else __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mc.call_text_raw.return_value = ai_text
        monkeypatch.setattr(engine, "create_llm_client", lambda: mc)

    def test_mthfr_draft_generated(self, monkeypatch):
        from unittest.mock import MagicMock
        import app.gene_index as gene_index
        import app.gene_cards as gene_cards
        import app.gene_knowledge as gene_knowledge

        ai_text = (
            "הגן MTHFR מקודד לאנזים methylenetetrahydrofolate reductase, "
            "המעורב במטבוליזם של פולאט. "
            "MTHFR קשור למסלול פולאט ולמטבוליזם של הומוציסטאין."
        )
        gene = "MTHFR"
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        summary = {"gene_symbol": gene, "total_variants": 80,
                   "by_significance": {"pathogenic": 2, "vus": 50, "benign": 28},
                   "phenotypes": []}
        monkeypatch.setattr(gene_index, "get_gene_summary",
                            lambda g: summary if g == gene else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            engine, "_extract_gene_with_correction",
            lambda text: (gene, None) if gene in text.upper() else (None, None),
        )
        mc = MagicMock()
        mc.call_text_raw.return_value = ai_text
        monkeypatch.setattr(engine, "create_llm_client", lambda: mc)

        data = client.post("/ask", json={
            "question": "מה זה הגן MTHFR?",
            "include_unverified_gene_draft": True,
        }).json()

        meta = data.get("gene_metadata", {})
        assert meta.get("answer_tier") == "tier2", f"Expected tier2. Got: {meta}"
        assert meta.get("ai_draft_attempted") is True, "ai_draft_attempted must be True"
        assert meta.get("ai_draft_generated") is True, (
            f"ai_draft_generated must be True. ai_draft_debug: {data.get('ai_draft_debug')}"
        )
        assert data.get("unverified_gene_draft") is not None, "unverified_gene_draft must exist"
        # Answer must not contain boilerplate referral phrases
        answer = data.get("answer", "")
        for phrase in self.BANNED_REFERRAL:
            assert phrase not in answer, (
                f"Banned referral phrase in answer: {phrase!r}\nAnswer: {answer[:200]!r}"
            )

    def test_sry_draft_generated(self, monkeypatch):
        from unittest.mock import MagicMock
        import app.gene_index as gene_index
        import app.gene_cards as gene_cards
        import app.gene_knowledge as gene_knowledge

        ai_text = (
            "הגן SRY ממוקם על כרומוזום Y ומקודד לגורם שעתוק. "
            "הגן SRY קשור להתפתחות מערכת המין הזכרית בעובר."
        )
        gene = "SRY"
        monkeypatch.setattr(gene_index, "_GENE_INDEX_AVAILABLE", True)
        summary = {"gene_symbol": gene, "total_variants": 30,
                   "by_significance": {"pathogenic": 5, "vus": 20, "benign": 5},
                   "phenotypes": []}
        monkeypatch.setattr(gene_index, "get_gene_summary",
                            lambda g: summary if g == gene else None)
        monkeypatch.setattr(gene_cards, "get_approved_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_patient_summary", lambda g: None)
        monkeypatch.setattr(gene_knowledge, "get_gene_vus_note", lambda g: None)
        monkeypatch.setattr(
            engine, "_extract_gene_with_correction",
            lambda text: (gene, None) if gene in text.upper() else (None, None),
        )
        mc = MagicMock()
        mc.call_text_raw.return_value = ai_text
        monkeypatch.setattr(engine, "create_llm_client", lambda: mc)

        data = client.post("/ask", json={
            "question": "מה זה הגן SRY?",
            "include_unverified_gene_draft": True,
        }).json()

        meta = data.get("gene_metadata", {})
        assert meta.get("ai_draft_generated") is True, (
            f"SRY ai_draft_generated must be True. debug: {data.get('ai_draft_debug')}"
        )
        draft = data.get("unverified_gene_draft")
        assert draft is not None
        assert "SRY" in (draft.get("text_he") or ""), "Draft text must mention SRY"


class TestFrontendStaticTextS25:
    """Session 25: app/static/app.js must not contain banned referral/verbose phrases
    in ordinary AI draft card rendering."""

    BANNED_JS_PHRASES = [
        "המידע הבא נוצר אוטומטית על ידי מודל שפה",
        "לתשובה אישית",
        "לפרשנות אישית",
        "המשמעות האישית נקבעת",
        "משמעות אישית נקבעת",
    ]

    def test_app_js_no_banned_phrases(self):
        import os
        js_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "static", "app.js"
        )
        with open(js_path, encoding="utf-8") as f:
            content = f.read()
        for phrase in self.BANNED_JS_PHRASES:
            assert phrase not in content, (
                f"Banned phrase found in app.js: {phrase!r}"
            )
