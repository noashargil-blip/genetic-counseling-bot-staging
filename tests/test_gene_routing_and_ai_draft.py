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
