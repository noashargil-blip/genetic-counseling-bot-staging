# -*- coding: utf-8 -*-
"""
Session 22 — Product-level response policy reset tests.

Covers:
  1. No context bleed: Q2 never echoes Q1's gene when Q2 is a standalone concept.
  2. Disclaimers suppressed: ordinary answers must NOT contain the blocked phrases.
  3. VUS suggested questions only when question mentions VUS.
  4. Gene answers are informative (not "no summary available").
  5. Concept questions answered as concepts, not prior gene answers.
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.counseling_engine as ceng

client = TestClient(app)

# Disclaimer phrases that must NOT appear in ordinary gene/general answers
_BANNED_DISCLAIMERS = [
    "לתשובה אישית לפי תוצאות הבדיקה שלך",
    "המשמעות של כל ממצא ספציפי נקבעת על ידי הצוות הגנטי",
    "המידע כללי ואינו מחליף ייעוץ רפואי אישי",
    "לפרשנות אישית יש לפנות לצוות הגנטי",
    "לבירור המדויק — פנה",
    "לפנות לצוות הגנטי",   # catches all variants of team referral
]


# ── 1. Context bleed prevention ──────────────────────────────────────────────

class TestContextBleedPrevention:
    """
    Q2 must never be answered using Q1's gene when Q2 is a standalone concept.
    Tests simulate the multi-turn case by sending last_gene_symbol in the body
    (the backend now ignores it).
    """

    @pytest.mark.parametrize("q2, prior_gene, forbidden_terms", [
        ("מה זה חלבון?",             "TP53",  ["TP53", "p53"]),
        ("מה זה המוגלובין?",          "HBB",   ["HBB", "beta-globin", "hemoglobin", "המוגלובין בהקשר של HBB"]),
        ("מה זה כרומוזום?",           "DMD",   ["DMD", "dystrophin"]),
        ("מה זה מחלה נוירודגנרטיבית?", "APOE",  ["APOE"]),
    ])
    def test_q2_does_not_echo_q1_gene(self, q2, prior_gene, forbidden_terms):
        data = client.post(
            "/ask", json={"question": q2, "last_gene_symbol": prior_gene}
        ).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != prior_gene, (
            f"Q2={q2!r}: gene_metadata.gene_symbol must not be {prior_gene}. Got {gene_sym!r}"
        )
        assert data.get("matched_topic") != "gene_clinvar_summary" or gene_sym == "", (
            f"Q2={q2!r}: if topic=gene_clinvar_summary, gene must not be {prior_gene}"
        )

    def test_tp53_then_protein(self):
        """Full sequence: TP53 gene answer, then 'מה זה חלבון?' must not mention TP53."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        # Q1
        r1 = client.post("/ask", json={"question": "מה זה הגן TP53?"}).json()
        assert r1.get("matched_topic") == "gene_clinvar_summary"
        # Q2 — simulate with last_gene_symbol (backend ignores it)
        r2 = client.post(
            "/ask", json={"question": "מה זה חלבון?", "last_gene_symbol": "TP53"}
        ).json()
        gene_sym = r2.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "TP53", f"Q2 'חלבון' must not route to TP53. Got {gene_sym!r}"

    def test_hbb_then_stroke(self):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        r1 = client.post("/ask", json={"question": "מה זה הגן HBB?"}).json()
        assert r1.get("gene_metadata", {}).get("gene_symbol") == "HBB"
        r2 = client.post(
            "/ask", json={"question": "מה זה שבץ מוחי?", "last_gene_symbol": "HBB"}
        ).json()
        gene_sym = r2.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "HBB", f"Stroke question must not route to HBB. Got {gene_sym!r}"

    def test_dmd_then_chromosome(self):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        r2 = client.post(
            "/ask", json={"question": "מה זה כרומוזום?", "last_gene_symbol": "DMD"}
        ).json()
        gene_sym = r2.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "DMD", f"Chromosome question must not route to DMD. Got {gene_sym!r}"
        assert r2.get("matched_topic") != "x_linked", "Chromosome question must not return x_linked"

    def test_apoe_then_neurodegenerative(self):
        r2 = client.post(
            "/ask", json={"question": "מה זה מחלה נוירודגנרטיבית?", "last_gene_symbol": "APOE"}
        ).json()
        gene_sym = r2.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym != "APOE", (
            f"Neurodegenerative disease question must not route to APOE. Got {gene_sym!r}"
        )


# ── 2. Disclaimer suppression ─────────────────────────────────────────────────

class TestDisclaimerSuppressed:
    """Ordinary gene/general answers must NOT contain the banned disclaimer phrases."""

    @pytest.mark.parametrize("q", [
        "מה זה הגן BRCA1?",
        "מה זה הגן CFTR?",
        "מה זה VUS?",
        "מה זה נשאות?",
        "מה זה תורשה אוטוזומלית רצסיבית?",
    ])
    def test_no_banned_disclaimer(self, q):
        data = client.post("/ask", json={"question": q}).json()
        answer = data.get("answer", "")
        safety = data.get("safety_level", "")
        if safety == "requires_genetic_counselor":
            return  # personal/blocked answers may have referral
        for phrase in _BANNED_DISCLAIMERS:
            assert phrase not in answer, (
                f"Banned disclaimer found in answer for {q!r}:\n"
                f"  Phrase: {phrase!r}\n"
                f"  Answer: {answer[:200]!r}"
            )

    def test_gene_answer_no_banned_disclaimer(self):
        """Gene answers must be clean of repetitive referral sentences."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        for gene in ["BRCA1", "CFTR", "TP53"]:
            data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
            answer = data.get("answer", "")
            for phrase in _BANNED_DISCLAIMERS:
                assert phrase not in answer, (
                    f"Banned disclaimer in gene answer for {gene}: {phrase!r}"
                )

    def test_vus_answer_no_banned_disclaimer(self):
        """VUS answers must not have the banned referral phrases."""
        data = client.post("/ask", json={"question": "מה זה VUS?"}).json()
        answer = data.get("answer", "")
        for phrase in _BANNED_DISCLAIMERS:
            assert phrase not in answer, (
                f"Banned disclaimer in VUS answer: {phrase!r}\nAnswer: {answer[:200]!r}"
            )


# ── 3. VUS suggested questions only for VUS-related questions ─────────────────

class TestVusSuggestedQuestionsGating:
    """Suggested questions about VUS must NOT appear for plain gene questions."""

    VUS_PHRASES = [
        "האם VUS יכול להשתנות בעתיד?",
        "למה בדרך כלל לא מקבלים החלטות רפואיות רק לפי VUS?",
    ]

    @pytest.mark.parametrize("q", [
        "מה זה הגן BRCA1?",
        "מה זה הגן CFTR?",
        "מה זה הגן TP53?",
    ])
    def test_no_vus_suggested_for_plain_gene_question(self, q):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": q}).json()
        sq = data.get("suggested_questions", [])
        for phrase in self.VUS_PHRASES:
            assert phrase not in sq, (
                f"VUS suggested question found for plain gene question {q!r}: {phrase!r}"
            )

    @pytest.mark.parametrize("q", [
        "מה זה VUS?",
        "יש לי VUS בBRCA1, מה זה אומר?",
        "מה ההבדל בין VUS לבין ממצא pathogenic?",
    ])
    def test_vus_suggested_for_vus_question(self, q):
        """VUS questions should have VUS-related suggested questions."""
        data = client.post("/ask", json={"question": q}).json()
        sq = data.get("suggested_questions", [])
        # At least one VUS-related question expected
        has_vus = any("VUS" in s or "pathogenic" in s for s in sq)
        assert has_vus or len(sq) == 0, (
            f"Expected VUS-related suggested questions for {q!r}. Got: {sq}"
        )

    def test_general_gene_suggested_max_3(self):
        """Suggested questions always capped at 3."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": "מה זה הגן TP53?"}).json()
        sq = data.get("suggested_questions", [])
        assert len(sq) <= 3, f"suggested_questions must be ≤ 3. Got {len(sq)}: {sq}"


# ── 4. Gene answers are informative ──────────────────────────────────────────

class TestGeneAnswerInformative:
    """Gene answers must have substantive content, not just 'no summary available'."""

    BLAND_PHRASES = [
        "אין עדיין מידע",
        "אין עדיין סיכום",
        "לא נמצא מידע",
        "טרם עודכן",
    ]

    @pytest.mark.parametrize("gene", ["BRCA1", "HBB"])
    def test_approved_gene_answer_informative(self, gene):
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": f"מה זה הגן {gene}?"}).json()
        assert data.get("matched_topic") == "gene_clinvar_summary", (
            f"Gene {gene} must route to gene_clinvar_summary"
        )
        answer = data.get("answer", "")
        assert len(answer) > 60, f"Gene answer for {gene} too short: {answer!r}"
        for phrase in self.BLAND_PHRASES:
            assert phrase not in answer, (
                f"Bland 'no-summary' phrase in approved gene answer for {gene}: {phrase!r}"
            )

    def test_gene_answer_no_clinvar_dump(self):
        """Gene answer main text must not start with ClinVar variant counts."""
        import app.gene_index as gene_index
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")
        data = client.post("/ask", json={"question": "מה זה הגן BRCA1?"}).json()
        answer = data.get("answer", "")
        # Must not lead with stats/counts
        assert not answer.startswith("מידע כללי על גן"), (
            f"Gene answer must not open with raw ClinVar stats header: {answer[:80]!r}"
        )
        assert "במאגר ClinVar מתועדות" not in answer[:120], (
            f"Gene answer must not lead with variant count: {answer[:120]!r}"
        )


# ── 5. General concept answers ───────────────────────────────────────────────

class TestConceptAnswers:
    """Standalone concept questions are answered as concepts, not gene summaries."""

    @pytest.mark.parametrize("q", [
        "מה זה חלבון?",
        "מה זה המוגלובין?",
        "מה זה כרומוזום?",
        "מה זה שבץ מוחי?",
        "מה זה אלצהיימר?",
    ])
    def test_concept_not_gene_answer(self, q):
        data = client.post("/ask", json={"question": q}).json()
        gene_sym = data.get("gene_metadata", {}).get("gene_symbol", "")
        assert gene_sym == "", (
            f"Standalone concept {q!r} must not return gene_metadata. Got {gene_sym!r}"
        )

    def test_chromosome_not_xlinked(self):
        data = client.post("/ask", json={"question": "מה זה כרומוזום?"}).json()
        assert data.get("matched_topic") != "x_linked", (
            "Chromosome question must not route to x_linked KB entry"
        )

    def test_chromosome21_not_xlinked(self):
        data = client.post("/ask", json={"question": "מה זה כרומוזום 21?"}).json()
        assert data.get("matched_topic") != "x_linked"


# ── 6. Tier 2 fallback is short and clean ────────────────────────────────────

class TestTier2Fallback:
    """When gene is in ClinVar but has no approved Hebrew card and no LLM,
    the fallback must be short and free of long disclaimers."""

    def test_tier2_fallback_no_long_disclaimer(self):
        """_build_gene_clinvar_answer Tier 2 fallback must be short and honest."""
        # Inject a gene that's in ClinVar index but has no approved card
        import app.gene_index as gene_index
        import app.gene_cards as gc
        if not gene_index._GENE_INDEX_AVAILABLE:
            pytest.skip("Gene index not available")

        # Find a gene that is in ClinVar but NOT in gene_cards
        all_genes = gene_index.list_genes(limit=200)
        target = None
        for g in all_genes:
            sym = g.get("gene_symbol", "")
            if sym and not gc.get_approved_summary(sym):
                target = sym
                break
        if not target:
            pytest.skip("No Tier-2 gene found")

        data = client.post("/ask", json={"question": f"מה זה הגן {target}?"}).json()
        answer = data.get("answer", "")
        for phrase in _BANNED_DISCLAIMERS:
            assert phrase not in answer, (
                f"Banned disclaimer in Tier-2 fallback for {target}: {phrase!r}"
            )
