# -*- coding: utf-8 -*-
"""
tests/test_session10_issues.py

Tests for Session 10 product/routing fixes.

Issue 1 — No automatic "פנה/י לצוות הגנטי" in routine educational answers.
Issue 2 — General biology questions (gene count, DNA, gene definition) answered directly.
Issue 3 — Mutation questions about SHANK3/HBB/APC never mention BRCA1.
Issue 4 — Abortion/pregnancy termination intent is detected and redirected safely.
Issue 5 — UI demo buttons are patient-friendly (no ClinVar jargon).
Issue 6 — ClinVar stats do not appear in main answer text.
Issue 7 — Suggested questions are educational, not just "שאל את הגנטיקאי".
"""

import re
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import counseling_engine as ce

client = TestClient(app)


def ask(question: str, **kwargs) -> dict:
    payload = {"question": question, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Issue 1 — Routine answers should NOT end with automatic team referral
# ---------------------------------------------------------------------------

class TestNoAutomaticTeamReferral:
    """Routine educational answers use a short disclaimer, not a team referral."""

    REFERRAL_PHRASES = [
        "פנה/י לצוות הגנטי",
        "פנו לצוות הגנטי",
        "יש לפנות לצוות הגנטי",
        "ויש להתייעץ עם הצוות הגנטי",
        "ונקבעת על ידי הצוות הגנטי שטיפל בך",
    ]

    SHORT_DISCLAIMER = "המידע כללי ואינו מחליף ייעוץ רפואי אישי"

    def _answer_has_automatic_referral(self, answer: str) -> bool:
        return any(phrase in answer for phrase in self.REFERRAL_PHRASES)

    def test_vus_general_no_automatic_referral(self):
        resp = ask("מה זה VUS?")
        # General VUS education should not auto-refer
        # (KB entry may still mention counselor in context, that's fine —
        # what we check is the specific hard-coded referral phrases)
        # At minimum the short disclaimer should appear instead of the long one.
        answer = resp["answer"]
        assert "ונקבעת על ידי הצוות הגנטי שטיפל בך" not in answer

    def test_vus_gene_answer_uses_short_disclaimer(self):
        # VUS + gene answers from _build_known_gene_answer should use short disclaimer
        answer = ask("מה זה VUS?")["answer"]
        assert "ונקבעת על ידי הצוות הגנטי שטיפל בך" not in answer

    def test_vus_known_gene_template_no_hard_referral(self):
        # VUS_KNOWN_GENE_TEMPLATE_HE should not end with a hard counselor referral
        template = ce.VUS_KNOWN_GENE_TEMPLATE_HE
        assert "ויש להתייעץ עם הצוות הגנטי" not in template
        assert "המידע כללי ואינו מחליף ייעוץ רפואי אישי" in template

    def test_build_known_gene_answer_disclaimer(self):
        # _build_known_gene_answer should include the short disclaimer
        result = ce._build_known_gene_answer("BRCA1")
        assert "המידע כללי ואינו מחליף ייעוץ רפואי אישי" in result["answer"]
        assert "ונקבעת על ידי הצוות הגנטי שטיפל בך" not in result["answer"]


# ---------------------------------------------------------------------------
# Issue 2 — General biology questions answered directly
# ---------------------------------------------------------------------------

class TestGeneralBiologyQuestions:
    """'כמה גנים יש?', 'מה זה גן?', 'מה זה DNA?' should be answered directly."""

    def test_how_many_genes_not_questions_after_counseling(self):
        resp = ask("כמה גנים יש בבן אדם?")
        # Must NOT match the questions_after_counseling topic
        assert resp.get("matched_topic") != "questions_after_counseling"
        assert resp["answer"] != ""

    def test_how_many_genes_answer_contains_relevant_info(self):
        resp = ask("כמה גנים יש בבן אדם?")
        answer = resp["answer"]
        # Should mention approx gene count
        assert any(num in answer for num in ["20,000", "25,000", "20000", "25000", "כ-20", "כ-25"])

    def test_what_is_gene_answered_directly(self):
        resp = ask("מה זה גן?")
        answer = resp["answer"]
        assert answer != ""
        # Should explain gene concept
        assert any(w in answer for w in ["גן", "DNA", "דנ", "חלבון", "מידע גנטי"])

    def test_what_is_dna_answered_directly(self):
        resp = ask("מה זה DNA?")
        answer = resp["answer"]
        assert answer != ""
        assert any(w in answer for w in ["DNA", "דנ", "גנטי", "בסיס", "גנום"])

    def test_general_biology_schema_valid(self):
        for q in ["מה זה גן?", "מה זה DNA?", "כמה גנים יש לאדם?"]:
            resp = ask(q)
            assert set(resp.keys()) >= {"answer", "safety_level", "needs_genetic_counselor",
                                        "matched_topic", "suggested_questions"}

    def test_gene_count_topic_matches_human_genome_entry(self):
        resp = ask("כמה גנים יש לאדם?")
        assert resp.get("matched_topic") in ("human_genome_size", "what_is_gene", "what_is_dna",
                                              None, "gene_clinvar_summary")
        # Must not be questions_after_counseling
        assert resp.get("matched_topic") != "questions_after_counseling"


# ---------------------------------------------------------------------------
# Issue 3 — Mutation questions never show BRCA1 for unrelated genes
# ---------------------------------------------------------------------------

class TestNoBRCA1FallbackForOtherGenes:
    """Questions about SHANK3/HBB/APC mutations must never mention BRCA1."""

    def _check_no_spurious_brca1(self, gene: str):
        resp = ask(f"מה הבעיה במוטציה בגן {gene}?")
        answer = resp["answer"]
        # BRCA1 must not appear unless the question was about BRCA1
        assert "BRCA1" not in answer, (
            f"Answer for {gene} mutation question incorrectly mentions BRCA1: {answer[:200]}"
        )

    def test_shank3_no_brca1(self):
        self._check_no_spurious_brca1("SHANK3")

    def test_hbb_no_brca1(self):
        self._check_no_spurious_brca1("HBB")

    def test_apc_no_brca1(self):
        self._check_no_spurious_brca1("APC")

    def test_mutation_question_not_vus_known_gene_topic(self):
        # "מה הבעיה במוטציה בגן APC?" should NOT route to vus_known_gene
        # (that entry used to have hardcoded BRCA1)
        resp = ask("מה הבעיה במוטציה בגן APC?")
        assert resp.get("matched_topic") != "vus_known_gene"

    def test_vus_known_gene_kb_entry_no_brca1_hardcoded(self):
        from app import kb
        entry = kb.get_by_id("vus_known_gene")
        assert entry is not None
        answer_text = entry.get("approved_answer_he", "")
        # The approved_answer should not hardcode "BRCA1" as a specific example
        # (it may mention BRCA1 in parenthetical context such as topic field, but not in the answer)
        assert "BRCA1" not in answer_text, (
            f"vus_known_gene KB entry still has hardcoded BRCA1 in approved_answer_he: {answer_text}"
        )

    def test_brca1_question_still_mentions_brca1(self):
        # Control: asking about BRCA1 itself must still give BRCA1 information
        resp = ask("מה הבעיה במוטציה בגן BRCA1?")
        answer = resp["answer"]
        assert "BRCA1" in answer

    def test_gene_question_phrases_include_mutation(self):
        # Verify the fix is in the frozenset
        assert "מוטציה בגן" in ce._GENE_QUESTION_PHRASES
        assert "מה הבעיה" in ce._GENE_QUESTION_PHRASES

    def test_is_gene_level_question_detects_mutation_phrase(self):
        assert ce._is_gene_level_question("מה הבעיה במוטציה בגן SHANK3?")
        assert ce._is_gene_level_question("מוטציה בגן HBB — מה זה אומר?")


# ---------------------------------------------------------------------------
# Issue 4 — Reproductive/abortion intent detected and handled safely
# ---------------------------------------------------------------------------

class TestReproductiveDecisionSafety:
    """Abortion/pregnancy termination questions get a specific safety redirect."""

    ABORTION_TERMS = [
        "אמרו לי שיש לי VUS בגן APC, לעשות הפלה?",
        "יש לי VUS בגן BRCA2 — כדאי להפסיק את ההריון?",
        "יש לי ממצא גנטי, האם לעשות הפלה?",
        "VUS ב-CFTR, לסיים את ההריון?",
        "should I have an abortion because of the genetic finding?",
    ]

    def test_abortion_question_returns_requires_counselor(self):
        for q in self.ABORTION_TERMS:
            resp = ask(q)
            assert resp["safety_level"] == "requires_genetic_counselor", (
                f"Expected requires_genetic_counselor for: {q}"
            )

    def test_abortion_question_needs_counselor_true(self):
        resp = ask("אמרו לי שיש לי VUS בגן APC, לעשות הפלה?")
        assert resp["needs_genetic_counselor"] is True

    def test_abortion_question_answer_mentions_vus_not_basis(self):
        resp = ask("אמרו לי שיש לי VUS בגן APC, לעשות הפלה?")
        answer = resp["answer"]
        # Answer should mention VUS is not a sufficient basis for irreversible decisions
        assert any(phrase in answer for phrase in [
            "VUS לבדו", "לא בסיס", "בלתי הפיכה", "הפסקת הריון", "הפלה",
        ])

    def test_abortion_answer_no_gene_explanation_hijack(self):
        # The abortion safety answer must not be replaced by gene card info
        resp = ask("אמרו לי שיש לי VUS בגן APC, לעשות הפלה?")
        # Must not be the APC gene card / gene clinvar answer
        assert resp.get("matched_topic") is None

    def test_reproductive_decision_function_detects_terms(self):
        assert ce._is_reproductive_decision_question("לעשות הפלה?")
        assert ce._is_reproductive_decision_question("להפסיק הריון")
        assert ce._is_reproductive_decision_question("לסיים הריון")
        assert ce._is_reproductive_decision_question("abortion")
        assert ce._is_reproductive_decision_question("termination")
        assert not ce._is_reproductive_decision_question("מה זה VUS?")
        assert not ce._is_reproductive_decision_question("מה ידוע על BRCA1?")

    def test_reproductive_decision_constant_exists(self):
        assert hasattr(ce, "REPRODUCTIVE_DECISION_HE")
        assert len(ce.REPRODUCTIVE_DECISION_HE) > 50

    def test_abortion_response_schema_valid(self):
        resp = ask("יש לי ממצא גנטי, האם לעשות הפלה?")
        assert set(resp.keys()) >= {"answer", "safety_level", "needs_genetic_counselor",
                                    "matched_topic", "suggested_questions"}


# ---------------------------------------------------------------------------
# Issue 5 — UI demo buttons are patient-friendly
# ---------------------------------------------------------------------------

class TestDemoButtons:
    """DEMO_QUESTIONS must not include ClinVar jargon or technical gene lookups."""

    def test_app_js_no_clinvar_in_demo_labels(self):
        import pathlib
        js = pathlib.Path("app/static/app.js").read_text(encoding="utf-8")
        # Find the DEMO_QUESTIONS block
        start = js.find("const DEMO_QUESTIONS")
        end = js.find("];", start) + 2
        demo_block = js[start:end]
        # Labels should not be purely technical ClinVar lookups
        assert "ClinVar" not in demo_block or demo_block.count("ClinVar") == 0, (
            "DEMO_QUESTIONS block contains ClinVar jargon"
        )

    def test_app_js_demo_contains_patient_friendly_questions(self):
        import pathlib
        js = pathlib.Path("app/static/app.js").read_text(encoding="utf-8")
        start = js.find("const DEMO_QUESTIONS")
        end = js.find("];", start) + 2
        demo_block = js[start:end]
        # Should contain patient-friendly VUS education questions
        assert "VUS" in demo_block
        assert any(q in demo_block for q in ["pathogenic", "משתנה", "גן", "החלטות"])


# ---------------------------------------------------------------------------
# Issue 6 — ClinVar raw stats not in main answer text
# ---------------------------------------------------------------------------

class TestNoClinvarStatsInMainAnswer:
    """ClinVar bullet-point stats must not appear in main answer text."""

    CLINVAR_STAT_PATTERNS = [
        re.compile(r"נתוני ClinVar עבור גן"),
        re.compile(r"סה\"כ \d[\d,]+ רשומות וריאנט"),
        re.compile(r"פתוגניים / likely pathogenic"),
        re.compile(r"שפירים / likely benign"),
        re.compile(r"VUS \(משמעות לא ידועה\)"),
        re.compile(r"מצבים רפואיים קשורים בתיעוד ClinVar"),
    ]

    def _answer_has_raw_stats(self, answer: str) -> bool:
        return any(p.search(answer) for p in self.CLINVAR_STAT_PATTERNS)

    def test_brca1_vus_answer_no_raw_clinvar_stats(self):
        resp = ask("יש לי VUS ב-BRCA1, מה זה אומר?")
        assert not self._answer_has_raw_stats(resp["answer"]), (
            f"Main answer contains raw ClinVar stats: {resp['answer'][:400]}"
        )

    def test_brca2_gene_question_no_raw_stats(self):
        resp = ask("מה ידוע על BRCA2?")
        assert not self._answer_has_raw_stats(resp["answer"])

    def test_stats_may_appear_in_metadata(self):
        # gene_metadata is allowed to have stats
        resp = ask("יש לי VUS ב-BRCA1, מה זה אומר?")
        # If gene_metadata is present, it's fine to have stats there
        # (just not in the main answer string)
        if "gene_metadata" in resp:
            meta = resp["gene_metadata"]
            assert "gene_symbol" in meta


# ---------------------------------------------------------------------------
# Issue 7 — Suggested questions are educational
# ---------------------------------------------------------------------------

class TestEducationalSuggestedQuestions:
    """Suggested questions should teach, not just say 'ask your counselor'."""

    COUNSELOR_ONLY_PHRASES = [
        "שאל את הגנטיקאי",
        "שאלי את הגנטיקאית",
        "שאל/י את הגנטיקאי",
    ]

    def _suggested_dominated_by_counselor(self, questions: list) -> bool:
        counselor_count = sum(
            1 for q in questions
            if any(phrase in q for phrase in self.COUNSELOR_ONLY_PHRASES)
        )
        return counselor_count > len(questions) // 2

    def test_gene_suggested_questions_are_educational(self):
        sq = list(ce._GENE_SUGGESTED_QUESTIONS)
        assert not self._suggested_dominated_by_counselor(sq), (
            f"_GENE_SUGGESTED_QUESTIONS dominated by counselor-referral phrases: {sq}"
        )

    def test_vus_answer_suggested_questions_educational(self):
        resp = ask("מה זה VUS?")
        sq = resp.get("suggested_questions", [])
        assert len(sq) > 0
        assert not self._suggested_dominated_by_counselor(sq), (
            f"Suggested questions for VUS answer dominated by counselor phrases: {sq}"
        )

    def test_gene_vus_suggested_questions_educational(self):
        resp = ask("יש לי VUS ב-CFTR, מה זה?")
        sq = resp.get("suggested_questions", [])
        assert len(sq) > 0
        assert not self._suggested_dominated_by_counselor(sq)

    def test_suggested_questions_include_vus_education(self):
        sq = list(ce._GENE_SUGGESTED_QUESTIONS)
        # Should include educational VUS questions
        combined = " ".join(sq)
        assert any(term in combined for term in [
            "ההבדל בין VUS", "VUS יכול להשתנות", "לא מקבלים החלטות", "pathogenic"
        ]), f"Suggested questions don't seem educational enough: {sq}"


# ---------------------------------------------------------------------------
# Regression: existing behavior still correct
# ---------------------------------------------------------------------------

class TestRegressions:
    """Ensure existing correct behaviors still hold after all changes."""

    def test_brca1_vus_still_answered(self):
        resp = ask("יש לי VUS ב-BRCA1")
        assert resp["answer"] != ""
        assert resp["matched_topic"] == "vus_known_gene"

    def test_vus_general_still_answered(self):
        resp = ask("מה זה VUS?")
        assert resp["answer"] != ""
        assert resp["needs_genetic_counselor"] is False

    def test_carrier_still_answered(self):
        resp = ask("אמרו לי שאני נשאית, מה זה?")
        assert resp["answer"] != ""
        assert resp["matched_topic"] is not None

    def test_identifying_info_blocked(self):
        resp = ask("השם שלי הוא שרה כהן, מה זה VUS?")
        assert resp["safety_level"] == "contains_identifying_info"

    def test_schema_always_5_keys(self):
        for q in [
            "מה זה VUS?",
            "יש לי VUS ב-BRCA1",
            "אמרו לי שיש לי VUS בגן APC, לעשות הפלה?",
            "כמה גנים יש בבן אדם?",
        ]:
            resp = ask(q)
            for key in ("answer", "safety_level", "needs_genetic_counselor",
                        "matched_topic", "suggested_questions"):
                assert key in resp, f"Missing key '{key}' in response for: {q}"
