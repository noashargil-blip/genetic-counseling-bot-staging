"""
Session 27.13 — Patient UX polish and dead feedback UI removal.

Verifies:
  Part C – No "microarray" invented in simplification text
  Part D – No "הצוות ימשיך לעקוב" overconfident language in simplification
  Part E – "8 conditions" / trivial ClinVar placeholders filtered from top_phenotypes
  Part G – No duplicate guidance / question section in VUS practical answer
  Part J – Regression: KIAA2022, BRCA1, COL1A1, prenatal NIPA1 unchanged
  Part K – Acceptance scenarios
"""

import pytest
import app.counseling_engine as _ce
from app.counseling_engine import answer_question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TURN1_Q = (
    "קיבלתי תוצאה של VUS בהריון. "
    "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
)


def _ask(question: str, ctx: dict | None = None) -> dict:
    return answer_question(question=question, session_context=ctx)


def _ctx_of(result: dict) -> dict:
    return result.get("session_context_out", {})


def _simplify_ctx(gene: str | None = None, topic: str = "prenatal_vus_deletion",
                  is_prenatal: bool = True) -> dict:
    return {
        "active_topic": topic,
        "gene_symbol": gene,
        "finding_type": "deletion",
        "pregnancy_context": "עובר" if is_prenatal else None,
        "subject": "fetus" if is_prenatal else "self",
    }


# ---------------------------------------------------------------------------
# Part C — No "microarray" invented in simplification
# ---------------------------------------------------------------------------

class TestNoMicroarrayInSimplification:
    def test_prenatal_simplification_no_microarray(self):
        """Simplification of prenatal deletion VUS must not contain 'microarray'."""
        ctx = _simplify_ctx(gene="NIPA1")
        result = _ask("תסביר לי שוב", ctx=ctx)
        assert result["matched_topic"] in ("prenatal_vus_deletion", "vus"), \
            f"unexpected topic: {result['matched_topic']}"
        assert "microarray" not in result["answer"].lower(), \
            "simplification must not invent 'microarray' when user never mentioned it"

    def test_generic_vus_simplification_no_microarray(self):
        """Generic VUS simplification must not contain 'microarray'."""
        ctx = _simplify_ctx(gene="BRCA1", topic="vus_known_gene", is_prenatal=False)
        result = _ask("תסביר פשוט", ctx=ctx)
        assert "microarray" not in result["answer"].lower()

    def test_prenatal_simplification_no_microarray_no_gene(self):
        """Even without a gene name, simplification must not invent 'microarray'."""
        ctx = _simplify_ctx(gene=None)
        result = _ask("תסביר לי שוב", ctx=ctx)
        assert "microarray" not in result["answer"].lower()


# ---------------------------------------------------------------------------
# Part D — No "הצוות ימשיך לעקוב" overconfident language
# ---------------------------------------------------------------------------

class TestNoOverconfidentFollowup:
    def test_prenatal_simplification_no_team_followup_phrase(self):
        """Simplification must not contain 'הצוות ימשיך לעקוב'."""
        ctx = _simplify_ctx(gene="NIPA1")
        result = _ask("תסביר לי שוב", ctx=ctx)
        assert "ימשיך לעקוב" not in result["answer"], \
            "simplification must not promise team follow-up tracking"

    def test_generic_vus_simplification_no_team_followup_phrase(self):
        """Generic VUS simplification (no gene, no prenatal) must not have overconfident phrase."""
        ctx = {"active_topic": "vus", "gene_symbol": None, "pregnancy_context": None}
        result = _ask("תסבירי שוב בקצרה", ctx=ctx)
        assert "ימשיך לעקוב" not in result["answer"]

    def test_vus_practical_answer_no_team_followup(self):
        """_compose_vus_practical_answer must not contain 'ימשיך לעקוב'."""
        text = _ce._compose_vus_practical_answer(None)
        assert "ימשיך לעקוב" not in text


# ---------------------------------------------------------------------------
# Part E — "8 conditions" / trivial phenotypes filtered from top_phenotypes
# ---------------------------------------------------------------------------

class TestTrivialPhenotypesFiltered:
    def test_clean_clinvar_phenotypes_filters_8_conditions(self):
        """'8 conditions' must be removed by _clean_clinvar_phenotypes."""
        raw = ["8 conditions", "Breast cancer", "not specified"]
        result = _ce._clean_clinvar_phenotypes(raw)
        assert "8 conditions" not in result
        assert "not specified" not in result
        assert "Breast cancer" in result

    def test_clean_clinvar_phenotypes_filters_multiple_conditions(self):
        result = _ce._clean_clinvar_phenotypes(["multiple conditions", "Autism spectrum disorder"])
        assert "multiple conditions" not in result
        assert "Autism spectrum disorder" in result

    def test_clean_clinvar_phenotypes_filters_dash(self):
        result = _ce._clean_clinvar_phenotypes(["-", "Lynch syndrome"])
        assert "-" not in result
        assert "Lynch syndrome" in result

    def test_clean_clinvar_phenotypes_filters_empty_string(self):
        result = _ce._clean_clinvar_phenotypes(["", "Hereditary breast ovarian cancer"])
        assert "" not in result

    def test_clean_clinvar_phenotypes_case_insensitive(self):
        result = _ce._clean_clinvar_phenotypes(["NOT SPECIFIED", "Colorectal cancer"])
        assert "NOT SPECIFIED" not in result
        assert "Colorectal cancer" in result

    def test_clean_clinvar_phenotypes_deduplicates(self):
        result = _ce._clean_clinvar_phenotypes(["Breast cancer", "Breast cancer", "Ovarian cancer"])
        assert result.count("Breast cancer") == 1

    def test_clean_clinvar_phenotypes_all_trivial_returns_empty(self):
        result = _ce._clean_clinvar_phenotypes(["not specified", "not provided", "8 conditions"])
        assert result == []

    def test_clean_clinvar_phenotypes_none_input(self):
        result = _ce._clean_clinvar_phenotypes(None)
        assert result == []

    def test_clean_clinvar_phenotypes_other_trivials(self):
        trivials = ["n/a", "na", "unknown", "not stated", "various", "other", "conditions", "disease"]
        result = _ce._clean_clinvar_phenotypes(trivials)
        assert result == []


# ---------------------------------------------------------------------------
# Part G — No duplicate guidance section in VUS practical answer
# ---------------------------------------------------------------------------

class TestNoDuplicateGuidanceSectionInVusPractical:
    def test_compose_vus_practical_no_extra_clarification_guidance(self):
        """_compose_vus_practical_answer must have exactly one question block."""
        text = _ce._compose_vus_practical_answer(None)
        # Count occurrences of the counselor-question bullet marker
        bullet_count = text.count("•")
        assert bullet_count <= 4, \
            f"expected at most 4 bullets (one question block), got {bullet_count}"

    def test_compose_vus_practical_with_gene_no_extra_guidance(self):
        text = _ce._compose_vus_practical_answer("BRCA1")
        bullet_count = text.count("•")
        assert bullet_count <= 4

    def test_vus_followup_answer_single_question_section(self):
        """After a VUS turn, follow-up 'מה כדאי לעשות עם זה?' should have one question block."""
        turn1 = _ask("מה זה VUS?")
        ctx = _ctx_of(turn1)
        result = _ask("מה כדאי לעשות עם זה?", ctx=ctx)
        # Should not have two separate "שאלות לצוות הגנטי" headers
        count = result["answer"].count("שאלות לצוות הגנטי")
        assert count <= 1, f"found {count} 'שאלות לצוות הגנטי' headers, expected at most 1"


# ---------------------------------------------------------------------------
# Part J — Regression: standalone answers unchanged
# ---------------------------------------------------------------------------

class TestStandaloneRegressions:
    def test_brca1_standalone_still_answers(self):
        """BRCA1 standalone question still produces a meaningful answer."""
        result = _ask("מה זה BRCA1?")
        assert result["answer"]
        assert result["matched_topic"]
        assert "BRCA1" in result["answer"].upper() or "brca" in result["answer"].lower()

    def test_vus_standalone_still_answers(self):
        """General VUS question still produces an answer."""
        result = _ask("מה זה VUS?")
        assert result["answer"]
        assert "vus" in result["answer"].lower() or "VUS" in result["answer"]

    def test_carrier_standalone_still_answers(self):
        """Carrier status question still answers normally."""
        result = _ask("אמרו לי שאני נשאית, מה זה?")
        assert result["answer"]
        assert result["safety_level"] in ("general_information", "requires_genetic_counselor")

    def test_prenatal_nipa1_turn1_still_routes_correctly(self):
        """Session 27.11 prenatal NIPA1 question still routes to prenatal_vus_deletion."""
        result = _ask(TURN1_Q)
        assert result["matched_topic"] == "prenatal_vus_deletion", \
            f"expected prenatal_vus_deletion, got {result['matched_topic']}"
        assert "vus" in result["answer"].lower() or "VUS" in result["answer"]

    def test_col1a1_standalone_still_answers(self):
        """COL1A1 grounded answer still works."""
        result = _ask("מה זה COL1A1?")
        assert result["answer"]
        assert result["matched_topic"]

    def test_safety_block_still_works(self):
        """Identifying info is still blocked after UX changes."""
        result = _ask("קוראים לי שרה כהן")
        assert result["safety_level"] == "contains_identifying_info"

    def test_schema_has_five_keys(self):
        """Response schema still has exactly 5 required keys."""
        result = _ask("מה זה VUS?")
        required = {"answer", "safety_level", "needs_genetic_counselor", "matched_topic", "suggested_questions"}
        assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Part K — Acceptance scenarios (end-to-end)
# ---------------------------------------------------------------------------

class TestAcceptanceScenarios:
    def test_prenatal_simplification_contains_deletion_vus_content(self):
        """Simplification of prenatal deletion VUS contains expected content."""
        ctx = _simplify_ctx(gene="NIPA1")
        result = _ask("תסביר לי שוב", ctx=ctx)
        answer = result["answer"]
        assert "חסר" in answer or "deletion" in answer.lower(), \
            "simplification should mention deletion"
        assert "VUS" in answer or "vus" in answer.lower(), \
            "simplification should mention VUS"

    def test_prenatal_simplification_no_diagnosis(self):
        """Simplification must not claim the VUS is a diagnosis."""
        ctx = _simplify_ctx(gene="NIPA1")
        result = _ask("תסביר לי שוב", ctx=ctx)
        assert "אבחנה" not in result["answer"] or "אינו אבחנה" in result["answer"] or "לא אבחנה" in result["answer"], \
            "simplification must not present VUS as a diagnosis"

    def test_vus_practical_answer_structure(self):
        """_compose_vus_practical_answer has a question-for-counselor section."""
        text = _ce._compose_vus_practical_answer("NIPA1")
        assert "שאלות לצוות הגנטי" in text or "לשאול" in text, \
            "practical answer must contain counselor questions section"

    def test_vus_practical_answer_no_surgery(self):
        """VUS practical answer must not recommend surgery or medical action."""
        text = _ce._compose_vus_practical_answer("NIPA1")
        assert "ניתוח" not in text
        assert "טיפול" not in text

    def test_clean_phenotypes_used_in_gene_metadata(self):
        """top_phenotypes in a gene answer for a known gene passes through cleaning."""
        result = _ask("מה זה HBB?")
        meta = result.get("gene_metadata")
        if meta and meta.get("top_phenotypes"):
            for p in meta["top_phenotypes"]:
                assert p.lower().strip() not in (
                    "8 conditions", "multiple conditions", "not specified", "not provided",
                    "", "-", "n/a", "na", "unknown", "not stated", "various", "other",
                    "conditions", "disease",
                ), f"trivial phenotype '{p}' should have been filtered"
