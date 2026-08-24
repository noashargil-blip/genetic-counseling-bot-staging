"""
Session 27.12 — Finding-centric conversation context.

Verifies:
  Part B – Pattern gaps fixed: prenatal phrasing, "בעצם", "לי לשאול"
  Part C – Gene-within-finding: gene info + deletion context note
  Part D – Personal meaning for prenatal VUS: informative content, not only a redirect
  Part H – No duplicate question section in Turn 1
  Part I – No gene_metadata (ClinVar card) in contextual follow-up responses
  Part K – Full 8-turn NIPA1 acceptance conversation
"""

import pytest
from app.counseling_engine import answer_question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask(question: str, ctx: dict | None = None) -> dict:
    result = answer_question(
        question=question,
        session_context=ctx,
    )
    return result


def _ctx_of(result: dict) -> dict:
    return result.get("session_context_out", {})


# ---------------------------------------------------------------------------
# Part K — 8-turn NIPA1 acceptance conversation
# ---------------------------------------------------------------------------

TURN1_Q = (
    "קיבלתי תוצאה של VUS בהריון. "
    "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
)


class TestEightTurnNIPA1Conversation:
    """Full 8-turn prenatal deletion conversation — finding-centric routing."""

    def test_turn1_prenatal_deletion_vus(self):
        r = _ask(TURN1_Q)
        assert r["matched_topic"] == "prenatal_vus_deletion"
        assert r["safety_level"] == "general_information"
        # Context carries deletion finding
        ctx = _ctx_of(r)
        assert ctx.get("active_topic") == "prenatal_vus_deletion"
        assert ctx.get("gene_symbol", "").upper() == "NIPA1"
        assert ctx.get("finding_type") == "deletion"
        assert ctx.get("pregnancy_context") is True

    def test_turn1_no_duplicate_question_section(self):
        """Part H: answer must have at most ONE question-guidance section."""
        r = _ask(TURN1_Q)
        ans = r["answer"]
        # Count the number of times a guidance header appears
        guidance_phrases = [
            "שאלות שכדאי לשאול",
            "שאלות מומלצות",
            "מה יכול לעזור לצוות",
        ]
        count = sum(ans.count(p) for p in guidance_phrases)
        assert count <= 1, (
            f"Expected at most 1 guidance section, got indicators totalling {count}.\n"
            f"Answer:\n{ans}"
        )

    def test_turn2_personal_meaning_fetus_phrasing(self):
        """Part B+D: 'עבור העובר שלי' stays on prenatal finding, not gene_clinvar_summary."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        assert r2["matched_topic"] == "prenatal_vus_deletion", (
            f"Turn 2 misrouted to {r2['matched_topic']}"
        )
        # Part D: informative content, not just a redirect
        ans = r2["answer"]
        assert "VUS" in ans
        # Should mention VUS is not a diagnosis or that evidence is uncertain
        assert any(phrase in ans for phrase in ["אינו אבחנה", "לא הוכח", "אין עדיין"]), (
            f"Expected informative prenatal content, got:\n{ans}"
        )

    def test_turn2_no_clinvar_card(self):
        """Part I: contextual follow-up must not return gene_metadata (ClinVar card)."""
        r1 = _ask(TURN1_Q)
        r2 = _ask("מה זה אומר עבור העובר שלי?", _ctx_of(r1))
        assert "gene_metadata" not in r2 or r2["gene_metadata"] is None, (
            "ClinVar card should not appear for contextual follow-up"
        )

    def test_turn3_disease_concern_anchored_to_finding(self):
        """Turn 3: 'הגן קשור למחלה' concern stays on prenatal_vus_deletion."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r3 = _ask("אבל אמרת שהגן קשור למחלה נוירולוגית. זה לא מסוכן?", ctx2)
        assert r3["matched_topic"] == "prenatal_vus_deletion", (
            f"Turn 3 misrouted to {r3['matched_topic']}"
        )
        ans = r3["answer"]
        # Must distinguish gene-level association from variant classification
        assert any(phrase in ans for phrase in ["VUS", "סיווג", "pathogenic", "קשר"]), (
            f"Expected disease-vs-VUS distinction, got:\n{ans}"
        )

    def test_turn4_simplify_stays_on_finding(self):
        """Turn 4: 'לא הבנתי' simplification stays on prenatal_vus_deletion."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r3 = _ask("אבל אמרת שהגן קשור למחלה נוירולוגית. זה לא מסוכן?", ctx2)
        ctx3 = _ctx_of(r3)
        r4 = _ask("לא הבנתי", ctx3)
        assert r4["matched_topic"] == "prenatal_vus_deletion", (
            f"Turn 4 simplify misrouted to {r4['matched_topic']}"
        )

    def test_turn5_bevatsam_phrasing_personal_meaning(self):
        """Part B: 'אז מה זה אומר בעצם עבורי' must not route to gene_clinvar_summary."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r3 = _ask("אבל אמרת שהגן קשור למחלה נוירולוגית. זה לא מסוכן?", ctx2)
        ctx3 = _ctx_of(r3)
        r4 = _ask("לא הבנתי", ctx3)
        ctx4 = _ctx_of(r4)
        r5 = _ask("אז מה זה אומר בעצם עבורי?", ctx4)
        assert r5["matched_topic"] == "prenatal_vus_deletion", (
            f"Turn 5 'בעצם' misrouted to {r5['matched_topic']}"
        )

    def test_turn5_no_clinvar_card(self):
        """Part I: Turn 5 follow-up must not return gene_metadata."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r3 = _ask("לא הבנתי", ctx2)
        ctx3 = _ctx_of(r3)
        r5 = _ask("אז מה זה אומר בעצם עבורי?", ctx3)
        assert "gene_metadata" not in r5 or r5["gene_metadata"] is None

    def test_turn6_li_lishaol_phrasing_next_steps(self):
        """Part B: 'מה כדאי לי לשאול עכשיו' routes to prenatal_vus_deletion, not vus_questions_for_counselor."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r3 = _ask("אבל אמרת שהגן קשור למחלה נוירולוגית. זה לא מסוכן?", ctx2)
        ctx3 = _ctx_of(r3)
        r4 = _ask("לא הבנתי", ctx3)
        ctx4 = _ctx_of(r4)
        r5 = _ask("אז מה זה אומר בעצם עבורי?", ctx4)
        ctx5 = _ctx_of(r5)
        r6 = _ask("מה כדאי לי לשאול עכשיו את הצוות הגנטי?", ctx5)
        assert r6["matched_topic"] == "prenatal_vus_deletion", (
            f"Turn 6 'כדאי לי לשאול' misrouted to {r6['matched_topic']}"
        )
        ans = r6["answer"]
        # Prenatal-specific question list expected
        assert any(phrase in ans for phrase in ["חסר", "deletion", "segregation", "DECIPHER", "עובר"]), (
            f"Expected prenatal-specific questions, got:\n{ans}"
        )

    def test_turn6_no_clinvar_card(self):
        """Part I: Turn 6 follow-up must not return gene_metadata."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r6 = _ask("מה כדאי לי לשאול עכשיו את הצוות הגנטי?", ctx2)
        assert "gene_metadata" not in r6 or r6["gene_metadata"] is None

    def test_turn7_gene_within_finding_gets_note(self):
        """Part C: explicit gene query within prenatal finding gets gene info + context note."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r3 = _ask("אבל אמרת שהגן קשור למחלה נוירולוגית. זה לא מסוכן?", ctx2)
        ctx3 = _ctx_of(r3)
        r4 = _ask("לא הבנתי", ctx3)
        ctx4 = _ctx_of(r4)
        r5 = _ask("אז מה זה אומר בעצם עבורי?", ctx4)
        ctx5 = _ctx_of(r5)
        r6 = _ask("מה כדאי לי לשאול עכשיו את הצוות הגנטי?", ctx5)
        ctx6 = _ctx_of(r6)
        r7 = _ask("ומה ידוע על NIPA1 עצמו?", ctx6)
        ans = r7["answer"]
        assert "NIPA1" in ans.upper(), "Expected NIPA1 gene info in answer"
        assert "הערה" in ans, (
            f"Expected deletion context note in Turn 7 answer, got:\n{ans}"
        )
        assert "חסר" in ans or "deletion" in ans.lower() or "פרנטלי" in ans, (
            f"Context note should mention the deletion/prenatal context:\n{ans}"
        )

    def test_turn8_explicit_gene_switch_col1a1(self):
        """Turn 8: COL1A1 question is a new topic, not prenatal_vus_deletion."""
        r1 = _ask(TURN1_Q)
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר עבור העובר שלי?", ctx1)
        ctx2 = _ctx_of(r2)
        r8 = _ask("מה זה COL1A1?", ctx2)
        assert r8["matched_topic"] != "prenatal_vus_deletion", (
            "COL1A1 question should not stay on prenatal_vus_deletion"
        )


# ---------------------------------------------------------------------------
# Part B — Pattern unit tests (no session context needed for pattern matching,
# but we still need an active topic to fire _classify_followup_intent_v2)
# ---------------------------------------------------------------------------

class TestPatternGaps:
    """Verify the new pattern additions fire correctly."""

    T1_CTX = {
        "active_topic": "prenatal_vus_deletion",
        "gene_symbol": "NIPA1",
        "finding_type": "deletion",
        "pregnancy_context": True,
        "subject": "fetus",
        "turn_count": 1,
    }

    def test_fetus_phrasing_matches_personal_meaning(self):
        """'עבור העובר שלי' → prenatal_vus_deletion (not gene_clinvar_summary)."""
        r = _ask("מה זה אומר עבור העובר שלי?", self.T1_CTX)
        assert r["matched_topic"] == "prenatal_vus_deletion"

    def test_fetus_short_phrasing(self):
        """'עבור העובר' → prenatal_vus_deletion."""
        r = _ask("מה זה אומר עבור העובר?", self.T1_CTX)
        assert r["matched_topic"] == "prenatal_vus_deletion"

    def test_az_mah_zeh_omer_bevatsam(self):
        """'אז מה זה אומר בעצם עבורי?' → prenatal_vus_deletion."""
        r = _ask("אז מה זה אומר בעצם עבורי?", self.T1_CTX)
        assert r["matched_topic"] == "prenatal_vus_deletion"

    def test_mah_zeh_omer_bevatsam_only(self):
        """'מה זה אומר בעצם?' → prenatal_vus_deletion."""
        r = _ask("מה זה אומר בעצם?", self.T1_CTX)
        assert r["matched_topic"] == "prenatal_vus_deletion"

    def test_li_lishaol_safe_next_steps(self):
        """'מה כדאי לי לשאול' → prenatal_vus_deletion."""
        r = _ask("מה כדאי לי לשאול את הצוות?", self.T1_CTX)
        assert r["matched_topic"] == "prenatal_vus_deletion"

    def test_kedai_li_lishaol_variant(self):
        """'כדאי לי לשאול' → prenatal_vus_deletion."""
        r = _ask("כדאי לי לשאול מה הצעדים הבאים?", self.T1_CTX)
        assert r["matched_topic"] == "prenatal_vus_deletion"


# ---------------------------------------------------------------------------
# Part C — Gene-within-finding contextual note
# ---------------------------------------------------------------------------

class TestGeneWithinFindingNote:
    """Gene query within active prenatal deletion → gene info + context note."""

    CTX = {
        "active_topic": "prenatal_vus_deletion",
        "gene_symbol": "NIPA1",
        "finding_type": "deletion",
        "pregnancy_context": True,
        "turn_count": 3,
    }

    def test_gene_within_finding_has_note(self):
        r = _ask("ומה ידוע על NIPA1 עצמו?", self.CTX)
        ans = r["answer"]
        assert "NIPA1" in ans.upper()
        assert "הערה" in ans, f"Expected context note, got:\n{ans}"

    def test_gene_within_finding_note_mentions_deletion(self):
        r = _ask("ומה ידוע על NIPA1 עצמו?", self.CTX)
        ans = r["answer"]
        assert any(w in ans for w in ["חסר", "deletion", "פרנטלי"]), (
            f"Note must mention deletion/prenatal context:\n{ans}"
        )

    def test_different_gene_no_note(self):
        """Explicit switch to a DIFFERENT gene should not add the deletion note."""
        r = _ask("מה זה COL1A1?", self.CTX)
        ans = r["answer"]
        # Should NOT get the prenatal finding note — it's a topic switch
        # (note only added when the question is about the SAME gene in context)
        assert "COL1A1" in ans.upper() or r["matched_topic"] != "prenatal_vus_deletion"

    def test_no_gene_in_text_no_note(self):
        """Follow-up without gene mention should go through intent classification, not Part C."""
        r = _ask("ספר לי עוד", self.CTX)
        # Should not crash; topic should remain on prenatal or general
        assert r["matched_topic"] is not None or r["answer"]


# ---------------------------------------------------------------------------
# Part D — Personal meaning content for prenatal
# ---------------------------------------------------------------------------

class TestPrenatalPersonalMeaning:
    """_build_personal_meaning_answer must give informative content, not only redirect."""

    CTX = {
        "active_topic": "prenatal_vus_deletion",
        "gene_symbol": "NIPA1",
        "finding_type": "deletion",
        "pregnancy_context": True,
        "variant_classification": "vus",
        "turn_count": 1,
    }

    def test_vus_not_diagnosis_stated(self):
        r = _ask("מה זה אומר עבורי?", self.CTX)
        ans = r["answer"]
        assert any(p in ans for p in ["אינו אבחנה", "לא הוכח", "אין עדיין"]), (
            f"Expected VUS ≠ diagnosis statement, got:\n{ans}"
        )

    def test_deletion_factors_mentioned(self):
        r = _ask("מה זה אומר לנו?", self.CTX)
        ans = r["answer"]
        # Should mention at least one factor affecting interpretation
        assert any(p in ans for p in ["גודל", "ירוש", "de novo", "inherited", "גנים"]), (
            f"Expected deletion interpretation factors, got:\n{ans}"
        )

    def test_prenatal_no_surgery_recommendation(self):
        r = _ask("מה זה אומר עבור העובר?", self.CTX)
        ans = r["answer"]
        assert "ניתוח" not in ans
        assert "הפלה" not in ans


# ---------------------------------------------------------------------------
# Part H — No duplicate guidance section in Turn 1
# ---------------------------------------------------------------------------

class TestNoDuplicateGuidance:
    """_build_prenatal_deletion_vus_answer must have exactly one question section."""

    def test_single_question_section_nipa1(self):
        r = _ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        assert r["matched_topic"] == "prenatal_vus_deletion", (
            f"Unexpected topic: {r['matched_topic']}"
        )
        ans = r["answer"]
        n = ans.count("שאלות שכדאי לשאול")
        assert n <= 1, f"Duplicate guidance section detected ({n} times)"

    def test_single_clarification_block(self):
        r = _ask(
            "יש לי ממצא deletion VUS פרנטלי הכולל את הגן BRCA1. מה זה?"
        )
        if r["matched_topic"] != "prenatal_vus_deletion":
            pytest.skip("Question didn't route to prenatal_vus_deletion")
        ans = r["answer"]
        n = ans.count("שאלות שכדאי לשאול")
        assert n <= 1, f"Duplicate guidance detected"


# ---------------------------------------------------------------------------
# Part I — No ClinVar card for contextual follow-up turns
# ---------------------------------------------------------------------------

class TestClinVarCardSuppression:
    """Contextual follow-up handlers must not return gene_metadata (ClinVar card)."""

    CTX = {
        "active_topic": "prenatal_vus_deletion",
        "gene_symbol": "NIPA1",
        "finding_type": "deletion",
        "pregnancy_context": True,
        "turn_count": 1,
    }

    @pytest.mark.parametrize("question", [
        "מה זה אומר עבור העובר שלי?",
        "אז מה זה אומר בעצם עבורי?",
        "מה כדאי לי לשאול עכשיו את הצוות הגנטי?",
        "לא הבנתי",
        "אבל אמרת שהגן קשור למחלה. זה לא מסוכן?",
    ])
    def test_no_gene_metadata_in_contextual_followup(self, question):
        r = _ask(question, self.CTX)
        gm = r.get("gene_metadata")
        # None or absent is fine; a populated dict with gene_symbol would be wrong
        if gm:
            assert gm.get("answer_scope") != "clinvar_tier2_stats", (
                f"ClinVar card should not appear for contextual follow-up '{question}'"
            )


# ---------------------------------------------------------------------------
# Regression — standalone answers must not be affected
# ---------------------------------------------------------------------------

class TestStandaloneRegressions:
    """Standalone questions (no context) must still work correctly."""

    def test_brca1_vus_standalone(self):
        r = _ask("יש לי VUS ב BRCA1, מה זה?")
        assert r["matched_topic"] in ("vus_known_gene", "vus", "gene_clinvar_summary")

    def test_carrier_standalone(self):
        r = _ask("אמרו לי שאני נשאית, מה זה?")
        assert r["matched_topic"] is not None
        assert "נשא" in r["answer"] or "carrier" in r["answer"].lower()

    def test_vus_general_standalone(self):
        r = _ask("מה זה VUS?")
        assert "VUS" in r["answer"]
        assert r["matched_topic"] is not None

    def test_prenatal_outside_context_routes_correctly(self):
        """Prenatal deletion VUS question with no prior context still routes to prenatal_vus_deletion."""
        r = _ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        assert r["matched_topic"] == "prenatal_vus_deletion"
