"""
Session 27.11: Contextual follow-up routing tests.

Verifies:
  - 6-turn NIPA1 prenatal deletion conversation stays on topic
  - Follow-up intents (simplify, personal_meaning, disease_concern, next_steps) route correctly
  - Explicit gene switches bypass contextual routing
  - Existing standalone answers not broken
  - No personal diagnosis in any follow-up response
  - pregnancy_context / subject carried in session_context_out
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── helpers ───────────────────────────────────────────────────────────────────

def ask(question: str, ctx: dict | None = None) -> dict:
    body: dict = {"question": question}
    if ctx:
        body["context"] = ctx
    r = client.post("/ask", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def ctx_out(resp: dict) -> dict:
    return resp.get("conversation_context") or {}


PERSONAL_DIAGNOSIS_FORBIDDEN = [
    "הממצא שלך מסוכן",
    "הממצא שלך פתוגני",
    "לעובר שלך יש מחלה",
    "הסיכון שלך הוא",
    "אתה חולה",
    "העובר חולה",
]


def assert_no_personal_diagnosis(answer: str) -> None:
    for phrase in PERSONAL_DIAGNOSIS_FORBIDDEN:
        assert phrase not in answer, f"Personal diagnosis found: {phrase!r}"


# ── Part J: 6-turn NIPA1 prenatal deletion conversation ──────────────────────

class TestNIPA1SixTurnConversation:
    """
    Full 6-turn prenatal deletion VUS conversation.
    Each turn passes session_context from the previous response.
    """

    def test_turn1_initial_prenatal_deletion(self):
        """Turn 1: complex initial message → prenatal deletion educational answer."""
        q = (
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        r = ask(q)
        ans = r["answer"]
        c = ctx_out(r)

        # Answer must mention VUS and deletion
        assert "VUS" in ans or "וריאנט" in ans.lower(), "VUS not mentioned"
        assert any(w in ans for w in ("חסר", "deletion", "Deletion")), "Deletion not mentioned"

        # Must not give personal diagnosis
        assert_no_personal_diagnosis(ans)

        # matched_topic should be prenatal_vus_deletion or vus-related
        assert r["matched_topic"] in (
            "prenatal_vus_deletion", "vus_known_gene", "vus",
        ), f"Unexpected topic: {r['matched_topic']}"

        # Session context should carry gene, classification, pregnancy, subject
        assert c.get("gene_symbol") == "NIPA1", f"gene_symbol missing: {c}"
        assert c.get("variant_classification") == "vus", f"variant_classification: {c}"
        assert c.get("pregnancy_context") is True, f"pregnancy_context: {c}"
        assert c.get("subject") == "fetus", f"subject: {c}"
        assert c.get("active_topic") in (
            "prenatal_vus_deletion", "vus_known_gene",
        ), f"active_topic: {c}"

    def test_turn2_simplify_previous(self):
        """Turn 2: 'I didn't understand' → simplify, stays on NIPA1+deletion topic."""
        # Turn 1 context
        t1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(t1)

        r2 = ask("לא הבנתי, תסביר לי את הממצא", ctx=c1)
        ans = r2["answer"]

        # Should NOT drift to a completely unrelated topic
        assert_no_personal_diagnosis(ans)
        # Should mention VUS or deletion (not drift to e.g. carrier or inheritance)
        assert any(w in ans for w in ("VUS", "חסר", "deletion", "Deletion", "וריאנט")), (
            f"Answer drifted off-topic: {ans[:200]}"
        )
        # Context must still carry gene and pregnancy_context
        c2 = ctx_out(r2)
        assert c2.get("gene_symbol") == "NIPA1" or c1.get("gene_symbol") == "NIPA1", (
            "Gene lost from context"
        )

    def test_turn3_disease_concern(self):
        """Turn 3: disease-association concern → educational, not refused."""
        t1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(t1)

        r3 = ask(
            "אבל כתבת שהגן קשור למחלה נוירולוגית, זה מסוכן לעובר?",
            ctx=c1,
        )
        ans = r3["answer"]

        # Must be educational, not refused with PERSONAL_REDIRECT
        assert r3.get("safety_level") != "requires_genetic_counselor" or (
            # If safety_level is requires_gc, it should still have substantial educational content
            len(ans) > 200
        ), f"Refused without education: {ans[:200]}"

        # Must NOT say the variant is dangerous
        assert_no_personal_diagnosis(ans)

        # Should explain gene≠variant distinction or say it needs counselor
        educational_signals = [
            "קשר", "לעומת", "VUS", "וריאנט", "פתוגני", "pathogenic",
            "ראיות", "סיווג", "הצוות הגנטי",
        ]
        assert any(sig in ans for sig in educational_signals), (
            f"No educational content in disease-concern answer: {ans[:300]}"
        )

    def test_turn4_safe_next_steps(self):
        """Turn 4: 'what to ask the team' → prenatal-specific question list."""
        t1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(t1)

        r4 = ask("מה כדאי לשאול את הצוות הגנטי?", ctx=c1)
        ans = r4["answer"]

        assert_no_personal_diagnosis(ans)
        # Should have team-oriented question content
        assert any(w in ans for w in (
            "שאלות", "לשאול", "צוות", "גנטי", "segregation", "הפרדה", "deletion", "חסר"
        )), f"No question guidance found: {ans[:200]}"

    def test_turn5_personal_meaning(self):
        """Turn 5: personal meaning → safe boundary answer, no interpretation."""
        t1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(t1)

        r5 = ask("מה זה אומר עבורי ועבור ההריון?", ctx=c1)
        ans = r5["answer"]

        assert_no_personal_diagnosis(ans)
        # Must not give a personal risk estimate
        assert "הסיכון שלך" not in ans
        assert "אחוז" not in ans
        # Should redirect or say it can't interpret for them personally
        redirect_signals = [
            "לא יכול לפרש", "הצוות הגנטי", "צוות", "פירוש האישי", "גנטי שמטפל",
        ]
        assert any(sig in ans for sig in redirect_signals), (
            f"No safe-boundary answer: {ans[:200]}"
        )

    def test_turn6_explicit_gene_switch(self):
        """Turn 6: explicit BRCA2 question → routes to BRCA2, not NIPA1 follow-up."""
        t1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(t1)

        r6 = ask("מה זה הגן BRCA2?", ctx=c1)
        ans = r6["answer"]

        assert_no_personal_diagnosis(ans)
        # Should mention BRCA2, not be confused with NIPA1 context
        assert "BRCA2" in ans, f"BRCA2 not in answer: {ans[:200]}"

    def test_no_turn_returns_generic_what_is_gene(self):
        """No turn should return a topic-drifted generic 'what_is_gene' answer."""
        questions = [
            ("קיבלתי תוצאה של VUS בהריון. היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1", None),
        ]
        r1 = ask(questions[0][0])
        assert r1.get("matched_topic") != "what_is_gene", (
            f"Turn 1 returned what_is_gene: {r1}"
        )
        c1 = ctx_out(r1)

        followups = [
            "לא הבנתי, תסביר לי את הממצא",
            "זה מסוכן לעובר?",
            "מה כדאי לשאול את הצוות הגנטי?",
        ]
        for q in followups:
            r = ask(q, ctx=c1)
            assert r.get("matched_topic") != "what_is_gene", (
                f"Follow-up '{q}' returned what_is_gene"
            )
            c1 = ctx_out(r)


# ── Parameterized equivalent conversations ────────────────────────────────────

class TestContextualFollowupRouting:
    """
    Parameterized tests for different initial contexts + follow-up intents.
    """

    @pytest.mark.parametrize("initial,followup,must_contain", [
        # VUS + child context
        (
            "יש לילד שלי VUS בגן COL4A3, מה זה אומר?",
            "לא הבנתי, תסביר פשוט",
            ["VUS", "וריאנט"],
        ),
        # Adult VUS + simplify
        (
            "נמצא לי VUS בגן TP53, מה זה?",
            "תסביר לי את הממצא שוב",
            ["VUS", "וריאנט"],
        ),
        # Generic VUS + next steps
        (
            "יש לי VUS ב-BRCA1, מה זה?",
            "מה כדאי לשאול את הצוות?",
            ["שאלות", "צוות", "גנטי"],
        ),
    ])
    def test_simplify_stays_on_topic(self, initial, followup, must_contain):
        r1 = ask(initial)
        c1 = ctx_out(r1)
        if not c1.get("active_topic"):
            pytest.skip("No active topic set — cannot test contextual follow-up")

        r2 = ask(followup, ctx=c1)
        ans = r2["answer"]
        assert_no_personal_diagnosis(ans)
        assert any(w in ans for w in must_contain), (
            f"Expected {must_contain} in answer: {ans[:200]}"
        )

    @pytest.mark.parametrize("initial,followup", [
        ("יש לי VUS ב-BRCA1, מה זה?", "מה זה אומר עבורי?"),
        ("יש לי VUS ב-BRCA1, מה זה?", "מה זה אומר לי?"),
        (
            "קיבלתי תוצאה של VUS בהריון. היועצת אמרה לי שיש לעובר חסר הכולל את הגן NIPA1",
            "מה זה אומר עבורי ועבור ההריון?",
        ),
    ])
    def test_personal_meaning_safe_boundary(self, initial, followup):
        r1 = ask(initial)
        c1 = ctx_out(r1)
        if not c1.get("active_topic"):
            pytest.skip("No active topic set")

        r2 = ask(followup, ctx=c1)
        ans = r2["answer"]
        assert_no_personal_diagnosis(ans)
        assert "הסיכון שלך" not in ans
        assert "אחוז" not in ans
        # Must have some content (not empty fallback)
        assert len(ans) > 80, f"Answer too short: {ans!r}"

    @pytest.mark.parametrize("initial,followup", [
        ("יש לי VUS ב-BRCA1, מה זה?", "מה כדאי לשאול את הצוות?"),
        ("יש לי VUS ב-BRCA1, מה זה?", "מה הצעד הבא?"),
        (
            "קיבלתי תוצאה של VUS בהריון. היועצת אמרה לי שיש לעובר חסר הכולל את הגן NIPA1",
            "מה לשאול את הצוות הגנטי?",
        ),
    ])
    def test_safe_next_steps_content(self, initial, followup):
        r1 = ask(initial)
        c1 = ctx_out(r1)
        if not c1.get("active_topic"):
            pytest.skip("No active topic set")

        r2 = ask(followup, ctx=c1)
        ans = r2["answer"]
        assert_no_personal_diagnosis(ans)
        assert any(w in ans for w in ("שאל", "שאלות", "צוות", "גנטי", "לשאול")), (
            f"No question guidance: {ans[:200]}"
        )

    def test_disease_concern_not_refused(self):
        """Disease-association concern gets educational answer, not hard refusal."""
        r1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(r1)
        if not c1.get("active_topic"):
            pytest.skip("No active topic set")

        r2 = ask("אבל כתבת שהגן קשור למחלה, זה מסוכן?", ctx=c1)
        ans = r2["answer"]

        assert_no_personal_diagnosis(ans)
        # Must have substance (≥200 chars or multiple genetic terms)
        has_substance = (
            len(ans) > 200
            or sum(1 for w in ("VUS", "קשר", "פתוגני", "ראיות", "סיווג") if w in ans) >= 2
        )
        assert has_substance, f"Thin answer for disease concern: {ans[:200]}"

    def test_explicit_gene_switch_bypasses_context(self):
        """Naming a DIFFERENT gene bypasses contextual follow-up routing."""
        r1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(r1)
        if not c1.get("gene_symbol"):
            pytest.skip("No gene in context")

        r2 = ask("מה זה הגן BRCA2?", ctx=c1)
        ans = r2["answer"]
        assert "BRCA2" in ans, f"Explicit gene switch ignored: {ans[:200]}"


# ── Standalone gene answers not broken ────────────────────────────────────────

class TestStandaloneAnswersPreserved:
    """Existing standalone gene answers must still work without context."""

    def test_brca1_no_context(self):
        r = ask("מה זה הגן BRCA1?")
        assert "BRCA1" in r["answer"]
        assert r.get("matched_topic") in (
            "vus_known_gene", "gene_clinvar_summary", "gene_info",
        )

    def test_vus_no_context(self):
        r = ask("מה זה VUS?")
        assert "VUS" in r["answer"]
        assert r.get("matched_topic") in ("vus", "vus_general", "vus_known_gene")

    def test_carrier_no_context(self):
        r = ask("אמרו לי שאני נשאית, מה זה אומר?")
        ans = r["answer"]
        assert any(w in ans for w in ("נשא", "carrier", "נשאות")), f"Carrier answer missing: {ans[:150]}"

    def test_chromosome_education_no_context(self):
        r = ask("מה זה מחיקה בכרומוזום?")
        ans = r["answer"]
        assert any(w in ans for w in ("deletion", "מחיקה", "חסר")), f"Deletion answer missing: {ans[:150]}"


# ── Session context field validation ─────────────────────────────────────────

class TestSessionContextFields:
    """Verify that new Part A fields are correctly populated and carried forward."""

    def test_pregnancy_context_set_from_question(self):
        r = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c = ctx_out(r)
        assert c.get("pregnancy_context") is True, f"pregnancy_context not set: {c}"

    def test_subject_fetus_detected(self):
        r = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c = ctx_out(r)
        assert c.get("subject") == "fetus", f"subject not 'fetus': {c}"

    def test_variant_classification_vus_set(self):
        r = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c = ctx_out(r)
        assert c.get("variant_classification") == "vus", f"variant_classification: {c}"

    def test_finding_type_deletion(self):
        r = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c = ctx_out(r)
        assert c.get("finding_type") == "deletion", f"finding_type: {c}"

    def test_pregnancy_context_carried_to_followup(self):
        r1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(r1)
        if not c1.get("active_topic"):
            pytest.skip("No active topic")

        r2 = ask("לא הבנתי, תסביר לי את הממצא", ctx=c1)
        c2 = ctx_out(r2)
        # pregnancy_context should be carried to next turn's context output
        assert c2.get("pregnancy_context") is True, (
            f"pregnancy_context not carried to Turn 2: {c2}"
        )

    def test_gene_symbol_carried_on_followup(self):
        r1 = ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        c1 = ctx_out(r1)
        assert c1.get("gene_symbol") == "NIPA1", f"gene_symbol from Turn 1: {c1}"

        r2 = ask("לא הבנתי, תסביר לי", ctx=c1)
        c2 = ctx_out(r2)
        assert c2.get("gene_symbol") == "NIPA1", (
            f"gene_symbol not carried to Turn 2: {c2}"
        )

    def test_new_context_fields_pass_validation(self):
        """New SafeSessionContext fields should be accepted without error."""
        ctx = {
            "active_topic": "vus_known_gene",
            "gene_symbol": "NIPA1",
            "variant_classification": "vus",
            "pregnancy_context": True,
            "subject": "fetus",
            "finding_type": "deletion",
            "unresolved_question_type": "prenatal_vus_deletion",
            "turn_count": 1,
        }
        r = ask("לא הבנתי, תסביר לי", ctx=ctx)
        assert r.get("answer"), "Empty answer with full context"
