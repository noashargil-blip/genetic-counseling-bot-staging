"""
Session 27.14 — Entity-aware follow-ups and empty UI artifact removal.

Covers:
  Part A  – session context carries mentioned_conditions, last_primary_condition,
             last_answer_topic; clears on explicit gene switch
  Part B  – explain_previous_answer and condition_list_inquiry intents recognized
  Part C  – "מה זה אומר?" after gene overview → concise gene interpretation
  Part D  – "מה המחלות האלה?" → explain mentioned conditions; no personal implication
  Part E  – ambiguous condition term resolves against mentioned_conditions
  Part F  – follow-up answers carry gene_clinvar_summary matched_topic
  Part G  – explicit gene switch clears mentioned_conditions
  Part H  – _strip_markdown_tables removes pipe lines; no table artifact in answers
  Part I  – ClinVar copy: no redundant referral sentence in stats interpretation
  Part J  – 6-turn COL1A1 acceptance test
  Part K  – regression preservation (BRCA1, VUS, carrier, safety, schema)
"""

import re
import pytest
import app.counseling_engine as _ce
from app.counseling_engine import answer_question, _classify_followup_intent_v2
from app.counseling_engine import (
    _strip_markdown_tables,
    _extract_conditions_for_context,
    _find_matching_condition,
    _build_explain_previous_answer,
    _build_condition_list_answer,
    _build_condition_terminology_answer,
    _build_clinvar_patient_interpretation,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ask(question: str, ctx: dict | None = None) -> dict:
    return answer_question(question=question, session_context=ctx)


def _ctx_of(result: dict) -> dict:
    return result.get("session_context_out", {})


def _gene_ctx(gene: str, topic: str = "gene_clinvar_summary",
              conditions: list | None = None) -> dict:
    base = {"active_topic": topic, "gene_symbol": gene}
    if conditions is not None:
        base["mentioned_conditions"] = conditions
        if conditions:
            base["last_primary_condition"] = conditions[0]
    return base


# ── Part H: _strip_markdown_tables utility ───────────────────────────────────

class TestStripMarkdownTables:
    def test_no_pipe_returns_unchanged(self):
        text = "גן COL1A1 קשור למספר מצבים רפואיים."
        assert _strip_markdown_tables(text) == text

    def test_pure_table_removed(self):
        text = "| Col1 | Col2 |\n| --- | --- |\n| א | ב |"
        result = _strip_markdown_tables(text)
        assert "|" not in result

    def test_table_lines_removed_content_kept(self):
        text = "פסקה ראשונה.\n| א | ב |\n| ג | ד |\nפסקה שנייה."
        result = _strip_markdown_tables(text)
        assert "|" not in result
        assert "פסקה ראשונה" in result
        assert "פסקה שנייה" in result

    def test_empty_string_unchanged(self):
        assert _strip_markdown_tables("") == ""

    def test_none_unchanged(self):
        assert _strip_markdown_tables(None) is None

    def test_single_pipe_mid_sentence_kept(self):
        # Single | inside sentence (not at line start) must NOT be stripped
        text = "VUS (variant of uncertain significance) | ספציפי"
        result = _strip_markdown_tables(text)
        # Line doesn't start with |, so it stays
        assert "VUS" in result

    def test_leading_whitespace_pipe_stripped(self):
        text = "שורה רגילה\n   | Col | Val |\nשורה נוספת"
        result = _strip_markdown_tables(text)
        assert "|" not in result
        assert "שורה רגילה" in result
        assert "שורה נוספת" in result

    def test_no_empty_table_in_real_gene_answer(self):
        result = _ask("מה זה COL1A1?")
        answer = result.get("answer", "")
        lines = answer.split("\n")
        table_lines = [l for l in lines if re.match(r"^\s*\|", l)]
        assert len(table_lines) == 0, f"answer contains table lines: {table_lines}"


# ── Part A: session context new fields ───────────────────────────────────────

class TestSessionContextNewFields:
    def test_extract_conditions_from_result_with_metadata(self):
        result = {
            "gene_metadata": {
                "top_phenotypes": ["Osteogenesis imperfecta", "Ehlers-Danlos syndrome"]
            }
        }
        conds = _extract_conditions_for_context(result)
        assert "Osteogenesis imperfecta" in conds
        assert len(conds) <= 5

    def test_extract_conditions_empty_when_no_metadata(self):
        result = {"answer": "some answer"}
        assert _extract_conditions_for_context(result) == []

    def test_extract_conditions_filters_short_names(self):
        result = {
            "gene_metadata": {"top_phenotypes": ["AB", "Breast cancer"]}
        }
        conds = _extract_conditions_for_context(result)
        assert "AB" not in conds
        assert "Breast cancer" in conds

    def test_extract_conditions_max_five(self):
        result = {
            "gene_metadata": {
                "top_phenotypes": [f"Disease {i}" for i in range(10)]
            }
        }
        conds = _extract_conditions_for_context(result)
        assert len(conds) == 5

    def test_gene_answer_sets_mentioned_conditions(self):
        result = _ask("מה זה COL1A1?")
        ctx = _ctx_of(result)
        meta = result.get("gene_metadata") or {}
        phenos = meta.get("top_phenotypes") or []
        if phenos:
            assert "mentioned_conditions" in ctx
            assert len(ctx["mentioned_conditions"]) > 0

    def test_gene_answer_sets_last_answer_topic(self):
        result = _ask("מה זה HBB?")
        ctx = _ctx_of(result)
        assert "last_answer_topic" in ctx

    def test_conditions_carried_forward_on_followup(self):
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        if not ctx1.get("mentioned_conditions"):
            pytest.skip("COL1A1 has no phenotypes available in this environment")
        r2 = _ask("ומה זה אומר?", ctx=ctx1)
        ctx2 = _ctx_of(r2)
        # conditions must be carried forward
        assert ctx2.get("mentioned_conditions") == ctx1.get("mentioned_conditions")

    def test_explicit_gene_switch_clears_conditions(self):
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        if not ctx1.get("mentioned_conditions"):
            pytest.skip("COL1A1 has no phenotypes in this environment")
        r2 = _ask("מה זה HBB?", ctx=ctx1)
        ctx2 = _ctx_of(r2)
        col1a1_conds = ctx1.get("mentioned_conditions", [])
        hbb_conds = ctx2.get("mentioned_conditions", [])
        # HBB conditions should be different from COL1A1 conditions
        # (may be empty if HBB has no phenotypes, but should not be COL1A1's)
        assert hbb_conds != col1a1_conds or not hbb_conds


# ── Part B: new follow-up intent recognition ─────────────────────────────────

class TestNewFollowupIntentRecognition:
    def _ctx(self, topic: str = "gene_clinvar_summary") -> dict:
        return {"active_topic": topic, "gene_symbol": "COL1A1"}

    def test_ma_ze_omer_without_personal_marker_is_explain(self):
        ctx = self._ctx()
        intent = _classify_followup_intent_v2("מה זה אומר?", ctx)
        assert intent == "explain_previous_answer"

    def test_ma_ze_omer_with_li_is_personal_meaning(self):
        ctx = self._ctx()
        intent = _classify_followup_intent_v2("מה זה אומר לי?", ctx)
        assert intent == "personal_meaning"

    def test_ma_ze_omer_leobr_is_personal_meaning(self):
        ctx = self._ctx()
        intent = _classify_followup_intent_v2("מה זה אומר לעובר?", ctx)
        assert intent == "personal_meaning"

    def test_uma_ze_is_explain(self):
        ctx = self._ctx()
        intent = _classify_followup_intent_v2("ומה זה?", ctx)
        assert intent == "explain_previous_answer"

    def test_condition_list_inquiry_recognized(self):
        ctx = self._ctx()
        intent = _classify_followup_intent_v2("מה המחלות האלה?", ctx)
        assert intent == "condition_list_inquiry"

    def test_ma_hamatzavim_recognized(self):
        ctx = self._ctx()
        intent = _classify_followup_intent_v2("מה המצבים האלה?", ctx)
        assert intent == "condition_list_inquiry"

    def test_explain_patterns_recognized(self):
        ctx = self._ctx()
        for phrase in ["מה הכוונה בזה", "תסביר את מה שאמרת"]:
            intent = _classify_followup_intent_v2(phrase, ctx)
            assert intent == "explain_previous_answer", f"phrase '{phrase}' not recognized"

    def test_no_context_returns_none(self):
        intent = _classify_followup_intent_v2("מה זה אומר?", None)
        assert intent is None


# ── Part C: explain_previous_answer handler ───────────────────────────────────

class TestExplainPreviousAnswer:
    def test_returns_none_without_gene(self):
        ctx = {"active_topic": "gene_clinvar_summary"}
        result = _build_explain_previous_answer(ctx)
        assert result is None

    def test_returns_dict_with_gene(self):
        ctx = _gene_ctx("COL1A1")
        result = _build_explain_previous_answer(ctx)
        assert result is not None
        assert "COL1A1" in result["answer"]
        assert result["safety_level"] == "general_information"
        assert result["matched_topic"] == "gene_clinvar_summary"

    def test_answer_contains_gene_name(self):
        ctx = _gene_ctx("BRCA1")
        result = _build_explain_previous_answer(ctx)
        assert "BRCA1" in result["answer"]

    def test_includes_conditions_in_answer(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta"])
        result = _build_explain_previous_answer(ctx)
        assert "Osteogenesis imperfecta" in result["answer"]

    def test_two_conditions_in_answer(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta", "Ehlers-Danlos syndrome"])
        result = _build_explain_previous_answer(ctx)
        assert "Osteogenesis imperfecta" in result["answer"]
        assert "Ehlers-Danlos" in result["answer"]

    def test_no_diagnosis_claim(self):
        ctx = _gene_ctx("BRCA2", conditions=["Breast cancer"])
        result = _build_explain_previous_answer(ctx)
        answer = result["answer"]
        assert "אבחנה" not in answer or "אינו אבחנה" in answer or "לא אבחנה" in answer

    def test_suggested_questions_present(self):
        ctx = _gene_ctx("HBB")
        result = _build_explain_previous_answer(ctx)
        assert len(result["suggested_questions"]) >= 2

    def test_full_turn_ma_ze_omer_after_gene(self):
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        if not ctx1.get("gene_symbol"):
            pytest.skip("COL1A1 not recognized in this environment")
        r2 = _ask("מה זה אומר?", ctx=ctx1)
        assert r2["answer"]
        assert "COL1A1" in r2["answer"].upper() or r2["matched_topic"] == "gene_clinvar_summary"


# ── Part D: condition_list_answer handler ─────────────────────────────────────

class TestConditionListAnswer:
    def test_returns_none_without_conditions(self):
        ctx = {"active_topic": "gene_clinvar_summary", "gene_symbol": "COL1A1"}
        result = _build_condition_list_answer(ctx)
        assert result is None

    def test_returns_dict_with_conditions(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta"])
        result = _build_condition_list_answer(ctx)
        assert result is not None
        assert "Osteogenesis imperfecta" in result["answer"]

    def test_no_personal_diagnosis_claim(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta", "Ehlers-Danlos syndrome"])
        result = _build_condition_list_answer(ctx)
        assert "אבחנה עבורך" in result["answer"] or "לא אבחנה" in result["answer"] or "אינם אבחנה" in result["answer"]

    def test_gene_reference_in_answer(self):
        ctx = _gene_ctx("HBB", conditions=["Sickle cell disease"])
        result = _build_condition_list_answer(ctx)
        assert "HBB" in result["answer"] or "גן" in result["answer"]

    def test_multiple_conditions_in_answer(self):
        ctx = _gene_ctx("COL1A1", conditions=[
            "Osteogenesis imperfecta",
            "Ehlers-Danlos syndrome",
            "Stickler syndrome",
        ])
        result = _build_condition_list_answer(ctx)
        assert "Osteogenesis" in result["answer"]

    def test_full_turn_condition_list_inquiry(self):
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        if not ctx1.get("mentioned_conditions"):
            pytest.skip("no conditions in context")
        r2 = _ask("מה המחלות האלה?", ctx=ctx1)
        assert r2["answer"]
        assert r2["safety_level"] == "general_information"


# ── Part E: condition terminology resolution ──────────────────────────────────

class TestConditionTerminologyResolution:
    def test_find_matching_condition_latin_substring(self):
        conds = ["Osteogenesis imperfecta", "Breast cancer"]
        match = _find_matching_condition("מה זה osteogenesis?", conds)
        assert match == "Osteogenesis imperfecta"

    def test_find_matching_condition_case_insensitive(self):
        conds = ["Ehlers-Danlos syndrome", "Lynch syndrome"]
        match = _find_matching_condition("EHLERS מה זה", conds)
        assert match == "Ehlers-Danlos syndrome"

    def test_find_matching_condition_returns_none_no_match(self):
        conds = ["Breast cancer", "Ovarian cancer"]
        match = _find_matching_condition("מה זה VUS?", conds)
        assert match is None

    def test_find_matching_condition_empty_conditions(self):
        assert _find_matching_condition("test", []) is None

    def test_condition_terminology_answer_structure(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta"])
        result = _build_condition_terminology_answer("Osteogenesis imperfecta", ctx)
        assert "Osteogenesis imperfecta" in result["answer"]
        assert result["safety_level"] == "general_information"
        assert not result["needs_genetic_counselor"]

    def test_condition_terminology_no_diagnosis(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta"])
        result = _build_condition_terminology_answer("Osteogenesis imperfecta", ctx)
        # Must not imply the user has the condition
        assert "יש לך" not in result["answer"]
        assert "אתה חולה" not in result["answer"]

    def test_condition_terminology_full_turn(self):
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        conds = ctx1.get("mentioned_conditions", [])
        if not conds:
            pytest.skip("no conditions available")
        first_cond_word = conds[0].split()[0]  # first word of first condition
        r2 = _ask(f"מה זה {first_cond_word}?", ctx=ctx1)
        assert r2["answer"]


# ── Part I: ClinVar copy cleanup ─────────────────────────────────────────────

class TestClinvarCopyCleanup:
    def test_no_redundant_referral_in_stats_interpretation(self):
        text = _build_clinvar_patient_interpretation("COL1A1", {
            "total_variants": 100,
            "by_significance": {"Pathogenic": 10, "VUS": 30},
        })
        assert "כדי להבין את הממצא האישי שלך יש לפנות לצוות הגנטי" not in text

    def test_no_summary_interpretation_has_focused_text(self):
        text = _build_clinvar_patient_interpretation("COL1A1", None)
        assert "ההתפלגות במאגר אינה קובעת" in text

    def test_interpretation_still_mentions_classification_boundary(self):
        text = _build_clinvar_patient_interpretation("BRCA1", {
            "total_variants": 500,
            "by_significance": {"Pathogenic": 200, "VUS": 100},
        })
        assert "ממצא" in text or "וריאנט" in text

    def test_clinvar_meta_answer_no_redundant_referral(self):
        result = _ask("מה המשמעות של הנתונים?")
        assert "כדי להבין את הממצא האישי שלך יש לפנות לצוות הגנטי" not in result["answer"]


# ── Part J: 6-turn COL1A1 acceptance test ────────────────────────────────────

class TestCOL1A1SixTurnConversation:
    def test_six_turn_col1a1_flow(self):
        # Turn 1: Gene overview
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        assert r1["answer"], "Turn 1: no answer"
        assert ctx1.get("gene_symbol") == "COL1A1" or ctx1.get("active_topic"), \
            f"Turn 1: unexpected context {ctx1}"

        # Turn 2: "מה זה אומר?" — explain previous
        r2 = _ask("מה זה אומר?", ctx=ctx1)
        ctx2 = _ctx_of(r2)
        assert r2["answer"], "Turn 2: no answer"
        assert r2["safety_level"] == "general_information", \
            f"Turn 2: wrong safety level {r2['safety_level']}"

        # Turn 3: Conditions list
        r3 = _ask("מה המחלות האלה?", ctx=ctx2)
        ctx3 = _ctx_of(r3)
        assert r3["answer"], "Turn 3: no answer"
        assert "אבחנה" not in r3["answer"] or "אינם אבחנה" in r3["answer"] or \
               "לא אבחנה" in r3["answer"] or "אינה קובע" in r3["answer"], \
            "Turn 3: answer should not present conditions as personal diagnosis"

        # Turn 4: Personal meaning — should redirect (not diagnose)
        r4 = _ask("מה זה אומר לי?", ctx=ctx3)
        ctx4 = _ctx_of(r4)
        assert r4["answer"], "Turn 4: no answer"
        assert r4["safety_level"] in ("general_information", "requires_genetic_counselor"), \
            f"Turn 4: unexpected safety level {r4['safety_level']}"

        # Turn 5: Safe next steps
        r5 = _ask("מה כדאי לשאול את הצוות?", ctx=ctx4)
        ctx5 = _ctx_of(r5)
        assert r5["answer"], "Turn 5: no answer"

        # Turn 6: Must not contain surgery or treatment recommendation
        r6 = _ask("מה עלי לעשות?", ctx=ctx5)
        assert r6["answer"], "Turn 6: no answer"
        assert "ניתוח" not in r6["answer"], "Turn 6: answer mentions surgery"
        assert "טיפול" not in r6["answer"], "Turn 6: answer mentions treatment"

    def test_col1a1_answer_no_pipe_table_artifacts(self):
        result = _ask("מה זה COL1A1?")
        answer = result.get("answer", "")
        lines = answer.split("\n")
        table_lines = [l for l in lines if re.match(r"^\s*\|", l)]
        assert len(table_lines) == 0, f"answer has table lines: {table_lines[:3]}"

    def test_hbb_six_turn_no_diagnosis(self):
        r1 = _ask("מה זה HBB?")
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר?", ctx=ctx1)
        assert r2["answer"]
        # No diagnosis for the user
        assert "יש לך" not in r2["answer"]
        assert "את חולה" not in r2["answer"]


# ── Part K: regression preservation ─────────────────────────────────────────

class TestRegressions27_14:
    def test_brca1_standalone_unchanged(self):
        result = _ask("מה זה BRCA1?")
        assert result["answer"]
        assert result["matched_topic"]

    def test_vus_standalone_unchanged(self):
        result = _ask("מה זה VUS?")
        assert result["answer"]
        assert "vus" in result["answer"].lower() or "VUS" in result["answer"]

    def test_carrier_standalone_unchanged(self):
        result = _ask("אמרו לי שאני נשאית, מה זה?")
        assert result["answer"]
        assert result["safety_level"] in ("general_information", "requires_genetic_counselor")

    def test_safety_block_unchanged(self):
        result = _ask("קוראים לי שרה כהן")
        assert result["safety_level"] == "contains_identifying_info"

    def test_schema_five_keys(self):
        result = _ask("מה זה VUS?")
        required = {"answer", "safety_level", "needs_genetic_counselor", "matched_topic",
                    "suggested_questions"}
        assert required.issubset(set(result.keys()))

    def test_prenatal_nipa1_still_routes(self):
        result = _ask(
            "קיבלתי תוצאה של VUS בהריון. "
            "היועצת אמרה לי שיש לעובר שינוי של חסר הכולל את הגן NIPA1"
        )
        assert result["matched_topic"] == "prenatal_vus_deletion", \
            f"expected prenatal_vus_deletion, got {result['matched_topic']}"

    def test_vus_brca1_followup_still_routes(self):
        r1 = _ask("יש לי VUS ב BRCA1, מה זה?")
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה כדאי לעשות עם זה?", ctx=ctx1)
        assert r2["answer"]
        # Answer must NOT be a carrier explanation — content check is more robust
        answer_lower = r2["answer"].lower()
        assert "נשא" not in answer_lower or "vus" in answer_lower or \
               "שאלות" in r2["answer"] or "לצוות" in r2["answer"], \
            f"answer looks like carrier content: {r2['answer'][:120]}"

    def test_personal_meaning_still_redirects(self):
        r1 = _ask("מה זה COL1A1?")
        ctx1 = _ctx_of(r1)
        r2 = _ask("מה זה אומר לי?", ctx=ctx1)
        # personal meaning → should mention counselor or have requires_gc
        answer_lower = r2["answer"].lower()
        assert (
            r2["needs_genetic_counselor"]
            or "צוות" in r2["answer"]
            or "יועץ" in r2["answer"]
        ), "personal meaning should mention the genetics team"

    def test_condition_list_intent_no_personal_implication(self):
        ctx = _gene_ctx("HBB", conditions=["Sickle cell disease", "Beta-thalassemia"])
        result = _build_condition_list_answer(ctx)
        assert result is not None
        answer = result["answer"]
        assert "יש לך" not in answer
        assert "חולה" not in answer

    def test_explain_previous_no_tables_in_answer(self):
        ctx = _gene_ctx("COL1A1", conditions=["Osteogenesis imperfecta"])
        result = _build_explain_previous_answer(ctx)
        answer = result["answer"]
        table_lines = [l for l in answer.split("\n") if re.match(r"^\s*\|", l)]
        assert len(table_lines) == 0
