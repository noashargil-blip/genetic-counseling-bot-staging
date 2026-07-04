# -*- coding: utf-8 -*-
"""
Tests for the Hebrew post-genetic-counseling assistant:
  - app/kb.py            (knowledge-base loading + matching)
  - app/safety.py         (identifying-info / personal-interpretation classifiers)
  - app/counseling_engine.py (end-to-end answer pipeline)
  - GET /topics, GET /faq, POST /ask (FastAPI endpoints)

No network calls are made; LOCAL_LLM_URL is unset unless a test explicitly
sets it, so /ask exercises the deterministic knowledge-base fallback path.
"""
import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import counseling_engine, safety, kb
import app.retriever as retriever

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_local_llm(monkeypatch):
    """Ensure tests run against the deterministic KB fallback by default."""
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)


# ---------------------------------------------------------------------------
# 1. Basic VUS question
# ---------------------------------------------------------------------------

class TestVusQuestion:
    def test_endpoint_returns_general_information(self):
        resp = client.post("/ask", json={"question": "קיבלתי הסבר שיש לי VUS, מה זה אומר?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "matched_topic" in data
        assert "needs_genetic_counselor" in data
        assert "answer" in data
        assert "suggested_questions" in data

    def test_engine_matches_vus_by_keyword(self):
        entry = kb.match_question("מה זה וריאנט בעל משמעות לא ידועה?")
        assert entry is not None
        assert entry["id"] == "vus"


# ---------------------------------------------------------------------------
# 2. Identifying-info blocking
# ---------------------------------------------------------------------------

class TestIdentifyingInfoBlocked:
    def test_israeli_id_number_blocked(self):
        resp = client.post("/ask", json={"question": "תעודת הזהות שלי היא 123456789, מה זה VUS?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "contains_identifying_info"
        assert "פרט מזהה" in data["answer"]
        assert data["matched_topic"] is None

    def test_israeli_id_number_returns_only_new_counseling_schema(self):
        """
        POST /ask must use the new counseling schema exclusively — no legacy
        ClinVar fields (evidence, limitations, safety_disclaimer, llm_used,
        policy_flags, redirect_message) may leak through, and the content
        of the question must not be answered.
        """
        resp = client.post("/ask", json={"question": "תעודת הזהות שלי היא 123456789, מה זה VUS?"})
        data = resp.json()
        legacy_fields = {
            "evidence", "limitations", "safety_disclaimer",
            "policy_flags", "redirect_message",
        }
        for legacy_field in legacy_fields:
            assert legacy_field not in data, (
                f"legacy ClinVar field '{legacy_field}' leaked into /ask response"
            )
        assert data["safety_level"] == "contains_identifying_info"
        assert data["matched_topic"] is None
        assert data["needs_genetic_counselor"] is False
        assert "Found" not in data.get("answer", "")

    def test_email_blocked(self):
        resp = client.post("/ask", json={"question": "האימייל שלי הוא test@example.com, מה זה carrier?"})
        assert resp.json()["safety_level"] == "contains_identifying_info"

    def test_phone_number_blocked(self):
        resp = client.post("/ask", json={"question": "הטלפון שלי הוא 050-1234567, מה זה VUS?"})
        assert resp.json()["safety_level"] == "contains_identifying_info"

    def test_safety_module_detects_bare_id(self):
        result = safety.contains_identifying_info("מספר הזהות שלי 123456789")
        assert result

    def test_safety_module_allows_clean_question(self):
        result = safety.contains_identifying_info("מה ההבדל בין VUS לפתוגני?")
        assert not result


# ---------------------------------------------------------------------------
# 3. Personal interpretation redirect
# ---------------------------------------------------------------------------

class TestPersonalInterpretationRedirect:
    def test_is_my_variant_dangerous(self):
        resp = client.post("/ask", json={"question": "האם הווריאנט שלי מסוכן?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True
        assert "גנטיקאי" in data["answer"] or "צוות הגנטי" in data["answer"]

    def test_what_is_my_risk(self):
        resp = client.post("/ask", json={"question": "מה הסיכון שלי לחלות?"})
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_should_my_children_be_sick(self):
        resp = client.post("/ask", json={"question": "האם הילדים שלי יהיו חולים?"})
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_english_interpret_my_variant(self):
        resp = client.post("/ask", json={"question": "Interpret my variant please"})
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_hgvs_notation_triggers_redirect(self):
        resp = client.post("/ask", json={"question": "I have c.123A>G, what does it mean?"})
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_safety_module_directly(self):
        result = safety.is_personal_interpretation_request("האם אני צריכה לעשות בדיקה?")
        assert result


# ---------------------------------------------------------------------------
# 4. LLM fallback
# ---------------------------------------------------------------------------

class TestLlmFallback:
    def test_engine_falls_back_to_kb_answer_without_llm_url(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        result = counseling_engine.answer_question("מה זה carrier?")
        carrier = kb.match_question("מה זה carrier?")
        assert result["answer"] == carrier["approved_answer_he"]
        assert result["matched_topic"] == "carrier"

    def test_endpoint_works_without_llm_url(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        resp = client.post("/ask", json={"question": "מה זה penetrance?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_topic"] == "penetrance"
        assert data["safety_level"] == "general_information"

    def test_call_local_llm_returns_none_without_url(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
        entry = kb.match_question("מה זה VUS?")
        result = counseling_engine._call_local_llm("מה זה VUS?", entry)
        assert result is None


# ---------------------------------------------------------------------------
# 5. Topics and FAQ endpoints
# ---------------------------------------------------------------------------

class TestTopicsEndpoint:
    def test_topics_endpoint_returns_all_kb_topics(self):
        resp = client.get("/topics")
        assert resp.status_code == 200
        topics = resp.json()["topics"]
        topic_ids = {t["id"] for t in topics}
        expected = {"vus", "carrier", "pathogenic", "penetrance"}
        assert expected.issubset(topic_ids)

    def test_faq_endpoint_returns_approved_answers(self):
        resp = client.get("/faq")
        assert resp.status_code == 200
        items = resp.json()["faq"]
        assert len(items) > 0
        assert all(item.get("approved_answer_he") for item in items)


# ---------------------------------------------------------------------------
# 6. Topic hint
# ---------------------------------------------------------------------------

class TestTopicHint:
    def test_explicit_topic_hint_overrides_keyword_scoring(self):
        resp = client.post("/ask", json={"question": "מה דעתך?", "topic": "franklin"})
        assert resp.json()["matched_topic"] == "franklin"

    def test_unknown_topic_hint_falls_back_to_keyword_scoring(self):
        resp = client.post("/ask", json={"question": "מה זה VUS?", "topic": "not-a-real-topic"})
        assert resp.json()["matched_topic"] != "not-a-real-topic"


# ---------------------------------------------------------------------------
# 7. Gene-name + VUS handling
# ---------------------------------------------------------------------------

class TestGeneNameVusHandling:
    def test_brca1_vus_returns_general_information(self):
        resp = client.post("/ask", json={"question": "קיבלתי VUS על הגן BRCA1, מה זה אומר?"})
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]
        assert "BRCA1" in data["answer"]
        assert data["matched_topic"] == "vus_known_gene"

    def test_braca1_typo_normalizes_to_brca1(self):
        resp = client.post("/ask", json={"question": "קיבלתי VUS על הגן BRACA 1, מה זה אומר?"})
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]
        assert "BRCA1" in data["answer"]

    def test_braca1_no_space_typo_also_normalizes(self):
        resp = client.post("/ask", json={"question": "יש לי VUS בגן BRACA1"})
        assert "BRCA1" in resp.json()["answer"]

    def test_nf1_gene_vus_handled(self):
        resp = client.post("/ask", json={"question": "קיבלתי VUS בגן NF1, מה זה אומר?"})
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["answer"]

    def test_gene_mention_without_vus_does_not_trigger_special_case(self):
        """Mentioning BRCA1 alone (no VUS) should not force the known-gene template."""
        resp = client.post("/ask", json={"question": "מה זה BRCA1?"})
        assert resp.json()["matched_topic"] != "vus_known_gene"

    def test_specific_variant_still_redirects_even_with_gene_name(self):
        """A specific HGVS variant must still win over the general gene+VUS case."""
        resp = client.post(
            "/ask",
            json={"question": "יש לי VUS בגן BRCA1, c.123A>G, האם הוא מסוכן עבורי?"},
        )
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_explicit_topic_hint_bypasses_gene_detection(self):
        """An explicit topic hint from the UI takes priority over free-text gene detection."""
        resp = client.post(
            "/ask",
            json={"question": "קיבלתי VUS על הגן BRCA1, מה זה אומר?", "topic": "carrier"},
        )
        assert resp.json()["matched_topic"] == "carrier"


# ---------------------------------------------------------------------------
# 8. VUS follow-up topic coverage
# ---------------------------------------------------------------------------

class TestVusFollowUpTopics:
    def test_family_testing_with_vus_not_fallback(self):
        resp = client.post("/ask", json={"question": "האם יש מקום לבדוק קרובי משפחה כשיש VUS?"})
        data = resp.json()
        assert data["safety_level"] != "out_of_scope"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_reclassification_question_not_fallback(self):
        resp = client.post("/ask", json={"question": "מה יכול לגרום לשינוי סיווג של VUS בעתיד?"})
        data = resp.json()
        assert data["safety_level"] != "out_of_scope"
        assert data["matched_topic"] == "vus_reclassification"

    def test_vus_vs_pathogenic_difference_not_fallback(self):
        resp = client.post("/ask", json={"question": "מה ההבדל בין VUS לבין pathogenic variant?"})
        assert resp.json()["matched_topic"] == "vus_vs_pathogenic"

    def test_why_vus_not_guide_decisions_not_fallback(self):
        resp = client.post("/ask", json={"question": "למה בדרך כלל לא מקבלים החלטות רפואיות רק לפי VUS?"})
        assert resp.json()["matched_topic"] == "vus_not_guide_decisions"

    def test_questions_for_counselor_not_fallback(self):
        resp = client.post("/ask", json={"question": "מה אפשר לשאול את הגנטיקאי/ת אחרי קבלת VUS?"})
        assert resp.json()["matched_topic"] == "vus_questions_for_counselor"

    def test_plain_vus_question_still_matches_main_vus_entry(self):
        """Ties between the generic 'vus' entry and specialized vus_* entries
        must resolve to the generic entry when no specialized phrase is present."""
        resp = client.post("/ask", json={"question": "מה זה VUS?"})
        assert resp.json()["matched_topic"]


# ---------------------------------------------------------------------------
# 9. All suggested questions answerable
# ---------------------------------------------------------------------------

class TestAllSuggestedQuestionsAnswerable:
    """Goal: no suggested_questions chip should ever lead to an out_of_scope
    fallback or get incorrectly blocked as identifying info."""

    def test_every_suggested_question_resolves_without_fallback(self):
        failures = []
        for entry in kb.all_entries():
            for sq in entry.get("suggested_questions", []):
                resp = client.post("/ask", json={"question": sq})
                if resp.json()["safety_level"] == "out_of_scope":
                    failures.append(sq)
        assert not failures, f"Suggested questions that fail to resolve: {failures}"


# ---------------------------------------------------------------------------
# 10. Variant evidence summary
# ---------------------------------------------------------------------------

class TestVariantEvidenceSummary:
    def test_hgvs_with_danger_question_gets_evidence_summary_not_just_refusal(self):
        resp = client.post("/ask", json={"question": "יש לי וריאנט c.123A>G, האם הוא מסוכן?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True
        assert data["matched_topic"] == "variant_evidence_summary"
        assert data["answer"]
        assert "פנה לגנטיקאי" in data["answer"] or "צוות הגנטי" in data["answer"]

    def test_answer_never_says_dangerous_or_benign_for_user_personally(self):
        resp = client.post("/ask", json={"question": "יש לי וריאנט c.123A>G, האם הוא מסוכן?"})
        answer = resp.json()["answer"]
        assert "מסוכן עבורך" not in answer
        assert "מסוכן לך" not in answer
        assert "תקין עבורך" not in answer
        assert "בטוח עבורך" not in answer

    def test_plain_hgvs_lookup_without_danger_framing_still_gets_summary(self):
        resp = client.post("/ask", json={"question": "מה ידוע על c.123A>G?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["matched_topic"] == "variant_evidence_summary"

    def test_rsid_lookup_attempts_evidence_summary(self):
        resp = client.post("/ask", json={"question": "מה ידוע על rs80357906?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["matched_topic"] == "variant_evidence_summary"
        assert data["answer"]

    def test_no_evidence_found_gives_educational_explanation(self):
        """A variant that doesn't exist in the local DB still gets a useful,
        non-refusal educational answer about what geneticists consider."""
        resp = client.post("/ask", json={"question": "מה ידוע על c.99999999Z>Q?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert "לא ניתן לקבוע" in data["answer"]
        assert any(
            kw in data["answer"]
            for kw in ["גנטיקאי", "צוות הגנטי", "בדיקה", "מאגר", "ראיות"]
        )

    def test_personal_danger_phrase_alone_without_variant_still_redirects(self):
        """No variant named -> falls back to the generic redirect, not an
        evidence summary (nothing to look up)."""
        resp = client.post("/ask", json={"question": "האם הווריאנט שלי מסוכן?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["matched_topic"] != "variant_evidence_summary"

    def test_identifying_info_still_blocks_before_variant_lookup(self):
        resp = client.post(
            "/ask",
            json={"question": "תעודת הזהות שלי היא 123456789, יש לי וריאנט c.123A>G, מסוכן?"},
        )
        data = resp.json()
        assert data["safety_level"] == "contains_identifying_info"
        legacy = {
            "evidence", "limitations", "safety_disclaimer",
            "policy_flags", "redirect_message",
        }
        assert not set(data.keys()).intersection(legacy)

    def test_schema_unchanged_for_variant_evidence_path(self):
        resp = client.post("/ask", json={"question": "מה ידוע על c.123A>G?"})
        data = resp.json()
        assert set(data.keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }


# ---------------------------------------------------------------------------
# 11. Personal medical action without variant
# ---------------------------------------------------------------------------

class TestPersonalMedicalActionWithoutVariant:
    """Medical-action questions with no specific variant named must still be
    redirected, not given an evidence summary or clinical recommendation."""

    def test_surgery_question_redirects(self):
        resp = client.post("/ask", json={"question": "האם אני צריכה ניתוח בגלל הווריאנט?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["matched_topic"] != "variant_evidence_summary"
        assert data["answer"]

    def test_mri_question_redirects(self):
        resp = client.post("/ask", json={"question": "האם אני צריכה MRI?"})
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_treatment_choice_question_redirects(self):
        resp = client.post("/ask", json={"question": "איזה טיפול לעשות?"})
        assert resp.json()["safety_level"] == "requires_genetic_counselor"

    def test_children_sick_question_redirects(self):
        resp = client.post("/ask", json={"question": "האם הילדים שלי יהיו חולים?"})
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert data["needs_genetic_counselor"] is True


# ---------------------------------------------------------------------------
# 12. Safety variant helpers
# ---------------------------------------------------------------------------

class TestSafetyVariantHelpers:
    def test_contains_specific_variant_detects_cdna(self):
        assert safety.contains_specific_variant("יש לי c.123A>G")

    def test_contains_specific_variant_detects_rsid(self):
        assert safety.contains_specific_variant("מה ידוע על rs80357906?")

    def test_contains_specific_variant_false_for_gene_name_only(self):
        assert not safety.contains_specific_variant("קיבלתי VUS בגן BRCA1")

    def test_extract_variant_query_rsid(self):
        query = "מה ידוע על rs80357906?"
        result = safety.extract_variant_query(query)
        assert result is not None and result.get("rsid") == "rs80357906"

    def test_extract_variant_query_cdna(self):
        query = "יש לי c.123A>G, מה זה אומר?"
        result = safety.extract_variant_query(query)
        assert result and result.get("variant")

    def test_is_personal_interpretation_request_no_longer_flags_bare_hgvs(self):
        """HGVS notation alone is handled by the variant-evidence path now,
        not by the generic personal-interpretation redirect."""
        assert not safety.is_personal_interpretation_request("c.123A>G")


# ---------------------------------------------------------------------------
# 13. Out of scope
# ---------------------------------------------------------------------------

class TestOutOfScope:
    def test_unrelated_question_is_out_of_scope(self):
        resp = client.post("/ask", json={"question": "מה השעה עכשיו בטוקיו?"})
        data = resp.json()
        assert data["safety_level"] == "out_of_scope"
        assert data["needs_genetic_counselor"] is True
        assert data["matched_topic"] is None

    def test_fallback_is_helpful_not_harsh(self):
        """The fallback message should explain scope and offer topic
        suggestions, not just a flat refusal."""
        resp = client.post("/ask", json={"question": "מה השעה עכשיו בטוקיו?"})
        data = resp.json()
        assert data["suggested_questions"]
        assert "מידע כללי" in data["answer"] or data["answer"]


# ---------------------------------------------------------------------------
# 14. Follow-up handling
# ---------------------------------------------------------------------------

class TestFollowUpHandling:
    def test_followup_after_vus_known_gene_uses_last_topic(self):
        first = client.post(
            "/ask", json={"question": "יש לי VUS ל BRACA1 מה זה אומר?"}
        ).json()
        assert first["matched_topic"] == "vus_known_gene"
        resp = client.post(
            "/ask",
            json={
                "question": "אתה יכול לפרט יותר?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "יש לי VUS ל BRACA1 מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]
        assert "BRCA1" in data["answer"]

    def test_followup_resolves_gene_from_context_when_last_topic_omitted(self):
        """Even without an explicit last_topic, scanning conversation_context
        for the matched_topic and gene mention should be enough."""
        first = client.post(
            "/ask", json={"question": "יש לי VUS בגן BRCA1, מה זה אומר?"}
        ).json()
        resp = client.post(
            "/ask",
            json={
                "question": "אפשר דוגמה?",
                "conversation_context": [
                    {"role": "user", "content": "יש לי VUS בגן BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": "vus_known_gene"},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"]
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_followup_without_any_context_falls_through_normally(self):
        """A follow-up phrase with no last_topic and no context should not
        crash — it just falls through to normal (likely fallback) handling."""
        resp = client.post("/ask", json={"question": "אתה יכול לפרט יותר?"})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }

    def test_followup_on_generic_kb_topic_expands_with_related_entry(self):
        resp = client.post(
            "/ask",
            json={
                "question": "תסביר יותר",
                "last_topic": "carrier",
            },
        )
        data = resp.json()
        assert data["matched_topic"] == "carrier"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_carrier_then_partner_followup_uses_new_kb_entry(self):
        first = client.post(
            "/ask", json={"question": "אמרו לי שאני נשאית, מה זה אומר?"}
        ).json()
        assert first["matched_topic"] == "carrier"
        resp = client.post("/ask", json={"question": "ומה זה אומר לבן זוג?"})
        data = resp.json()
        assert data["safety_level"] != "out_of_scope"
        assert "אין לי מידע מאושר" not in data["answer"]
        assert data["matched_topic"] == "carrier_partner_testing"

    def test_identifying_info_in_current_question_blocks_even_with_context(self):
        resp = client.post(
            "/ask",
            json={"question": "תעודת הזהות שלי היא 123456789, אפשר לפרט?"},
        )
        assert resp.json()["safety_level"] == "contains_identifying_info"

    def test_identifying_info_in_old_context_is_sanitized_out(self):
        """Past context containing identifying info must never influence the
        current answer (defense in depth — frontend should also avoid this,
        but the backend must not trust it either)."""
        first = client.post("/ask", json={"question": "מה זה VUS?"}).json()
        resp = client.post(
            "/ask",
            json={
                "question": "אתה יכול לפרט יותר?",
                "conversation_context": [
                    {"role": "user", "content": "קוראים לי דנה, תעודת זהות 123456789"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": "vus_known_gene"},
                ],
            },
        )
        data = resp.json()
        assert set(data.keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }


# ---------------------------------------------------------------------------
# 15. Conversation context schema
# ---------------------------------------------------------------------------

class TestConversationContextSchema:
    def test_request_accepts_conversation_context_and_last_topic(self):
        resp = client.post(
            "/ask",
            json={
                "question": "מה זה VUS?",
                "last_topic": "carrier",
                "conversation_context": [
                    {"role": "user", "content": "שאלה קודמת"},
                    {"role": "assistant", "content": "...", "matched_topic": "carrier"},
                ],
            },
        )
        assert resp.status_code == 200

    def test_response_schema_unchanged_with_context_fields_present(self):
        resp = client.post(
            "/ask",
            json={
                "question": "מה זה VUS?",
                "conversation_context": [
                    {"role": "user", "content": "שאלה קודמת"},
                ],
            },
        )
        data = resp.json()
        assert set(data.keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }

    def test_request_without_context_fields_still_works(self):
        """Backwards compatible — old-style {question, topic} requests still work."""
        resp = client.post("/ask", json={"question": "מה זה VUS?"})
        assert resp.status_code == 200
        assert resp.json()["safety_level"] == "general_information"


# ---------------------------------------------------------------------------
# 16. Safety helper context sanitization
# ---------------------------------------------------------------------------

class TestSafetyHelperContextSanitization:
    def test_sanitize_context_drops_identifying_info_messages(self):
        context = [
            {"role": "user", "content": "קוראים לי דנה, תעודת זהות 123456789"},
            {"role": "assistant", "content": "מה זה VUS?"},
        ]
        safe = counseling_engine._sanitize_context(context)
        assert len(safe) < len(context)
        assert not any("123456789" in m.get("content", "") for m in safe)

    def test_sanitize_context_handles_none(self):
        assert counseling_engine._sanitize_context(None) == []

    def test_is_followup_question_detects_known_phrases(self):
        for phrase in counseling_engine._FOLLOWUP_PHRASES:
            assert counseling_engine._is_followup_question(phrase), phrase

    def test_is_followup_question_false_for_normal_questions(self):
        assert not counseling_engine._is_followup_question("מה זה VUS?")
        assert not counseling_engine._is_followup_question("יש לי וריאנט c.123A>G")


# ---------------------------------------------------------------------------
# 17. No DB required
# ---------------------------------------------------------------------------

class TestNoDbRequired:
    @pytest.fixture
    def _simulate_missing_db(self, monkeypatch):
        monkeypatch.setattr(retriever, "_DB_AVAILABLE", False)

    def test_match_uploaded_variant_degrades_gracefully(self, _simulate_missing_db):
        result = retriever.match_uploaded_variant({"rsid": "rs80357906"})
        assert result["match_confidence"] == "no_match"
        assert not result["matches"]

    def test_variant_question_still_works_without_db(self, _simulate_missing_db):
        resp = client.post("/ask", json={"question": "יש לי וריאנט c.123A>G, האם הוא מסוכן?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety_level"] == "requires_genetic_counselor"
        assert "לא ניתן לקבוע" in data["answer"]

    def test_general_kb_questions_unaffected_without_db(self, _simulate_missing_db):
        resp = client.post("/ask", json={"question": "מה זה VUS?"})
        assert resp.json()["safety_level"] == "general_information"

    def test_app_routes_unaffected_without_db(self, _simulate_missing_db):
        assert client.get("/topics").status_code == 200


# ---------------------------------------------------------------------------
# 18. Exact two-turn scenario: VUS in NF1 → follow-up about implications
# ---------------------------------------------------------------------------

class TestVusNf1FollowUpScenario:
    """
    Regression tests for the bug where follow-up questions after a
    VUS-in-NF1 first turn returned generic fallback suggestions instead
    of a substantive general explanation.

    Turn 1: "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"
    Turn 2: "בכל זאת מה ההשלכות של זה?"  (or the variant below)
            "מה אתה יכול לספר לי על זה?"

    Required behaviour:
    - safety_level == "general_information"
    - matched_topic == "vus_known_gene"
    - Answer is substantive (expanded vs turn 1)
    - Covers: VUS ≠ pathogenic, interpretation factors, reclassification
    - Never diagnoses, estimates personal risk, or recommends treatment
    """

    def _turn1(self):
        resp = client.post(
            "/ask",
            json={"question": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_topic"] == "vus_known_gene", (
            f"Turn-1 sanity check failed: expected vus_known_gene"
        )
        return data

    def test_is_followup_detects_hashlacha_phrase(self):
        "'מה ההשלכות' must be recognised as a follow-up continuation."
        assert counseling_engine._is_followup_question("בכל זאת מה ההשלכות של זה?") is True

    def test_is_followup_detects_saper_li_phrase(self):
        "'ספר לי' embedded in a question must be recognised as a follow-up."
        assert counseling_engine._is_followup_question("מה אתה יכול לספר לי על זה?") is True

    def test_is_followup_detects_ma_ze_omer_bifual(self):
        "'מה זה אומר בפועל' must fire the follow-up path (already in list)."
        assert counseling_engine._is_followup_question("מה זה אומר בפועל?") is True

    def test_is_followup_detects_bkhol_zot_ma(self):
        assert counseling_engine._is_followup_question("בכל זאת מה הכוונה?") is True

    def test_implications_followup_returns_general_information(self):
        "'בכל זאת מה ההשלכות של זה?' after VUS/NF1 must not fall back to out_of_scope."
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_implications_followup_answer_is_expanded(self):
        "The follow-up answer must be substantially longer than turn 1."
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert len(resp.json()["answer"]) > 400, (
            "Follow-up answer should be substantive (>400 chars)"
        )

    def test_tell_me_more_followup_returns_general_information(self):
        "'מה אתה יכול לספר לי על זה?' must give a substantive answer, not a fallback."
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה אתה יכול לספר לי על זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"
        assert "אין לי מידע מאושר" not in data["answer"]
        assert len(data["answer"]) > 400, (
            "Follow-up answer should be substantive (>400 chars)"
        )

    def test_followup_answer_explains_vus_not_same_as_pathogenic(self):
        "The expanded answer must convey that VUS ≠ pathogenic variant."
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        answer = resp.json()["answer"]
        assert "pathogenic" in answer.lower() or "פתוגני" in answer

    def test_followup_answer_covers_interpretation_factors(self):
        """Answer must include at least one interpretation factor
        (e.g. inheritance pattern, population frequency, functional evidence)."""
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        answer = resp.json()["answer"]
        keywords = (
            "דפוס התורשה", "שכיחות", "עדויות חישוביות", "הקשר הקליני",
            "ראיות", "מידע מדעי", "ייעוץ גנטי", "מאגרי מידע",
        )
        assert any(kw in answer for kw in keywords)

    def test_followup_never_diagnoses_or_recommends_treatment(self):
        """Safety: follow-up answer must not contain personal diagnosis,
        risk estimate, or treatment/surveillance recommendation."""
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        answer = resp.json()["answer"]
        forbidden = (
            "מסוכן לך", "מסוכן עבורך", "בטוח לך", "הסיכון שלך", "הסיכון האישי",
            "צריך ניתוח", "צריכה ניתוח", "צריך טיפול", "צריכה טיפול",
            "צריך מעקב", "צריכה מעקב",
        )
        for phrase in forbidden:
            assert phrase not in answer, f"Forbidden phrase found in answer: {phrase!r}"

    def test_followup_without_last_topic_resolves_via_conversation_context(self):
        """Without an explicit last_topic the engine must still resolve the
        topic from matched_topic in the conversation_context messages."""
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_response_schema_unchanged_for_followup(self):
        "API schema must stay stable: exactly the five counseling fields."
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "בכל זאת מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שיש לי VUS ב NF1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }


# ---------------------------------------------------------------------------
# 19. VUS in BRCA1/BRCA2 — follow-up questions
# ---------------------------------------------------------------------------

class TestVusBrca1Brca2FollowUp:
    """Two-turn conversation: initial BRCA1/BRCA2 VUS answer followed by
    vague follow-up questions that should return expanded, substantive replies."""

    def _turn1_brca1(self):
        resp = client.post("/ask", json={"question": "קיבלתי VUS ב BRCA1, מה זה אומר?"})
        assert resp.status_code == 200
        return resp.json()

    def _turn1_brca2(self):
        resp = client.post("/ask", json={"question": "נמצא לי VUS בגן BRCA2, מה המשמעות?"})
        assert resp.status_code == 200
        return resp.json()

    def test_brca1_initial_answer_is_vus_known_gene(self):
        data = self._turn1_brca1()
        assert data["matched_topic"] == "vus_known_gene"
        assert data["safety_level"] == "general_information"

    def test_brca2_initial_answer_is_vus_known_gene(self):
        data = self._turn1_brca2()
        assert data["matched_topic"] == "vus_known_gene"
        assert data["safety_level"] == "general_information"

    def test_brca1_initial_answer_mentions_gene(self):
        assert "BRCA1" in self._turn1_brca1()["answer"]

    def test_brca2_initial_answer_mentions_gene(self):
        assert "BRCA2" in self._turn1_brca2()["answer"]

    def test_brca1_hashlacha_followup_returns_general_information(self):
        first = self._turn1_brca1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_brca1_hashlacha_followup_is_expanded(self):
        first = self._turn1_brca1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert len(resp.json()["answer"]) > 400, (
            "Follow-up answer should be substantive (>400 chars)"
        )

    def test_brca1_bifual_followup_is_substantive(self):
        first = self._turn1_brca1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה זה אומר בפועל?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_brca2_explain_more_followup(self):
        first = self._turn1_brca2()
        resp = client.post(
            "/ask",
            json={
                "question": "תסביר יותר",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "נמצא לי VUS בגן BRCA2, מה המשמעות?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"

    def test_brca2_tell_me_more_followup(self):
        first = self._turn1_brca2()
        resp = client.post(
            "/ask",
            json={
                "question": "מה אתה יכול לספר לי על זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "נמצא לי VUS בגן BRCA2, מה המשמעות?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_brca1_space_variant_is_detected(self):
        """'BRCA 1' (with a space) must be normalised to BRCA1."""
        resp = client.post("/ask", json={"question": "קיבלתי VUS ב BRCA 1, מה זה?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_topic"] == "vus_known_gene"
        assert "BRCA1" in data["answer"]

    def test_brca2_dash_variant_is_detected(self):
        """'BRCA-2' (with a dash) must be normalised to BRCA2."""
        resp = client.post("/ask", json={"question": "קיבלתי VUS ב BRCA-2, מה זה?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_topic"] == "vus_known_gene"
        assert "BRCA2" in data["answer"]

    def test_brca1_followup_answer_explains_pathogenic_distinction(self):
        first = self._turn1_brca1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert "pathogenic" in resp.json()["answer"].lower()

    def test_brca1_followup_never_gives_personal_risk(self):
        first = self._turn1_brca1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        answer = resp.json()["answer"]
        forbidden = ["הסיכון שלך", "הסיכון שלי", "מסוכן לך", "מסוכן לי",
                     "מומלץ לך לעשות", "עליך לעשות", "עליך לעבור"]
        for phrase in forbidden:
            assert phrase not in answer, f"Forbidden personal-risk phrase: {phrase!r}"

    def test_brca1_followup_schema_unchanged(self):
        first = self._turn1_brca1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות של זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }


# ---------------------------------------------------------------------------
# 20. Carrier status — follow-up questions
# ---------------------------------------------------------------------------

class TestCarrierFollowUp:
    """Two-turn conversation: initial carrier-status question followed by
    vague follow-up questions that must return expanded, substantive replies."""

    def _turn1(self):
        resp = client.post("/ask", json={"question": "אמרו לי שאני נשא, מה זה אומר?"})
        assert resp.status_code == 200
        return resp.json()

    def test_carrier_initial_matched_topic(self):
        assert self._turn1()["matched_topic"] == "carrier"

    def test_carrier_initial_safety_level(self):
        assert self._turn1()["safety_level"] == "general_information"

    def test_carrier_hashlacha_followup_returns_general_information(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_carrier_hashlacha_followup_is_expanded(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert len(resp.json()["answer"]) > len(first["answer"])

    def test_carrier_bifual_followup(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה זה אומר בפועל?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["matched_topic"] == "carrier"
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_carrier_tell_me_more_followup(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה אתה יכול לספר לי על זה?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "carrier"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_carrier_explain_more_followup(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "תסביר יותר",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "carrier"

    def test_carrier_followup_contains_relevant_content(self):
        """Expanded carrier answer must mention partner or inheritance concepts."""
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        answer = resp.json()["answer"]
        keywords = ["נשא", "נשאות", "בן זוג", "ירושה", "הורשה", "גנטי", "carrier"]
        assert any(kw in answer for kw in keywords), (
            f"Expected carrier-related keyword in follow-up answer. Got: {answer[:200]}"
        )

    def test_carrier_followup_no_personal_risk(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        answer = resp.json()["answer"]
        forbidden = [
            "הסיכון שלך", "הסיכון שלי", "מסוכן לך",
            "צריך לעשות בדיקה", "צריכה לעשות בדיקה",
        ]
        for phrase in forbidden:
            assert phrase not in answer, f"Forbidden personal-risk phrase: {phrase!r}"

    def test_carrier_followup_schema_unchanged(self):
        first = self._turn1()
        resp = client.post(
            "/ask",
            json={
                "question": "מה ההשלכות?",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אמרו לי שאני נשא, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }


# ---------------------------------------------------------------------------
# 21. Vague follow-up phrase detection
# ---------------------------------------------------------------------------

class TestVagueFollowUpPatterns:
    """Unit-level tests for _is_followup_question() covering the new phrases,
    plus integration smoke tests confirming they resolve via prior context."""

    @pytest.mark.parametrize("phrase", [
        "תרחיב",
        "תרחיבי",
        "מידע נוסף",
        "פרטים נוספים",
        "ספר עוד",
        "ספרי עוד",
        "ומה עוד",
        "יש עוד",
        "מה נוסף",
    ])
    def test_new_phrase_detected_as_followup(self, phrase):
        assert counseling_engine._is_followup_question(phrase), (
            f"Expected {phrase!r} to be detected as a follow-up phrase"
        )

    @pytest.mark.parametrize("phrase", [
        "תסביר יותר",
        "מה ההשלכות",
        "ספר לי",
        "מה זה אומר בפועל",
        "תוכל לפרט",
        "can you elaborate",
        "tell me more",
    ])
    def test_original_phrase_still_detected(self, phrase):
        assert counseling_engine._is_followup_question(phrase), (
            f"Original phrase {phrase!r} should still be detected as follow-up"
        )

    def test_tarchev_resolves_vus_known_gene(self):
        first_resp = client.post("/ask", json={"question": "קיבלתי VUS ב BRCA1, מה זה אומר?"})
        first = first_resp.json()
        resp = client.post(
            "/ask",
            json={
                "question": "תרחיב",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA1, מה זה אומר?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_more_info_resolves_carrier(self):
        first_resp = client.post("/ask", json={"question": "אני נשאית, מה המשמעות?"})
        first = first_resp.json()
        resp = client.post(
            "/ask",
            json={
                "question": "מידע נוסף",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "אני נשאית, מה המשמעות?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_more_details_resolves_nf1_vus(self):
        first_resp = client.post("/ask", json={"question": "יש לי VUS ב NF1, מה זה?"})
        first = first_resp.json()
        resp = client.post(
            "/ask",
            json={
                "question": "פרטים נוספים",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "יש לי VUS ב NF1, מה זה?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        data = resp.json()
        assert data["safety_level"] == "general_information"
        assert data["matched_topic"] == "vus_known_gene"
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_new_followup_phrase_schema_unchanged(self):
        first_resp = client.post("/ask", json={"question": "קיבלתי VUS ב BRCA2, מה זה?"})
        first = first_resp.json()
        resp = client.post(
            "/ask",
            json={
                "question": "ספר עוד",
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "קיבלתי VUS ב BRCA2, מה זה?"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }


# ---------------------------------------------------------------------------
# 22. VUS follow-up routing — regression for "מה כדאי לעשות" → carrier bug
# ---------------------------------------------------------------------------

class TestVusFollowUpRouting:
    """
    Regression tests for the bug where follow-up questions like
    "מה כדאי לעשות עם זה?" after a VUS topic incorrectly returned
    the carrier_vs_affected answer (caused by difflib fuzzy-matching
    "לעשות" → "לעומת" at exactly 0.80 cutoff).
    """

    def _turn1_brca1(self):
        r = client.post("/ask", json={"question": "אמרו לי שיש VUS בגן BRACA1 מה זה אומר?"})
        assert r.status_code == 200
        return r.json()

    def _turn1_vus_general(self):
        r = client.post("/ask", json={"question": "מה זה VUS?"})
        assert r.status_code == 200
        return r.json()

    def _turn1_nf1(self):
        r = client.post("/ask", json={"question": "יש לי VUS ב NF1, מה זה?"})
        assert r.status_code == 200
        return r.json()

    def _followup(self, first, question):
        return client.post(
            "/ask",
            json={
                "question": question,
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "שאלה על VUS"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        )

    def test_ma_kdai_laasot_detected_as_followup(self):
        assert counseling_engine._is_followup_question("מה כדאי לעשות עם זה?") is True

    def test_ma_osim_detected_as_followup(self):
        assert counseling_engine._is_followup_question("מה עושים עם זה?") is True

    def test_ma_hatsad_haba_detected_as_followup(self):
        assert counseling_engine._is_followup_question("מה הצעד הבא?") is True

    def test_brca1_vus_kdai_laasot_not_carrier(self):
        first = self._turn1_brca1()
        data = self._followup(first, "מה כדאי לעשות עם זה?").json()
        assert data["matched_topic"] not in ("carrier_vs_affected", "carrier")

    def test_brca1_vus_kdai_laasot_returns_vus_topic(self):
        first = self._turn1_brca1()
        data = self._followup(first, "מה כדאי לעשות עם זה?").json()
        assert data["matched_topic"] == "vus_known_gene"
        assert data["safety_level"] == "general_information"

    def test_brca1_vus_kdai_laasot_mentions_vus_and_pathogenic(self):
        first = self._turn1_brca1()
        answer = self._followup(first, "מה כדאי לעשות עם זה?").json()["answer"]
        assert "VUS" in answer
        assert "pathogenic" in answer.lower()

    def test_general_vus_kdai_laasot_not_fallback(self):
        first = self._turn1_vus_general()
        data = self._followup(first, "מה כדאי לעשות עם זה?").json()
        assert data["safety_level"] == "general_information"
        assert "VUS" in data["answer"]
        assert "אין לי מידע מאושר" not in data["answer"]

    def test_general_vus_followup_covers_cannot_conclude(self):
        first = self._turn1_vus_general()
        answer = self._followup(first, "מה כדאי לעשות עם זה?").json()["answer"]
        # Accepts both old-style ("לא ניתן"/"בלבד") and new conversational style
        # ("לא מקבלים" / "לא ברור") — both correctly convey that VUS alone
        # cannot drive medical decisions.
        assert any(kw in answer for kw in [
            "לא ניתן", "אינו מאפשר", "אינו אבחנה", "בלבד",
            "לא מקבלים", "לא ברור",
        ])

    def test_general_vus_followup_covers_reclassification(self):
        first = self._turn1_vus_general()
        answer = self._followup(first, "מה כדאי לעשות עם זה?").json()["answer"]
        assert any(kw in answer for kw in ["תעד", "מסווג מחדש", "סיווג מחדש", "ראיות", "עתיד"])

    def test_general_vus_followup_covers_genetics_team_questions(self):
        first = self._turn1_vus_general()
        answer = self._followup(first, "מה כדאי לעשות עם זה?").json()["answer"]
        assert any(kw in answer for kw in ["צוות הגנטי", "ייעוץ גנטי", "לשאול", "שאלות"])

    def test_nf1_vus_hashlacha_returns_vus_known_gene(self):
        first = self._turn1_nf1()
        data = self._followup(first, "מה ההשלכות?").json()
        assert data["matched_topic"] == "vus_known_gene"
        assert data["safety_level"] == "general_information"

    def test_nf1_vus_hashlacha_mentions_nf1(self):
        first = self._turn1_nf1()
        assert "NF1" in self._followup(first, "מה ההשלכות?").json()["answer"]

    def test_nf1_vus_hashlacha_pathogenic_distinction(self):
        first = self._turn1_nf1()
        assert "pathogenic" in self._followup(first, "מה ההשלכות?").json()["answer"].lower()

    def test_nf1_vus_no_personal_diagnosis(self):
        first = self._turn1_nf1()
        answer = self._followup(first, "מה ההשלכות?").json()["answer"]
        for phrase in ["מסוכן לך", "הסיכון שלך", "יש לך מחלה", "מאובחן"]:
            assert phrase not in answer

    def test_carrier_question_still_returns_carrier(self):
        r = client.post("/ask", json={"question": "אמרו לי שאני נשא, מה זה אומר?"})
        d = r.json()
        assert r.status_code == 200
        assert d["matched_topic"] == "carrier"
        assert d["safety_level"] == "general_information"

    def test_carrier_answer_not_about_vus(self):
        r = client.post("/ask", json={"question": "מה זה נשאות?"})
        d = r.json()
        assert r.status_code == 200
        assert d["safety_level"] == "general_information"
        assert "VUS" not in d["answer"]

    def test_vus_followup_no_personal_risk(self):
        first = self._turn1_brca1()
        answer = self._followup(first, "מה כדאי לעשות עם זה?").json()["answer"]
        for phrase in [
            "הסיכון שלך", "הסיכון שלי", "מסוכן לך", "מסוכן לי",
            "יש לך", "את חולה", "אתה חולה",
            "מומלץ לך לעשות ניתוח", "עליך לעשות", "עליך לעבור",
        ]:
            assert phrase not in answer

    def test_vus_followup_schema_unchanged(self):
        first = self._turn1_brca1()
        resp = self._followup(first, "מה כדאי לעשות עם זה?")
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {
            "answer", "safety_level", "needs_genetic_counselor",
            "matched_topic", "suggested_questions",
            "llm_used", "fallback_used", "llm_mode",
        }

    # ------------------------------------------------------------------
    # BRCA2 — same routing guarantee as BRCA1
    # ------------------------------------------------------------------

    def _turn1_brca2(self):
        r = client.post("/ask", json={"question": "נמצא לי VUS בגן BRCA2, מה המשמעות?"})
        assert r.status_code == 200
        return r.json()

    def test_brca2_vus_kdai_laasot_not_carrier(self):
        """BRCA2 VUS + 'מה כדאי לעשות עם זה?' must not return carrier or carrier_vs_affected."""
        first = self._turn1_brca2()
        data = self._followup(first, "מה כדאי לעשות עם זה?").json()
        assert data["matched_topic"] not in ("carrier_vs_affected", "carrier")

    def test_brca2_vus_kdai_laasot_returns_vus_topic(self):
        first = self._turn1_brca2()
        data = self._followup(first, "מה כדאי לעשות עם זה?").json()
        assert data["matched_topic"] == "vus_known_gene"
        assert data["safety_level"] == "general_information"

    def test_brca2_vus_kdai_laasot_mentions_brca2_and_pathogenic(self):
        first = self._turn1_brca2()
        answer = self._followup(first, "מה כדאי לעשות עם זה?").json()["answer"]
        assert "BRCA2" in answer
        assert "pathogenic" in answer.lower()


# ---------------------------------------------------------------------------
# 23. VUS follow-up answer quality — concise, conversational, non-repetitive
# ---------------------------------------------------------------------------

class TestVusFollowUpAnswerQuality:
    """
    Checks that VUS follow-up answers are short, conversational, and do not
    repeat the initial VUS definition verbatim.

    The composer must open with a practical phrase ("בפועל..."), cover the
    key practical points, and include suggested questions — all within a
    reasonable word count (~120–220 words).
    """

    def _initial(self, question):
        r = client.post("/ask", json={"question": question})
        assert r.status_code == 200
        return r.json()

    def _followup(self, first, question):
        return client.post(
            "/ask",
            json={
                "question": question,
                "last_topic": first["matched_topic"],
                "conversation_context": [
                    {"role": "user", "content": "שאלה על VUS"},
                    {"role": "assistant", "content": first["answer"],
                     "matched_topic": first["matched_topic"]},
                ],
            },
        ).json()

    # ------------------------------------------------------------------
    # 1. Does not repeat the initial VUS opening sentence
    # ------------------------------------------------------------------
    def test_brca1_followup_does_not_repeat_initial_opening(self):
        """Follow-up must not paste the initial VUS template opening."""
        first = self._initial("אמרו לי שיש VUS בגן BRACA1 מה זה אומר?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        # VUS_KNOWN_GENE_TEMPLATE_HE starts with this phrase
        assert "כאשר מתקבל VUS" not in answer

    # ------------------------------------------------------------------
    # 2. Answer length is within the target range (not a long concatenation)
    # ------------------------------------------------------------------
    def test_brca1_followup_answer_word_count_in_range(self):
        first = self._initial("אמרו לי שיש VUS בגן BRACA1 מה זה אומר?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        words = len(answer.split())
        assert 40 <= words <= 250, (
            f"Word count {words} outside expected range 40–250"
        )

    # ------------------------------------------------------------------
    # 3. Answer includes a practical opening phrase
    # ------------------------------------------------------------------
    def test_brca1_followup_opens_practically(self):
        first = self._initial("אמרו לי שיש VUS בגן BRACA1 מה זה אומר?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        assert any(kw in answer for kw in ["בפועל", "לא מקבלים", "לא ברור", "תיעוד"])

    # ------------------------------------------------------------------
    # 4. Answer includes suggested questions for the genetics team
    # ------------------------------------------------------------------
    def test_brca1_followup_includes_genetics_team_questions(self):
        first = self._initial("אמרו לי שיש VUS בגן BRACA1 מה זה אומר?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        assert "צוות הגנטי" in answer or "שאלות" in answer
        # At least one question mark from the suggested-question list
        assert answer.count("?") >= 1

    # ------------------------------------------------------------------
    # 5. Answer does not mention carrier / נשאות
    # ------------------------------------------------------------------
    def test_brca1_followup_does_not_mention_carrier(self):
        first = self._initial("אמרו לי שיש VUS בגן BRACA1 מה זה אומר?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        assert "נשא" not in answer
        assert "נשאות" not in answer
        assert "carrier" not in answer.lower()

    # ------------------------------------------------------------------
    # 6. NF1 and BRCA2 follow-ups work with the concise composer
    # ------------------------------------------------------------------
    def test_nf1_followup_quality(self):
        first = self._initial("יש לי VUS ב NF1, מה זה?")
        answer = self._followup(first, "מה ההשלכות?")["answer"]
        assert "NF1" in answer
        assert "pathogenic" in answer.lower()
        words = len(answer.split())
        assert words <= 250, f"NF1 follow-up too long: {words} words"

    def test_brca2_followup_quality(self):
        first = self._initial("נמצא לי VUS בגן BRCA2, מה המשמעות?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        assert "BRCA2" in answer
        assert "pathogenic" in answer.lower()
        words = len(answer.split())
        assert words <= 250, f"BRCA2 follow-up too long: {words} words"

    # ------------------------------------------------------------------
    # 7. Safety invariants still hold
    # ------------------------------------------------------------------
    def test_followup_no_personal_risk_or_treatment_phrases(self):
        first = self._initial("אמרו לי שיש VUS בגן BRACA1 מה זה אומר?")
        answer = self._followup(first, "מה כדאי לעשות עם זה?")["answer"]
        forbidden = [
            "הסיכון שלך", "הסיכון שלי", "מסוכן לך", "מסוכן לי",
            "יש לך מחלה", "מאובחן", "עליך לעשות ניתוח", "מומלץ לך לעשות",
        ]
        found = [p for p in forbidden if p in answer]
        assert not found, f"Forbidden safety phrases found: {found}"

# APPEND MARKER — do not remove
