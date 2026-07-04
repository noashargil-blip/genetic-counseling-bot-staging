"""
tests/test_llm_prompt.py

Tests for the upload-specific LLM prompt layer.

No API calls are made. Tests cover:
  1. Prompt structure — required sections present
  2. Uploaded variant details rendered correctly
  3. No-match instruction injected when matches == []
  4. Match confidence labels and their instructions
  5. Gene consistency mismatch included
  6. Warnings rendered
  7. UPLOAD_SYSTEM_PROMPT contains required safety prohibitions
  8. Safe question examples — expect no policy flags
  9. Unsafe question examples — expect policy flags / redirect message
"""

import pytest

from app.llm_client import (
    UPLOAD_SYSTEM_PROMPT,
    _format_upload_user_prompt,
    _format_uploaded_variant_block,
    _format_evidence_block,
)
from app import policy


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _variant(
    gene="BRCA1",
    rsid="rs80357382",
    variant="c.68_69delAG",
    chromosome="17",
    position="43045717",
    protein_change="",
    zygosity="heterozygous",
):
    return {
        "gene": gene,
        "rsid": rsid,
        "variant": variant,
        "chromosome": chromosome,
        "position": position,
        "protein_change": protein_change,
        "zygosity": zygosity,
        "ref": "",
        "alt": "",
        "clinical_significance": "",
    }


def _clinvar_result(
    confidence="exact_rsid",
    gene_consistency="match",
    matches=None,
    warnings=None,
):
    return {
        "match_confidence": confidence,
        "gene_consistency": gene_consistency,
        "matches": matches if matches is not None else [],
        "warnings": warnings if warnings is not None else [],
    }


def _one_match():
    return [
        {
            "gene_symbol": "BRCA1",
            "clinical_significance": "Pathogenic",
            "review_status": "reviewed by expert panel",
            "phenotype_list": "Hereditary breast and ovarian cancer syndrome",
            "dbsnp_id": 80357382,
            "variant_type": "single nucleotide variant",
            "chromosome": "17",
            "start_pos": 43045717,
            "stop_pos": 43045717,
            "last_evaluated": "2023-01-01",
        }
    ]


# ---------------------------------------------------------------------------
# _format_uploaded_variant_block
# ---------------------------------------------------------------------------

class TestFormatUploadedVariantBlock:
    def test_gene_included(self):
        block = _format_uploaded_variant_block(_variant(gene="BRCA2"))
        assert "BRCA2" in block

    def test_rsid_included(self):
        block = _format_uploaded_variant_block(_variant(rsid="rs80358538"))
        assert "rs80358538" in block

    def test_variant_included(self):
        block = _format_uploaded_variant_block(_variant(variant="c.5946delT"))
        assert "c.5946delT" in block

    def test_empty_fields_omitted(self):
        uv = _variant(rsid="", chromosome="", position="")
        block = _format_uploaded_variant_block(uv)
        assert "rsid" not in block
        assert "chromosome" not in block

    def test_empty_variant_returns_placeholder(self):
        block = _format_uploaded_variant_block({})
        assert "no variant details" in block.lower()

    def test_zygosity_included(self):
        block = _format_uploaded_variant_block(_variant(zygosity="homozygous"))
        assert "homozygous" in block


# ---------------------------------------------------------------------------
# _format_evidence_block
# ---------------------------------------------------------------------------

class TestFormatEvidenceBlock:
    def test_empty_list_returns_no_records_message(self):
        block = _format_evidence_block([])
        assert "No ClinVar records" in block

    def test_record_fields_present(self):
        block = _format_evidence_block(_one_match())
        assert "Pathogenic" in block
        assert "BRCA1" in block

    def test_multiple_records_numbered(self):
        block = _format_evidence_block(_one_match() + _one_match())
        assert "Record 1:" in block
        assert "Record 2:" in block


# ---------------------------------------------------------------------------
# _format_upload_user_prompt — structure
# ---------------------------------------------------------------------------

class TestFormatUploadUserPrompt:
    def test_question_in_prompt(self):
        prompt = _format_upload_user_prompt(
            "What does this variant mean?",
            _variant(),
            _clinvar_result(),
        )
        assert "What does this variant mean?" in prompt

    def test_uploaded_section_present(self):
        prompt = _format_upload_user_prompt("q", _variant(), _clinvar_result())
        assert "UPLOADED VARIANT" in prompt

    def test_clinvar_section_present(self):
        prompt = _format_upload_user_prompt("q", _variant(), _clinvar_result())
        # Header format: "=== CLINVAR MATCH ==="
        assert "CLINVAR MATCH" in prompt

    def test_task_section_present(self):
        prompt = _format_upload_user_prompt("q", _variant(), _clinvar_result())
        # Task instructions are inline (no separate "TASK" header)
        assert "ONLY the data above" in prompt
        assert "valid JSON" in prompt

    def test_match_confidence_in_prompt(self):
        prompt = _format_upload_user_prompt(
            "q", _variant(), _clinvar_result(confidence="position_match")
        )
        assert "position_match" in prompt

    def test_gene_consistency_in_prompt(self):
        prompt = _format_upload_user_prompt(
            "q", _variant(), _clinvar_result(gene_consistency="mismatch")
        )
        assert "mismatch" in prompt

    def test_record_count_shown(self):
        prompt = _format_upload_user_prompt(
            "q", _variant(), _clinvar_result(matches=_one_match())
        )
        # Format: "=== CLINVAR RECORDS (1 found) ==="
        assert "(1 found)" in prompt

    def test_zero_records_shown(self):
        prompt = _format_upload_user_prompt(
            "q", _variant(), _clinvar_result(matches=[])
        )
        # Format: "=== CLINVAR RECORDS (0 found) ==="
        assert "(0 found)" in prompt


# ---------------------------------------------------------------------------
# No-match instruction
# ---------------------------------------------------------------------------

class TestNoMatchInstruction:
    def test_no_match_note_injected_when_no_records(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(confidence="no_match", matches=[]),
        )
        assert "No ClinVar records were found" in prompt
        # Instruction uses "do not speculate" (equivalent to "must not speculate")
        assert "do not speculate" in prompt.lower() or "must not speculate" in prompt.lower()

    def test_no_match_note_absent_when_records_present(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(confidence="exact_rsid", matches=_one_match()),
        )
        assert "No ClinVar records were found" not in prompt

    def test_gene_only_note_injected(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(confidence="gene_only", matches=_one_match()),
        )
        assert "gene_only" in prompt
        assert "not specific" in prompt.lower()

    def test_gene_hgvs_partial_note_injected(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(confidence="gene_hgvs_partial", matches=_one_match()),
        )
        assert "gene_hgvs_partial" in prompt
        assert "not specific" in prompt.lower()

    def test_exact_rsid_has_no_extra_note(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(confidence="exact_rsid", matches=_one_match()),
        )
        assert "IMPORTANT:" not in prompt


# ---------------------------------------------------------------------------
# Warnings rendering
# ---------------------------------------------------------------------------

class TestWarningsRendering:
    def test_warnings_block_shown(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(warnings=["rsID rs123 was not found in ClinVar."]),
        )
        assert "rsID rs123 was not found in ClinVar." in prompt

    def test_no_warnings_label(self):
        prompt = _format_upload_user_prompt(
            "q", _variant(), _clinvar_result(warnings=[])
        )
        assert "Warnings: none" in prompt

    def test_multiple_warnings_all_present(self):
        prompt = _format_upload_user_prompt(
            "q",
            _variant(),
            _clinvar_result(warnings=["Warning A", "Warning B", "Warning C"]),
        )
        assert "Warning A" in prompt
        assert "Warning B" in prompt
        assert "Warning C" in prompt


# ---------------------------------------------------------------------------
# UPLOAD_SYSTEM_PROMPT content requirements
# ---------------------------------------------------------------------------

class TestUploadSystemPrompt:
    def test_forbids_diagnosis(self):
        assert "diagnos" in UPLOAD_SYSTEM_PROMPT.lower()

    def test_forbids_treatment(self):
        assert "treatment" in UPLOAD_SYSTEM_PROMPT.lower()

    def test_forbids_personal_risk(self):
        assert "personal" in UPLOAD_SYSTEM_PROMPT.lower()

    def test_forbids_medical_action(self):
        assert "medical action" in UPLOAD_SYSTEM_PROMPT.lower()

    def test_requires_genetics_professional_referral(self):
        lower = UPLOAD_SYSTEM_PROMPT.lower()
        assert "genetics professional" in lower or "genetic counselor" in lower

    def test_requires_json_format(self):
        assert "JSON" in UPLOAD_SYSTEM_PROMPT

    def test_no_match_rule_present(self):
        # Check non-lowercased so mixed-case "ClinVar" is found correctly
        assert "no ClinVar records" in UPLOAD_SYSTEM_PROMPT or \
               "no records" in UPLOAD_SYSTEM_PROMPT

    def test_gene_only_rule_present(self):
        assert "gene_only" in UPLOAD_SYSTEM_PROMPT

    def test_mismatch_rule_present(self):
        assert "mismatch" in UPLOAD_SYSTEM_PROMPT.lower()

    def test_safety_disclaimer_verbatim_required(self):
        assert "educational purposes only" in UPLOAD_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Safe question examples — policy must not flag these
# ---------------------------------------------------------------------------

SAFE_QUESTIONS = [
    "What is the clinical significance of this variant in ClinVar?",
    "What conditions is this variant associated with?",
    "How many ClinVar submissions are there for this variant?",
    "What does 'Pathogenic' mean in this context?",
    "What is the review status for this ClinVar entry?",
    "Can you explain what a Variant of Uncertain Significance means?",
    "What gene is this variant located in?",
    "How confident is the ClinVar match for this variant?",
    "What is the difference between Likely Pathogenic and Pathogenic?",
    "Is this variant present in ClinVar at all?",
]

UNSAFE_QUESTIONS = [
    "Should I have surgery because of this variant?",
    "Should I start treatment based on this result?",
    "Can I stop treatment given this variant?",
    "What is my risk percentage of getting cancer?",
    "Should I have amniocentesis based on this finding?",
    "Is my family risk high for this condition?",
    "What oncology workup should I ask for?",
    "Does this mean I need CVS?",
]


class TestPolicySafeQuestions:
    def test_safe_questions_have_no_policy_flags(self):
        for q in SAFE_QUESTIONS:
            result = policy.check_question(q)
            assert result.allowed is True, f"allowed should be True for: {q!r}"
            assert result.flags == [], \
                f"Expected no flags for safe question {q!r}, got: {result.flags}"

    def test_safe_questions_have_no_redirect(self):
        for q in SAFE_QUESTIONS:
            result = policy.check_question(q)
            assert result.redirect_message is None, \
                f"Expected no redirect for safe question {q!r}"


class TestPolicyUnsafeQuestions:
    def test_unsafe_questions_produce_flags_or_redirect(self):
        for q in UNSAFE_QUESTIONS:
            result = policy.check_question(q)
            has_issue = bool(result.flags) or bool(result.redirect_message)
            assert has_issue, \
                f"Expected policy flags or redirect for unsafe question: {q!r}"

    def test_unsafe_questions_redirect_mentions_professional(self):
        for q in UNSAFE_QUESTIONS:
            result = policy.check_question(q)
            if result.redirect_message:
                lower = result.redirect_message.lower()
                has_ref = "genetic counselor" in lower or "physician" in lower
                assert has_ref, \
                    f"Redirect for {q!r} should mention a professional: {result.redirect_message!r}"

    def test_surgery_flagged(self):
        result = policy.check_question("Should I have surgery?")
        assert "surgery" in result.flags

    def test_amniocentesis_flagged(self):
        result = policy.check_question("Should I have amniocentesis?")
        assert "amniocentesis" in result.flags

    def test_risk_percentage_flagged(self):
        result = policy.check_question("What is my risk percentage?")
        assert "risk percentage" in result.flags

    def test_family_risk_flagged(self):
        result = policy.check_question("Is my family risk high?")
        assert "family risk" in result.flags


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------

class TestSanitisation:
    def test_strips_control_characters(self):
        dirty = "What is \x00this \x1f variant?"
        clean = policy.sanitise_question(dirty)
        assert "\x00" not in clean
        assert "\x1f" not in clean
        assert "What is" in clean

    def test_strips_leading_trailing_whitespace(self):
        assert policy.sanitise_question("  hello  ") == "hello"

    def test_preserves_normal_text(self):
        q = "What is the significance of BRCA1 c.68_69del?"
        assert policy.sanitise_question(q) == q


# ---------------------------------------------------------------------------
# ask_upload method wiring (mocked — no real API call)
# ---------------------------------------------------------------------------

class TestAskUploadMethod:
    """
    Verify that ask_upload passes UPLOAD_SYSTEM_PROMPT to _call_api
    and returns the parsed response.
    """

    def _make_client(self, response_json: str):
        """Return an AnthropicLLMClient with _call_api fully mocked."""
        from unittest.mock import MagicMock, patch
        from app.llm_client import AnthropicLLMClient
        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client._call_api = MagicMock(return_value=response_json)
        return client

    # Payload that satisfies _validate_upload_answer (5 section headers, 80+ words, no forbidden phrases)
    _VALID_UPLOAD_ANSWER = (
        "1. What was found in the report\n"
        "Gene BRCA1, variant c.68_69delAG, zygosity heterozygous.\n\n"
        "2. What ClinVar says\n"
        "ClinVar contains one record classified as Pathogenic with expert panel review. "
        "The match confidence is exact_rsid, meaning the rsID matched a ClinVar entry directly.\n\n"
        "3. What this classification means\n"
        "Pathogenic means the variant has been associated with a hereditary condition in ClinVar. "
        "This is a ClinVar classification based on submitted evidence.\n\n"
        "4. Key limitations\n"
        "ClinVar data reflects submitted evidence and may not be complete. "
        "Classification may change as new evidence emerges.\n\n"
        "5. Recommended next step\n"
        "Please discuss these findings with a certified genetics professional or genetic counselor "
        "before drawing any clinical conclusions from this information."
    )

    def test_ask_upload_uses_upload_system_prompt(self):
        import json
        from app.llm_client import UPLOAD_SYSTEM_PROMPT
        payload = {
            "answer": self._VALID_UPLOAD_ANSWER,
            "evidence": ["Gene: BRCA1 | Significance: Pathogenic"],
            "limitations": ["Gene-level only."],
            "safety_disclaimer": (
                "This information is for educational purposes only and does not "
                "constitute medical advice. Please consult a certified genetics "
                "professional or genetic counselor for clinical interpretation and "
                "personalized recommendations."
            ),
        }
        client = self._make_client(json.dumps(payload))

        result = client.ask_upload(
            "What is the significance?",
            _variant(),
            _clinvar_result(confidence="exact_rsid", matches=_one_match()),
        )

        # _call_api should have been called with UPLOAD_SYSTEM_PROMPT
        call_kwargs = client._call_api.call_args
        assert call_kwargs is not None
        # system_prompt is the second positional arg or keyword
        args, kwargs = call_kwargs
        used_prompt = kwargs.get("system_prompt") or (args[1] if len(args) > 1 else None)
        assert used_prompt == UPLOAD_SYSTEM_PROMPT

        # Result is parsed correctly
        assert result["answer"] == payload["answer"]
        assert isinstance(result["evidence"], list)
        assert "safety_disclaimer" in result

    def test_ask_upload_no_match_prompt_contains_warning(self):
        """The user prompt sent to the LLM must include the no-match instruction."""
        import json
        payload = {
            "answer": (
                "1. What was found in the report\n"
                "Gene BRCA1, variant c.68_69delAG.\n\n"
                "2. What ClinVar says\n"
                "No ClinVar records were found for this variant. "
                "No classification can be provided.\n\n"
                "3. What this classification means\n"
                "No ClinVar record means classification status is unknown for this variant. "
                "This does not indicate the variant is benign or pathogenic.\n\n"
                "4. Key limitations\n"
                "No ClinVar data available. Absence of a record does not imply benign status. "
                "The variant may exist in other databases.\n\n"
                "5. Recommended next step\n"
                "Please discuss these findings with a certified genetics professional or "
                "genetic counselor before drawing any clinical conclusions."
            ),
            "evidence": [],
            "limitations": ["No records in ClinVar for this variant."],
            "safety_disclaimer": (
                "This information is for educational purposes only and does not constitute "
                "medical advice. Please consult a certified genetics professional or genetic "
                "counselor for clinical interpretation and personalized recommendations."
            ),
        }
        client = self._make_client(json.dumps(payload))

        client.ask_upload(
            "What is the significance?",
            _variant(),
            _clinvar_result(confidence="no_match", matches=[]),
        )

        user_prompt_sent = client._call_api.call_args[0][0]
        assert "No ClinVar records were found" in user_prompt_sent
