"""
tests/test_unverified_draft_clinvar.py

Tests for the ClinVar-context unverified gene draft feature.

Covers:
  1. _DRAFT_QUALITY_RE rejects hype/biology-claim language
  2. _validate_unverified_draft accepts/rejects correctly
  3. _build_clinvar_context_block builds the right metadata block
  4. _generate_unverified_gene_draft — based_on / source_note_he fields
  5. LLM prompts explicitly forbid biology invention
  6. Draft does not contain a raw ClinVar phenotype list
  7. Existing test suite not broken (spot-check via import)

All LLM calls are mocked — no live server needed.
"""

import os
import pathlib
import sys
import importlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def _engine():
    import app.retriever as retriever
    retriever._DB_AVAILABLE = False
    import app.counseling_engine as ce
    importlib.reload(ce)
    return ce


_VALID_CLINVAR_SUMMARY = (
    "הגן POLE מופיע לעיתים בהקשרים של סרטן מעי גס ולסרטן אנדומטריום. "
    "ממצא VUS בגן זה נותר בגדר אי-ודאות ומשמעותו טרם הוברה."
)

_POLE_CLINVAR_CONTEXT = {
    "total_variants": 532,
    "top_phenotypes": [
        "Colorectal cancer",
        "Endometrial cancer",
        "Lynch syndrome",
        "Familial cancer of breast",
        "Hereditary cancer-predisposing syndrome",
    ],
    "significance_breakdown": {
        "Pathogenic": 45,
        "Likely pathogenic": 12,
        "Uncertain significance": 310,
        "Benign": 5,
        "Likely benign": 20,
    },
}


class _MockLLMClient:
    """Minimal stub that returns a preset response for both _call_api and call_text_raw."""

    def __init__(self, url="mock"):
        self.url = url
        self.calls = []

    def _call_api(self, user_content, system_prompt=None):
        self.calls.append({"user_content": user_content, "system_prompt": system_prompt})
        return _MockLLMClient._response

    def call_text_raw(self, user_content, system_prompt=None):
        self.calls.append({"user_content": user_content, "system_prompt": system_prompt})
        return _MockLLMClient._response

    _response = _VALID_CLINVAR_SUMMARY


# ── 1. _DRAFT_QUALITY_RE: hype and biology-claim rejection ───────────────────

class TestDraftQualityRegex:
    """_DRAFT_QUALITY_RE must block hype language and unsupported biology claims."""

    @pytest.fixture(autouse=True)
    def _load_re(self):
        ce = _engine()
        self.re = ce._DRAFT_QUALITY_RE

    def _matches(self, text):
        return bool(self.re.search(text))

    def test_rejects_kesem(self):
        assert self._matches("POLE גן מעורר עניין בשל מוטציות קסם-החיים שלו")

    def test_rejects_kesem_standalone(self):
        assert self._matches("יש לו מוטציות קסם")

    def test_rejects_meourer_inyan(self):
        assert self._matches("גן מעורר עניין ביותר")

    def test_rejects_marthik(self):
        assert self._matches("המחקר מרתק")

    def test_rejects_madim(self):
        assert self._matches("תגלית מדהימה")

    def test_rejects_hagen_ose(self):
        assert self._matches("הגן עושה תהליכי תיקון DNA")

    def test_rejects_hagen_achrai(self):
        assert self._matches("הגן אחראי לתיקון שגיאות בDNA")

    def test_rejects_mekoded_le(self):
        assert self._matches("הגן מקודד לחלבון פולימראז")

    def test_rejects_meshatef_batahlich(self):
        assert self._matches("הגן משתתף בתהליך שכפול DNA")

    def test_rejects_meyatzer_halbon(self):
        assert self._matches("הגן מייצר חלבון חשוב")

    def test_rejects_halbon_prefix(self):
        assert self._matches("חלבון הפולימראז")

    def test_accepts_valid_clinvar_summary(self):
        assert not self._matches(_VALID_CLINVAR_SUMMARY)

    def test_accepts_vus_uncertainty_sentence(self):
        text = "ממצא VUS בגן POLE נותר בגדר אי-ודאות ואינו מהווה אבחנה."
        assert not self._matches(text)

    def test_rejects_clinvar_mention(self):
        # ClinVar is detected by _CLINVAR_IN_DRAFT_RE (separate from _DRAFT_QUALITY_RE)
        # so use the combined validator to verify ClinVar is still rejected
        from app.counseling_engine import _validate_unverified_draft
        text = "גן POLE מדווח במאגר ClinVar בהקשרים קשורים לסרטן מעי גס ולסרטן אנדומטריום."
        assert _validate_unverified_draft(text) is not None

    def test_accepts_contexts_sentence_without_clinvar(self):
        text = "גן POLE מופיע לעיתים בהקשרים קשורים לסרטן מעי גס ולסרטן אנדומטריום."
        assert not self._matches(text)


# ── 2. _validate_unverified_draft ────────────────────────────────────────────

class TestValidateUnverifiedDraft:
    """_validate_unverified_draft must accept valid and reject invalid text."""

    @pytest.fixture(autouse=True)
    def _load_fn(self):
        ce = _engine()
        self.validate = ce._validate_unverified_draft

    def test_valid_text_passes(self):
        assert self.validate(_VALID_CLINVAR_SUMMARY) is None

    def test_empty_text_rejected(self):
        assert self.validate("") is not None

    def test_only_spaces_rejected(self):
        assert self.validate("   ") is not None

    def test_too_long_text_rejected(self):
        text = "א" * 601
        assert self.validate(text) is not None

    def test_hype_kesem_rejected(self):
        assert self.validate("POLE גן מעורר עניין בשל מוטציות קסם-החיים שלו.") is not None

    def test_biology_claim_rejected(self):
        assert self.validate("הגן מקודד לחלבון פולימראז אפסילון.") is not None

    def test_no_hebrew_rejected(self):
        assert self.validate("POLE gene has many variants in ClinVar.") is not None

    def test_question_mark_rejected(self):
        assert self.validate("האם הגן POLE מסוכן?") is not None

    def test_single_valid_sentence_passes(self):
        text = "הגן POLE מופיע לעיתים בהקשרים הקשורים לסרטן."
        assert self.validate(text) is None


# ── 3. _build_clinvar_context_block ──────────────────────────────────────────

class TestBuildClinvarContextBlock:
    """_build_clinvar_context_block must produce the right structured text."""

    @pytest.fixture(autouse=True)
    def _load_fn(self):
        ce = _engine()
        self.build = ce._build_clinvar_context_block

    def test_contains_gene_symbol(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "Gene: POLE" in block

    def test_contains_total_variants(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "532" in block

    def test_contains_pathogenic_count(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        # 45 + 12 = 57 pathogenic/likely pathogenic
        assert "57" in block

    def test_contains_vus_count(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "310" in block

    def test_contains_phenotype_item(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        # Phenotypes are now normalized to Hebrew labels
        assert "נטייה לסרטן המעי הגס" in block or "נטייה לסרטן הרחם" in block

    def test_caps_phenotypes_at_five(self):
        context = dict(_POLE_CLINVAR_CONTEXT)
        # Use real phenotype names that normalize to Hebrew; 6th should be excluded
        context["top_phenotypes"] = [
            "Colorectal cancer",
            "Endometrial cancer",
            "Breast cancer",
            "Ovarian cancer",
            "Pancreatic cancer",
            "Thyroid cancer",  # 6th — must be excluded
        ]
        block = self.build("POLE", context)
        assert "נטייה לסרטן בלוטת התריס" not in block
        assert "נטייה לסרטן הלבלב" in block

    def test_instruction_line_forbids_biology(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "Do NOT describe" in block

    def test_alternative_key_names_total(self):
        ctx = {"total": 100, "phenotypes": ["Breast cancer"], "by_significance": {}}
        block = self.build("BRCA1", ctx)
        assert "100" in block
        # Phenotype normalized to Hebrew
        assert "נטייה לסרטן השד" in block

    def test_empty_context_graceful(self):
        block = self.build("TTN", {})
        assert "Gene: TTN" in block


# ── 4. _generate_unverified_gene_draft: based_on / source_note_he ────────────

class TestGenerateUnverifiedGeneDraftFields:
    """Draft dict must carry correct provenance fields."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.ce = _engine()
        _MockLLMClient._response = _VALID_CLINVAR_SUMMARY
        monkeypatch.setattr(self.ce, "create_llm_client", lambda: _MockLLMClient())

    def test_with_clinvar_context_based_on_is_clinvar_metadata(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["based_on"] == "clinvar_metadata"

    def test_with_clinvar_context_has_source_note_he(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert "source_note_he" in draft
        assert draft["source_note_he"]  # non-empty

    def test_with_clinvar_context_source_note_he_matches_constant(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["source_note_he"] == self.ce._UNVERIFIED_DRAFT_SOURCE_NOTE_HE

    def test_without_clinvar_context_based_on_is_llm_knowledge(self):
        draft = self.ce._generate_unverified_gene_draft("POLE")
        assert draft is not None
        assert draft["based_on"] == "llm_knowledge"

    def test_without_clinvar_context_no_source_note_he(self):
        draft = self.ce._generate_unverified_gene_draft("POLE")
        assert draft is not None
        assert "source_note_he" not in draft

    def test_approved_is_always_false(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["approved"] is False

    def test_review_status_is_unreviewed(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["review_status"] == "unreviewed"

    def test_status_is_ai_generated_unreviewed(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["status"] == "ai_generated_unreviewed"


# ── 5. Hype text → draft rejected → returns None ─────────────────────────────

class TestDraftRejectedOnHypeText:
    """When LLM returns hype/biology text, draft is rejected and None returned."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.ce = _engine()
        monkeypatch.setattr(self.ce, "create_llm_client", lambda: _MockLLMClient())

    def test_kesem_chayyim_rejected(self):
        _MockLLMClient._response = "POLE גן מעורר עניין בשל מוטציות קסם-החיים שלו."
        # No clinvar_context → no deterministic fallback → must return None
        draft = self.ce._generate_unverified_gene_draft("POLE")
        assert draft is None

    def test_meourer_inyan_rejected(self):
        _MockLLMClient._response = "הגן POLE מעורר עניין רב בקהילה המחקרית."
        draft = self.ce._generate_unverified_gene_draft("POLE")
        assert draft is None

    def test_biology_claim_rejected(self):
        _MockLLMClient._response = "הגן POLE מקודד לחלבון פולימראז שמתקן שגיאות DNA."
        draft = self.ce._generate_unverified_gene_draft("POLE")
        assert draft is None

    def test_hagen_ose_rejected(self):
        _MockLLMClient._response = "הגן עושה תיקון DNA בתאים."
        draft = self.ce._generate_unverified_gene_draft("POLE")
        assert draft is None

    def test_hype_rejected_with_clinvar_context_returns_deterministic_fallback(self):
        """When LLM text is rejected but ClinVar context available, returns deterministic draft."""
        _MockLLMClient._response = "הגן POLE מקודד לחלבון פולימראז שמתקן שגיאות DNA."
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["status"] == "deterministic_clinvar_summary"
        assert draft["based_on"] == "clinvar_metadata"
        assert draft["generated_by_model"] == "deterministic"


# ── 6. Prompts explicitly forbid biology invention ────────────────────────────

class TestClinvarDraftPromptsForbidBiology:
    """System prompts for the ClinVar path must forbid biology claims."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.ce = _engine()

    def test_clinvar_prompt_forbids_protein(self):
        p = self.ce._UNVERIFIED_CLINVAR_DRAFT_SYSTEM_PROMPT
        assert "protein" in p.lower() or "encodes" in p.lower()

    def test_clinvar_prompt_forbids_biology_invention(self):
        p = self.ce._UNVERIFIED_CLINVAR_DRAFT_SYSTEM_PROMPT
        assert "NOT" in p or "MUST NOT" in p

    def test_clinvar_prompt_forbids_pathway(self):
        p = self.ce._UNVERIFIED_CLINVAR_DRAFT_SYSTEM_PROMPT
        assert "pathway" in p.lower() or "mechanism" in p.lower()

    def test_clinvar_retry_prompt_forbids_biology(self):
        p = self.ce._UNVERIFIED_CLINVAR_DRAFT_RETRY_SYSTEM_PROMPT
        assert "biology" in p.lower() or "protein" in p.lower() or "mechanism" in p.lower()

    def test_clinvar_prompt_mentions_vus_uncertain(self):
        p = self.ce._UNVERIFIED_CLINVAR_DRAFT_SYSTEM_PROMPT
        assert "uncertain" in p.lower() or "VUS" in p

    def test_source_note_he_does_not_mention_clinvar(self):
        # Patient-facing source note must not name "ClinVar" directly
        note = self.ce._UNVERIFIED_DRAFT_SOURCE_NOTE_HE
        assert "ClinVar" not in note

    def test_source_note_he_not_personal_interpretation(self):
        note = self.ce._UNVERIFIED_DRAFT_SOURCE_NOTE_HE
        assert "אישי" in note or "אישית" in note


# ── 7. Draft text vs raw ClinVar list ────────────────────────────────────────

class TestRawClinvarListNotInDraftText:
    """The draft text_he must not be a verbatim dump of ClinVar phenotype names."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.ce = _engine()
        _MockLLMClient._response = _VALID_CLINVAR_SUMMARY
        monkeypatch.setattr(self.ce, "create_llm_client", lambda: _MockLLMClient())

    def test_draft_text_he_is_not_raw_phenotype_list(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        text = draft["text_he"]
        # Raw phenotype enumeration would have most phenotype names joined verbatim;
        # a summarised Hebrew sentence will not.
        raw_joined = " | ".join(_POLE_CLINVAR_CONTEXT["top_phenotypes"])
        assert raw_joined not in text

    def test_context_block_contains_normalized_hebrew_phenotypes_for_llm(self):
        block = self.ce._build_clinvar_context_block("POLE", _POLE_CLINVAR_CONTEXT)
        # Phenotypes are normalized to Hebrew in the context block
        assert "נטייה לסרטן המעי הגס" in block or "נטייה לסרטן הרחם" in block

    def test_draft_text_he_is_in_hebrew(self):
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        heb_chars = [c for c in draft["text_he"] if "א" <= c <= "׿"]
        assert len(heb_chars) > 5


# ── 8. No LOCAL_LLM_URL → returns None (no crash) ────────────────────────────

class TestNollmUrlReturnsNone:
    """When no LLM is configured, _generate_unverified_gene_draft must return None."""

    def test_returns_none_when_no_llm_configured(self, monkeypatch):
        ce = _engine()
        def _no_llm():
            raise ValueError("No LLM configured")
        monkeypatch.setattr(ce, "create_llm_client", _no_llm)
        draft = ce._generate_unverified_gene_draft("POLE", clinvar_context=_POLE_CLINVAR_CONTEXT)
        assert draft is None


# ── 9. Phenotype normalization ────────────────────────────────────────────────

class TestPhenotypeNormalization:
    """_normalize_clinvar_phenotypes_for_patient and _map_phenotype_to_hebrew."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.ce = _engine()
        self.norm = self.ce._normalize_clinvar_phenotypes_for_patient
        self.map1 = self.ce._map_phenotype_to_hebrew

    # _map_phenotype_to_hebrew
    def test_colorectal_cancer_maps_to_hebrew(self):
        result = self.map1("Colorectal cancer")
        assert result == "נטייה לסרטן המעי הגס"

    def test_lynch_syndrome_maps_to_hebrew(self):
        result = self.map1("Lynch syndrome")
        assert result == "תסמונת לינץ׳"

    def test_not_provided_maps_to_none(self):
        assert self.map1("not provided") is None

    def test_not_specified_maps_to_none(self):
        assert self.map1("not specified") is None

    def test_unrecognised_entry_maps_to_none(self):
        assert self.map1("Some rare obscure condition XYZ") is None

    def test_gene_specific_related_disorder(self):
        result = self.map1("POLE-related disorder", gene="POLE")
        assert result == "מצבים הקשורים לגן POLE"

    def test_gene_specific_associated_syndrome(self):
        result = self.map1("MSH2-associated syndrome", gene="MSH2")
        assert result == "מצבים הקשורים לגן MSH2"

    def test_case_insensitive_match(self):
        result = self.map1("colorectal CANCER")
        assert result == "נטייה לסרטן המעי הגס"

    # _normalize_clinvar_phenotypes_for_patient
    def test_pole_phenotype_list_normalizes_to_hebrew(self):
        result = self.norm(
            ["Colorectal cancer", "Hereditary cancer-predisposing syndrome"],
            "POLE",
        )
        assert "נטייה לסרטן המעי הגס" in result
        assert "נטייה תורשתית למצבים סרטניים" in result

    def test_semicolon_split_yields_multiple_labels(self):
        result = self.norm(["Lynch syndrome;Familial adenomatous polyposis"], "MSH2")
        assert "תסמונת לינץ׳" in result
        assert "פוליפוזיס אדנומטוטית משפחתית" in result

    def test_not_provided_filtered_out(self):
        result = self.norm(["not provided", "Breast cancer"], "BRCA1")
        assert "נטייה לסרטן השד" in result
        assert len(result) == 1

    def test_deduplication_preserves_order(self):
        result = self.norm(
            ["Colorectal cancer", "Colorectal cancer", "Breast cancer"], "GENE"
        )
        assert result.count("נטייה לסרטן המעי הגס") == 1
        assert result[0] == "נטייה לסרטן המעי הגס"

    def test_returns_at_most_five(self):
        result = self.norm(
            [
                "Colorectal cancer",
                "Endometrial cancer",
                "Breast cancer",
                "Ovarian cancer",
                "Pancreatic cancer",
                "Thyroid cancer",
                "Prostate cancer",
            ],
            "GENE",
        )
        assert len(result) == 5

    def test_long_entry_skipped(self):
        long_entry = "A" * 121
        result = self.norm([long_entry, "Breast cancer"], "GENE")
        assert result == ["נטייה לסרטן השד"]

    def test_empty_list_returns_empty(self):
        assert self.norm([], "GENE") == []

    def test_all_uninformative_returns_empty(self):
        result = self.norm(["not provided", "not specified", "see cases"], "GENE")
        assert result == []

    def test_hereditary_cancer_predisposing_maps_correctly(self):
        result = self.norm(["Hereditary cancer-predisposing syndrome"], "POLE")
        assert "נטייה תורשתית למצבים סרטניים" in result

    def test_polymerase_polyposis_maps_to_specific_label(self):
        result = self.norm(
            ["Polymerase proofreading-related adenomatous polyposis"], "POLE"
        )
        assert "פוליפוזיס אדנומטוטית הקשורה למנגנון הגהה של שכפול DNA" in result


# ── 10. Mixed Hebrew/Latin rejection ─────────────────────────────────────────

class TestMixedHebrewLatinRejection:
    """_validate_unverified_draft must reject mixed Hebrew/Latin token drafts."""

    @pytest.fixture(autouse=True)
    def _load(self):
        ce = _engine()
        self.validate = ce._validate_unverified_draft

    def _rejected(self, text):
        return self.validate(text) is not None

    def test_rejects_sindromot_mixed(self):
        assert self._rejected("סינדromות פREDISפוזיציה")

    def test_rejects_predis_in_uppercase(self):
        assert self._rejected("פREDISפוזיצIONיות")

    def test_rejects_yrachayim_wrong_translation(self):
        assert self._rejected("סרטן הירכיים")

    def test_rejects_polyposis_garbled(self):
        assert self._rejected("פוליאפולויפוזיס")

    def test_rejects_chronic_garbled(self):
        assert self._rejected("כרוניקליים")

    def test_rejects_predis_fragment(self):
        assert self._rejected("predisposition to various conditions")

    def test_rejects_redis_fragment(self):
        assert self._rejected("redis pattern")

    def test_rejects_hebrew_immediately_followed_by_latin(self):
        assert self._rejected("גןPOLE מדווח במאגר")

    def test_rejects_latin_immediately_followed_by_hebrew(self):
        assert self._rejected("GENEגן מדווח")

    def test_rejects_too_many_non_gene_latin_words(self):
        # "mixed" and "token" are non-gene Latin words (count = 2 > 1)
        assert self._rejected("mixed בעיה token")

    def test_accepts_gene_symbol_adjacent_with_space(self):
        # Gene symbols separated by space are fine (without ClinVar brand name)
        assert not self._rejected("הגן POLE מופיע לעיתים בהקשרים של סרטן.")

    def test_accepts_clinvar_as_latin(self):
        # "ClinVar" is whitelisted, should not count as a non-gene Latin word
        ce = _engine()
        count = ce._count_non_gene_latin_words("הגן POLE מדווח במאגר ClinVar.")
        assert count == 0

    def test_count_non_gene_latin_words_excludes_gene_symbols(self):
        ce = _engine()
        count = ce._count_non_gene_latin_words("BRCA1 POLE MLH1 הם גנים.")
        assert count == 0

    def test_count_non_gene_latin_words_counts_lowercase(self):
        ce = _engine()
        count = ce._count_non_gene_latin_words("the gene encodes something.")
        assert count == 4  # "the", "gene", "encodes", "something"


# ── 11. Deterministic fallback ────────────────────────────────────────────────

class TestDeterministicFallback:
    """_build_deterministic_clinvar_draft and its use as a fallback."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.ce = _engine()
        self.build_det = self.ce._build_deterministic_clinvar_draft
        self.validate = self.ce._validate_unverified_draft

    def test_returns_none_for_empty_contexts(self):
        assert self.build_det("POLE", []) is None

    def test_returns_dict_for_non_empty_contexts(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert isinstance(result, dict)

    def test_based_on_is_clinvar_metadata(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert result["based_on"] == "clinvar_metadata"

    def test_approved_is_false(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert result["approved"] is False

    def test_status_is_deterministic_clinvar_summary(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert result["status"] == "deterministic_clinvar_summary"

    def test_generated_by_model_is_deterministic(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert result["generated_by_model"] == "deterministic"

    def test_review_status_is_unreviewed(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert result["review_status"] == "unreviewed"

    def test_text_passes_validate(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert self.validate(result["text_he"]) is None

    def test_text_contains_gene_symbol(self):
        result = self.build_det("BRCA1", ["נטייה לסרטן השד"])
        assert "BRCA1" in result["text_he"]

    def test_text_contains_context_label(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert "נטייה לסרטן המעי הגס" in result["text_he"]

    def test_two_contexts_joined_with_vav(self):
        result = self.build_det(
            "POLE", ["נטייה לסרטן המעי הגס", "נטייה תורשתית למצבים סרטניים"]
        )
        # Two contexts joined with "ו" conjunction
        assert "ו" in result["text_he"]

    def test_three_contexts_joined(self):
        result = self.build_det(
            "POLE",
            ["נטייה לסרטן המעי הגס", "נטייה לסרטן הרחם", "תסמונת לינץ׳"],
        )
        assert "נטייה לסרטן המעי הגס" in result["text_he"]
        assert "נטייה לסרטן הרחם" in result["text_he"]

    def test_has_warning_he_field(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert "warning_he" in result
        assert result["warning_he"]

    def test_has_source_note_he_field(self):
        result = self.build_det("POLE", ["נטייה לסרטן המעי הגס"])
        assert "source_note_he" in result

    def test_uses_only_first_three_contexts(self):
        contexts = [
            "נטייה לסרטן המעי הגס",
            "נטייה לסרטן הרחם",
            "תסמונת לינץ׳",
            "נטייה לסרטן השד",  # 4th — not in text
        ]
        result = self.build_det("POLE", contexts)
        assert "נטייה לסרטן השד" not in result["text_he"]

    def test_fallback_returned_when_llm_fails_and_clinvar_available(self, monkeypatch):
        """Integration: LLM returns bad text → deterministic fallback returned."""
        _MockLLMClient._response = "הגן POLE מקודד לחלבון פולימראז."  # rejected
        monkeypatch.setattr(self.ce, "create_llm_client", lambda: _MockLLMClient())
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=_POLE_CLINVAR_CONTEXT
        )
        assert draft is not None
        assert draft["status"] == "deterministic_clinvar_summary"

    def test_fallback_not_returned_when_clinvar_context_has_no_normalisable_phenotypes(
        self, monkeypatch
    ):
        """If ClinVar phenotypes don't normalise to Hebrew, fallback is None."""
        _MockLLMClient._response = "הגן POLE מקודד לחלבון פולימראז."  # rejected
        monkeypatch.setattr(self.ce, "create_llm_client", lambda: _MockLLMClient())
        empty_ctx = {"total_variants": 5, "top_phenotypes": ["not provided", "see cases"]}
        draft = self.ce._generate_unverified_gene_draft(
            "POLE", clinvar_context=empty_ctx
        )
        assert draft is None


# ── 12. Context block uses normalized Hebrew (not raw English) ─────────────────

class TestContextBlockUsesNormalizedHebrew:
    """_build_clinvar_context_block must output Hebrew phenotype labels, not raw English."""

    @pytest.fixture(autouse=True)
    def _load(self):
        ce = _engine()
        self.build = ce._build_clinvar_context_block

    def test_does_not_contain_raw_colorectal_cancer(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "Colorectal cancer" not in block

    def test_does_not_contain_raw_endometrial_cancer(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "Endometrial cancer" not in block

    def test_does_not_contain_hereditary_cancer_predisposing_raw(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "Hereditary cancer-predisposing syndrome" not in block

    def test_contains_hebrew_colorectal(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "נטייה לסרטן המעי הגס" in block

    def test_contains_hebrew_endometrial(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "נטייה לסרטן הרחם" in block

    def test_contains_hebrew_hereditary_cancer(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "נטייה תורשתית למצבים סרטניים" in block

    def test_contains_hebrew_instruction_phrase(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "נסח רק על בסיס ההקשרים העבריים שסופקו" in block

    def test_contains_no_raw_translate_instruction(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "אל תתרגם מחדש מונחים באנגלית גולמיים" in block

    def test_not_provided_not_in_block(self):
        ctx = {
            "total_variants": 10,
            "top_phenotypes": ["not provided", "Breast cancer"],
        }
        block = self.build("BRCA1", ctx)
        assert "not provided" not in block
        assert "נטייה לסרטן השד" in block

    def test_block_section_header_in_hebrew(self):
        block = self.build("POLE", _POLE_CLINVAR_CONTEXT)
        assert "הקשרים קליניים מדווחים" in block
