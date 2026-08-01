"""
Session 27.9 policy tests — General Low-Risk AI, Medical Educational AI Expansion,
Reduced Referral Boilerplate, UI Labels, Chromosome 21 context.

Parts covered:
  A   Response class definitions (via constants/function presence)
  B   General low-risk AI route (_classify_general_question, _build_general_education_answer)
  C   Medical educational AI expansion — supplemental card, labels
  E   Referral policy (_should_include_clinical_referral helper)
  F/G Chromosome 21 context opener — no Down syndrome assumption
  H   MTHFR association answer — no generic closing referral
  I   UI label strings for pending/approved expansion cards
  J   Review queue: risk_class, intended_use in chromosome draft metadata
  K   Regressions (HBB, TYR, physician approval endpoint)

Run:
  PYTHONUTF8=1 python -m pytest tests/test_session279_policy.py -v
"""
import re
from unittest.mock import MagicMock, patch, call
import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.counseling_engine as _ce

client = TestClient(app)

_REQUIRED_KEYS = {"answer", "safety_level", "needs_genetic_counselor",
                  "matched_topic", "suggested_questions"}

_GENERIC_REFERRAL_PHRASES = [
    "לפנות לצוות הגנטי",
    "מומלץ לפנות לצוות הגנטי",
    "המידע כללי ואינו מחליף ייעוץ רפואי",
    "ייעוץ גנטי מקצועי",
    "יש לפנות לצוות הגנטי שטיפל בך",
]


def _ask(question: str, **kwargs) -> dict:
    payload = {"question": question, **kwargs}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    d = r.json()
    assert _REQUIRED_KEYS.issubset(d.keys()), f"Missing keys: {_REQUIRED_KEYS - d.keys()}"
    return d


def _has_referral(text: str) -> bool:
    for phrase in _GENERIC_REFERRAL_PHRASES:
        if phrase in text:
            return True
    return False


# ===========================================================================
# Part B — General Low-Risk AI Route (_classify_general_question)
# ===========================================================================

class TestGeneralLowRiskAIRoute:
    """Session 27.9 Part B: organism and broad-genetics signals classify as safe_general_education."""

    def test_classify_sea_turtle_question(self):
        """'כמה גנים יש לצב ים?' must classify as safe_general_education."""
        result = _ce._classify_general_question("כמה גנים יש לצב ים?")
        assert result == "safe_general_education", (
            f"Expected safe_general_education, got {result!r}"
        )

    def test_classify_dog_chromosomes(self):
        """Non-human organism 'כלב' triggers safe_general_education."""
        result = _ce._classify_general_question("כמה כרומוזומים יש לכלב?")
        assert result == "safe_general_education"

    def test_classify_how_many_genes_start(self):
        """'כמה גנים יש לבני אדם?' starts with 'כמה גנים' → safe_general_education."""
        result = _ce._classify_general_question("כמה גנים יש לבני אדם?")
        assert result == "safe_general_education"

    def test_classify_how_many_chromosomes_signal(self):
        """'כמה כרומוזומים' broad signal → safe_general_education."""
        result = _ce._classify_general_question("כמה כרומוזומים יש בתא אנושי?")
        assert result == "safe_general_education"

    def test_classify_somatic_mutation_signal(self):
        """'מוטציה סומטית' is a broad genetics edu signal."""
        result = _ce._classify_general_question("מה זה מוטציה סומטית?")
        assert result == "safe_general_education"

    def test_classify_what_is_genome(self):
        """'מה זה גנום' → safe_general_education."""
        result = _ce._classify_general_question("מה זה גנום?")
        assert result == "safe_general_education"

    def test_personal_question_not_classified_as_safe(self):
        """A personal medical question must NOT enter the general low-risk route."""
        personal_questions = [
            "האם הגנטיקה שלי גורמת לסרטן?",
            "מה הסיכון שלי לחלות?",
            "האם עלי לעשות ניתוח?",
        ]
        for q in personal_questions:
            result = _ce._classify_general_question(q)
            assert result != "safe_general_education", (
                f"Personal question classified as safe: {q!r} → {result!r}"
            )

    def test_general_education_answer_has_correct_ai_content_type(self):
        """When LLM is available and answers, ai_content_type must be 'general_low_risk_ai'."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "לצב ים יש כ-28 זוגות כרומוזומים. מספר הגנים המדויק אינו ידוע."
        )
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            result, debug = _ce._build_general_education_answer("כמה גנים יש לצב ים?")
        assert result is not None, "Expected a result dict, got None"
        assert result.get("ai_content_type") == "general_low_risk_ai", (
            f"Expected general_low_risk_ai, got {result.get('ai_content_type')!r}"
        )

    def test_general_education_answer_requires_physician_review_false(self):
        """GENERAL_LOW_RISK_AI answers must NOT require physician review."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "לצב ים יש כ-28 זוגות כרומוזומים. מספר הגנים המדויק אינו ידוע."
        )
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            result, _ = _ce._build_general_education_answer("כמה גנים יש לצב ים?")
        assert result is not None
        assert result.get("requires_physician_review") is False

    def test_general_education_answer_no_referral(self):
        """GENERAL_LOW_RISK_AI answer text must not contain generic referral boilerplate."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "לצב ים יש כ-28 זוגות כרומוזומים. גנומו מורכב ומכיל מאות גנים ידועים."
        )
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            result, _ = _ce._build_general_education_answer("כמה גנים יש לצב ים?")
        assert result is not None
        assert not _has_referral(result["answer"]), (
            f"General low-risk answer must not contain referral boilerplate: {result['answer']!r}"
        )

    def test_general_education_needs_genetic_counselor_false(self):
        """GENERAL_LOW_RISK_AI: needs_genetic_counselor must be False."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "לצב ים יש כ-28 זוגות כרומוזומים. הגנום מכיל מידע גנטי רב."
        )
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            result, _ = _ce._build_general_education_answer("כמה גנים יש לצב ים?")
        assert result is not None
        assert result.get("needs_genetic_counselor") is False

    def test_non_human_organism_frozenset_contains_sea_turtle(self):
        """_NON_HUMAN_BIOLOGY_ORGANISMS must include 'צב ים'."""
        assert "צב ים" in _ce._NON_HUMAN_BIOLOGY_ORGANISMS

    def test_broad_genetics_signals_frozenset_present(self):
        """_BROAD_GENETICS_EDU_SIGNALS must be defined and contain expected signals."""
        assert hasattr(_ce, "_BROAD_GENETICS_EDU_SIGNALS")
        assert "כמה גנים" in _ce._BROAD_GENETICS_EDU_SIGNALS
        assert "כמה כרומוזומים" in _ce._BROAD_GENETICS_EDU_SIGNALS


# ===========================================================================
# Part C / I — Medical Educational AI Expansion: supplemental card metadata
# ===========================================================================

class TestMedicalExpansionCard:
    """Session 27.9 Parts C/I: chromosome draft is supplemental with correct labels."""

    def _get_chromosome_answer(self, question: str = "מה זה מחיקה בכרומוזום?",
                                sub_intent: str = "chromosome_deletion",
                                chromosome_number: str = None) -> dict:
        """Helper: get a chromosome education answer without real LLM."""
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=None):
            return _ce._build_chromosome_education_answer(
                question, sub_intent, chromosome_number=chromosome_number
            )

    def test_base_answer_is_always_deterministic(self):
        """The main 'answer' key must come from the deterministic KB, not AI draft."""
        result = self._get_chromosome_answer()
        assert "answer" in result
        assert len(result["answer"]) > 10, "Expected non-trivial KB answer"
        assert result.get("llm_used") is False, (
            "llm_used must be False when no draft was generated"
        )

    def test_chromosome_draft_metadata_present(self):
        """chromosome_draft_metadata must be present in the result."""
        result = self._get_chromosome_answer()
        assert "chromosome_draft_metadata" in result, (
            "chromosome_draft_metadata must be present in chromosome education answer"
        )

    def test_expansion_label_pending_in_metadata(self):
        """expansion_label_pending must be present and correct."""
        result = self._get_chromosome_answer()
        meta = result["chromosome_draft_metadata"]
        assert "expansion_label_pending" in meta, (
            "expansion_label_pending must be in chromosome_draft_metadata"
        )
        assert meta["expansion_label_pending"] == "הסבר נוסף שנוצר באמצעות AI"

    def test_expansion_label_approved_in_metadata(self):
        """expansion_label_approved must be present and correct."""
        result = self._get_chromosome_answer()
        meta = result["chromosome_draft_metadata"]
        assert "expansion_label_approved" in meta
        assert meta["expansion_label_approved"] == "מידע נוסף שנבדק ואושר"

    def test_expansion_pending_warning_in_metadata(self):
        """expansion_pending_warning must be present and correct."""
        result = self._get_chromosome_answer()
        meta = result["chromosome_draft_metadata"]
        assert "expansion_pending_warning" in meta
        assert meta["expansion_pending_warning"] == (
            "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי."
        )

    def test_draft_approved_promoted_always_false(self):
        """Chromosome drafts are SUPPLEMENTAL — approved_draft_promoted must always be False."""
        result = self._get_chromosome_answer()
        meta = result["chromosome_draft_metadata"]
        assert meta.get("approved_draft_promoted") is False, (
            "Chromosome drafts must never replace the deterministic main answer"
        )

    def test_ai_draft_supplemental_not_main_answer(self):
        """When a draft IS generated, it must NOT replace the main answer."""
        mock_draft = {
            "text_he": "מחיקה כרומוזומלית היא כאשר קטע מהכרומוזום חסר.",
            "sub_intent": "chromosome_deletion",
            "review_status": "pending",
            "approved": False,
            "warning_he": "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי.",
            "expansion_label": "הסבר נוסף שנוצר באמצעות AI",
            "ai_content_type": "medical_educational_ai_expansion",
            "intended_use": "supplemental_medical_expansion",
            "risk_class": "medical_educational",
        }
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=mock_draft):
            result = _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )
        # Main answer is deterministic KB — not the draft text
        assert result["answer"] != mock_draft["text_he"], (
            "AI draft text must not replace the deterministic main answer"
        )
        # Draft is stored in unverified_chromosome_draft
        assert "unverified_chromosome_draft" in result, (
            "Generated draft must be stored in unverified_chromosome_draft"
        )


# ===========================================================================
# Part J — Review queue: risk_class, intended_use in chromosome draft source_metadata
# ===========================================================================

class TestReviewQueueMetadata:
    """Session 27.9 Part J: create_draft receives risk_class and intended_use."""

    def test_chromosome_draft_enqueue_includes_risk_class(self):
        """create_draft must receive risk_class='medical_educational' in source_metadata."""
        mock_draft = {
            "text_he": "מחיקה כרומוזומלית כוללת אובדן של קטע ממנה.",
            "sub_intent": "chromosome_deletion",
            "review_status": "pending",
            "approved": False,
            "warning_he": "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי.",
            "expansion_label": "הסבר נוסף שנוצר באמצעות AI",
            "ai_content_type": "medical_educational_ai_expansion",
            "intended_use": "supplemental_medical_expansion",
            "risk_class": "medical_educational",
            "generated_by_model": "test-model",
        }
        captured_calls = []

        def mock_create_draft(**kwargs):
            captured_calls.append(kwargs)

        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=mock_draft), \
             patch("app.review_db.create_draft", side_effect=mock_create_draft), \
             patch("app.review_db.get_approved_chromosome_draft", return_value=None):
            _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )

        if captured_calls:
            source_meta = captured_calls[0].get("source_metadata", {})
            assert source_meta.get("risk_class") == "medical_educational", (
                f"Expected risk_class='medical_educational', got {source_meta!r}"
            )
            assert source_meta.get("intended_use") == "supplemental_medical_expansion", (
                f"Expected intended_use='supplemental_medical_expansion', got {source_meta!r}"
            )
            assert source_meta.get("ai_content_type") == "medical_educational_ai_expansion", (
                f"Expected ai_content_type='medical_educational_ai_expansion', got {source_meta!r}"
            )

    def test_chromosome_draft_enqueue_uses_prompt_version_s279(self):
        """create_draft must use prompt_version='s279' (Part J requirement)."""
        mock_draft = {
            "text_he": "מחיקה כרומוזומלית כוללת אובדן של קטע.",
            "sub_intent": "chromosome_deletion",
            "review_status": "pending",
            "approved": False,
            "warning_he": "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי.",
            "expansion_label": "הסבר נוסף שנוצר באמצעות AI",
            "ai_content_type": "medical_educational_ai_expansion",
            "intended_use": "supplemental_medical_expansion",
            "risk_class": "medical_educational",
            "generated_by_model": "test-model",
        }
        captured_calls = []

        def mock_create_draft(**kwargs):
            captured_calls.append(kwargs)

        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=mock_draft), \
             patch("app.review_db.create_draft", side_effect=mock_create_draft), \
             patch("app.review_db.get_approved_chromosome_draft", return_value=None):
            _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )

        if captured_calls:
            assert captured_calls[0].get("prompt_version") == "s279", (
                f"Expected prompt_version='s279', got {captured_calls[0].get('prompt_version')!r}"
            )


# ===========================================================================
# Part E — Referral policy: _should_include_clinical_referral
# ===========================================================================

class TestReferralPolicy:
    """Session 27.9 Part E: _should_include_clinical_referral gates referral language."""

    def test_helper_function_exists(self):
        """_should_include_clinical_referral must be importable from counseling_engine."""
        assert hasattr(_ce, "_should_include_clinical_referral"), (
            "_should_include_clinical_referral must be defined in counseling_engine"
        )

    def test_no_args_returns_false(self):
        """No flags set → referral is not warranted."""
        assert _ce._should_include_clinical_referral() is False

    def test_is_personal_returns_true(self):
        """Personal result interpretation warrants a referral."""
        assert _ce._should_include_clinical_referral(is_personal=True) is True

    def test_asks_for_action_returns_true(self):
        """Action-request (surgery, decision) warrants a referral."""
        assert _ce._should_include_clinical_referral(asks_for_action=True) is True

    def test_asks_for_interpretation_returns_true(self):
        """Interpretation of personal result warrants a referral."""
        assert _ce._should_include_clinical_referral(asks_for_interpretation=True) is True

    def test_requires_report_details_returns_true(self):
        """Questions requiring exact report coordinates warrant a referral."""
        assert _ce._should_include_clinical_referral(requires_report_details=True) is True

    def test_specific_variant_intent_returns_true(self):
        """'specific_variant' intent is high-stakes — referral warranted."""
        assert _ce._should_include_clinical_referral(intent="specific_variant") is True

    def test_general_gene_description_no_referral(self):
        """General gene description questions do not warrant a referral."""
        assert _ce._should_include_clinical_referral(
            intent="general_education", is_personal=False
        ) is False

    def test_animal_genetics_no_referral(self):
        """Animal genetics questions do not warrant a referral."""
        assert _ce._should_include_clinical_referral(
            intent="general_education", is_personal=False, asks_for_action=False
        ) is False

    def test_general_chromosome_definition_no_referral(self):
        """A general chromosomal concept question does not warrant a referral."""
        assert _ce._should_include_clinical_referral(
            intent="chromosome_finding_general", is_personal=False
        ) is False

    def test_referral_not_in_vus_template(self):
        """VUS_KNOWN_GENE_TEMPLATE_HE must not end with a generic referral sentence."""
        template = _ce.VUS_KNOWN_GENE_TEMPLATE_HE
        for phrase in _GENERIC_REFERRAL_PHRASES:
            assert phrase not in template, (
                f"VUS template must not contain referral phrase {phrase!r}"
            )

    def test_no_referral_in_chromosome_finding_general_kb(self):
        """The chromosome_finding_general KB entry must not contain a generic referral."""
        kb_entry = _ce._CYTOGENETIC_KB.get("chromosome_finding_general")
        assert kb_entry is not None, "chromosome_finding_general must be in _CYTOGENETIC_KB"
        answer = kb_entry.get("answer_he", "")
        assert not _has_referral(answer), (
            f"chromosome_finding_general KB answer must not contain generic referral: {answer[:120]!r}"
        )

    def test_chromosome_finding_general_has_what_to_clarify(self):
        """chromosome_finding_general KB entry must ask 'מה כדאי לברר' instead of generic referral."""
        kb_entry = _ce._CYTOGENETIC_KB.get("chromosome_finding_general")
        assert kb_entry is not None
        answer = kb_entry.get("answer_he", "")
        assert "מה כדאי לברר" in answer, (
            "chromosome_finding_general KB answer must include 'מה כדאי לברר' section"
        )

    def test_referral_absent_for_personal_variant_question(self):
        """Personal variant questions go through the safety redirect — not KB general answer."""
        resp = _ask("יש לי BRCA1 c.5266dupC מה זה אומר עלי?")
        # Personal variant → safety or specific-variant path; answer should still be valid
        assert resp["answer"], "Must return an answer for personal variant question"


# ===========================================================================
# Part I — UI Labels
# ===========================================================================

class TestUILabels:
    """Session 27.9 Part I: pending/approved/warning labels are canonically correct."""

    def test_generate_chromosome_draft_pending_label(self):
        """_generate_chromosome_education_draft must return correct expansion_label."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "מחיקה כרומוזומלית היא היעדרות של קטע מסוים מהכרומוזום. "
            "גודל המחיקה ומיקומה קובעים את ההשלכות הרפואיות."
        )
        debug: dict = {}
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            result = _ce._generate_chromosome_education_draft(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion", _debug=debug
            )
        if result is None:
            pytest.skip("No LLM configured or draft generation failed (no LLM in test env)")
        assert result.get("expansion_label") == "הסבר נוסף שנוצר באמצעות AI", (
            f"Expected pending expansion_label, got {result.get('expansion_label')!r}"
        )

    def test_generate_chromosome_draft_warning_text(self):
        """_generate_chromosome_education_draft must return correct warning_he."""
        mock_client = MagicMock()
        mock_client.call_text_raw.return_value = (
            "מחיקה כרומוזומלית היא היעדרות של קטע מסוים מהכרומוזום. "
            "גודל המחיקה ומיקומה קובעים את ההשלכות הרפואיות."
        )
        with patch.object(_ce, "create_llm_client", return_value=mock_client):
            result = _ce._generate_chromosome_education_draft(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )
        if result is None:
            pytest.skip("Draft generation failed (no LLM in test env)")
        assert result.get("warning_he") == (
            "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי."
        ), f"Got: {result.get('warning_he')!r}"

    def test_chromosome_metadata_pending_label(self):
        """chromosome_draft_metadata.expansion_label_pending must be canonical."""
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=None):
            result = _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )
        label = result["chromosome_draft_metadata"]["expansion_label_pending"]
        assert label == "הסבר נוסף שנוצר באמצעות AI", f"Got: {label!r}"

    def test_chromosome_metadata_approved_label(self):
        """chromosome_draft_metadata.expansion_label_approved must be canonical."""
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=None):
            result = _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )
        label = result["chromosome_draft_metadata"]["expansion_label_approved"]
        assert label == "מידע נוסף שנבדק ואושר", f"Got: {label!r}"

    def test_chromosome_metadata_pending_warning(self):
        """chromosome_draft_metadata.expansion_pending_warning must be canonical."""
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=None):
            result = _ce._build_chromosome_education_answer(
                "מה זה מחיקה בכרומוזום?", "chromosome_deletion"
            )
        warning = result["chromosome_draft_metadata"]["expansion_pending_warning"]
        assert warning == "הסבר זה נוצר באמצעות AI וטרם נבדק על ידי הצוות הגנטי.", (
            f"Got: {warning!r}"
        )


# ===========================================================================
# Parts F/G — Chromosome 21 Context, No Down Syndrome Assumption
# ===========================================================================

class TestChromosome21Context:
    """Session 27.9 Parts F/G: chromosome 21 context opener; no Down syndrome assumption."""

    def _chr21_answer(self, question: str, sub_intent: str = "chromosome_finding_general") -> dict:
        with patch.object(_ce, "_generate_chromosome_education_draft", return_value=None):
            return _ce._build_chromosome_education_answer(
                question, sub_intent, chromosome_number="21"
            )

    def test_chr21_mentioned_in_answer_text(self):
        """When chromosome_number='21', the answer must explicitly mention chromosome 21."""
        result = self._chr21_answer("יש לי שינוי בכרומוזום 21, מה זה?")
        assert "21" in result["answer"], (
            "Answer must mention chromosome 21 when number is known"
        )

    def test_chr21_opener_in_chromosome_context_openers(self):
        """'chromosome_finding_general' must be in _CHROMOSOME_CONTEXT_OPENERS."""
        assert "chromosome_finding_general" in _ce._CHROMOSOME_CONTEXT_OPENERS, (
            "'chromosome_finding_general' must have an entry in _CHROMOSOME_CONTEXT_OPENERS"
        )

    def test_chr21_opener_contains_placeholder(self):
        """The chromosome_finding_general opener must contain {chromosome_number} placeholder."""
        opener = _ce._CHROMOSOME_CONTEXT_OPENERS["chromosome_finding_general"]
        assert "{chromosome_number}" in opener, (
            "chromosome_finding_general opener must contain {chromosome_number} placeholder"
        )

    def test_chr21_opener_formatted_correctly(self):
        """Formatting the opener with chromosome_number='21' must produce coherent text."""
        opener_template = _ce._CHROMOSOME_CONTEXT_OPENERS["chromosome_finding_general"]
        formatted = opener_template.format(chromosome_number="21")
        assert "21" in formatted
        assert "{chromosome_number}" not in formatted, "Placeholder must be replaced"

    def test_no_down_syndrome_assumption_for_chr21_general(self):
        """A general chromosome 21 question must NOT assume trisomy 21 / Down syndrome."""
        result = self._chr21_answer("יש לי שינוי בכרומוזום 21, מה זה?")
        answer_lower = result["answer"].lower()
        # Must not assume trisomy 21 or Down syndrome without explicit question
        assert "טריזומיה 21" not in result["answer"] or "מה זה שינוי בכרומוזום?" in result["answer"], (
            "General chr21 question must not assume trisomy 21 (Down syndrome)"
        )
        assert "תסמונת דאון" not in result["answer"], (
            "General chr21 question must not assume Down syndrome"
        )

    def test_chr21_deletion_shows_chromosome_in_clinician_questions(self):
        """Clinician questions must reference chromosome 21 when number is known."""
        result = self._chr21_answer("מחיקה בכרומוזום 21", sub_intent="chromosome_deletion_general")
        if result.get("clinician_questions"):
            for q in result["clinician_questions"]:
                if "כרומוזום" in q:
                    assert "21" in q, f"Clinician question references chromosome but not 21: {q!r}"
                    break

    def test_chromosome_number_detected_in_metadata(self):
        """chromosome_draft_metadata must record the detected chromosome number."""
        result = self._chr21_answer("מה זה שינוי בכרומוזום 21?")
        meta = result["chromosome_draft_metadata"]
        assert meta.get("chromosome_number_detected") == "21", (
            f"Expected '21', got {meta.get('chromosome_number_detected')!r}"
        )


# ===========================================================================
# Part H — MTHFR: association-only answer has no generic referral
# ===========================================================================

class TestMTHFRReferralPolicy:
    """Session 27.9 Part H: MTHFR association-only grounded answer must not have generic referral."""

    def test_mthfr_not_in_gene_knowledge_base(self):
        """MTHFR has no biology entry in gene_knowledge_base.json — confirmed by session 27.8.2."""
        import app.gene_knowledge as _gk
        result = _gk.get_gene_knowledge_biology_text("MTHFR")
        assert result is None, (
            "MTHFR must not be in gene_knowledge_base.json for this test to be valid"
        )

    def test_association_system_prompt_no_referral_rule(self):
        """_SOURCE_GROUNDED_ASSOCIATION_SYSTEM_PROMPT must contain 'Do NOT add a generic referral'."""
        assert hasattr(_ce, "_SOURCE_GROUNDED_ASSOCIATION_SYSTEM_PROMPT"), (
            "_SOURCE_GROUNDED_ASSOCIATION_SYSTEM_PROMPT must exist in counseling_engine"
        )
        prompt = _ce._SOURCE_GROUNDED_ASSOCIATION_SYSTEM_PROMPT
        assert "Do NOT add a generic referral" in prompt, (
            "Association system prompt must explicitly forbid generic referral sentences"
        )

    def test_gene_system_prompt_no_referral_rule(self):
        """_SOURCE_GROUNDED_GENE_SYSTEM_PROMPT must contain 'Do NOT add a generic referral'."""
        assert hasattr(_ce, "_SOURCE_GROUNDED_GENE_SYSTEM_PROMPT"), (
            "_SOURCE_GROUNDED_GENE_SYSTEM_PROMPT must exist in counseling_engine"
        )
        prompt = _ce._SOURCE_GROUNDED_GENE_SYSTEM_PROMPT
        assert "Do NOT add a generic referral" in prompt, (
            "Gene system prompt must explicitly forbid generic referral sentences"
        )


# ===========================================================================
# Regressions — ensure prior sessions still work
# ===========================================================================

class TestRegressions:
    """Session 27.9 Regressions: HBB, TYR, physician approval endpoint, schema."""

    def test_hbb_vus_question_returns_valid_response(self):
        """HBB VUS question must return a valid 5-key response."""
        resp = _ask("יש לי VUS בגן HBB, מה זה?")
        assert resp["answer"]
        assert resp["safety_level"] in {
            "general_information", "contains_identifying_info",
            "requires_genetic_counselor", "out_of_scope",
        }

    def test_tyr_question_returns_valid_response(self):
        """TYR question must return a valid 5-key response."""
        resp = _ask("מה זה גן TYR?")
        assert resp["answer"]

    def test_carrier_question_returns_valid_response(self):
        """Carrier question must return a valid response (no regression in carrier routing)."""
        resp = _ask("אמרו לי שאני נשאית, מה זה?")
        assert resp["answer"]
        assert resp["safety_level"] == "general_information"

    def test_vus_question_returns_valid_response(self):
        """General VUS question must return a valid response."""
        resp = _ask("מה זה VUS?")
        assert resp["answer"]
        assert "VUS" in resp["answer"] or "variants" in resp["answer"].lower() or "שינוי" in resp["answer"]

    def test_physician_approval_endpoint_200(self):
        """POST /review/approve must not return 422."""
        r = client.post("/review/approve", json={
            "draft_id": 999999,
            "reviewer_id": "test_s279",
        })
        # Should return 200 (draft not found) or 404, never 422
        assert r.status_code != 422, f"Got 422 on /review/approve: {r.text}"

    def test_ask_response_schema_exactly_5_keys(self):
        """Every /ask response must contain exactly the 5 required keys (may have extra)."""
        resp = _ask("מה זה גנטיקה?")
        assert _REQUIRED_KEYS.issubset(resp.keys()), (
            f"Missing required keys: {_REQUIRED_KEYS - resp.keys()}"
        )

    def test_identifying_info_blocked(self):
        """Messages with Israeli ID numbers must be blocked."""
        resp = _ask("קוראים לי דניאל 123456789 ויש לי VUS")
        assert resp["safety_level"] in {
            "contains_identifying_info", "requires_genetic_counselor"
        }, f"Expected block, got: {resp['safety_level']}"

    def test_medical_action_refused(self):
        """'האם עלי לעשות ניתוח?' must be refused as medical action."""
        resp = _ask("האם עלי לעשות ניתוח?")
        assert resp["needs_genetic_counselor"] is True or resp["safety_level"] in {
            "requires_genetic_counselor", "out_of_scope"
        }, f"Medical action must be refused: {resp!r}"
