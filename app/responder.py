# All comments are in English.

from typing import Dict, Any, List
from .policy import POLICY_TEXT

# ------------------------------------------------------------
# Phase 2A (ACTIVE): Safe, template-based response (no LLM/GPU)
# ------------------------------------------------------------

def build_safe_answer(
    question: str,
    gene: str,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Template-based response: safe, deterministic, and does not require an LLM.
    Used by the /ask endpoint.
    """
    n = len(records)
    unique_clinsig = sorted(
        {r.get("clinical_significance") for r in records if r.get("clinical_significance")}
    )

    answer: List[str] = []
    answer.append(f"Question: {question}")
    answer.append("")
    answer.append(f"Gene: {gene}")
    answer.append(f"Found {n} matching ClinVar records (showing a limited sample).")

    if unique_clinsig:
        shown = ", ".join(unique_clinsig[:10])
        suffix = "..." if len(unique_clinsig) > 10 else ""
        answer.append(f"ClinicalSignificance values in this sample: {shown}{suffix}")

    answer.append("")
    answer.append("What this means (general):")
    answer.append("- ClinVar aggregates submissions about genetic variants and their reported clinical significance.")
    answer.append("- 'Variant of Uncertain Significance (VUS)' means current evidence is insufficient to label the variant as benign or pathogenic.")
    answer.append("- VUS should not be used alone for clinical decision-making; interpretation depends on the full clinical context and genetics consultation.")

    answer.append("")
    answer.append("Safety note:")
    answer.append("- This tool does not provide medical advice or recommendations. Please consult a genetic counselor/physician for any decisions.")

    return {
        "answer": "\n".join(answer),
        "policy": POLICY_TEXT,
        "evidence": records[:10],  # keep response compact
    }


# ------------------------------------------------------------
# Phase 2B (INACTIVE for now): LLM-based response (requires GPU)
# Keep this commented/disabled until you have funding & RunPod.
# ------------------------------------------------------------

# Uncomment these only when you are ready to use an LLM server:
# from .llm_client import generate_with_llm


# ------------------------------------------------------------
# Module-level API expected by main.py
# ------------------------------------------------------------

def build_response(question: str, evidence_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Thin adapter so main.py can call build_response(question, evidence_records).
    Delegates to the safe template responder and converts the output to the
    AskResponse schema (evidence as list[str], adds limitations and safety_disclaimer).
    """
    gene = str(evidence_records[0].get("gene_symbol", "")) if evidence_records else ""
    base = build_safe_answer(question, gene, evidence_records)

    _KEEP = {"gene_symbol", "clinical_significance", "phenotype_list", "dbsnp_id", "review_status"}
    evidence_strings: List[str] = []
    for rec in evidence_records[:10]:
        parts = [
            str(v)
            for k, v in rec.items()
            if k in _KEEP and v is not None and str(v) not in ("", "nan", "None")
        ]
        evidence_strings.append(" | ".join(parts) if parts else repr(rec))

    limitations: List[str] = [
        "This information is based on ClinVar submissions and may not reflect the most current evidence.",
        "Variant interpretation requires full clinical context and should not be made from database records alone.",
    ]
    if any(
        "uncertain" in str(r.get("clinical_significance", "")).lower()
        for r in evidence_records
    ):
        limitations.append(
            "One or more variants are classified as Variant of Uncertain Significance (VUS). "
            "VUS findings must not be used for clinical decision-making without expert review."
        )

    return {
        "answer": base["answer"],
        "evidence": evidence_strings,
        "limitations": limitations,
        "safety_disclaimer": (
            "This information is for educational purposes only and does not constitute "
            "medical advice. Please consult a certified genetics professional or genetic "
            "counselor for clinical interpretation and personalized recommendations."
        ),
    }

# def build_llm_answer(question: str, gene: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     LLM-based answer. Requires an LLM backend (e.g., Ollama on RunPod).
#     """
#     evidence_text = "\n".join(
#         f"- {r.get('gene_symbol')} | {r.get('clinical_significance')} | {r.get('phenotype_list', '')}"
#         for r in records[:10]
#     )
#
#     prompt = f"""
# You are a clinical genetics assistant.
# You must strictly follow the safety policy below.
#
# SAFETY POLICY:
# {POLICY_TEXT}
#
# CONTEXT (ClinVar evidence):
# {evidence_text}
#
# USER QUESTION:
# {question}
#
# INSTRUCTIONS:
# - Explain concepts clearly and cautiously.
# - Do NOT provide medical advice or recommendations.
# - Do NOT estimate risk or suggest actions.
# - Emphasize uncertainty when relevant.
# - Refer the user to a genetic counselor or physician.
#
# ANSWER:
# """
#
#     text = generate_with_llm(prompt)
#
#     return {
#         "answer": text,
#         "policy": POLICY_TEXT,
#         "evidence": records[:10],
#     }
