# All comments are in English.

import re
from dataclasses import dataclass, field
from typing import List, Optional

POLICY_TEXT = """
Safety & scope:
- This tool provides general information and explains what ClinVar shows.
- It does NOT provide medical advice or clinical recommendations.
- It must NOT recommend: pregnancy termination/continuation, treatment changes, surgeries, invasive tests (e.g., amniocentesis, CVS),
  changes in pregnancy follow-up, oncologic/cardiologic/neurologic workup, or any other clinical action.
- It must NOT interpret familial genetic risk or give probability estimates.
- For any decision-making, refer the user to a licensed physician/genetic counselor.
"""

FORBIDDEN_ACTIONS = [
    "pregnancy termination", "stop treatment", "start treatment", "surgery",
    "amniocentesis", "CVS", "invasive test", "change follow-up",
    "oncology workup", "cardiology workup", "neurology workup",
    "risk percentage", "family risk"
]

_SAFETY_DISCLAIMER = (
    "This information is for educational purposes only and does not constitute "
    "medical advice. Please consult a certified genetics professional or genetic "
    "counselor for clinical interpretation and personalized recommendations."
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class PolicyResult:
    allowed: bool = True
    flags: List[str] = field(default_factory=list)
    redirect_message: Optional[str] = None


def sanitise_question(question: str) -> str:
    """Strip control characters and leading/trailing whitespace."""
    return _CONTROL_CHARS.sub("", question.strip())


def check_question(question: str) -> PolicyResult:
    """
    Detect forbidden clinical-action keywords.
    Always returns allowed=True — the caller may still respond,
    but policy_flags and redirect_message are populated so the
    response can be appropriately scoped.
    """
    q_lower = question.lower()
    flags = [action for action in FORBIDDEN_ACTIONS if action.lower() in q_lower]
    redirect_message = (
        "This question touches on clinical decision-making. "
        "Please consult a licensed physician or certified genetic counselor."
        if flags
        else None
    )
    return PolicyResult(allowed=True, flags=flags, redirect_message=redirect_message)


def enforce_disclaimer(result: dict) -> dict:
    """
    Unconditionally overwrite the safety_disclaimer field and ensure
    a non-empty limitations list. Does not mutate the caller's dict.
    """
    out = dict(result)
    out["safety_disclaimer"] = _SAFETY_DISCLAIMER
    if not out.get("limitations"):
        out["limitations"] = [
            "This information is based on ClinVar data and may not reflect the most recent evidence.",
            "Variant interpretation depends on full clinical context.",
        ]
    return out
