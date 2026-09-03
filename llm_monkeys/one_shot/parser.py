"""Parser and evaluator for extracting multiple-choice answers from LLM outputs."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    r"FINAL\s+ANSWER\s*:\s*(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"(?:The\s+)?(?:correct|best)\s+(?:answer|option)\s+is\s+(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"(?:Correct\s+)?\*{0,2}Answer\*{0,2}\s*:\*{0,2}\s*(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"(?:Therefore|Hence|Thus|In conclusion),?\s+(?:the\s+correct\s+answer\s+is\s+)?(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"Option\s+([A-E])\s+is\s+(?:the\s+)?(?:most\s+appropriate|correct)",
    r"^\(?([A-E])\)?(?:\.|\:|\n|\s|$)",
]


def extract_predicted_option(
    response_text: str,
    options: dict[str, str] | list[str] | Any = None,
) -> str | None:
    """Extracts the predicted multiple choice option (A-E) from LLM response text."""
    if not response_text or not response_text.strip():
        return None

    text = response_text.strip()

    for pattern in _PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if isinstance(options, dict):
        tail = "\n".join(text.splitlines()[-5:]).lower()
        for opt_key, opt_val in options.items():
            if opt_val.strip() and opt_val.strip().lower() in tail:
                return opt_key.upper()

    bold_matches = re.findall(r"\*\*(?:Option\s+)?([A-E])\*\*", text, re.IGNORECASE)
    return bold_matches[-1].upper() if bold_matches else None


def evaluate_prediction(
    prediction: str | None,
    ground_truth_idx: str,
) -> bool:
    """Evaluates if the predicted option matches the ground truth answer index."""
    if not prediction or not ground_truth_idx:
        return False
    return prediction.strip().upper() == ground_truth_idx.strip().upper()

