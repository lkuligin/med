"""Parser and evaluator for extracting multiple-choice answers from LLM outputs."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    # Accepts "C", "**C**", "(C)", "Option C" and "[Option C]"; \b keeps it from
    # reading the first letter of a word ("FINAL ANSWER: Based on...").
    r"FINAL\s+ANSWER\s*:\s*\**\s*[\[\(]?\s*(?:Option\s+)?\**\s*([A-E])\b",
    r"(?:The\s+)?(?:correct|best)\s+(?:answer|option)\s+is\s+(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"(?:Correct\s+)?\*{0,2}Answer\*{0,2}\s*:\*{0,2}\s*(?:Option\s+)?[\*\[\(]?([A-E])\b",
    r"(?:Therefore|Hence|Thus|In conclusion),?\s+(?:the\s+correct\s+answer\s+is\s+)?(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"Option\s+([A-E])\s+is\s+(?:the\s+)?(?:most\s+appropriate|correct)",
    r"###\s*(?:Final\s+)?(?:Answer|Option)\s*:?\s*(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
    r"(?:The\s+)?Answer\s+is\s+(?:Option\s+)?[\*\[\(]?([A-E])[\*\]\)]?",
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
    # Strip <think>...</think> reasoning blocks if present to avoid matching internal deliberation
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    search_text = cleaned_text if cleaned_text else text

    for pattern in _PATTERNS:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if isinstance(options, dict):
        tail = "\n".join(search_text.splitlines()[-5:]).lower()
        for opt_key, opt_val in options.items():
            if opt_val.strip() and opt_val.strip().lower() in tail:
                return opt_key.upper()

    bold_matches = re.findall(
        r"\*\*(?:Option\s+)?([A-E])\*\*", search_text, re.IGNORECASE
    )
    return bold_matches[-1].upper() if bold_matches else None


def evaluate_prediction(
    prediction: str | None,
    ground_truth_idx: str,
) -> bool:
    """Evaluates if the predicted option matches the ground truth answer index."""
    if not prediction or not ground_truth_idx:
        return False
    return prediction.strip().upper() == ground_truth_idx.strip().upper()


def rescore_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-parse stored raw responses with the current parser.

    Stored runs keep the predicted_option and is_correct their parser produced
    at the time. Scoring from raw_response instead makes a parser fix apply to
    runs already on disk. Attempts without a raw response keep their stored
    values. The question-level fields derived from attempts are recomputed the
    way the workflow computes them. The input is not modified.
    """
    rescored = []
    for item in items:
        gt = item.get("ground_truth") if isinstance(item, dict) else None
        attempts = item.get("attempts") if isinstance(item, dict) else None
        if not gt or not isinstance(attempts, list):
            rescored.append(item)
            continue
        new_attempts = []
        for a in attempts:
            raw = a.get("raw_response") if isinstance(a, dict) else None
            if raw and raw.strip():
                pred = extract_predicted_option(raw, item.get("options"))
                a = {**a, "predicted_option": pred,
                     "is_correct": evaluate_prediction(pred, gt)}
            new_attempts.append(a)
        correct = sum(1 for a in new_attempts if isinstance(a, dict) and a.get("is_correct"))
        all_correct = bool(new_attempts) and correct == len(new_attempts)
        first = new_attempts[0] if new_attempts and isinstance(new_attempts[0], dict) else {}
        rescored.append({
            **item,
            "attempts": new_attempts,
            "predicted_option": first.get("predicted_option", item.get("predicted_option")),
            "is_correct": all_correct,
            "is_all_correct": all_correct,
            "correct_attempts": correct,
        })
    return rescored
