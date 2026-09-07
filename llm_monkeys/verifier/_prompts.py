"""System instructions and prompt templates for the LLM-as-a-judge fact verifier."""

from __future__ import annotations

DEFAULT_VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an expert physician, medical licensing board examiner, and clinical judge. "
    "Your task is to critically evaluate whether a stated medical fact or clinical assertion "
    "is factually accurate, scientifically sound, and clinically correct in the context of the medical question. "
    "Provide a binary classification: 1 if the statement is factually and clinically correct, "
    "or 0 if the statement is factually incorrect, misleading, false, or scientifically flawed."
)


def format_fact_verification_prompt(
    question: str,
    options: dict[str, str],
    fact: str,
) -> str:
    """Format prompt for the verifier agent to evaluate a single medical fact.

    Args:
        question: Medical examination question text.
        options: Dict mapping option letters (e.g. 'A', 'B') to option text.
        fact: Candidate atomic medical fact statement to be verified.

    Returns:
        Formatted prompt string.
    """
    options_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        f"Clinical Context and Examination Question:\n"
        f"{question}\n\n"
        f"Options:\n"
        f"{options_str}\n\n"
        f"Candidate Medical Fact to Verify:\n"
        f'"{fact}"\n\n'
        f"Task:\n"
        f"Determine whether the candidate medical fact above is factually and clinically accurate (1) "
        f"or inaccurate/false/misleading (0).\n\n"
        f"Respond with:\n"
        f"- is_correct: 1 if correct and scientifically sound, 0 if incorrect or flawed\n"
        f"- rationale: A brief 1-2 sentence medical explanation for your judgment\n"
    )


__all__ = [
    "DEFAULT_VERIFIER_SYSTEM_INSTRUCTION",
    "format_fact_verification_prompt",
]
