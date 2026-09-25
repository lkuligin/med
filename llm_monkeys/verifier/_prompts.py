"""System instructions and prompt templates for the LLM-as-a-judge fact verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

DEFAULT_VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an expert physician, medical licensing board examiner, and clinical judge. "
    "Your task is to critically evaluate whether a stated medical fact or clinical assertion "
    "is factually accurate, scientifically sound, and clinically correct in the context of the medical question. "
    "Provide a binary classification: YES (1) if the statement is factually and clinically correct, "
    "or NO (0) if the statement is factually incorrect, misleading, false, or scientifically flawed."
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
        f"Determine whether the candidate medical fact above is factually and clinically accurate (YES / 1) "
        f"or inaccurate/false/misleading (NO / 0).\n\n"
        f"Respond with:\n"
        f"- is_correct: YES if correct and scientifically sound, NO if incorrect or flawed (1 for YES, 0 for NO)\n"
        f"- rationale: A brief 1-2 sentence medical explanation for your judgment\n"
    )


# Experimental: the judge sees the fact and nothing else, so it cannot solve
# the question itself and a verdict is about the fact alone. Meant for facts
# written to stand on their own (the "background" fact prompt of step 2).
FACT_ONLY_VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an expert physician and medical fact checker. "
    "You are given a single medical statement and nothing else. "
    "Your task is to judge whether it is accurate according to current mainstream "
    "medical knowledge and clinical guidelines. "
    "Provide a binary classification: 1 if the statement is accurate, or 0 if it is "
    "inaccurate, outdated, misleading, or cannot be judged true on its own."
)


def format_fact_only_verification_prompt(
    question: str,
    options: dict[str, str],
    fact: str,
) -> str:
    """Format a verification prompt that carries the fact alone.

    Takes the question and options only to share a signature with the
    reference prompt; neither reaches the judge.
    """
    return (
        f"Medical Statement to Verify:\n"
        f'"{fact}"\n\n'
        f"Task:\n"
        f"Determine whether the statement above is accurate as a general medical fact (1) "
        f"or not (0).\n"
        f"Mark it 0 if any part of it is wrong, if it misstates how common, strong or "
        f"specific something is, or if it refers to a patient, case or context that is "
        f"not given.\n\n"
        f"Respond with:\n"
        f"- is_correct: 1 if the statement is accurate, 0 otherwise\n"
        f"- rationale: A brief 1-2 sentence medical explanation for your judgment\n"
    )


# Experiment case-grounded-facts-question-aware-judge: the judge sees the
# question but not the options, so it can check a fact against the findings
# the question gives without being handed a list to pick an answer from.
QUESTION_AWARE_VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an expert physician and medical fact checker. "
    "You are given a clinical question and a single medical statement written while "
    "reasoning about it. Your task is to judge whether the statement is accurate according "
    "to current mainstream medical knowledge and clinical guidelines, and, where it cites "
    "clinical findings, whether they match the question. "
    "Provide a binary classification: 1 if the statement is accurate, or 0 if it is not."
)


def format_question_aware_verification_prompt(
    question: str,
    options: dict[str, str],
    fact: str,
) -> str:
    """Format a verification prompt that carries the question but not the options.

    Takes the options only to share a signature with the reference prompt.
    """
    return (
        f"Clinical Question:\n"
        f"{question}\n\n"
        f"Medical Statement to Verify:\n"
        f'"{fact}"\n\n'
        f"Task:\n"
        f"Determine whether the statement above is accurate (1) or not (0).\n"
        f"Mark it 0 if any part of it is medically wrong, if it misstates how common, strong "
        f"or specific something is, or if it cites a finding that contradicts or is absent "
        f"from the question.\n"
        f"Do not mark a statement 0 only because it is not needed to answer the question.\n\n"
        f"Respond with:\n"
        f"- is_correct: 1 if the statement is accurate, 0 otherwise\n"
        f"- rationale: A brief 1-2 sentence medical explanation for your judgment\n"
    )


# As question-no-options, but the statement is judged on its medical accuracy
# alone and the question is there only to check the findings it quotes. The
# question-no-options judge, handed the case, mostly checked whether a fact
# fitted it and rejected far less.
QUESTION_AS_REFERENCE_VERIFIER_SYSTEM_INSTRUCTION = (
    "You are an expert physician and medical fact checker. "
    "Your task is to judge whether a single medical statement is accurate according to "
    "current mainstream medical knowledge and clinical guidelines. "
    "A clinical question is provided for reference only. "
    "Provide a binary classification: 1 if the statement is accurate, or 0 if it is not."
)


def format_question_as_reference_verification_prompt(
    question: str,
    options: dict[str, str],
    fact: str,
) -> str:
    """Format a verification prompt that puts the statement first and gives the
    question, without the options, only as a reference for quoted findings."""
    return (
        f"Medical Statement to Verify:\n"
        f'"{fact}"\n\n'
        f"Clinical Question (for reference only):\n"
        f"{question}\n\n"
        f"Task:\n"
        f"Judge the statement on its medical accuracy alone. Do not judge whether it fits "
        f"the question, is relevant to it, or supports any answer.\n"
        f"Mark it 0 if any part of it is medically wrong, or if it misstates how common, "
        f"strong or specific something is.\n"
        f"Use the question only to check numbers and findings the statement quotes: mark it "
        f"0 if a quoted finding differs from the question.\n\n"
        f"Respond with:\n"
        f"- is_correct: 1 if the statement is accurate, 0 otherwise\n"
        f"- rationale: A brief 1-2 sentence medical explanation for your judgment\n"
    )


@dataclass(frozen=True)
class JudgePrompt:
    """How the judge is asked: its system instruction and its user message."""

    system_instruction: str
    format: Callable[[str, dict[str, str], str], str]


JUDGE_PROMPTS: dict[str, JudgePrompt] = {
    # The authors' prompt, unchanged: every stored verdict was made with it.
    "reference": JudgePrompt(DEFAULT_VERIFIER_SYSTEM_INSTRUCTION,
                             format_fact_verification_prompt),
    "fact-only": JudgePrompt(FACT_ONLY_VERIFIER_SYSTEM_INSTRUCTION,
                             format_fact_only_verification_prompt),
    "question-no-options": JudgePrompt(QUESTION_AWARE_VERIFIER_SYSTEM_INSTRUCTION,
                                       format_question_aware_verification_prompt),
    "question-as-reference": JudgePrompt(QUESTION_AS_REFERENCE_VERIFIER_SYSTEM_INSTRUCTION,
                                         format_question_as_reference_verification_prompt),
}
DEFAULT_JUDGE_PROMPT = "reference"


__all__ = [
    "DEFAULT_VERIFIER_SYSTEM_INSTRUCTION",
    "FACT_ONLY_VERIFIER_SYSTEM_INSTRUCTION",
    "format_fact_verification_prompt",
    "format_fact_only_verification_prompt",
    "QUESTION_AWARE_VERIFIER_SYSTEM_INSTRUCTION",
    "format_question_aware_verification_prompt",
    "QUESTION_AS_REFERENCE_VERIFIER_SYSTEM_INSTRUCTION",
    "format_question_as_reference_verification_prompt",
    "JudgePrompt",
    "JUDGE_PROMPTS",
    "DEFAULT_JUDGE_PROMPT",
]
