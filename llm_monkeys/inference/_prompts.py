"""Workflow-specific system instructions and prompt definitions for candidate inference."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FACT_SYSTEM_INSTRUCTION = (
    "You are an expert physician and clinical knowledge specialist. "
    "For a given medical licensing examination question and options, identify and generate "
    "atomic, verifiable medical facts, pathophysiological mechanisms, pharmacological properties, "
    "and clinical guidelines relevant to answering the question accurately."
)

DEFAULT_ANSWER_SYSTEM_INSTRUCTION = (
    "You are an expert physician taking a medical licensing board examination. "
    "Reason carefully through clinical questions using the provided medical facts and options, "
    "determine the single most accurate option, and state your final answer."
)


@dataclass(frozen=True)
class FactPrompt:
    """How the fact generator is asked: its system instruction, and the task
    text placed before the question and options in the user message."""

    system_instruction: str
    task: str


# The reference prompt, as the authors wrote it. Kept byte for byte: it is what
# every stored step 2 run was generated with.
REFERENCE_FACT_PROMPT = FactPrompt(
    system_instruction=DEFAULT_FACT_SYSTEM_INSTRUCTION,
    task=(
        "Analyze the following multiple-choice medical examination question and options. "
        "Extract and generate all key atomic, verifiable medical facts, clinical principles, "
        "pathophysiological mechanisms, and pharmacological properties relevant to solving this question accurately."
    ),
)

# The reference system instruction without "relevant to answering the question
# accurately", which would pull the facts towards an answer. Shared by the
# experimental prompts below.
UNANCHORED_FACT_SYSTEM_INSTRUCTION = (
    "You are an expert physician and clinical knowledge specialist. "
    "For a given medical licensing examination question and options, identify and generate "
    "atomic, verifiable medical facts, pathophysiological mechanisms, pharmacological properties, "
    "and clinical guidelines."
)

# Experiment general-facts-question-unaware-judge: background knowledge that a
# judge can verify with no access to the question or the options. The facts
# describe the conditions, drugs and mechanisms the question mentions, not the
# patient, and do not answer it; applying them to the case is left to the
# answer step.
BACKGROUND_FACT_PROMPT = FactPrompt(
    system_instruction=UNANCHORED_FACT_SYSTEM_INSTRUCTION,
    task=(
        "Read the following multiple-choice medical examination question and its options. "
        "Write the background medical facts needed to reason about it.\n\n"
        "Rules for every fact:\n"
        "- One atomic claim that an expert can mark true or false on its own, without seeing "
        "the question, the options or the other facts.\n"
        "- General medical knowledge only: never mention the patient, this case, the question, "
        "the options or an option letter, and do not restate findings from the question.\n"
        "- Specific and falsifiable: avoid vague wording such as \"may\", \"can\" or "
        "\"is associated with\" unless you say how often or how strongly; name the population "
        "for any superlative (for example, \"the most common cause of X in adults\").\n"
        "- Useful: prefer facts that distinguish the conditions, drugs or mechanisms mentioned "
        "in the options from one another, cover every option, and skip trivial definitions.\n\n"
        "Write two or three facts for each condition, drug or mechanism the options name, "
        "and no more than 30 facts in total."
    ),
)

# Experiment case-grounded-facts-question-aware-judge: as background, but facts
# may also interpret the findings the question gives, provided each states the
# finding it is about, so it still reads on its own and a judge who sees the
# question can check it against the case.
CASE_GROUNDED_FACT_PROMPT = FactPrompt(
    system_instruction=UNANCHORED_FACT_SYSTEM_INSTRUCTION,
    task=(
        "Read the following multiple-choice medical examination question and its options. "
        "Write the medical facts needed to reason about it: both general knowledge about the "
        "conditions, drugs and mechanisms involved, and interpretations of the specific clinical "
        "findings the question gives.\n\n"
        "Rules for every fact:\n"
        "- One atomic claim that an expert can mark true or false on its own.\n"
        "- Self-contained: never write \"the patient\", \"this case\", \"the question\", "
        "\"the options\" or an option letter. When a fact is about a finding from the question, "
        "state the finding itself inside the fact (for example, \"A platelet count of "
        "450,000/mm3 is within the normal range\", not \"The patient's platelet count is "
        "normal\").\n"
        "- Specific and falsifiable: avoid vague wording such as \"may\", \"can\" or "
        "\"is associated with\" unless you say how often or how strongly; name the population "
        "for any superlative.\n"
        "- Useful: include facts that distinguish the conditions, drugs or mechanisms named in "
        "the options from one another, and facts that connect the specific findings to them.\n\n"
        "Write two or three facts for each condition, drug or mechanism the options name, plus "
        "the facts needed to interpret the key findings, and no more than 30 facts in total."
    ),
)

# Experiment grounded-unframed-facts-cited-answer: the form rules of case-grounded, but
# nothing about what to cover, so the model spends its facts where its own
# reasoning goes instead of spreading them evenly over the options.
GROUNDED_UNFRAMED_FACT_PROMPT = FactPrompt(
    system_instruction=UNANCHORED_FACT_SYSTEM_INSTRUCTION,
    task=(
        "Read the following multiple-choice medical examination question and its options. "
        "Write the medical facts you need to reason about it.\n\n"
        "Rules for every fact:\n"
        "- One atomic claim that an expert can mark true or false on its own.\n"
        "- Self-contained: never write \"the patient\", \"this case\", \"the question\", "
        "\"the options\" or an option letter. When a fact is about a finding from the question, "
        "state the finding itself inside the fact (for example, \"A platelet count of "
        "450,000/mm3 is within the normal range\", not \"The patient's platelet count is "
        "normal\").\n"
        "- Specific and falsifiable: avoid vague wording such as \"may\", \"can\" or "
        "\"is associated with\" unless you say how often or how strongly; name the population "
        "for any superlative.\n\n"
        "Write no more than 50 facts."
    ),
)

FACT_PROMPTS: dict[str, FactPrompt] = {
    "reference": REFERENCE_FACT_PROMPT,
    "background": BACKGROUND_FACT_PROMPT,
    "case-grounded": CASE_GROUNDED_FACT_PROMPT,
    "grounded-unframed": GROUNDED_UNFRAMED_FACT_PROMPT,
}
DEFAULT_FACT_PROMPT = "reference"


@dataclass(frozen=True)
class AnswerPrompt:
    """How the answer step is asked: its system instruction, whether the facts
    are numbered, and the instructions placed after them."""

    system_instruction: str
    numbered_facts: bool
    instructions: str


ANSWER_PROMPTS: dict[str, AnswerPrompt] = {
    # The authors' wording, unchanged.
    "reference": AnswerPrompt(
        system_instruction=DEFAULT_ANSWER_SYSTEM_INSTRUCTION,
        numbered_facts=False,
        instructions=(
            "Based on these clinical facts, reason through the scenario and determine the "
            "single best option. Conclude your response on a new line with:\n"
            "FINAL ANSWER: [Option Letter]"
        ),
    ),
    # Experiment grounded-unframed-facts-cited-answer: the answer names the facts it
    # relies on, so a judge can later check only those.
    "cited-facts": AnswerPrompt(
        system_instruction=DEFAULT_ANSWER_SYSTEM_INSTRUCTION,
        numbered_facts=True,
        instructions=(
            "Based on these clinical facts, reason through the scenario and determine the "
            "single best option.\n"
            "Then list the numbers of the facts your answer relies on, and conclude on new "
            "lines with:\n"
            "FACTS USED: [numbers, comma-separated]\n"
            "FINAL ANSWER: [Option Letter]"
        ),
    ),
}
DEFAULT_ANSWER_PROMPT = "reference"

__all__ = [
    "DEFAULT_FACT_SYSTEM_INSTRUCTION",
    "DEFAULT_ANSWER_SYSTEM_INSTRUCTION",
    "FactPrompt",
    "FACT_PROMPTS",
    "DEFAULT_FACT_PROMPT",
    "AnswerPrompt",
    "ANSWER_PROMPTS",
    "DEFAULT_ANSWER_PROMPT",
]
