"""Unit tests for verifier._prompts module."""

from __future__ import annotations

from verifier._prompts import (
    DEFAULT_VERIFIER_SYSTEM_INSTRUCTION,
    format_fact_verification_prompt,
)


def test_default_verifier_system_instruction():
    assert "expert physician" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "binary classification" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "YES" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "NO" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "1" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert "0" in DEFAULT_VERIFIER_SYSTEM_INSTRUCTION


def test_format_fact_verification_prompt():
    question = "A 45-year-old man presents with acute chest pain."
    options = {"A": "Aspirin", "B": "Morphine"}
    fact = "Aspirin inhibits platelet cyclooxygenase irreversibly."

    prompt = format_fact_verification_prompt(question, options, fact)

    assert question in prompt
    assert "A. Aspirin" in prompt
    assert "B. Morphine" in prompt
    assert fact in prompt
    assert "is_correct: YES" in prompt
    assert "NO" in prompt
    assert "rationale:" in prompt


def test_reference_judge_prompt_is_the_default_and_unchanged():
    from verifier._prompts import DEFAULT_JUDGE_PROMPT, JUDGE_PROMPTS

    assert DEFAULT_JUDGE_PROMPT == "reference"
    ref = JUDGE_PROMPTS["reference"]
    assert ref.system_instruction == DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    assert ref.format is format_fact_verification_prompt


def test_fact_only_prompt_carries_neither_question_nor_options():
    from verifier._prompts import JUDGE_PROMPTS

    prompt = JUDGE_PROMPTS["fact-only"].format(
        "SECRET QUESTION", {"A": "SECRET OPTION"}, "Aspirin inhibits COX-1."
    )
    assert '"Aspirin inhibits COX-1."' in prompt
    assert "SECRET" not in prompt
    assert "is_correct" in prompt and "rationale" in prompt


def test_judge_prompt_selects_the_system_instruction():
    from config import VerifierConfig
    from verifier._prompts import FACT_ONLY_VERIFIER_SYSTEM_INSTRUCTION
    from verifier.agent import create_fact_verifier_agent

    assert VerifierConfig().resolved_system_instruction == DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    agent = create_fact_verifier_agent(VerifierConfig(judge_prompt="fact-only"))
    assert agent.instruction == FACT_ONLY_VERIFIER_SYSTEM_INSTRUCTION


def test_cli_selects_the_judge_prompt_and_the_run_records_it():
    from results_store import sampling_of
    from verifier.cli import build_config, create_parser

    default = build_config(create_parser().parse_args([]))
    chosen = build_config(create_parser().parse_args(["--judge-prompt", "fact-only"]))
    assert default.judge_prompt == "reference"
    assert chosen.judge_prompt == "fact-only"
    assert sampling_of(chosen)["judge_prompt"] == "fact-only"


def test_question_aware_prompt_carries_the_question_but_not_the_options():
    from verifier._prompts import JUDGE_PROMPTS

    prompt = JUDGE_PROMPTS["question-no-options"].format(
        "A 9-month-old boy has recurrent otitis media.", {"A": "SECRET OPTION"}, "Fact X."
    )
    assert "A 9-month-old boy has recurrent otitis media." in prompt
    assert '"Fact X."' in prompt
    assert "SECRET" not in prompt


def test_question_as_reference_prompt_puts_the_statement_first_without_options():
    from verifier._prompts import JUDGE_PROMPTS

    prompt = JUDGE_PROMPTS["question-as-reference"].format(
        "A 9-month-old boy has recurrent otitis media.", {"A": "SECRET OPTION"}, "Fact X."
    )
    assert prompt.index('"Fact X."') < prompt.index("A 9-month-old boy")
    assert "SECRET" not in prompt
