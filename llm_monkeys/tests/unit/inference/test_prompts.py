"""Unit tests for the fact-generation prompt variants."""

from __future__ import annotations

import pytest

from config import CandidateInferenceConfig
from dataset import MedQAQuestion
from inference._dataset import format_fact_generation_prompt
from inference._prompts import FACT_PROMPTS
from inference.run import build_config, parse_args
from results_store import sampling_of

QUESTION = MedQAQuestion(
    question_id="1",
    question="Q?",
    options={"A": "a", "B": "b"},
    answer_idx="A",
    answer="a",
    meta_info=None,
)

# The reference prompt as it was before the variants existed. Every stored
# step 2 run was generated with exactly this text.
REFERENCE_PROMPT = (
    "Analyze the following multiple-choice medical examination question and options. "
    "Extract and generate all key atomic, verifiable medical facts, clinical principles, "
    "pathophysiological mechanisms, and pharmacological properties relevant to solving "
    "this question accurately.\n\n"
    "Question: Q?\n\n"
    "Options:\nA. a\nB. b\n\n"
    "Generate atomic, verifiable statements as structured JSON conforming to the schema."
)


def test_default_fact_prompt_is_unchanged():
    assert format_fact_generation_prompt(QUESTION) == REFERENCE_PROMPT
    assert format_fact_generation_prompt(QUESTION, "reference") == REFERENCE_PROMPT


def test_background_prompt_keeps_question_and_options():
    prompt = format_fact_generation_prompt(QUESTION, "background")
    assert prompt.startswith(FACT_PROMPTS["background"].task)
    assert "Question: Q?" in prompt
    assert "Options:\nA. a\nB. b" in prompt
    assert "never mention the patient" in prompt


def test_unknown_fact_prompt_is_rejected():
    with pytest.raises(ValueError, match="fact_prompt"):
        CandidateInferenceConfig(fact_prompt="nope").validate()


def test_cli_selects_the_prompt_and_the_run_records_it():
    default = build_config(parse_args([]))
    chosen = build_config(parse_args(["--fact-prompt", "background"]))
    assert default.fact_prompt == "reference"
    assert chosen.fact_prompt == "background"
    assert sampling_of(chosen)["fact_prompt"] == "background"


def test_case_grounded_prompt_keeps_question_and_options():
    prompt = format_fact_generation_prompt(QUESTION, "case-grounded")
    assert prompt.startswith(FACT_PROMPTS["case-grounded"].task)
    assert "Options:\nA. a\nB. b" in prompt
    assert "state the finding itself inside the fact" in prompt


def test_experimental_prompts_drop_the_answer_anchor():
    for name in ("background", "case-grounded"):
        assert "relevant to answering" not in FACT_PROMPTS[name].system_instruction


REFERENCE_ANSWER_PROMPT = (
    "The following is a multiple-choice medical examination question.\n\n"
    "Question: Q?\n\n"
    "Options:\nA. a\nB. b\n\n"
    "Relevant Medical Facts:\n- f1\n- f2\n\n"
    "Based on these clinical facts, reason through the scenario and determine the single "
    "best option. Conclude your response on a new line with:\n"
    "FINAL ANSWER: [Option Letter]"
)


def test_default_answer_prompt_is_unchanged():
    from inference._dataset import format_answer_generation_prompt

    assert format_answer_generation_prompt(QUESTION, ["f1", "f2"]) == REFERENCE_ANSWER_PROMPT


def test_cited_facts_prompt_numbers_the_facts():
    from inference._dataset import format_answer_generation_prompt

    prompt = format_answer_generation_prompt(QUESTION, ["f1", "f2"], "cited-facts")
    assert "[1] f1\n[2] f2" in prompt
    assert "FACTS USED:" in prompt and "FINAL ANSWER:" in prompt


def test_grounded_unframed_prompt_sets_no_coverage():
    task = FACT_PROMPTS["grounded-unframed"].task
    assert "no more than 50 facts" in task
    assert "each condition" not in task and "cover every option" not in task


@pytest.mark.parametrize(
    "text,n,expected",
    [
        ("Reasoning.\nFACTS USED: 1, 4, 7\nFINAL ANSWER: B", 10, [0, 3, 6]),
        ("FACTS USED: [2,3]\nFINAL ANSWER: C", 5, [1, 2]),
        ("FACTS USED: 3, 99\nFINAL ANSWER: C", 5, [2]),
        ("FINAL ANSWER: C", 5, None),
    ],
)
def test_extract_cited_facts(text, n, expected):
    from inference.parser import extract_cited_facts

    assert extract_cited_facts(text, n) == expected


def test_cli_selects_the_answer_prompt():
    chosen = build_config(parse_args(["--answer-prompt", "cited-facts"]))
    assert chosen.answer_prompt == "cited-facts"
    assert sampling_of(chosen)["answer_prompt"] == "cited-facts"
