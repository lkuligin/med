"""Unit tests for inference.workflow module."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import CandidateInferenceConfig
from dataset import MedQAQuestion
from inference.workflow import CandidateInferenceWorkflow, is_rate_limit_error


def make_mock_event(text: str, usage: Any = None) -> MagicMock:
    part = MagicMock(text=text, thought=False)
    return MagicMock(content=MagicMock(parts=[part]), usage_metadata=usage)


@pytest.mark.asyncio
async def test_workflow_run_single_candidate():
    mock_fact_runner = MagicMock()
    mock_ans_runner = MagicMock()

    usage_fact = MagicMock(
        prompt_token_count=100,
        candidates_token_count=40,
        total_token_count=140,
        cached_content_token_count=0,
        thoughts_token_count=0,
    )
    usage_ans = MagicMock(
        prompt_token_count=150,
        candidates_token_count=60,
        total_token_count=210,
        cached_content_token_count=0,
        thoughts_token_count=0,
    )

    async def mock_fact_run(**kwargs):
        yield make_mock_event(
            '{"facts": ["Verified clinical fact 1", "Verified clinical fact 2"]}',
            usage_fact,
        )

    async def mock_ans_run(**kwargs):
        yield make_mock_event(
            "Reasoning through options...\nFINAL ANSWER: A", usage_ans
        )

    mock_fact_runner.run_async = mock_fact_run
    mock_ans_runner.run_async = mock_ans_run

    config = CandidateInferenceConfig(output_filepath="")
    workflow = CandidateInferenceWorkflow(
        config=config,
        fact_runner=mock_fact_runner,
        answer_runner=mock_ans_runner,
    )

    q = MedQAQuestion(
        question_id="0",
        question="Which drug causes ototoxicity?",
        options={"A": "Cisplatin", "B": "Paracetamol"},
        answer_idx="A",
        answer="Cisplatin",
    )

    semaphore = asyncio.Semaphore(1)
    result = await workflow.run_single_candidate(
        q, candidate_idx=0, semaphore=semaphore
    )

    assert result.candidate_index == 0
    assert result.facts == ["Verified clinical fact 1", "Verified clinical fact 2"]
    assert result.predicted_option == "A"
    assert result.is_correct is True
    assert result.total_prompt_tokens == 250
    assert result.total_candidate_tokens == 100
    assert result.total_tokens == 350
    assert result.error is None


@pytest.mark.asyncio
async def test_workflow_run_candidate_fact_error():
    mock_fact_runner = MagicMock()
    mock_ans_runner = MagicMock()

    async def mock_fact_fail(**kwargs):
        raise RuntimeError("Fact service timeout")
        yield  # Make it generator

    mock_fact_runner.run_async = mock_fact_fail

    config = CandidateInferenceConfig(max_retries=0, rate_limit_max_retries=0)
    workflow = CandidateInferenceWorkflow(
        config=config,
        fact_runner=mock_fact_runner,
        answer_runner=mock_ans_runner,
    )

    q = MedQAQuestion(
        question_id="0",
        question="Q?",
        options={"A": "Opt A"},
        answer_idx="A",
        answer="Opt A",
    )

    result = await workflow.run_single_candidate(
        q, candidate_idx=0, semaphore=asyncio.Semaphore(1)
    )
    assert result.error is not None
    assert "Fact generation error" in result.error
    assert result.is_correct is False


@pytest.mark.asyncio
async def test_workflow_end_to_end_with_resume(tmp_path):
    output_file = tmp_path / "test_candidates.json"

    mock_fact_runner = MagicMock()
    mock_ans_runner = MagicMock()

    async def mock_fact_run(**kwargs):
        yield make_mock_event(
            '{"facts": ["Fact 1"]}',
            MagicMock(
                prompt_token_count=50,
                candidates_token_count=20,
                total_token_count=70,
                cached_content_token_count=0,
                thoughts_token_count=0,
            ),
        )

    async def mock_ans_run(**kwargs):
        yield make_mock_event(
            "Reasoning...\nFINAL ANSWER: A",
            MagicMock(
                prompt_token_count=70,
                candidates_token_count=30,
                total_token_count=100,
                cached_content_token_count=0,
                thoughts_token_count=0,
            ),
        )

    mock_fact_runner.run_async = mock_fact_run
    mock_ans_runner.run_async = mock_ans_run

    config = CandidateInferenceConfig(
        output_filepath=str(output_file),
        n_candidates=2,
        concurrency=2,
    )
    workflow = CandidateInferenceWorkflow(
        config=config,
        fact_runner=mock_fact_runner,
        answer_runner=mock_ans_runner,
    )

    questions = [
        MedQAQuestion(
            question_id="0",
            question="Q0",
            options={"A": "Opt A"},
            answer_idx="A",
            answer="Opt A",
        ),
        MedQAQuestion(
            question_id="1",
            question="Q1",
            options={"A": "Opt A"},
            answer_idx="A",
            answer="Opt A",
        ),
    ]

    summary, results = await workflow.run(questions=questions)
    assert summary.total_questions == 2
    assert summary.total_candidates_generated == 4
    assert summary.total_correct_candidates == 4
    assert output_file.exists()

    # Test resume: running again should load existing results without error
    summary2, results2 = await workflow.run(questions=questions)
    assert summary2.total_questions == 2
    assert summary2.total_candidates_generated == 4


def test_is_rate_limit_error():
    assert is_rate_limit_error(Exception("429 Resource exhausted")) is True
    assert is_rate_limit_error(Exception("The request queue is full.")) is True
    assert is_rate_limit_error(Exception("quota exceeded")) is True
    assert is_rate_limit_error(ValueError("Invalid argument")) is False
