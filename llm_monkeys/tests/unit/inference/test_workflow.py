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


@pytest.mark.asyncio
async def test_workflow_adds_missing_candidates_only(tmp_path):
    output_file = tmp_path / "test_missing_candidates.json"

    call_count = 0

    async def mock_fact_run(**kwargs):
        nonlocal call_count
        call_count += 1
        yield make_mock_event(
            '{"facts": ["Fact"]}',
            MagicMock(
                prompt_token_count=10, candidates_token_count=10, total_token_count=20
            ),
        )

    async def mock_ans_run(**kwargs):
        yield make_mock_event(
            "FINAL ANSWER: A",
            MagicMock(
                prompt_token_count=10, candidates_token_count=10, total_token_count=20
            ),
        )

    mock_fact = MagicMock()
    mock_fact.run_async = mock_fact_run
    mock_ans = MagicMock()
    mock_ans.run_async = mock_ans_run

    q0 = MedQAQuestion(
        question_id="0",
        question="Q0",
        options={"A": "Opt A"},
        answer_idx="A",
        answer="Opt A",
    )

    # Run 1: generate 2 candidates
    config1 = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=2, concurrency=1
    )
    wf1 = CandidateInferenceWorkflow(
        config=config1, fact_runner=mock_fact, answer_runner=mock_ans
    )
    summary1, results1 = await wf1.run(questions=[q0])
    assert summary1.total_candidates_generated == 2
    assert call_count == 2
    assert len(results1[0].candidates) == 2
    assert [c.candidate_index for c in results1[0].candidates] == [0, 1]

    # Run 2: request 4 candidates -> only 2 missing candidates should be generated
    config2 = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=4, concurrency=1
    )
    wf2 = CandidateInferenceWorkflow(
        config=config2, fact_runner=mock_fact, answer_runner=mock_ans
    )
    summary2, results2 = await wf2.run(questions=[q0])
    assert summary2.total_candidates_generated == 4
    # call_count should have increased by 2 (candidates 2 and 3), not 4
    assert call_count == 4
    assert len(results2[0].candidates) == 4
    assert [c.candidate_index for c in results2[0].candidates] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_workflow_skips_existing_and_only_runs_new_question(tmp_path):
    output_file = tmp_path / "test_skip_and_new.json"

    async def mock_fact_run(**kwargs):
        yield make_mock_event('{"facts": ["Fact"]}', None)

    async def mock_ans_run(**kwargs):
        yield make_mock_event("FINAL ANSWER: A", None)

    mock_fact = MagicMock()
    mock_fact.run_async = mock_fact_run
    mock_ans = MagicMock()
    mock_ans.run_async = mock_ans_run

    q0 = MedQAQuestion(
        question_id="0", question="Q0", options={"A": "A"}, answer_idx="A", answer="A"
    )
    q1 = MedQAQuestion(
        question_id="1", question="Q1", options={"A": "A"}, answer_idx="A", answer="A"
    )

    # Step 1: run Q0 only
    config1 = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=2, concurrency=1
    )
    wf1 = CandidateInferenceWorkflow(
        config=config1, fact_runner=mock_fact, answer_runner=mock_ans
    )
    await wf1.run(questions=[q0])

    # Track invocations for Step 2
    fact_call_count = 0

    async def mock_fact_run2(**kwargs):
        nonlocal fact_call_count
        fact_call_count += 1
        yield make_mock_event('{"facts": ["Fact"]}', None)

    mock_fact2 = MagicMock()
    mock_fact2.run_async = mock_fact_run2

    # Step 2: run with Q0 and Q1. Q0 should be skipped completely, only Q1 should be run.
    config2 = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=2, concurrency=1
    )
    wf2 = CandidateInferenceWorkflow(
        config=config2, fact_runner=mock_fact2, answer_runner=mock_ans
    )
    summary2, results2 = await wf2.run(questions=[q0, q1])

    # Only 2 calls to mock_fact2 (for Q1's 2 candidates; Q0 was skipped)
    assert fact_call_count == 2
    assert len(results2) == 2
    assert summary2.total_questions == 2
    assert summary2.total_candidates_generated == 4

    # File should contain both Q0 and Q1
    saved = wf2.load_existing_results(output_file)
    assert "0" in saved
    assert "1" in saved
    assert len(saved["0"].candidates) == 2
    assert len(saved["1"].candidates) == 2


@pytest.mark.asyncio
async def test_workflow_preserves_other_questions_when_running_disjoint_subset(
    tmp_path,
):
    output_file = tmp_path / "test_disjoint.json"

    async def mock_fact_run(**kwargs):
        yield make_mock_event('{"facts": ["Fact"]}', None)

    async def mock_ans_run(**kwargs):
        yield make_mock_event("FINAL ANSWER: A", None)

    mock_fact = MagicMock()
    mock_fact.run_async = mock_fact_run
    mock_ans = MagicMock()
    mock_ans.run_async = mock_ans_run

    q0 = MedQAQuestion(
        question_id="100",
        question="Q100",
        options={"A": "A"},
        answer_idx="A",
        answer="A",
    )
    q1 = MedQAQuestion(
        question_id="200",
        question="Q200",
        options={"A": "A"},
        answer_idx="A",
        answer="A",
    )

    # Save Q0 first
    config1 = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=1, concurrency=1
    )
    wf1 = CandidateInferenceWorkflow(
        config=config1, fact_runner=mock_fact, answer_runner=mock_ans
    )
    await wf1.run(questions=[q0])

    # Run with Q1 only
    config2 = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=1, concurrency=1
    )
    wf2 = CandidateInferenceWorkflow(
        config=config2, fact_runner=mock_fact, answer_runner=mock_ans
    )
    summary2, results2 = await wf2.run(questions=[q1])

    # Results returned should be for Q1
    assert len(results2) == 1
    assert results2[0].question_id == "200"

    # But file must contain BOTH Q100 and Q200
    saved = wf2.load_existing_results(output_file)
    assert len(saved) == 2
    assert "100" in saved
    assert "200" in saved
    assert summary2.total_questions == 2


@pytest.mark.asyncio
async def test_workflow_retries_failed_candidate(tmp_path):
    from inference._schemas import (
        CandidateQuestionResult,
        CandidateResult,
        StepTokenUsage,
    )

    output_file = tmp_path / "test_failed_retry.json"

    # Create an existing result where candidate 0 succeeded, candidate 1 failed
    c0 = CandidateResult(
        candidate_index=0,
        facts=["Fact 0"],
        facts_raw_response="",
        answer_raw_response="",
        predicted_option="A",
        is_correct=True,
        fact_latency_seconds=0.1,
        answer_latency_seconds=0.1,
        total_latency_seconds=0.2,
        fact_tokens=StepTokenUsage(),
        answer_tokens=StepTokenUsage(),
        total_prompt_tokens=10,
        total_candidate_tokens=10,
        total_tokens=20,
        error=None,
    )
    c1 = CandidateResult(
        candidate_index=1,
        facts=[],
        facts_raw_response="",
        answer_raw_response="",
        predicted_option=None,
        is_correct=False,
        fact_latency_seconds=0.1,
        answer_latency_seconds=0.0,
        total_latency_seconds=0.1,
        fact_tokens=StepTokenUsage(),
        answer_tokens=StepTokenUsage(),
        total_prompt_tokens=5,
        total_candidate_tokens=0,
        total_tokens=5,
        error="Fact generation error: timeout",
    )
    qres = CandidateQuestionResult(
        question_id="5",
        meta_info=None,
        question="Q5?",
        options={"A": "Option A"},
        ground_truth="A",
        ground_truth_answer="Option A",
        candidates=[c0, c1],
    )
    qres.update_aggregates()

    config = CandidateInferenceConfig(
        output_filepath=str(output_file), n_candidates=2, concurrency=1
    )
    wf = CandidateInferenceWorkflow(config=config)
    wf.save_results(output_file, [qres])

    async def mock_fact_run(**kwargs):
        yield make_mock_event('{"facts": ["Retried fact"]}', None)

    async def mock_ans_run(**kwargs):
        yield make_mock_event("FINAL ANSWER: A", None)

    mock_fact = MagicMock()
    mock_fact.run_async = mock_fact_run
    mock_ans = MagicMock()
    mock_ans.run_async = mock_ans_run

    wf.fact_runner = mock_fact
    wf.answer_runner = mock_ans

    q5 = MedQAQuestion(
        question_id="5",
        question="Q5?",
        options={"A": "Option A"},
        answer_idx="A",
        answer="Option A",
    )
    summary, results = await wf.run(questions=[q5])

    # Only candidate 1 should have been retried
    assert len(results[0].candidates) == 2
    res_c0 = results[0].candidates[0]
    res_c1 = results[0].candidates[1]
    assert res_c0.candidate_index == 0
    assert res_c0.error is None
    assert res_c0.facts == ["Fact 0"]  # preserved from before!

    assert res_c1.candidate_index == 1
    assert res_c1.error is None
    assert res_c1.facts == ["Retried fact"]  # retried successfully!


def test_load_existing_results_formats(tmp_path):
    wf = CandidateInferenceWorkflow()

    # None and empty path
    assert wf.load_existing_results(None) == {}
    assert wf.load_existing_results("") == {}
    assert wf.load_existing_results(tmp_path / "nonexistent.json") == {}

    # File with list format
    list_file = tmp_path / "list.json"
    list_file.write_text(
        '[{"question_id": 42, "question": "Q?", "options": {}, "ground_truth": "A", "ground_truth_answer": "A"}]'
    )
    res_list = wf.load_existing_results(list_file)
    assert "42" in res_list
    assert res_list["42"].question_id == "42"

    # File with dict "results" format
    dict_file = tmp_path / "dict.json"
    dict_file.write_text(
        '{"summary": {}, "results": [{"question_id": "99", "question": "Q?", "options": {}, "ground_truth": "B", "ground_truth_answer": "B"}]}'
    )
    res_dict = wf.load_existing_results(dict_file)
    assert "99" in res_dict
    assert res_dict["99"].question_id == "99"


def test_is_rate_limit_error():
    assert is_rate_limit_error(Exception("429 Resource exhausted")) is True
    assert is_rate_limit_error(Exception("The request queue is full.")) is True
    assert is_rate_limit_error(Exception("quota exceeded")) is True
    assert is_rate_limit_error(ValueError("Invalid argument")) is False
