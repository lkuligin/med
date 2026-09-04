import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from config import DEFAULT_MODEL, resolve_model_name
from dataset import MedQAQuestion
from one_shot.workflow import (
    AttemptResult,
    InferenceItemResult,
    TokenUsageAccumulator,
    WorkflowSummary,
    is_rate_limit_error,
)


def make_mock_event(text: str, thought: bool = False, usage: Any = None) -> MagicMock:
    """Helper to create a mock ADK runner stream event."""
    part = MagicMock(text=text, thought=thought)
    return MagicMock(content=MagicMock(parts=[part]), usage_metadata=usage)


def make_result_dict(
    question_id: str = "0",
    predicted_option: str | None = "A",
    ground_truth: str = "A",
    is_correct: bool = True,
    error: str | None = None,
    n_attempts: int = 3,
    attempt_errors: list[str | None] | None = None,
    raw_response: str = "Answer: A",
) -> dict[str, Any]:
    """Helper to generate serializable InferenceItemResult dictionaries for resume tests."""
    attempts = []
    for i in range(n_attempts):
        att_err = (
            attempt_errors[i] if (attempt_errors and i < len(attempt_errors)) else None
        )
        att_pred = None if att_err else predicted_option
        att_correct = is_correct and (att_err is None)
        attempts.append(
            {
                "attempt_index": i,
                "predicted_option": att_pred,
                "is_correct": att_correct,
                "raw_response": raw_response if not att_err else "",
                "latency_seconds": 0.2,
                "error": att_err,
            }
        )
    has_error = error is not None or any(a["error"] for a in attempts)
    all_correct = is_correct and not has_error
    return {
        "question_id": question_id,
        "question": f"Sample question {question_id}",
        "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
        "ground_truth": ground_truth,
        "ground_truth_answer": f"Opt {ground_truth}",
        "predicted_option": predicted_option if not error else None,
        "is_correct": all_correct,
        "is_all_correct": all_correct,
        "correct_attempts": sum(1 for a in attempts if a["is_correct"]),
        "total_attempts": len(attempts),
        "raw_response": raw_response if not error else "",
        "prompt": f"prompt {question_id}",
        "latency_seconds": 0.5,
        "prompt_tokens": 100,
        "candidate_tokens": 20,
        "total_tokens": 120,
        "cached_tokens": 0,
        "error": error,
        "parse_retries": 0,
        "attempts": attempts,
    }


def write_results_file(
    filepath: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> None:
    """Helper to dump pre-existing results to JSON file."""
    data: dict[str, Any] = {"results": results}
    if summary:
        data["summary"] = summary
    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core Workflow Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_run_with_mocked_runner(
    sample_questions, tmp_path, workflow_factory
):
    output_file = tmp_path / "test_results.json"
    workflow = workflow_factory(
        output_filepath=str(output_file),
        concurrency=2,
        n_attempts=3,
    )

    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        ans = (
            "Answer: A\nExplanation: test"
            if "Sample question 1" in prompt_text
            else "Answer: C\nExplanation: test"
        )
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120,
        )
        return ans, usage

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    # 2 questions x 3 attempts = 6 model invocations
    assert call_count == 6

    assert isinstance(summary, WorkflowSummary)
    assert summary.n_attempts == 3
    assert summary.total_questions == 2
    assert summary.completed == 2
    assert summary.failed == 0
    assert summary.correct == 1
    assert summary.simple_questions == 1
    assert summary.difficult_questions == 1
    assert summary.accuracy == 0.5
    # 6 attempts x 100 prompt tokens = 600
    assert summary.total_prompt_tokens == 600
    # 6 attempts x 20 candidate tokens = 120
    assert summary.total_candidate_tokens == 120
    assert summary.total_tokens == 720

    assert len(results) == 2
    # Question 0: All 3 attempts correct
    assert results[0].question_id == "0"
    assert results[0].predicted_option == "A"
    assert results[0].is_correct is True
    assert results[0].is_all_correct is True
    assert results[0].correct_attempts == 3
    assert results[0].total_attempts == 3
    assert len(results[0].attempts) == 3
    for i, att in enumerate(results[0].attempts):
        assert att.attempt_index == i
        assert att.predicted_option == "A"
        assert att.is_correct is True
        assert att.prompt_tokens == 100
        assert att.candidate_tokens == 20
        assert att.total_tokens == 120

    # Question 1: All 3 attempts predicted C instead of B
    assert results[1].question_id == "1"
    assert results[1].predicted_option == "C"
    assert results[1].is_correct is False
    assert results[1].is_all_correct is False
    assert results[1].correct_attempts == 0
    assert results[1].total_attempts == 3
    assert len(results[1].attempts) == 3

    assert output_file.exists()
    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "summary" in saved_data
    assert "results" in saved_data
    assert saved_data["summary"]["total_questions"] == 2
    assert saved_data["summary"]["n_attempts"] == 3
    assert saved_data["summary"]["simple_questions"] == 1
    assert saved_data["summary"]["difficult_questions"] == 1
    assert len(saved_data["results"]) == 2
    assert saved_data["results"][0]["ground_truth"] == "A"
    assert saved_data["results"][0]["predicted_option"] == "A"
    assert len(saved_data["results"][0]["attempts"]) == 3
    assert saved_data["results"][0]["attempts"][0]["predicted_option"] == "A"
    assert saved_data["results"][0]["attempts"][0]["is_correct"] is True


@pytest.mark.asyncio
async def test_workflow_simple_vs_difficult_split(sample_questions, workflow_factory):
    workflow = workflow_factory(concurrency=2, n_attempts=3)
    q1_calls = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal q1_calls
        if "Sample question 1" in prompt_text:
            # Q1 is consistently correct (all 3 attempts) -> Simple
            return "Answer: A\nExplanation: text", None
        else:
            # Q2 is partially correct (2 out of 3) -> Difficult
            q1_calls += 1
            if q1_calls <= 2:
                return "Answer: B\nExplanation: text", None
            return "Answer: D\nExplanation: text", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    assert summary.simple_questions == 1
    assert summary.difficult_questions == 1
    assert summary.correct == 1
    assert summary.accuracy == 0.5

    assert results[0].is_all_correct is True
    assert results[0].correct_attempts == 3

    assert results[1].is_all_correct is False
    assert results[1].correct_attempts == 2
    assert results[1].total_attempts == 3
    assert len(results[1].attempts) == 3
    assert results[1].attempts[0].is_correct is True
    assert results[1].attempts[1].is_correct is True
    assert results[1].attempts[2].is_correct is False


@pytest.mark.asyncio
async def test_workflow_error_handling(sample_questions, workflow_factory):
    workflow = workflow_factory(concurrency=2, n_attempts=3)

    async def mock_invoke_with_error(prompt_text, session_id, user_id="med_eval_user"):
        if "Sample question 1" in prompt_text:
            raise RuntimeError("API rate limit exceeded")
        return "Answer: B", None

    workflow._invoke_agent_with_retry = mock_invoke_with_error

    progress_events = []

    def callback(completed, total, res):
        progress_events.append((completed, total, res))

    summary, results = await workflow.run(
        questions=sample_questions, progress_callback=callback
    )

    assert summary.total_questions == 2
    assert summary.completed == 1
    assert summary.failed == 1
    assert summary.correct == 1
    assert summary.accuracy == 0.5

    assert results[0].error == "API rate limit exceeded"
    assert results[0].is_correct is False
    assert results[0].is_all_correct is False
    assert len(results[0].attempts) == 3
    for att in results[0].attempts:
        assert att.error == "API rate limit exceeded"

    assert results[1].error is None
    assert results[1].is_correct is True
    assert results[1].is_all_correct is True
    assert len(results[1].attempts) == 3

    assert len(progress_events) == 2


@pytest.mark.asyncio
async def test_workflow_empty_questions(workflow_factory):
    workflow = workflow_factory(n_attempts=3)
    summary, results = await workflow.run(questions=[])
    assert summary.total_questions == 0
    assert summary.completed == 0
    assert summary.failed == 0
    assert summary.correct == 0
    assert summary.accuracy == 0.0
    assert summary.simple_questions == 0
    assert summary.difficult_questions == 0
    assert summary.average_latency_seconds == 0.0
    assert results == []


@pytest.mark.asyncio
async def test_workflow_dumps_periodically_every_n_examples(tmp_path, workflow_factory):
    output_file = tmp_path / "periodic_results.json"
    questions = [
        MedQAQuestion(
            question_id=f"q_{i}",
            question=f"Question {i}",
            options={"A": "Opt A", "B": "Opt B"},
            answer_idx="A",
            answer="Opt A",
        )
        for i in range(5)
    ]

    workflow = workflow_factory(
        output_filepath=str(output_file),
        concurrency=1,
        n_attempts=2,
        save_every_n=2,  # dump every 2 examples
    )

    save_counts = []
    original_save = workflow._save_to_json

    def tracked_save(filepath, summary, results):
        save_counts.append(len(results))
        return original_save(filepath, summary, results)

    workflow._save_to_json = tracked_save

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        return "Answer: A\nExplanation: test", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=questions)

    assert len(results) == 5
    # For 5 items with save_every_n=2:
    # Intermediate saves at count 2 and 4, plus final save at count 5
    assert save_counts == [2, 4, 5]
    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved_data["results"]) == 5
    for r in saved_data["results"]:
        assert len(r["attempts"]) == 2


# ---------------------------------------------------------------------------
# Parse Retry Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_retries_on_unparsed_option_success(
    single_question, workflow_factory
):
    workflow = workflow_factory(max_parse_retries=3, concurrency=1, n_attempts=1)
    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120,
        )
        if call_count == 1:
            # First attempt: unparseable output
            return "I am not completely certain about the exact answer.", usage
        # Second attempt: parseable output
        return "Answer: A\nExplanation: Cisplatin is ototoxic.", usage

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[single_question])

    assert call_count == 2
    assert summary.total_questions == 1
    assert summary.completed == 1
    assert summary.correct == 1
    assert summary.accuracy == 1.0
    assert summary.total_tokens == 240
    assert summary.total_prompt_tokens == 200
    assert summary.total_candidate_tokens == 40

    assert len(results) == 1
    assert results[0].predicted_option == "A"
    assert results[0].is_correct is True
    assert results[0].parse_retries == 1
    assert len(results[0].attempts) == 1
    assert results[0].attempts[0].parse_retries == 1


@pytest.mark.asyncio
async def test_workflow_retries_on_unparsed_option_exhausted(
    single_question, workflow_factory
):
    workflow = workflow_factory(max_parse_retries=3, concurrency=1, n_attempts=1)
    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50,
            candidates_token_count=10,
            total_token_count=60,
        )
        return "Unknown unparseable text", usage

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[single_question])

    # Initial attempt (1) + max_parse_retries (3) = 4 total attempts
    assert call_count == 4
    assert summary.total_questions == 1
    assert summary.completed == 1
    assert summary.correct == 0
    assert summary.accuracy == 0.0
    assert summary.total_tokens == 240

    assert len(results) == 1
    assert results[0].predicted_option is None
    assert results[0].is_correct is False
    assert results[0].parse_retries == 3
    assert len(results[0].attempts) == 1
    assert results[0].attempts[0].parse_retries == 3


@pytest.mark.asyncio
async def test_workflow_retries_configurable_k(single_question, workflow_factory):
    workflow = workflow_factory(max_parse_retries=1, concurrency=1, n_attempts=1)
    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        return "Still unparseable", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[single_question])

    # 1 initial + 1 retry = 2 attempts
    assert call_count == 2
    assert results[0].predicted_option is None
    assert results[0].parse_retries == 1


@pytest.mark.asyncio
async def test_workflow_skips_already_processed_questions(
    sample_questions, tmp_path, workflow_factory
):
    output_file = tmp_path / "resume_results.json"
    cached_q0 = make_result_dict(
        "0",
        predicted_option="A",
        raw_response="Answer: A\nExplanation: cached answer",
    )
    summary_data = {
        "model": "test-model",
        "dataset": "bigbio/med_qa",
        "config": "med_qa_en_source",
        "split": "test",
        "n_attempts": 3,
        "total_questions": 1,
        "completed": 1,
        "failed": 0,
        "correct": 1,
        "accuracy": 1.0,
        "simple_questions": 1,
        "difficult_questions": 0,
        "total_time_seconds": 1.5,
        "average_latency_seconds": 1.5,
        "total_tokens": 120,
        "total_prompt_tokens": 100,
        "total_candidate_tokens": 20,
        "created_at": "2026-09-02T12:00:00Z",
    }
    write_results_file(output_file, [cached_q0], summary=summary_data)

    workflow = workflow_factory(
        output_filepath=str(output_file),
        concurrency=2,
        n_attempts=3,
    )

    call_count = 0
    invoked_prompts = []

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        invoked_prompts.append(prompt_text)
        return "Answer: B\nExplanation: fresh run", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    # Only question "1" should have triggered mock_invoke (3 attempts)
    assert call_count == 3
    assert all("Sample question 2" in p for p in invoked_prompts)

    assert summary.total_questions == 2
    assert summary.completed == 2
    assert summary.correct == 2

    assert len(results) == 2
    assert results[0].question_id == "0"
    assert results[0].raw_response == "Answer: A\nExplanation: cached answer"
    assert len(results[0].attempts) == 3
    assert results[1].question_id == "1"
    assert results[1].raw_response == "Answer: B\nExplanation: fresh run"
    assert len(results[1].attempts) == 3

    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 2
    assert saved["results"][0]["question_id"] == "0"
    assert len(saved["results"][0]["attempts"]) == 3
    assert saved["results"][1]["question_id"] == "1"
    assert len(saved["results"][1]["attempts"]) == 3


@pytest.mark.asyncio
async def test_workflow_skips_all_when_all_processed(
    sample_questions, tmp_path, workflow_factory
):
    output_file = tmp_path / "all_processed.json"
    write_results_file(
        output_file,
        [
            make_result_dict("0", predicted_option="A", raw_response="Cached 0"),
            make_result_dict(
                "1",
                predicted_option="B",
                ground_truth="B",
                raw_response="Cached 1",
            ),
        ],
    )

    workflow = workflow_factory(output_filepath=str(output_file), n_attempts=3)
    mock_invoke = AsyncMock()
    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    assert mock_invoke.call_count == 0
    assert summary.total_questions == 2
    assert summary.completed == 2
    assert summary.correct == 2
    assert len(results) == 2


@pytest.mark.asyncio
async def test_workflow_reprocesses_failed_questions_from_existing_file(
    sample_questions, tmp_path, workflow_factory
):
    """Test that failed questions in existing file are re-processed while successful ones are skipped."""
    output_file = tmp_path / "partially_failed.json"
    write_results_file(
        output_file,
        [
            make_result_dict(
                "0",
                predicted_option="A",
                raw_response="Cached answer for Q0",
            ),
            make_result_dict(
                "1",
                predicted_option=None,
                ground_truth="B",
                is_correct=False,
                error="litellm.RateLimitError: 429 RESOURCE_EXHAUSTED",
                attempt_errors=[
                    None,
                    "litellm.RateLimitError: 429 RESOURCE_EXHAUSTED",
                    "litellm.RateLimitError: 429 RESOURCE_EXHAUSTED",
                ],
            ),
        ],
        summary={
            "model": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
            "total_questions": 2,
            "completed": 1,
            "failed": 1,
        },
    )

    workflow = workflow_factory(
        output_filepath=str(output_file),
        concurrency=2,
        n_attempts=3,
    )

    call_count = 0
    invoked_prompts = []

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        invoked_prompts.append(prompt_text)
        return (
            "Final Answer: Option B\nExplanation: successfully re-processed",
            None,
        )

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    # Question "0" was successful -> skipped (0 calls)
    # Question "1" was failed -> re-processed (3 attempts = 3 calls)
    assert call_count == 3
    assert all("Sample question 2" in p for p in invoked_prompts)

    assert summary.total_questions == 2
    assert summary.completed == 2
    assert summary.failed == 0
    assert summary.correct == 2

    assert len(results) == 2
    # Q0 kept cached successful result
    assert results[0].question_id == "0"
    assert results[0].error is None
    assert results[0].raw_response == "Cached answer for Q0"

    # Q1 updated with fresh result
    assert results[1].question_id == "1"
    assert results[1].error is None
    assert results[1].predicted_option == "B"
    assert results[1].is_correct is True
    assert results[1].is_all_correct is True
    assert "successfully re-processed" in results[1].raw_response

    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_data["summary"]["completed"] == 2
    assert saved_data["summary"]["failed"] == 0
    assert saved_data["results"][1]["error"] is None
    assert saved_data["results"][1]["is_all_correct"] is True


@pytest.mark.asyncio
async def test_workflow_reprocesses_attempt_level_error(
    sample_questions, tmp_path, workflow_factory
):
    """Test that question is re-processed if any attempt has an error even if top-level error is None."""
    output_file = tmp_path / "attempt_error.json"
    write_results_file(
        output_file,
        [
            make_result_dict(
                "0",
                predicted_option="A",
                error=None,
                attempt_errors=[None, "ConnectionResetError", None],
            )
        ],
    )

    workflow = workflow_factory(output_filepath=str(output_file), n_attempts=3)
    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        return "Final Answer: Option A\nExplanation: re-run", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[sample_questions[0]])

    assert call_count == 3
    assert summary.completed == 1
    assert summary.failed == 0
    assert results[0].error is None
    assert all(a.error is None for a in results[0].attempts)


@pytest.mark.asyncio
async def test_workflow_reprocesses_incomplete_attempts(
    sample_questions, tmp_path, workflow_factory
):
    """Test that a question with fewer attempts than configured n_attempts is re-processed."""
    output_file = tmp_path / "incomplete.json"
    write_results_file(
        output_file,
        [make_result_dict("0", predicted_option="A", n_attempts=1)],
    )

    workflow = workflow_factory(output_filepath=str(output_file), n_attempts=3)
    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        return "Final Answer: Option A\nExplanation: completed all 3", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[sample_questions[0]])

    assert call_count == 3
    assert len(results[0].attempts) == 3
    assert summary.completed == 1
    assert summary.failed == 0


@pytest.mark.asyncio
async def test_workflow_handles_corrupted_destination_file(
    sample_questions, tmp_path, workflow_factory
):
    output_file = tmp_path / "corrupted.json"
    output_file.write_text("{corrupted json content...", encoding="utf-8")

    workflow = workflow_factory(
        output_filepath=str(output_file), concurrency=2, n_attempts=1
    )
    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        return "Answer: A", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    # Should have processed all questions afresh without crashing
    assert call_count == 2
    assert summary.total_questions == 2
    assert len(results) == 2
    assert output_file.exists()


@pytest.mark.parametrize("thoughts_tokens", [None, 35])
def test_attempt_result_serialization(thoughts_tokens):
    att = AttemptResult(
        attempt_index=1,
        predicted_option="B",
        is_correct=True,
        raw_response="Answer: B",
        latency_seconds=1.23,
        prompt_tokens=50,
        candidate_tokens=10,
        total_tokens=60,
        cached_tokens=0,
        thoughts_tokens=thoughts_tokens,
        error=None,
        parse_retries=1,
    )
    d = att.to_dict()
    assert d["attempt_index"] == 1
    assert d["predicted_option"] == "B"
    assert d["is_correct"] is True
    assert d["thoughts_tokens"] == thoughts_tokens

    reconstructed = AttemptResult.from_dict(d)
    assert reconstructed.thoughts_tokens == thoughts_tokens
    assert reconstructed == att


def test_inference_item_result_serialization():
    attempt = AttemptResult(
        attempt_index=0,
        predicted_option="A",
        is_correct=True,
        raw_response="Answer: A",
        latency_seconds=2.0,
        prompt_tokens=100,
        candidate_tokens=50,
        total_tokens=150,
        cached_tokens=0,
        thoughts_tokens=40,
    )
    item = InferenceItemResult(
        question_id="q100",
        meta_info="step1",
        question="What is the treatment?",
        options={"A": "Drug A", "B": "Drug B"},
        ground_truth="A",
        ground_truth_answer="Drug A",
        prompt="prompt",
        predicted_option="A",
        is_correct=True,
        is_all_correct=True,
        correct_attempts=1,
        total_attempts=1,
        raw_response="Answer: A",
        latency_seconds=2.0,
        prompt_tokens=100,
        candidate_tokens=50,
        total_tokens=150,
        cached_tokens=0,
        thoughts_tokens=40,
        attempts=[attempt],
    )
    d = item.to_dict()
    assert d["thoughts_tokens"] == 40
    assert d["attempts"][0]["thoughts_tokens"] == 40

    reconstructed = InferenceItemResult.from_dict(d)
    assert reconstructed.thoughts_tokens == 40
    assert reconstructed.attempts[0].thoughts_tokens == 40
    assert reconstructed == item


def test_inference_item_result_legacy_deserialization():
    legacy_data = {
        "question_id": "legacy_1",
        "question": "Sample legacy question",
        "options": {"A": "Opt A", "B": "Opt B"},
        "ground_truth": "A",
        "ground_truth_answer": "Opt A",
        "predicted_option": "A",
        "is_correct": True,
        "raw_response": "Answer: A",
        "prompt": "legacy prompt",
        "latency_seconds": 0.42,
        "prompt_tokens": 100,
        "candidate_tokens": 20,
        "total_tokens": 120,
        "cached_tokens": 0,
        "error": None,
        "parse_retries": 0,
    }
    item = InferenceItemResult.from_dict(legacy_data)
    assert item.question_id == "legacy_1"
    assert item.is_correct is True
    assert item.is_all_correct is True
    assert item.correct_attempts == 1
    assert item.total_attempts == 1
    assert len(item.attempts) == 1
    assert item.attempts[0].predicted_option == "A"
    assert item.attempts[0].is_correct is True


def test_resolve_model_name():
    """Test model alias resolution for GPT-OSS and Gemma models."""
    assert resolve_model_name(None) == DEFAULT_MODEL
    assert resolve_model_name("") == DEFAULT_MODEL
    assert (
        resolve_model_name("openai/gpt-oss-20b-maas")
        == "vertex_ai/openai/gpt-oss-20b-maas"
    )
    assert (
        resolve_model_name("vertex_ai/openai/gpt-oss-120b-maas")
        == "vertex_ai/openai/gpt-oss-120b-maas"
    )
    assert (
        resolve_model_name("vertex_ai/openai/gpt-oss-20b-maas")
        == "vertex_ai/openai/gpt-oss-20b-maas"
    )
    assert resolve_model_name("custom-model-id") == "custom-model-id"


def test_token_usage_accumulator_with_thoughts():
    """Test accumulator handling thoughts/reasoning tokens."""
    acc = TokenUsageAccumulator()
    assert acc.thoughts_tokens == 0
    assert not acc.had_usage

    usage1 = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=150,
        cached_content_token_count=10,
        thoughts_token_count=30,
    )

    acc.add(usage1)
    assert acc.had_usage is True
    assert acc.prompt_tokens == 100
    assert acc.candidate_tokens == 50
    assert acc.total_tokens == 150
    assert acc.cached_tokens == 10
    assert acc.thoughts_tokens == 30

    usage2 = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=80,
        candidates_token_count=40,
        total_token_count=120,
        thoughts_token_count=25,
    )
    acc.add(usage2)
    assert acc.prompt_tokens == 180
    assert acc.candidate_tokens == 90
    assert acc.total_tokens == 270
    assert acc.thoughts_tokens == 55


def test_workflow_summary_with_thoughts():
    """Test WorkflowSummary aggregating thoughts tokens."""
    item1 = InferenceItemResult(
        question_id="q1",
        meta_info=None,
        question="Q1",
        options={"A": "A1"},
        ground_truth="A",
        ground_truth_answer="A1",
        prompt="p",
        is_all_correct=True,
        total_tokens=100,
        prompt_tokens=60,
        candidate_tokens=40,
        thoughts_tokens=25,
    )
    item2 = InferenceItemResult(
        question_id="q2",
        meta_info=None,
        question="Q2",
        options={"B": "B1"},
        ground_truth="B",
        ground_truth_answer="B1",
        prompt="p",
        is_all_correct=True,
        total_tokens=120,
        prompt_tokens=70,
        candidate_tokens=50,
        thoughts_tokens=35,
    )
    summary = WorkflowSummary.from_results(
        results=[item1, item2],
        model="vertex_ai/openai/gpt-oss-120b-maas",
        dataset="bigbio/med_qa",
        config="med_qa_en_source",
        split="test",
        n_attempts=1,
        total_time_seconds=3.5,
    )
    assert summary.total_thoughts_tokens == 60
    assert summary.model == "vertex_ai/openai/gpt-oss-120b-maas"
    d = summary.to_dict()
    assert d["total_thoughts_tokens"] == 60


# ---------------------------------------------------------------------------
# Runner Event Stream and Thought Processing Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_agent_separates_thoughts_and_output(workflow_factory):
    """Test _invoke_agent_with_retry separates thought chunks and final text."""
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=120,
        candidates_token_count=60,
        total_token_count=180,
    )
    setattr(usage, "thoughts_token_count", 40)

    events = [
        make_mock_event("Thinking about option A versus B... ", thought=True),
        make_mock_event("Deciding on B.\n", thought=True),
        make_mock_event(
            "Final Answer: Option B\nExplanation: clinical reasoning.",
            usage=usage,
        ),
    ]

    async def mock_run_async(*args, **kwargs):
        for event in events:
            yield event

    mock_runner = MagicMock(run_async=mock_run_async)
    workflow = workflow_factory(
        model_name="vertex_ai/openai/gpt-oss-20b-maas",
        max_retries=1,
        runner=mock_runner,
    )
    assert workflow.config.model_name == "vertex_ai/openai/gpt-oss-20b-maas"

    output_text, usage_meta = await workflow._invoke_agent_with_retry(
        prompt_text="Solve medical question",
        session_id="test_sess",
    )

    assert "Thinking about option" not in output_text
    assert output_text == "Final Answer: Option B\nExplanation: clinical reasoning."
    assert usage_meta == usage


@pytest.mark.asyncio
async def test_invoke_agent_fallback_when_only_thoughts(workflow_factory):
    """Test _invoke_agent_with_retry falls back to thought parts if no non-thought text emitted."""
    events = [make_mock_event("Answer: Option A", thought=True)]

    async def mock_run_async(*args, **kwargs):
        for event in events:
            yield event

    mock_runner = MagicMock(run_async=mock_run_async)
    workflow = workflow_factory(model_name="gpt-oss", max_retries=1, runner=mock_runner)

    output_text, _ = await workflow._invoke_agent_with_retry(
        prompt_text="Solve medical question",
        session_id="test_sess",
    )

    assert output_text == "Answer: Option A"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, thoughts_per_attempt",
    [
        ("vertex_ai/openai/gpt-oss-20b-maas", 50),
        ("vertex_ai/gemini-3.8-flash", 40),
    ],
)
async def test_workflow_run_models_with_thoughts(
    model_name,
    thoughts_per_attempt,
    sample_questions,
    tmp_path,
    workflow_factory,
):
    """Test running workflow with different models that produce reasoning tokens."""
    safe_name = model_name.replace("/", "_")
    output_file = tmp_path / f"{safe_name}_results.json"
    workflow = workflow_factory(
        model_name=model_name,
        output_filepath=str(output_file),
        concurrency=2,
        n_attempts=2,
    )
    assert workflow.config.model_name == model_name

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        ans = (
            "Final Answer: Option A"
            if "Sample question 1" in prompt_text
            else "Final Answer: Option B"
        )
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=120,
            candidates_token_count=60,
            total_token_count=180,
        )
        setattr(usage, "thoughts_token_count", thoughts_per_attempt)
        return ans, usage

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    assert summary.model == model_name
    assert summary.total_questions == 2
    assert summary.correct == 2
    assert summary.accuracy == 1.0
    # 2 questions x 2 attempts = 4 invocations x thoughts_per_attempt
    expected_total_thoughts = 4 * thoughts_per_attempt
    assert summary.total_thoughts_tokens == expected_total_thoughts

    assert len(results) == 2
    assert results[0].predicted_option == "A"
    assert results[0].thoughts_tokens == 2 * thoughts_per_attempt
    assert results[1].predicted_option == "B"
    assert results[1].thoughts_tokens == 2 * thoughts_per_attempt

    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_data["summary"]["model"] == model_name
    assert saved_data["summary"]["total_thoughts_tokens"] == expected_total_thoughts


# ---------------------------------------------------------------------------
# Rate Limiting and Resilience Tests
# ---------------------------------------------------------------------------


def test_is_rate_limit_error_detection():
    """Test is_rate_limit_error correctly identifies 429 and RESOURCE_EXHAUSTED errors."""
    exact_json_msg = json.dumps(
        {
            "error": {
                "code": 429,
                "message": "The request is throttled due to too many concurrent requests.",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
    )
    assert is_rate_limit_error(Exception(exact_json_msg)) is True

    class Http429Error(Exception):
        status_code = 429

    assert is_rate_limit_error(Http429Error("Too many requests")) is True

    class ApiResponseError(Exception):
        response = {
            "error": {
                "code": 429,
                "message": "The request is throttled due to too many concurrent requests.",
                "status": "RESOURCE_EXHAUSTED",
            }
        }

    assert is_rate_limit_error(ApiResponseError("API call failed")) is True
    assert is_rate_limit_error(ValueError("Invalid syntax")) is False
    assert is_rate_limit_error(KeyError("missing key")) is False


@pytest.mark.asyncio
async def test_invoke_agent_retries_on_429_resource_exhausted(
    monkeypatch, workflow_factory
):
    """Test that _invoke_agent_with_retry catches 429 RESOURCE_EXHAUSTED error and retries."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    attempts_made = 0
    sessions_seen = []

    async def mock_run_async(user_id, session_id, new_message):
        nonlocal attempts_made
        attempts_made += 1
        sessions_seen.append(session_id)
        if attempts_made == 1:
            raise Exception(
                'litellm.RateLimitError: Vertex_aiException - [{"error": {"code": 429, "message": "The request is throttled due to too many concurrent requests.", "status": "RESOURCE_EXHAUSTED"}}]'
            )
        yield make_mock_event(
            "Final Answer: Option C\nExplanation: recovered after 429 retry."
        )

    mock_runner = MagicMock(run_async=mock_run_async)
    workflow = workflow_factory(
        max_retries=2,
        rate_limit_max_retries=3,
        base_retry_delay=0.01,
        max_retry_delay=0.1,
        runner=mock_runner,
    )

    output_text, _ = await workflow._invoke_agent_with_retry(
        prompt_text="Solve medical question",
        session_id="test_sess_429",
    )

    assert attempts_made == 2
    assert "Final Answer: Option C" in output_text
    assert sessions_seen == ["test_sess_429", "test_sess_429_try1"]


@pytest.mark.asyncio
async def test_invoke_agent_rate_limit_exhaustion(monkeypatch, workflow_factory):
    """Test that 429 rate limit retries up to rate_limit_max_retries and raises if not resolved."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    attempts_made = 0

    async def mock_run_async(user_id, session_id, new_message):
        nonlocal attempts_made
        attempts_made += 1
        if False:
            yield None
        raise Exception(
            '{"error": {"code": 429, "message": "The request is throttled due to too many concurrent requests.", "status": "RESOURCE_EXHAUSTED"}}'
        )

    mock_runner = MagicMock(run_async=mock_run_async)
    workflow = workflow_factory(
        max_retries=1,
        rate_limit_max_retries=3,
        base_retry_delay=0.01,
        max_retry_delay=0.1,
        runner=mock_runner,
    )

    with pytest.raises(Exception) as exc_info:
        await workflow._invoke_agent_with_retry(
            prompt_text="Solve medical question",
            session_id="test_sess_exhaust",
        )

    # 1 initial attempt + 3 rate limit retries = 4 attempts total
    assert attempts_made == 4
    assert "RESOURCE_EXHAUSTED" in str(exc_info.value)
