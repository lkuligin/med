import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from config import InferenceConfig
from dataset import MedQAQuestion
from one_shot.workflow import (
    AttemptResult,
    InferenceItemResult,
    OneShotInferenceWorkflow,
    WorkflowSummary,
)


@pytest.fixture
def sample_questions():
    return [
        MedQAQuestion(
            question_id="0",
            question="Sample question 1",
            options={"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
            answer_idx="A",
            answer="Opt A",
            meta_info="step1",
        ),
        MedQAQuestion(
            question_id="1",
            question="Sample question 2",
            options={"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
            answer_idx="B",
            answer="Opt B",
            meta_info="step2",
        ),
    ]


@pytest.mark.asyncio
async def test_workflow_run_with_mocked_runner(sample_questions, tmp_path):
    output_file = tmp_path / "test_results.json"
    config = InferenceConfig(
        output_filepath=str(output_file),
        concurrency=2,
        n_attempts=3,
    )

    mock_agent = MagicMock()
    mock_runner = MagicMock()
    mock_session_service = MagicMock()
    mock_session_service.create_session = AsyncMock()

    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=mock_agent,
        runner=mock_runner,
        session_service=mock_session_service,
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
    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

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
async def test_workflow_simple_vs_difficult_split(sample_questions):
    config = InferenceConfig(concurrency=2, n_attempts=3, output_filepath="")
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )

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
            else:
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
async def test_workflow_error_handling(sample_questions):
    config = InferenceConfig(concurrency=2, n_attempts=3, output_filepath="")
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )

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
async def test_workflow_empty_questions():
    workflow = OneShotInferenceWorkflow(
        config=InferenceConfig(output_filepath="", n_attempts=3),
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )
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
async def test_workflow_retries_on_unparsed_option_success():
    question = MedQAQuestion(
        question_id="test_retry_1",
        question="Which drug causes ototoxicity?",
        options={"A": "Cisplatin", "B": "Paracetamol", "C": "Amoxicillin", "D": "Metformin"},
        answer_idx="A",
        answer="Cisplatin",
    )
    # n_attempts=1 to isolate parse retry behavior
    config = InferenceConfig(max_parse_retries=3, concurrency=1, n_attempts=1, output_filepath="")
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )

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

    summary, results = await workflow.run(questions=[question])

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
async def test_workflow_retries_on_unparsed_option_exhausted():
    question = MedQAQuestion(
        question_id="test_retry_exhausted",
        question="Which drug causes ototoxicity?",
        options={"A": "Cisplatin", "B": "Paracetamol", "C": "Amoxicillin", "D": "Metformin"},
        answer_idx="A",
        answer="Cisplatin",
    )
    # n_attempts=1 to isolate parse retry behavior
    config = InferenceConfig(max_parse_retries=3, concurrency=1, n_attempts=1, output_filepath="")
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )

    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50,
            candidates_token_count=10,
            total_token_count=60,
        )
        # Always return unparseable response
        return "Unknown unparseable text", usage

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[question])

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
async def test_workflow_retries_configurable_k():
    question = MedQAQuestion(
        question_id="test_retry_k1",
        question="Sample question",
        options={"A": "Opt A", "B": "Opt B"},
        answer_idx="A",
        answer="Opt A",
    )
    # k = 1 retry, n_attempts = 1
    config = InferenceConfig(max_parse_retries=1, concurrency=1, n_attempts=1, output_filepath="")
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )

    call_count = 0

    async def mock_invoke(prompt_text, session_id, user_id="med_eval_user"):
        nonlocal call_count
        call_count += 1
        return "Still unparseable", None

    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=[question])

    # 1 initial + 1 retry = 2 attempts
    assert call_count == 2
    assert results[0].predicted_option is None
    assert results[0].parse_retries == 1


@pytest.mark.asyncio
async def test_workflow_skips_already_processed_questions(sample_questions, tmp_path):
    output_file = tmp_path / "resume_results.json"
    # Pre-populate output file with result for question_id="0" including attempts
    pre_existing_data = {
        "summary": {
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
        },
        "results": [
            {
                "question_id": "0",
                "meta_info": "step1",
                "question": "Sample question 1",
                "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
                "ground_truth": "A",
                "ground_truth_answer": "Opt A",
                "predicted_option": "A",
                "is_correct": True,
                "is_all_correct": True,
                "correct_attempts": 3,
                "total_attempts": 3,
                "raw_response": "Answer: A\nExplanation: cached answer",
                "prompt": "prompt 0",
                "latency_seconds": 0.5,
                "prompt_tokens": 100,
                "candidate_tokens": 20,
                "total_tokens": 120,
                "cached_tokens": 0,
                "error": None,
                "parse_retries": 0,
                "attempts": [
                    {
                        "attempt_index": 0,
                        "predicted_option": "A",
                        "is_correct": True,
                        "raw_response": "Answer: A",
                        "latency_seconds": 0.1,
                    },
                    {
                        "attempt_index": 1,
                        "predicted_option": "A",
                        "is_correct": True,
                        "raw_response": "Answer: A",
                        "latency_seconds": 0.2,
                    },
                    {
                        "attempt_index": 2,
                        "predicted_option": "A",
                        "is_correct": True,
                        "raw_response": "Answer: A",
                        "latency_seconds": 0.2,
                    },
                ],
            }
        ],
    }
    output_file.write_text(json.dumps(pre_existing_data), encoding="utf-8")

    config = InferenceConfig(
        output_filepath=str(output_file),
        concurrency=2,
        n_attempts=3,
    )
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
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
    assert summary.correct == 2  # question 0 correct (from file) + question 1 correct (B)

    assert len(results) == 2
    assert results[0].question_id == "0"
    assert results[0].raw_response == "Answer: A\nExplanation: cached answer"
    assert len(results[0].attempts) == 3
    assert results[1].question_id == "1"
    assert results[1].raw_response == "Answer: B\nExplanation: fresh run"
    assert len(results[1].attempts) == 3

    # Verify output file was updated with both results
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 2
    assert saved["results"][0]["question_id"] == "0"
    assert len(saved["results"][0]["attempts"]) == 3
    assert saved["results"][1]["question_id"] == "1"
    assert len(saved["results"][1]["attempts"]) == 3


@pytest.mark.asyncio
async def test_workflow_skips_all_when_all_processed(sample_questions, tmp_path):
    output_file = tmp_path / "all_processed.json"
    pre_existing_data = {
        "results": [
            {
                "question_id": "0",
                "question": "Sample question 1",
                "options": {},
                "ground_truth": "A",
                "ground_truth_answer": "Opt A",
                "predicted_option": "A",
                "is_correct": True,
                "is_all_correct": True,
                "correct_attempts": 3,
                "total_attempts": 3,
                "raw_response": "Cached 0",
                "prompt": "",
                "latency_seconds": 0.1,
                "attempts": [],
            },
            {
                "question_id": "1",
                "question": "Sample question 2",
                "options": {},
                "ground_truth": "B",
                "ground_truth_answer": "Opt B",
                "predicted_option": "B",
                "is_correct": True,
                "is_all_correct": True,
                "correct_attempts": 3,
                "total_attempts": 3,
                "raw_response": "Cached 1",
                "prompt": "",
                "latency_seconds": 0.1,
                "attempts": [],
            },
        ]
    }
    output_file.write_text(json.dumps(pre_existing_data), encoding="utf-8")

    config = InferenceConfig(output_filepath=str(output_file), n_attempts=3)
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
    )

    mock_invoke = AsyncMock()
    workflow._invoke_agent_with_retry = mock_invoke

    summary, results = await workflow.run(questions=sample_questions)

    assert mock_invoke.call_count == 0
    assert summary.total_questions == 2
    assert summary.completed == 2
    assert summary.correct == 2
    assert len(results) == 2


@pytest.mark.asyncio
async def test_workflow_handles_corrupted_destination_file(sample_questions, tmp_path):
    output_file = tmp_path / "corrupted.json"
    output_file.write_text("{corrupted json content...", encoding="utf-8")

    config = InferenceConfig(output_filepath=str(output_file), concurrency=2, n_attempts=1)
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
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


@pytest.mark.asyncio
async def test_workflow_dumps_periodically_every_n_examples(tmp_path):
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

    config = InferenceConfig(
        output_filepath=str(output_file),
        concurrency=1,
        n_attempts=2,
        save_every_n=2,  # dump every 2 examples
    )
    workflow = OneShotInferenceWorkflow(
        config=config,
        agent=MagicMock(),
        runner=MagicMock(),
        session_service=MagicMock(),
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


def test_attempt_result_to_dict_and_from_dict():
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
        error=None,
        parse_retries=1,
    )
    d = att.to_dict()
    assert d["attempt_index"] == 1
    assert d["predicted_option"] == "B"
    assert d["is_correct"] is True

    reconstructed = AttemptResult.from_dict(d)
    assert reconstructed == att


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

