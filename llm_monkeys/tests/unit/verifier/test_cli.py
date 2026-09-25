"""Unit tests for verifier.cli module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from config import DEFAULT_VERIFIER_MODEL, VerifierConfig
from results_store import DEFAULT_RESULTS_DIR
from verifier._schemas import VerifierWorkflowSummary
from verifier.cli import (
    async_main,
    build_config,
    create_parser,
    format_summary,
    main,
    main_async,
    parse_args,
)


def test_create_parser_defaults():
    parser = create_parser()
    args = parser.parse_args([])

    assert args.model == DEFAULT_VERIFIER_MODEL
    assert args.results_dir == DEFAULT_RESULTS_DIR
    # Both default to the model name: the run being verified has to be named,
    # and the judge directory is named after the judge unless told otherwise.
    assert args.run_name is None
    assert args.judge_name is None
    # Tied to the config rather than a literal: the two defaults must agree,
    # and a literal here silently went stale when the default moved to 1.0.
    assert args.temperature == VerifierConfig().temperature
    assert args.concurrency == 4
    assert args.max_candidates_per_question is None
    assert args.early_stop_facts is False
    assert args.early_stop_candidates is True
    assert args.save_every_n_questions == 1


def test_parse_args():
    args = parse_args(["--run-name", "my-run", "--concurrency", "2"])
    assert args.run_name == "my-run"
    assert args.concurrency == 2
    assert args.early_stop_candidates is True

    args_no_early = parse_args(["--no-early-stop-candidates"])
    assert args_no_early.early_stop_candidates is False


def test_build_config():
    parser = create_parser()
    args = parser.parse_args(
        [
            "--model",
            "gemini-3-flash-preview",
            "--run-name",
            "custom-run",
            "--judge-name",
            "custom-judge",
            "--concurrency",
            "8",
            "--temperature",
            "0.1",
            "--max-candidates-per-question",
            "10",
            "--early-stop-facts",
            "--save-every-n-questions",
            "5",
        ]
    )
    config = build_config(args)

    assert config.model_name == "gemini-3-flash-preview"
    assert config.run_name == "custom-run"
    assert config.judge_name == "custom-judge"
    assert config.resolved_judge_name == "custom-judge"
    assert config.concurrency == 8
    assert config.temperature == 0.1
    assert config.max_candidates_per_question == 10
    assert config.early_stop_facts is True
    assert config.early_stop_candidates is True
    assert config.save_every_n_questions == 5


def _dummy_summary(failed_questions: int = 0) -> VerifierWorkflowSummary:
    return VerifierWorkflowSummary(
        model="gemini-3-flash-preview",
        run_name="test-run",
        judge_name="test-judge",
        total_questions=10,
        completed_questions=10 - failed_questions,
        failed_questions=failed_questions,
        questions_with_valid_candidate=8,
        correct_answers=6,
        accuracy=0.75,
        overall_accuracy=0.60,
        average_attempts_to_valid=3.25,
        total_candidates_evaluated=35,
        total_facts_verified=150,
        total_facts_correct=120,
        total_facts_incorrect=30,
        total_time_seconds=45.2,
        average_question_latency_seconds=4.52,
        total_tokens=50000,
        total_prompt_tokens=35000,
        total_candidate_tokens=15000,
    )


def test_format_summary():
    summary = _dummy_summary()
    out = format_summary(summary, "results/facts-pipeline/test-run")
    assert "FACT VERIFICATION SUMMARY" in out
    assert "Questions with Valid Candidate" in out
    assert "75.00%" in out
    assert "3.25" in out


@pytest.mark.asyncio
async def test_async_main_success(capsys):
    args = parse_args(["--run-name", "dummy-run"])
    summary = _dummy_summary(failed_questions=0)

    with patch("verifier.cli.VerifierWorkflow.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (summary, [])
        exit_code = await async_main(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "FACT VERIFICATION SUMMARY" in captured.out


@pytest.mark.asyncio
async def test_async_main_failure():
    args = parse_args([])
    summary = _dummy_summary(failed_questions=2)

    with patch("verifier.cli.VerifierWorkflow.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (summary, [])
        exit_code = await async_main(args)
        assert exit_code == 1


@pytest.mark.asyncio
async def test_main_async_alias():
    assert main_async is async_main


def test_main():
    with (
        patch("verifier.cli.async_main", new_callable=AsyncMock) as mock_async_main,
        patch("sys.exit") as mock_exit,
    ):
        mock_async_main.return_value = 0
        main(["--run-name", "test-run"])
        mock_exit.assert_called_once_with(0)


def test_difficult_questions_reaches_the_config():
    config = build_config(parse_args(["--difficult-questions", "difficult_questions_mb.csv"]))
    assert config.difficult_questions == "difficult_questions_mb.csv"
    assert build_config(parse_args([])).difficult_questions is None
