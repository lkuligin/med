"""Command-line interface for running the MedQA fact verification workflow (Step 3)."""

from __future__ import annotations

import argparse
import asyncio
import sys

from cli_utils import add_common_arguments, setup_logging
from config import DEFAULT_VERIFIER_MODEL, VerifierConfig
from verifier._schemas import VerifierWorkflowSummary
from verifier.workflow import VerifierWorkflow

_CLI_CONFIG_MAP = {
    "model": "model_name",
    "input": "input_filepath",
    "output": "output_filepath",
    "project": "project_id",
    "dataset": "dataset_name",
    "split": "dataset_split",
}


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for the verifier workflow."""
    parser = argparse.ArgumentParser(
        description="Run MedQA Step 3 Fact Verification on candidate reasoning paths using an LLM-as-a-judge on Vertex AI."
    )
    add_common_arguments(
        parser,
        default_model=DEFAULT_VERIFIER_MODEL,
        default_output="results_step3_verified.json",
        default_temperature=1.0,
        default_concurrency=4,
        include_attempts=False,
    )
    parser.add_argument(
        "--input",
        "-i",
        default="results_step2_gemma4_candidates.json",
        help="Path to Step 2 JSON candidate output file (default: results_step2_gemma4_candidates.json)",
    )
    parser.add_argument(
        "--max-candidates-per-question",
        type=int,
        default=None,
        help="Maximum number of candidates to evaluate per question (default: evaluate all until first pass)",
    )
    parser.add_argument(
        "--early-stop-facts",
        action="store_true",
        default=False,
        help="Stop evaluating candidate facts immediately on the first incorrect fact (default: False)",
    )
    parser.add_argument(
        "--early-stop-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop evaluating candidates immediately once a candidate has all facts correct (default: True)",
    )
    parser.add_argument(
        "--save-every-n-questions",
        type=int,
        default=1,
        help="Persist intermediate results after every N questions (default: 1)",
    )
    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the verifier workflow."""
    return create_parser().parse_args(args)


def build_config(args: argparse.Namespace) -> VerifierConfig:
    """Build VerifierConfig from parsed CLI arguments."""
    raw = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    params = {_CLI_CONFIG_MAP.get(k, k): v for k, v in raw.items() if v is not None}
    return VerifierConfig.from_dict(params)


def format_summary(summary: VerifierWorkflowSummary, output_path: str = "") -> str:
    """Format verification workflow summary for console output."""
    out_path = output_path or summary.output_filepath
    return (
        f"\n{'=' * 70}\n"
        f"FACT VERIFICATION SUMMARY (Step 3 - LLM-as-a-Judge Rejection Sampling):\n"
        f"Model: {summary.model}\n"
        f"Input File: {summary.input_filepath}\n"
        f"Output File: {out_path}\n"
        f"Total Questions Evaluated: {summary.total_questions}\n"
        f"Completed Questions: {summary.completed_questions}\n"
        f"Failed Questions: {summary.failed_questions}\n"
        f"Questions with Valid Candidate (All Facts Correct): {summary.questions_with_valid_candidate}\n"
        f"Correct Final Answers: {summary.correct_answers}\n"
        f"Total Correct Candidates (All Facts Correct): {summary.total_correct_candidates:,}\n"
        f"  - Right Answers: {summary.total_correct_candidates_right_answers:,}\n"
        f"  - Wrong Answers: {summary.total_correct_candidates_wrong_answers:,}\n"
        f"First Correct Candidate Position (Only First Assumptions): avg {summary.avg_position_assumptions_only:.2f}\n"
        f"First Correct Candidate Position (All Assumptions and Answer): avg {summary.avg_position_all_and_answer:.2f}\n"
        f"Accuracy (Valid Candidates): {summary.accuracy * 100:.2f}%\n"
        f"Overall Accuracy: {summary.overall_accuracy * 100:.2f}%\n"
        f"Average Attempts to First Valid Candidate: {summary.average_attempts_to_valid:.2f}\n"
        f"Total Candidates Evaluated: {summary.total_candidates_evaluated:,}\n"
        f"Total Facts Verified: {summary.total_facts_verified:,} "
        f"(Correct: {summary.total_facts_correct:,}, Incorrect: {summary.total_facts_incorrect:,})\n"
        f"Total Tokens: {summary.total_tokens:,} "
        f"(Prompt: {summary.total_prompt_tokens:,}, Candidate: {summary.total_candidate_tokens:,})\n"
        f"Total Time: {summary.total_time_seconds:.2f}s "
        f"(Avg/Question: {summary.average_question_latency_seconds:.2f}s)\n"
        f"{'=' * 70}\n"
    )


async def async_main(args: argparse.Namespace | None = None) -> int:
    """Execute the verification workflow asynchronously."""
    parsed_args = parse_args() if args is None else args
    setup_logging(parsed_args.log_level)

    config = build_config(parsed_args)
    config.validate()

    workflow = VerifierWorkflow(config=config)
    summary, _ = await workflow.run()

    print(format_summary(summary, config.output_filepath))
    return 0 if summary.failed_questions == 0 else 1


main_async = async_main


def main(args: list[str] | None = None) -> None:
    """Synchronous CLI entry point."""
    sys.exit(asyncio.run(async_main(parse_args(args))))


if __name__ == "__main__":
    main()
