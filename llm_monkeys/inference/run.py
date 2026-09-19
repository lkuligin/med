"""CLI for running candidate-based MedQA / MedBullets inference workflow on difficult questions."""

from __future__ import annotations

import argparse
import asyncio
import sys

from cli_utils import add_common_arguments, setup_logging
from config import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_MODEL,
    MEDBULLETS_DATASET,
    MEDBULLETS_SPLIT,
    CandidateInferenceConfig,
    is_medbullets_dataset,
    resolve_dataset_name,
    resolve_model_name,
)
from inference._schemas import CandidateWorkflowSummary
from inference.workflow import CandidateInferenceWorkflow

__all__ = [
    "async_main",
    "build_config",
    "create_parser",
    "format_summary",
    "main",
    "parse_args",
]


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for candidate inference workflow."""
    parser = argparse.ArgumentParser(
        description="Run MedQA / MedBullets candidate-based reasoning workflow on difficult questions using Vertex AI."
    )
    add_common_arguments(
        parser,
        default_model=DEFAULT_MODEL,
        default_output="results_step2_gemma4_candidates.json",
        default_temperature=0.8,
        default_concurrency=4,
        include_attempts=False,
    )
    parser.add_argument(
        "--difficult-questions",
        default="difficult_questions.csv",
        help=(
            "Path to CSV containing difficult question IDs "
            "(default: difficult_questions.csv, or difficult_questions_mb.csv for MedBullets)"
        ),
    )
    parser.add_argument(
        "--n-candidates",
        "-n",
        type=int,
        default=1000,
        help="Number of reasoning candidate paths to generate per question (default: 1000)",
    )
    parser.add_argument(
        "--save-every-n-candidates",
        type=int,
        default=25,
        help="Save results incrementally every N candidates (default: 25)",
    )
    parser.add_argument(
        "--save-every-n-questions",
        type=int,
        default=1,
        help="Save results after every N questions (default: 1)",
    )
    return parser


def parse_args(
    args: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for candidate inference workflow."""
    if parser is None:
        parser = create_parser()
    return parser.parse_args(args)


def build_config(args: argparse.Namespace) -> CandidateInferenceConfig:
    """Build CandidateInferenceConfig from parsed arguments."""
    output_arg = getattr(args, "output", "") or ""
    difficult_arg = getattr(args, "difficult_questions", "") or ""
    is_medbullets = (
        getattr(args, "medbullets", False)
        or is_medbullets_dataset(getattr(args, "dataset", None))
        or getattr(args, "split", None) == MEDBULLETS_SPLIT
        or "_mb" in output_arg.lower()
        or "medbullets" in output_arg.lower()
        or "_mb" in difficult_arg.lower()
        or "medbullets" in difficult_arg.lower()
    )

    if is_medbullets:
        dataset_name = (
            MEDBULLETS_DATASET
            if (
                not getattr(args, "dataset", None)
                or args.dataset == DEFAULT_DATASET
                or is_medbullets_dataset(args.dataset)
            )
            else resolve_dataset_name(args.dataset)
        )
        dataset_config = (
            None
            if getattr(args, "dataset_config", None) == DEFAULT_DATASET_CONFIG
            else getattr(args, "dataset_config", None)
        )
        dataset_split = (
            MEDBULLETS_SPLIT
            if (not getattr(args, "split", None) or args.split == "test")
            else args.split
        )
        output_filepath = (
            "results_step2_gemma4_candidates_mb.json"
            if getattr(args, "output", "")
            in ("", "results_step2_gemma4_candidates.json")
            else getattr(args, "output", "results_step2_gemma4_candidates_mb.json")
        )
        difficult_questions_path = (
            "difficult_questions_mb.csv"
            if getattr(args, "difficult_questions", "")
            in ("", "difficult_questions.csv")
            else getattr(args, "difficult_questions", "difficult_questions_mb.csv")
        )
    else:
        dataset_name = resolve_dataset_name(getattr(args, "dataset", None))
        dataset_config = getattr(args, "dataset_config", DEFAULT_DATASET_CONFIG)
        dataset_split = getattr(args, "split", DEFAULT_DATASET_SPLIT)
        output_filepath = (
            getattr(args, "output", "results_step2_gemma4_candidates.json")
            or "results_step2_gemma4_candidates.json"
        )
        difficult_questions_path = (
            getattr(args, "difficult_questions", "difficult_questions.csv")
            or "difficult_questions.csv"
        )

    if hasattr(args, "output"):
        args.output = output_filepath
    if hasattr(args, "difficult_questions"):
        args.difficult_questions = difficult_questions_path

    kwargs = {
        "model_name": resolve_model_name(getattr(args, "model", None)),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "difficult_questions_path": difficult_questions_path,
        "n_candidates": getattr(args, "n_candidates", 1000),
        "concurrency": args.concurrency,
        "limit": args.limit,
        "offset": args.offset,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "output_filepath": output_filepath,
        "save_every_n_candidates": getattr(args, "save_every_n_candidates", 25),
        "save_every_n_questions": getattr(args, "save_every_n_questions", 1),
        "max_retries": args.max_retries,
        "rate_limit_max_retries": args.rate_limit_max_retries,
    }
    if getattr(args, "project", None):
        kwargs["project_id"] = args.project
    if getattr(args, "location", None):
        kwargs["location"] = args.location

    return CandidateInferenceConfig(**kwargs)


def format_summary(summary: CandidateWorkflowSummary, output_path: str) -> str:
    """Format workflow summary for console output."""
    config_str = f"{summary.config}, " if summary.config else ""
    return (
        f"\n{'=' * 65}\n"
        f"CANDIDATE INFERENCE SUMMARY (Step 2 - Repeated Sampling):\n"
        f"Model: {summary.model}\n"
        f"Dataset: {summary.dataset} ({config_str}split={summary.split})\n"
        f"Candidates per Question (N): {summary.n_candidates}\n"
        f"Total Difficult Questions: {summary.total_questions}\n"
        f"Completed Questions: {summary.completed_questions}\n"
        f"Failed Questions: {summary.failed_questions}\n"
        f"Total Candidates Generated: {summary.total_candidates_generated:,}\n"
        f"Successful Candidates: {summary.total_successful_candidates:,}\n"
        f"Correct Candidates: {summary.total_correct_candidates:,}\n"
        f"Overall Candidate Accuracy: {summary.overall_accuracy * 100:.2f}%\n"
        f"Total Tokens: {summary.total_tokens:,} "
        f"(Prompt: {summary.total_prompt_tokens:,}, Candidate: {summary.total_candidate_tokens:,})\n"
        f"Total Time: {summary.total_time_seconds:.2f}s "
        f"(Avg/Question: {summary.average_question_latency_seconds:.2f}s, "
        f"Avg/Candidate: {summary.average_candidate_latency_seconds:.2f}s)\n"
        f"Results saved to: {output_path}\n"
        f"{'=' * 65}\n"
    )


async def async_main(args: argparse.Namespace) -> int:
    """Asynchronous entrypoint for CLI."""
    setup_logging(args.log_level)
    config = build_config(args)
    workflow = CandidateInferenceWorkflow(config=config)
    summary, _ = await workflow.run()
    print(format_summary(summary, args.output))
    return 0 if summary.failed_questions == 0 else 1


def main(args: list[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parsed_args = parse_args(args)
    sys.exit(asyncio.run(async_main(parsed_args)))


if __name__ == "__main__":
    main()
