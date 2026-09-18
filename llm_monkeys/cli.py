"""Command-line interface for running MedQA inference experiments and workflows."""

from __future__ import annotations

import argparse
import asyncio
import sys

from cli_utils import add_common_arguments, create_base_parser, setup_logging
from config import InferenceConfig
from one_shot.workflow import OneShotInferenceWorkflow, WorkflowSummary

__all__ = [
    "add_common_arguments",
    "async_main",
    "build_config",
    "create_base_parser",
    "format_summary",
    "main",
    "parse_args",
    "setup_logging",
]


def parse_args(
    args: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse command line arguments for MedQA inference workflows."""
    if parser is None:
        parser = create_base_parser()
    return parser.parse_args(args)


def build_config(args: argparse.Namespace) -> InferenceConfig:
    """Construct an InferenceConfig instance from parsed CLI arguments."""
    kwargs = {
        "model_name": args.model,
        "dataset_name": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "output_filepath": args.output,
        "n_attempts": getattr(args, "n_attempts", 3),
        "concurrency": args.concurrency,
        "max_parse_retries": args.max_parse_retries,
        "save_every_n": getattr(args, "save_every_n", 10),
        "max_retries": getattr(args, "max_retries", 5),
        "rate_limit_max_retries": getattr(args, "rate_limit_max_retries", 10),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.project:
        kwargs["project_id"] = args.project
    if args.location:
        kwargs["location"] = args.location
    return InferenceConfig(**kwargs)


def format_summary(summary: WorkflowSummary, output_path: str) -> str:
    """Format workflow summary statistics into a printable string."""
    return (
        f"\n{'=' * 60}\n"
        f"INFERENCE SUMMARY:\n"
        f"Model: {summary.model}\n"
        f"Dataset: {summary.dataset} ({summary.config}, split={summary.split})\n"
        f"Attempts per Question (n): {summary.n_attempts}\n"
        f"Total Questions: {summary.total_questions}\n"
        f"Completed: {summary.completed}\n"
        f"Failed: {summary.failed}\n"
        f"All Correct (Simple): {summary.simple_questions}\n"
        f"Difficult Questions: {summary.difficult_questions}\n"
        f"Accuracy (All Correct): {summary.accuracy * 100:.2f}%\n"
        f"Total Tokens: {summary.total_tokens} (Prompt: {summary.total_prompt_tokens}, Candidate: {summary.total_candidate_tokens})\n"
        f"Total Time: {summary.total_time_seconds:.2f}s (Avg Latency: {summary.average_latency_seconds:.2f}s)\n"
        f"Results saved to: {output_path}\n"
        f"{'=' * 60}\n"
    )


async def async_main(args: argparse.Namespace) -> int:
    """Main asynchronous execution flow for CLI."""
    setup_logging(args.log_level)
    config = build_config(args)
    summary, _ = await OneShotInferenceWorkflow(config=config).run()
    print(format_summary(summary, args.output))
    return 0 if summary.failed == 0 else 1


def main(args: list[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parsed_args = parse_args(args)
    sys.exit(asyncio.run(async_main(parsed_args)))


if __name__ == "__main__":
    main()
