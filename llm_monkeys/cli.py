"""Command-line interface for running MedQA inference experiments and workflows."""

from __future__ import annotations

import argparse
import asyncio
import sys

from cli_utils import add_common_arguments, create_base_parser, setup_logging
from config import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    MEDBULLETS_DATASET,
    MEDBULLETS_SPLIT,
    InferenceConfig,
    is_medbullets_dataset,
    resolve_dataset_name,
    resolve_model_name,
)
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
    output_arg = getattr(args, "output", "") or ""
    is_medbullets = (
        getattr(args, "medbullets", False)
        or is_medbullets_dataset(getattr(args, "dataset", None))
        or getattr(args, "split", None) == MEDBULLETS_SPLIT
        or "_mb" in output_arg.lower()
        or "medbullets" in output_arg.lower()
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
            "results_one_shot_medbullets.json"
            if getattr(args, "output", "") == "results_one_shot_gemma4.json"
            else getattr(args, "output", "results_one_shot_medbullets.json")
        )
    else:
        dataset_name = resolve_dataset_name(getattr(args, "dataset", None))
        dataset_config = getattr(args, "dataset_config", DEFAULT_DATASET_CONFIG)
        dataset_split = getattr(args, "split", DEFAULT_DATASET_SPLIT)
        output_filepath = getattr(args, "output", "results_one_shot_gemma4.json")

    if hasattr(args, "output"):
        args.output = output_filepath

    kwargs = {
        "model_name": resolve_model_name(args.model),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "limit": args.limit,
        "offset": args.offset,
        "output_filepath": output_filepath,
        "results_dir": args.results_dir,
        "run_name": args.run_name,
        "n_attempts": getattr(args, "n_attempts", 3),
        "concurrency": args.concurrency,
        "max_parse_retries": args.max_parse_retries,
        "save_every_n": getattr(args, "save_every_n", 10),
        "max_retries": getattr(args, "max_retries", 5),
        "rate_limit_max_retries": getattr(args, "rate_limit_max_retries", 10),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if getattr(args, "project", None):
        kwargs["project_id"] = args.project
    if getattr(args, "location", None):
        kwargs["location"] = args.location
    return InferenceConfig(**kwargs)


def format_summary(summary: WorkflowSummary, destination: str) -> str:
    """Format workflow summary statistics into a printable string."""
    config_str = f"{summary.config}, " if summary.config else ""
    return (
        f"\n{'=' * 60}\n"
        f"INFERENCE SUMMARY:\n"
        f"Model: {summary.model}\n"
        f"Dataset: {summary.dataset} ({config_str}split={summary.split})\n"
        f"Attempts per Question (n): {summary.n_attempts}\n"
        f"Total Questions: {summary.total_questions}\n"
        f"Completed: {summary.completed}\n"
        f"Failed: {summary.failed}\n"
        f"All Correct (Simple): {summary.simple_questions}\n"
        f"Difficult Questions: {summary.difficult_questions}\n"
        f"Accuracy (All Correct): {summary.accuracy * 100:.2f}%\n"
        f"Total Tokens: {summary.total_tokens} (Prompt: {summary.total_prompt_tokens}, Candidate: {summary.total_candidate_tokens})\n"
        f"Total Time: {summary.total_time_seconds:.2f}s (Avg Latency: {summary.average_latency_seconds:.2f}s)\n"
        f"Results saved to: {destination}\n"
        f"{'=' * 60}\n"
    )


async def async_main(args: argparse.Namespace) -> int:
    """Main asynchronous execution flow for CLI."""
    setup_logging(args.log_level)
    config = build_config(args)
    workflow = OneShotInferenceWorkflow(config=config)
    summary, _ = await workflow.run()
    print(format_summary(summary, str(workflow.store)))
    return 0 if summary.failed == 0 else 1


def main(args: list[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parsed_args = parse_args(args)
    sys.exit(asyncio.run(async_main(parsed_args)))


if __name__ == "__main__":
    main()
