"""Command-line interface for running MedQA inference experiments and workflows."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from config import InferenceConfig
from one_shot.workflow import OneShotInferenceWorkflow, WorkflowSummary


def create_base_parser(
    description: str = "Run MedQA inference experiments using Vertex MAAS with ADK and LiteLLM.",
    default_model: str = "vertex_ai/google/gemma-4-26b-a4b-it-maas",
    default_output: str = "results_one_shot_gemma4.json",
) -> argparse.ArgumentParser:
    """Create a base argument parser with common arguments reusable across workflows."""
    parser = argparse.ArgumentParser(description=description)
    return add_common_arguments(
        parser, default_model=default_model, default_output=default_output
    )


def add_common_arguments(
    parser: argparse.ArgumentParser,
    default_model: str = "vertex_ai/google/gemma-4-26b-a4b-it-maas",
    default_output: str = "results_one_shot_gemma4.json",
) -> argparse.ArgumentParser:
    """Add standard arguments used across MedQA evaluation workflows."""
    parser.add_argument("--model", default=default_model, help="Model identifier")
    parser.add_argument("--dataset", default="bigbio/med_qa", help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", default="med_qa_en_source", help="Dataset configuration name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Starting index offset in dataset")
    parser.add_argument("--output", default=default_output, help="Output JSON file path")
    parser.add_argument(
        "--n-attempts",
        "-n",
        type=int,
        default=3,
        help="Number of inference attempts per question (default: 3)",
    )
    parser.add_argument("--concurrency", type=int, default=2, help="Maximum concurrent model requests")
    parser.add_argument("--max-parse-retries", type=int, default=3, help="Maximum retries if predicted option is not parsed")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--project", default=None, help="GCP Project ID")
    parser.add_argument("--location", default=None, help="Vertex AI location")
    parser.add_argument("--save-every-n", type=int, default=10, help="Dump results every N completed questions")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


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


def setup_logging(log_level: str = "INFO") -> None:
    """Configure basic logging for CLI workflows."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
