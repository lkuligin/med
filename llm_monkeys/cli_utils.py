"""Shared CLI utilities and argument parser configuration for MedQA workflows."""

from __future__ import annotations

import argparse
import logging

from results_store import DEFAULT_RESULTS_DIR
from config import DEFAULT_MODEL


def setup_logging(log_level: str = "INFO") -> None:
    """Configure basic logging for CLI workflows."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if level == logging.DEBUG:
        try:
            import litellm

            litellm.set_verbose = True
        except ImportError:
            pass


def add_common_arguments(
    parser: argparse.ArgumentParser,
    default_model: str = DEFAULT_MODEL,
    default_output: str = "results_one_shot_gemma4.json",
    default_temperature: float = 0.0,
    default_concurrency: int = 2,
    include_attempts: bool = True,
) -> argparse.ArgumentParser:
    """Add standard arguments used across MedQA evaluation workflows."""
    parser.add_argument("--model", default=default_model, help="Model identifier")
    parser.add_argument(
        "--dataset",
        default="bigbio/med_qa",
        help=(
            "HuggingFace dataset name or alias (e.g., 'bigbio/med_qa', 'medbullets', 'mkieffer/Medbullets')"
        ),
    )
    parser.add_argument(
        "--dataset-config",
        default="med_qa_en_source",
        help="Dataset configuration name (default: 'med_qa_en_source' for MedQA; None for MedBullets)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: 'test' for MedQA; 'op5_test' for MedBullets)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Starting index offset in dataset"
    )
    parser.add_argument(
        "--output", default=default_output, help="Output JSON file path"
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory results are stored under, for a store that keeps one "
        f"(default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the directory this run's results go in, for a store that "
        "keeps one (default: the model name)",
    )
    if include_attempts:
        parser.add_argument(
            "--n-attempts",
            "-n",
            type=int,
            default=3,
            help="Number of inference attempts per question (default: 3)",
        )
        parser.add_argument(
            "--max-parse-retries",
            type=int,
            default=3,
            help="Maximum retries if predicted option is not parsed",
        )
        parser.add_argument(
            "--save-every-n",
            type=int,
            default=10,
            help="Dump results every N completed questions",
        )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=default_concurrency,
        help="Maximum concurrent model requests",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_temperature,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max output tokens"
    )
    parser.add_argument("--project", default=None, help="GCP Project ID")
    parser.add_argument("--location", default=None, help="Vertex AI location")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for model invocation errors (default: 5)",
    )
    parser.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=10,
        help="Maximum retries specifically for 429 / RESOURCE_EXHAUSTED rate limit errors (default: 10)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def create_base_parser(
    description: str = "Run MedQA inference experiments using Vertex MAAS with ADK and LiteLLM.",
    default_model: str = DEFAULT_MODEL,
    default_output: str = "results_one_shot_gemma4.json",
) -> argparse.ArgumentParser:
    """Create a base argument parser with common arguments reusable across workflows."""
    parser = argparse.ArgumentParser(description=description)
    return add_common_arguments(
        parser, default_model=default_model, default_output=default_output
    )
