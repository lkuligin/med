"""CLI for running candidate-based MedQA inference workflow on difficult questions."""

from __future__ import annotations

import argparse
import asyncio
import sys

from cli import add_common_arguments, setup_logging
from config import DEFAULT_MODEL, CandidateInferenceConfig
from inference._schemas import CandidateWorkflowSummary
from inference.workflow import CandidateInferenceWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for candidate inference workflow."""
    parser = argparse.ArgumentParser(
        description="Run MedQA candidate-based reasoning workflow on difficult questions using Gemma 4 on Vertex AI."
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
        help="Path to CSV containing difficult question IDs (default: difficult_questions.csv)",
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


def build_config(args: argparse.Namespace) -> CandidateInferenceConfig:
    """Build CandidateInferenceConfig from parsed arguments."""
    return CandidateInferenceConfig(
        model_name=args.model,
        difficult_questions_path=getattr(
            args, "difficult_questions", "difficult_questions.csv"
        ),
        n_candidates=getattr(args, "n_candidates", 1000),
        concurrency=args.concurrency,
        limit=args.limit,
        offset=args.offset,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.split,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_filepath=args.output,
        save_every_n_candidates=getattr(args, "save_every_n_candidates", 25),
        save_every_n_questions=getattr(args, "save_every_n_questions", 1),
        max_retries=args.max_retries,
        rate_limit_max_retries=args.rate_limit_max_retries,
        project_id=args.project,
        location=args.location,
    )


def format_summary(summary: CandidateWorkflowSummary, output_path: str) -> str:
    """Format workflow summary for console output."""
    return (
        f"\n{'=' * 65}\n"
        f"CANDIDATE INFERENCE SUMMARY (Step 2 - Repeated Sampling):\n"
        f"Model: {summary.model}\n"
        f"Dataset: {summary.dataset} ({summary.config}, split={summary.split})\n"
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
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    sys.exit(asyncio.run(async_main(parsed_args)))


if __name__ == "__main__":
    main()
