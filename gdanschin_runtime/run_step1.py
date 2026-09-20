"""Run step 1 on exactly the difficult questions, as a baseline for step 2.

    python3 gdanschin_runtime/run_step1.py --base gemma-4-26b

llm_monkeys' own step 1 CLI slices the dataset by --limit/--offset, so it would
measure the first N questions of the split rather than the 483 the pipeline
works on. The workflow itself accepts a question list, which is what this uses:
the comparison only means anything on the same questions, with the same model.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from gdanschin_runtime import _bootstrap  # noqa: F401

from config import InferenceConfig
from inference._dataset import load_difficult_questions
from one_shot.workflow import OneShotInferenceWorkflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="base model name from models.py; also "
                                       "names the directory results go in")
    parser.add_argument("--difficult-questions", default="difficult_questions.csv")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    questions = load_difficult_questions(csv_path=args.difficult_questions,
                                         limit=args.limit)
    print(f"{len(questions)} difficult questions, {args.n_attempts} attempt(s) each")

    config = InferenceConfig(
        n_attempts=args.n_attempts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        run_name=args.base,
    )
    workflow = OneShotInferenceWorkflow(config=config)
    summary, _ = asyncio.run(workflow.run(questions=questions))
    print(f"accuracy {summary.accuracy * 100:.1f}%  "
          f"({summary.correct}/{summary.completed})  -> {workflow.store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
