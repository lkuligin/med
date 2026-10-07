"""Step 1, the one-shot prompt, many times over on MedQA's difficult questions.

    python -m gdanschin_runtime.one_shot_many --model M --run-name R \\
        --results-dir results/experiments/one-shot-k --n-attempts 50 \\
        --temperature 0.8 --max-tokens 4096 --concurrency 400

Run from llm_monkeys/ with gdanschin_runtime/env.sh sourced. The stock step 1
CLI takes the whole split; a majority vote over k attempts is wanted only where
step 2 drew k candidates, so this answers the difficult questions alone, into
a results directory of its own. Resumable: a question is topped up to its
attempt count.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from config import OneShotInferenceConfig
from inference._dataset import load_difficult_questions
from one_shot.workflow import OneShotInferenceWorkflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--n-attempts", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--difficult-questions", default="difficult_questions.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = OneShotInferenceConfig(
        model_name=args.model, run_name=args.run_name, results_dir=args.results_dir,
        dataset_name="bigbio/med_qa", n_attempts=args.n_attempts,
        temperature=args.temperature, max_tokens=args.max_tokens,
        concurrency=args.concurrency)
    config.validate()
    questions = load_difficult_questions(args.difficult_questions)
    summary, _ = asyncio.run(OneShotInferenceWorkflow(config=config).run(questions=questions))
    logging.info("%d questions, %d failed", summary.total_questions, summary.failed)
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
