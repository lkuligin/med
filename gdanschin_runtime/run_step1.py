"""Run step 1 on exactly the difficult questions, as a baseline for step 2.

    python3 gdanschin_runtime/run_step1.py --base gemma-4-26b

llm_monkeys' own step 1 CLI slices the dataset by --limit/--offset, so it would
measure the first N questions of the split rather than the 483 the pipeline
works on. The workflow itself accepts a question list, which is what this uses:
the comparison only means anything on the same questions, with the same model.

--base names an entry in models.py and decides both the model called and the
directory the answers go in, exactly as it does for step 2.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from gdanschin_runtime import _bootstrap  # noqa: F401

from gdanschin_runtime.models import BASE_MODELS

from config import InferenceConfig, resolve_dataset_name
from inference._dataset import load_difficult_questions
from one_shot.workflow import OneShotInferenceWorkflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="entry in models.py: the model called, "
                                       "and the directory results go in")
    parser.add_argument("--difficult-questions", default="difficult_questions.csv")
    parser.add_argument("--dataset", default=None,
                        help="dataset name or alias (default: MedQA)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default: the budget models.py records for --base")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    # --base has to reach the model, not only the directory name. Setting the
    # run name alone left the config on its default model, so every base
    # measured the same Gemma and stored it under the requested name: a
    # baseline that looked plausible, matched no model, and was wrong in the
    # one direction nothing downstream could detect.
    model = None
    if args.base:
        model = BASE_MODELS.get(args.base)
        if model is None:
            print(f"unknown base model: {args.base}   "
                  f"(known: {', '.join(BASE_MODELS)})", file=sys.stderr)
            return 2

    config = InferenceConfig(
        n_attempts=args.n_attempts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens or (model.max_tokens if model else 1024),
        run_name=args.base,
        **({"dataset_name": resolve_dataset_name(args.dataset)} if args.dataset else {}),
        **({"model_name": model.gateway_model} if model else {}),
    )

    questions = load_difficult_questions(csv_path=args.difficult_questions,
                                         dataset_name=config.dataset_name,
                                         config_name=config.dataset_config,
                                         split=config.dataset_split,
                                         limit=args.limit)
    print(f"{len(questions)} difficult questions, {args.n_attempts} attempt(s) each, "
          f"model {config.resolved_model_name}, {config.max_tokens} tokens "
          f"-> results/single-step/{config.resolved_run_name}", flush=True)

    workflow = OneShotInferenceWorkflow(config=config)
    summary, _ = asyncio.run(workflow.run(questions=questions))
    print(f"accuracy {summary.accuracy * 100:.1f}%  "
          f"({summary.correct}/{summary.completed})  -> {workflow.store}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
