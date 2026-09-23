"""Run step 1 on exactly the difficult questions, as a baseline for step 2.

    python3 gdanschin_runtime/run_step1.py --base gemma-4-26b

llm_monkeys' own step 1 CLI slices the dataset by --limit/--offset, so it would
measure the first N questions of the split rather than the 483 the pipeline
works on. The workflow itself accepts a question list, which is what this uses:
the comparison only means anything on the same questions, with the same model.

--base names an entry in models.py and decides both the model called and the
directory the answers go in: <base>-difficult, since <base> itself is step 1
over the whole split.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from gdanschin_runtime import _bootstrap  # noqa: F401

from gdanschin_runtime.local_endpoint import EndpointNotReady, require_ready
from gdanschin_runtime.models import BASE_MODELS, difficult_run

from config import (MEDBULLETS_SPLIT, InferenceConfig,
                    is_medbullets_dataset, resolve_dataset_name)
from inference._dataset import load_difficult_questions
from one_shot.workflow import OneShotInferenceWorkflow


def create_parser() -> argparse.ArgumentParser:
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
    return parser


def _dataset_kwargs(dataset: str | None) -> dict:
    """Name, config and split for a dataset argument.

    Medbullets needs all three, not just the name: it has no config and its
    split is op5_test, where the default is test. llm_monkeys' own CLI works
    this out in build_config and this script did not, so a medbullets run
    stopped on "Unknown split" before it reached a model.
    """
    if not dataset:
        return {}
    name = resolve_dataset_name(dataset)
    if not is_medbullets_dataset(name):
        return {"dataset_name": name}
    return {"dataset_name": name, "dataset_config": None,
            "dataset_split": MEDBULLETS_SPLIT}


def build_config(args: argparse.Namespace) -> InferenceConfig:
    """The configuration a run of these arguments uses.

    Separate from main() so that what --base actually selects can be read
    without calling a model: the one thing that went wrong here was invisible
    until a run had finished and its summary was read.

    Raises:
        KeyError: If --base names no entry in models.py.
    """
    # --base has to reach the model, not only the directory name. Setting the
    # run name alone left the config on its default model, so every base
    # measured the same Gemma and stored it under the requested name: a
    # baseline that looked plausible, matched no model, and was wrong in the
    # one direction nothing downstream could detect.
    model = None
    if args.base:
        if args.base not in BASE_MODELS:
            raise KeyError(args.base)
        model = BASE_MODELS[args.base]

    return InferenceConfig(
        n_attempts=args.n_attempts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens or (model.max_tokens if model else 1024),
        run_name=difficult_run(args.base) if args.base else None,
        **_dataset_kwargs(args.dataset),
        **({"model_name": model.gateway_model} if model else {}),
    )


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        config = build_config(args)
    except KeyError as unknown:
        print(f"unknown base model: {unknown.args[0]}   "
              f"(known: {', '.join(BASE_MODELS)})", file=sys.stderr)
        return 2

    # Before the dataset, so a server that is down costs a second rather than
    # a loaded dataset and a wall of per-question failures - and so a server
    # running the WRONG model is caught at all, which it otherwise is not: it
    # would fill a directory with plausible answers under another model's name.
    if args.base:
        try:
            require_ready(BASE_MODELS[args.base])
        except EndpointNotReady as not_ready:
            print(not_ready, file=sys.stderr)
            return 3

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
