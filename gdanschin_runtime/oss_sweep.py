"""The stage 2 and stage 3 sweep over our locally served models.

Written as a table rather than a loop because the two things that vary are
not derivable from anything: how many candidates each model generates, and
whether it is judged at all.

Judging is the expensive half. It runs on the external gateway against a
paid frontier model, so a model judged by accident costs real money and
nothing in the output says it was not meant to happen. The stock plan() in
run_plan.py pairs a verdicts step with every generation step, which is
correct for its own purpose and wrong for this one - hence a separate table,
an explicit flag per row, and a guard that refuses to build a verdicts step
for a model the table does not mark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS

MEDBULLETS = "mkieffer/Medbullets"
MEDQA = "bigbio/med_qa"
DATASETS = (MEDBULLETS, MEDQA)

JUDGE = os.getenv("MEDQA_JUDGE_MODEL", "gemini-3.8-flash")
GEN_CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "32"))
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "32"))

DIFFICULT = {MEDBULLETS: "difficult_questions_mb.csv",
             MEDQA: "difficult_questions.csv"}


@dataclass(frozen=True)
class Target:
    """One model in the sweep: how many candidates, and whether it is judged."""

    base: str            # the entry in models.py, which is served locally
    serve: str           # the entry in gpu_serving's catalogue
    k: int
    judged: bool


# The order is the order they run in. Judged models come first, so that the
# expensive half starts early and overlaps the generation that follows it.
SWEEP: tuple[Target, ...] = (
    Target("qwen3.6-35b-a3b-nr-local", "qwen3.6-35b-a3b", 50, True),
    Target("qwen3.5-9b-nr-local", "qwen3.5-9b", 100, True),
    Target("qwen3.5-4b-nr-local", "qwen3.5-4b", 100, True),
    Target("gemma-4-e2b-local", "gemma-4-e2b", 100, True),
    Target("gpt-oss-20b-local", "gpt-oss-20b", 100, False),
    Target("qwen3.6-27b-nr-local", "qwen3.6-27b", 50, False),
    Target("qwen3.8-27b-nr-local", "qwen3.8-27b", 50, False),
    Target("qwen3.5-2b-nr-local", "qwen3.5-2b", 100, False),
    Target("qwen3.5-0.8b-nr-local", "qwen3.5-0.8b", 100, False),
)

JUDGED = tuple(t.base for t in SWEEP if t.judged)


class NotJudged(RuntimeError):
    """Asked to judge a model the sweep does not judge."""


def candidates_command(target: Target, dataset: str, python: str) -> list[str]:
    """Stage 2: generate candidates from the locally served model."""
    entry = BASE_MODELS[target.base]
    return [python, "-m", "inference.cli",
            "--model", entry.gateway_model,
            "--run-name", target.base,
            "--difficult-questions", DIFFICULT[dataset],
            "--dataset", dataset,
            "--n-candidates", str(target.k),
            "--concurrency", str(GEN_CONCURRENCY),
            "--max-tokens", str(entry.max_tokens)]


def verdicts_command(target: Target, dataset: str, python: str) -> list[str]:
    """Stage 3: judge the candidates. Refuses a model the table does not mark.

    The refusal is the point. This step spends money on an external gateway,
    and an accidental run would look exactly like an intended one.
    """
    if not target.judged:
        raise NotJudged(
            f"{target.base} is not judged in this sweep. Stage 3 costs money "
            f"on the external gateway; judged models are: {', '.join(JUDGED)}")
    judge = JUDGE_MODELS[JUDGE]
    return [python, "-m", "verifier.cli",
            "--run-name", target.base,
            "--judge-name", JUDGE,
            "--model", judge.gateway_model,
            "--temperature", str(judge.temperature),
            "--max-tokens", str(judge.max_tokens),
            "--dataset", dataset,
            "--concurrency", str(JUDGE_CONCURRENCY)]


def describe() -> None:
    print(f"  {'model':26} {'served as':18} {'k':>4} {'stage 3':>9}")
    for t in SWEEP:
        print(f"  {t.base:26} {t.serve:18} {t.k:4} "
              f"{'JUDGED' if t.judged else '-':>9}")
    print(f"\n  stage 3 runs for {len(JUDGED)} of {len(SWEEP)} models, "
          f"on {len(DATASETS)} datasets each")


if __name__ == "__main__":
    describe()


__all__ = ["Target", "SWEEP", "JUDGED", "NotJudged", "DATASETS",
           "candidates_command", "verdicts_command", "describe"]
