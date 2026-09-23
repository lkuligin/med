"""The step 1 and judging runs of the OSS-judge comparison, as a table.

    python3 -m gdanschin_runtime.run_judges        # run it
    python3 -m gdanschin_runtime.judge_plan        # print it and stop

Two candidate judges are measured against gemini: GLM-5.3-Flash and
Qwen3.6-35B-A3B with thinking on. Each needs step 1 of its own over both
splits, and then judges every generator that already has candidates.

Step 1 groups by served model, because a switch costs minutes and an FP8
first start has cost half an hour. Judging does not: it goes generator by
generator, both judges on each, so that whatever it gets through is a set of
finished comparisons. Within any pair, medbullets before med_qa - the small
split fails fast if something is wrong with the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

# The two candidates. `serve` is the catalogue entry, `base` the registry
# entry - they differ because one describes weights and flags, the other
# describes what a run is called and how requests are shaped.
# serve, registry entry, and the budget this judge answers within. The
# budgets differ because the models do: GLM writes long and is served at a
# 32768 context, Qwen3.6-35B-A3B at 65536 but answers shorter.
JUDGES = (("glm-5.3-flash", "glm-5.3-flash-local", 16384),
          ("qwen3.6-35b-a3b", "qwen3.6-35b-a3b-local", 8192))

# Generators to judge. Every one has candidates on both splits already; the
# two gateway-era runs also carry candidates for the simple questions, which
# MEDQA_ONLY_QUESTIONS trims away.
GENERATORS = (
    "gemma-4-26b",
    "qwen3.6-27b-nr-local",
    "gpt-oss-20b-local",
    "qwen3.5-9b-nr-local",
    "gemma-4-e2b-local",
    "qwen3.5-4b-nr-local",
)

DATASETS = (("medbullets", "mkieffer/Medbullets", "difficult_questions_mb.csv"),
            ("med_qa", "bigbio/med_qa", "difficult_questions.csv"))

ATTEMPTS = 3

# The two steps sample differently, and neither number is ours to pick. 0.8 is
# what BaseInferenceConfig sets and what every step 1 run already measured has
# used; 1.0 is what VerifierConfig sets and what every gemini verdict we have
# was produced at. A judge run at 0.8 would not be comparable with those, which
# is the whole point of the exercise.
STEP1_TEMPERATURE = 0.8
JUDGE_TEMPERATURE = 1.0

# Budgets live on each judge above, not here. The reference 512 left 6% of
# gemini's verdicts with no verdict in them, because thinking is charged to
# the same allowance - and both candidates think. An unused allowance costs
# nothing; a short one costs verdicts we cannot tell from rejections.


@dataclass(frozen=True)
class Step:
    """One run: what to serve, what to do, and where it lands."""

    serve: str          # catalogue entry, the model the cards hold
    kind: str           # "step1" | "judge"
    base: str           # registry entry: the generator, or the judge itself
    dataset: str        # canonical dataset name
    dataset_dir: str    # where results live
    questions: str      # the frozen difficult list for this split
    judge: str = ""     # registry entry of the judge, for kind == "judge"
    max_tokens: int = 0  # what this run answers within

    @property
    def label(self) -> str:
        if self.kind == "step1":
            return f"step 1  {self.base} on {self.dataset_dir}"
        return f"judge   {self.base} on {self.dataset_dir} by {self.judge}"


def plan() -> list[Step]:
    """Every run, in the order they are done."""
    steps: list[Step] = []
    for serve, judge, budget in JUDGES:
        for dataset_dir, dataset, questions in DATASETS:
            steps.append(Step(serve, "step1", judge, dataset, dataset_dir,
                              questions, max_tokens=budget))
    # Generator first, then both judges, then both splits. This costs a
    # server switch per judge per generator rather than one per judge, and
    # that is the point: a generator finishes complete, judged both ways, so
    # a stop at any moment leaves whole comparisons rather than half of each.
    for generator in GENERATORS:
        for serve, judge, budget in JUDGES:
            for dataset_dir, dataset, questions in DATASETS:
                steps.append(Step(serve, "judge", generator, dataset,
                                  dataset_dir, questions, judge=judge,
                                  max_tokens=budget))
    return steps


def describe() -> None:
    steps = plan()
    served = None
    for index, step in enumerate(steps, 1):
        if step.serve != served:
            print(f"\n  --- serve {step.serve} ---")
            served = step.serve
        print(f"  {index:3}. {step.label}")
    switches = sum(1 for a, b in zip(steps, steps[1:]) if a.serve != b.serve)
    print(f"\n  {len(steps)} runs, {switches + 1} server starts")


if __name__ == "__main__":
    describe()


__all__ = ["Step", "plan", "describe", "JUDGES", "GENERATORS", "DATASETS",
           "ATTEMPTS", "STEP1_TEMPERATURE", "JUDGE_TEMPERATURE"]
