"""Run a sequence of steps unattended, logging where it is and how fast it goes.

    ./gdanschin_runtime/run_plan.sh              # start the whole plan
    ./gdanschin_runtime/run_plan.sh --from 3     # resume at stage 3
    ./gdanschin_runtime/run_plan.sh --list       # what the stages are
    tail -f logs/plan.log

Every step is resumable on its own - the generator tops a question up to the
target number of candidates, the verifier skips questions it has judged and
continues the ones that found nothing - so re-running the plan repeats no work.

The log carries a timestamp per line and a progress line every couple of
minutes with the rate and an estimate of what is left, which is the only way to
tell in the morning whether a stage was slow or stuck.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from gdanschin_runtime import _bootstrap
from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS, difficult_run

from results_store import (
    CandidateResults,
    OneShotResults,
    VerificationResults,
    dataset_dir_for,
)

REPO = Path(__file__).resolve().parents[1]
MONKEYS = REPO / "llm_monkeys"
RESULTS = MONKEYS / "results"
PYTHON = str(REPO / ".venv" / "bin" / "python")
LOG = REPO / "logs" / "plan.log"

GEN_CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "8"))
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "32"))
JUDGE = os.getenv("MEDQA_JUDGE_MODEL", "gemini-3.8-flash")
# Which dataset the whole plan is about. It decides the questions, the list of
# difficult ones, and the directory the runs are kept in - a plan cannot be
# half one dataset and half another, so it is read once, here.
DATASET_NAME = os.getenv("MEDQA_DATASET", "bigbio/med_qa")
MEDBULLETS = "mkieffer/Medbullets"
MODELS = ("gemma-4-26b", "gpt-oss-120b", "qwen3.8-27b-nr")

# A model's budget in models.py was measured on MedQA. MedBullets asks longer
# questions, so the budget is raised here rather than there: the MedQA runs a
# result is compared against were generated with the old one, and changing it
# in the registry would quietly change them too the next time they are run.
# How far a second pass over MedBullets takes each question, or None for no
# second pass. Off, because on this dataset the verified selector stops
# improving around k=20: past that the judge does still certify a candidate
# now and then, and on gpt-oss every one of those late certifications was
# wrong, so the pass costs about ten hours and moves the metric by nothing.
# Set it to 100 to run it - it does keep raising the ceiling, which is the
# measurement worth coming back for.
TOP_UP_TO: int | None = None

MAX_TOKENS: dict[tuple[str, str], int] = {
    ("medbullets", "gemma-4-26b"): 4096,
}


def difficult_file(dataset: str) -> str:
    """The list of questions the pipeline works on for this dataset."""
    from gdanschin_runtime.fetch_dataset import KNOWN

    return KNOWN[dataset_dir_for(dataset)].difficult.name
# Temperature comes from the judge's entry in models.py, and this overrides it
# for one run without inventing a registry entry for a setting being tried out.
# A judge run at a temperature other than its own is stored under a name of its
# own, or its verdicts mix with the ones it is being compared against.
JUDGE_TEMPERATURE = os.getenv("MEDQA_JUDGE_TEMPERATURE")
REPORT_EVERY = int(os.getenv("REPORT_EVERY", "120"))
JUDGE_CYCLE = int(os.getenv("JUDGE_CYCLE", "300"))

# Wall-clock seconds per candidate at GEN_CONCURRENCY, measured: gemma over its
# whole 483-question run, the others from probes, so treat them as a starting
# guess that the live rate replaces as soon as a step gets going.
SECONDS_PER_CANDIDATE: dict[str, float] = {
    "gemma-4-26b": 1.0,
    "gpt-oss-120b": 1.2,
    "qwen3.8-27b-nr": 2.4,
}
DEFAULT_SECONDS_PER_CANDIDATE = 2.0
# One answer per question with no candidates, measured at 110 questions/minute.
SECONDS_PER_ONE_SHOT_QUESTION = 0.6


def judge_temperature() -> float:
    """What the judge samples at: its entry, unless the environment says otherwise."""
    if JUDGE_TEMPERATURE is not None:
        return float(JUDGE_TEMPERATURE)
    return JUDGE_MODELS[JUDGE].temperature


@dataclass
class Step:
    stage: str
    name: str
    base: str
    kind: str           # one-shot | one-shot-full | candidates | verdicts
    target: int = 0     # candidates per question, for the generator
    dataset: str = DATASET_NAME     # the dataset this step is about

    @property
    def label(self) -> str:
        return f"{self.stage} / {self.name}"

    @property
    def dataset_dir(self) -> str:
        return dataset_dir_for(self.dataset)

    @property
    def run_name(self) -> str:
        """Where this step's results are stored.

        A run over a whole split is kept apart from a run over the questions
        the pipeline works on: same model, different coverage, and blending
        them would leave a directory nothing could describe. The split is
        step 1 proper and takes the model's name; the list run is the aside.
        """
        return difficult_run(self.base) if self.kind == "one-shot" else self.base

    @property
    def max_tokens(self) -> int:
        """The output budget this step runs with."""
        return MAX_TOKENS.get((self.dataset_dir, self.base),
                              BASE_MODELS[self.base].max_tokens)

    @property
    def questions(self) -> int:
        """How many questions this step covers."""
        from gdanschin_runtime.fetch_dataset import KNOWN

        known = KNOWN[self.dataset_dir]
        if self.kind == "one-shot-full":
            return known.rows
        return len(_question_ids(self.dataset))

    def command(self) -> list[str]:
        model = BASE_MODELS[self.base]
        if self.kind == "one-shot":
            return [PYTHON, str(REPO / "gdanschin_runtime" / "run_step1.py"),
                    "--base", self.base, "--max-tokens", str(self.max_tokens),
                    "--dataset", self.dataset,
                    "--difficult-questions", difficult_file(self.dataset),
                    "--concurrency", str(GEN_CONCURRENCY)]
        if self.kind == "one-shot-full":
            # Step 1 over the whole split, which run_step1.py cannot do: it
            # runs the list the pipeline works on, by design.
            return [PYTHON, "-m", "cli",
                    "--model", model.gateway_model, "--run-name", self.run_name,
                    "--dataset", self.dataset, "--n-attempts", "3",
                    "--concurrency", str(GEN_CONCURRENCY),
                    "--max-tokens", str(self.max_tokens),
                    "--temperature", "0.8"]
        if self.kind == "candidates":
            return [PYTHON, "-m", "inference.cli",
                    "--model", model.gateway_model, "--run-name", self.base,
                    "--difficult-questions", difficult_file(self.dataset),
                    "--dataset", self.dataset,
                    "--n-candidates", str(self.target),
                    "--concurrency", str(GEN_CONCURRENCY),
                    "--max-tokens", str(self.max_tokens)]
        # --judge-name only names the directory the verdicts go in. Without the
        # rest of the entry the verifier keeps its own defaults, so a judge
        # named for one setting would quietly have run at another.
        judge = JUDGE_MODELS[JUDGE]
        return [PYTHON, "-m", "verifier.cli",
                "--run-name", self.base, "--judge-name", JUDGE,
                "--model", judge.gateway_model,
                "--temperature", str(judge_temperature()),
                "--max-tokens", str(judge.max_tokens),
                "--dataset", self.dataset,
                "--concurrency", str(JUDGE_CONCURRENCY)]

    def done_and_total(self) -> tuple[float, float, str]:
        """(done, total, unit) for the progress line.

        Measured against the questions the step covers, which for steps 2 and 3
        is the list the run was given. A run can hold more than the list - a
        list can be shortened between runs, and nothing stored is ever thrown
        away - and counting what it holds against the list it is working now is
        how a finished stage came to report 308/165.
        """
        if self.kind in ("one-shot", "one-shot-full"):
            store = OneShotResults(RESULTS, self.run_name, self.dataset_dir)
            return len(store.read()), self.questions, "questions"
        candidates = CandidateResults(RESULTS, self.base, self.dataset_dir)
        wanted = _question_ids(self.dataset)
        if self.kind == "candidates":
            stored = sum(candidates.candidate_count(q) for q in wanted)
            return stored, len(wanted) * self.target, "candidates"
        judged = sum(1 for q in wanted
                     if (candidates.question_dir(q) / JUDGE).is_dir())
        return judged, len(wanted), "questions judged"


def plan() -> list[Step]:
    """The stages, in the order they run.

    Judging is started alongside each generation stage - the two use different
    gateways, so it costs the generator no throughput - and each dataset then
    gets a pass of its own afterwards for whatever the last one missed.
    """
    steps: list[Step] = []

    # 1. MedBullets candidates, every question, fifty each.
    for base in MODELS:
        stage = f"medbullets {base} k=50"
        steps += [
            Step(stage, "candidates k=50", base, "candidates", 50, MEDBULLETS),
            Step(stage, "verdicts", base, "verdicts", dataset=MEDBULLETS),
        ]

    # 2. MedQA step 1 over the whole split, which is what makes the one-shot
    #    numbers comparable between the two datasets.
    for base in MODELS:
        steps.append(Step(f"med_qa {base} step 1 (full split)", "single-step full",
                          base, "one-shot-full", dataset=DATASET_NAME))

    # 3. MedBullets verdicts at k=50, a pass of its own.
    for base in MODELS:
        steps.append(Step(f"medbullets {base} verdicts k=50", "verdicts",
                          base, "verdicts", dataset=MEDBULLETS))

    # 4. MedBullets candidates from fifty up to TOP_UP_TO. The generator tops
    #    each question up, so this adds the difference rather than redoing what
    #    is stored. Verdicts follow: no filtering is needed, since the verifier
    #    skips questions that already found a valid candidate and continues the
    #    rest from the candidate it stopped at.
    if TOP_UP_TO:
        for base in MODELS:
            stage = f"medbullets {base} k={TOP_UP_TO}"
            steps += [
                Step(stage, f"candidates k={TOP_UP_TO}", base, "candidates",
                     TOP_UP_TO, MEDBULLETS),
                Step(stage, "verdicts", base, "verdicts", dataset=MEDBULLETS),
            ]
        for base in MODELS:
            steps.append(Step(f"medbullets {base} verdicts k={TOP_UP_TO}",
                              "verdicts", base, "verdicts", dataset=MEDBULLETS))

    return steps


def remaining_candidates(step: Step) -> int:
    """How many candidates this step still has to generate."""
    if step.kind != "candidates":
        return 0
    store = CandidateResults(RESULTS, step.base, step.dataset_dir)
    stored = {q: store.candidate_count(q) for q in store.questions()}
    return sum(max(step.target - stored.get(str(q), 0), 0)
               for q in _question_ids(step.dataset))


def _question_ids(dataset: str) -> list[str]:
    from inference._dataset import load_difficult_question_ids

    return list(load_difficult_question_ids(MONKEYS / difficult_file(dataset)))


def _rate(base: str, rates: dict[str, float]) -> float:
    return rates.get(base, SECONDS_PER_CANDIDATE.get(
        base, DEFAULT_SECONDS_PER_CANDIDATE))


def forecast(steps: list[Step], rates: dict[str, float]) -> list[tuple[Step, float]]:
    """Seconds each step still needs, walking the plan in order.

    Later stages are estimated against what the earlier ones will have
    produced, not against what is on disk now: taking a run from k=20 to k=50
    generates thirty candidates a question, not fifty, and a forecast that
    missed that would be out by most of a day.
    """
    projected: dict[tuple[str, str], int] = {}
    out: list[tuple[Step, float]] = []
    for step in steps:
        questions = step.questions
        if step.kind == "verdicts":
            seconds = 0.0  # judged alongside generation, so it adds no wall clock
        elif step.kind in ("one-shot", "one-shot-full"):
            stored = len(OneShotResults(RESULTS, step.run_name, step.dataset_dir).read())
            attempts = 3 if step.kind == "one-shot-full" else 1
            seconds = (max(questions - stored, 0)
                       * SECONDS_PER_ONE_SHOT_QUESTION * attempts)
        else:
            key = (step.base, step.dataset_dir)
            have = projected.get(key)
            if have is None:
                have = stored_for_questions(step.base, step.dataset_dir,
                                            _question_ids(step.dataset))
            wanted = step.target * questions
            seconds = max(wanted - have, 0) * _rate(step.base, rates)
            projected[key] = max(have, wanted)
        out.append((step, seconds))
    return out


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def difficult_total() -> int:
    from inference._dataset import load_difficult_question_ids

    return len(load_difficult_question_ids(MONKEYS / DIFFICULT_QUESTIONS))


def watch(step: Step, started: float, stop: threading.Event,
          rates: dict[str, float], rest: list[Step] | None = None) -> None:
    """Log progress, what the rate so far says is left, and the plan's own eta."""
    first_done, _, _ = step.done_and_total()
    while not stop.wait(REPORT_EVERY):
        done, total, unit = step.done_and_total()
        elapsed = time.time() - started
        # Rate over this run only: what was already stored when the step began
        # was not produced now, and counting it would flatter the estimate.
        made = done - first_done
        per_unit = elapsed / made if made else 0.0
        # Only candidates go in: `rates` is seconds per candidate, and a
        # one-shot step measures seconds per question. Mixing the two made the
        # forecast for the rest of the plan read 16 hours instead of 28.
        if per_unit and step.kind == "candidates":
            rates[step.base] = per_unit      # the live rate replaces the guess
        left = max(total - done, 0)
        eta = (f", eta {timedelta(seconds=int(left * per_unit))}"
               if per_unit and left else "")
        plan_left = sum(sec for _, sec in forecast(rest or [], rates))
        ahead = (f", then {timedelta(seconds=int(plan_left))} of plan left"
                 if plan_left else "")
        log(f"    {step.label}: {done:.0f}/{total:.0f} {unit}, "
            f"{made:.0f} this run in {timedelta(seconds=int(elapsed))} "
            f"({60 / per_unit if per_unit else 0:.1f}/min){eta}{ahead}")


def step_log(step: Step) -> Path:
    """Where a step's own output goes: one file per step, model and dataset.

    The dataset is in the name because the plan spans two of them, and a log
    that mixed them would be unreadable exactly when it is needed.
    """
    name = {"one-shot": "step1", "one-shot-full": "step1-full",
            "candidates": "step2", "verdicts": "step3"}[step.kind]
    return REPO / "logs" / f"{name}_{step.dataset_dir}_{step.base}.log"


def spawn(step: Step) -> subprocess.Popen:
    """Start a step, with its own output going to its own log."""
    out = step_log(step)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("a", encoding="utf-8")
    handle.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} "
                 f"{step.label} =====\n")
    handle.flush()
    return subprocess.Popen(step.command(), cwd=MONKEYS, stdout=handle,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def stored_candidates(base: str, dataset: str) -> int:
    """How many candidates are on disk for a run, right now.

    Everything the run holds, which is what the verifier will walk - use
    stored_for_questions() for the work a generation step still has to do.
    """
    store = CandidateResults(RESULTS, base, dataset)
    return sum(store.candidate_count(q) for q in store.questions())


def stored_for_questions(base: str, dataset: str, ids: list[str]) -> int:
    """How many candidates are on disk for the questions a step covers.

    A run generated against a longer list keeps those questions, and counting
    them would make a run that has every question of the current list at fifty
    look like it was most of the way to a hundred.
    """
    store = CandidateResults(RESULTS, base, dataset)
    return sum(store.candidate_count(q) for q in ids)


def judge_while(alive: Callable[[], bool], step: Step) -> None:
    """Judge what has been generated so far, again and again, until generation
    stops - then once more for whatever the last pass missed.

    Each pass skips the questions that already found a valid candidate and
    continues the ones that did not, so judging a question that has only four
    candidates so far costs nothing later: the next pass picks it up at the
    fifth, and the judge would have read those four in order anyway.

    Two things this has to get right. An empty run is not a failure - at the
    start of a stage there is simply nothing generated yet, and the verifier
    raises rather than returning empty, so waiting is the correct response and
    must not count towards giving up. And a pass over a run that has not grown
    since the last one only re-reads every candidate on disk to conclude there
    is nothing to do.

    Generation and judging use different gateways, so this costs the generator
    no throughput.
    """
    failures = 0
    judged_at = -1
    while True:
        generating = alive()
        stored = stored_candidates(step.base, step.dataset_dir)

        if stored == 0:
            if not generating:
                log(f"    {step.label}: nothing was generated, nothing to judge")
                return
            log(f"    {step.label}: waiting for the first candidates")
        elif stored == judged_at:
            if not generating:
                return
            log(f"    {step.label}: no new candidates since the last pass")
        else:
            started = time.time()
            code = spawn(step).wait()
            judged_at = stored
            failures = failures + 1 if code != 0 else 0
            judged, total, _ = step.done_and_total()
            log(f"    {step.label}: pass over {stored} candidates finished, "
                f"exit {code}, {timedelta(seconds=int(time.time() - started))}, "
                f"{judged:.0f}/{total:.0f} questions judged")
            if failures >= 5:
                log(f"    {step.label}: five passes failed in a row, leaving it")
                return
            # The pass saw everything only if generation had already stopped
            # when it started; otherwise go round again for the tail.
            if not generating:
                return
            if not alive():
                continue

        if not alive():
            continue
        time.sleep(JUDGE_CYCLE if not failures else JUDGE_CYCLE * 2)


def run_step(step: Step, rates: dict[str, float],
             rest: list[Step] | None = None, judge: Step | None = None) -> int:
    log(f"  START {step.label}"
        + (f"  (+ {judge.name} alongside)" if judge else ""))
    started = time.time()
    stop = threading.Event()
    watcher = threading.Thread(target=watch,
                               args=(step, started, stop, rates, rest),
                               daemon=True)
    watcher.start()

    process = spawn(step)
    if judge is not None:
        judging = threading.Thread(
            target=judge_while,
            args=(lambda: process.poll() is None, judge),
            daemon=True,
        )
        judging.start()
    code = process.wait()
    if judge is not None:
        judging.join()

    stop.set()
    watcher.join(timeout=5)
    done, total, unit = step.done_and_total()
    log(f"  END   {step.label}: exit {code}, {timedelta(seconds=int(time.time() - started))}, "
        f"{done:.0f}/{total:.0f} {unit}  (log: {step_log(step).relative_to(REPO)})")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="stage number to start at (see --list)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    steps = plan()
    stages: list[str] = []
    for step in steps:
        if step.stage not in stages:
            stages.append(step.stage)

    if args.list:
        for number, stage in enumerate(stages, 1):
            print(f"  {number}. {stage}")
            for step in steps:
                if step.stage == stage:
                    print(f"       {step.name:22} {step.dataset_dir}, "
                          f"{step.questions} questions")
        return 0

    for base in {step.base for step in steps}:
        if base not in BASE_MODELS:
            print(f"unknown base model: {base}", file=sys.stderr)
            return 2
    if JUDGE not in JUDGE_MODELS:
        print(f"unknown judge: {JUDGE}", file=sys.stderr)
        return 2

    rates: dict[str, float] = {}
    started = time.time()
    todo = [s for s in steps if stages.index(s.stage) + 1 >= args.start]

    log("=" * 72)
    judge_model = JUDGE_MODELS[JUDGE]
    log(f"PLAN starts at stage {args.start}/{len(stages)}, "
        f"generation concurrency {GEN_CONCURRENCY}, judge {JUDGE} "
        f"({judge_model.gateway_model} at temperature {judge_temperature()}"
        + (" from MEDQA_JUDGE_TEMPERATURE" if JUDGE_TEMPERATURE is not None else "")
        + f") at {JUDGE_CONCURRENCY}, judging alongside generation every {JUDGE_CYCLE}s")
    predicted = forecast(todo, rates)
    total_estimate = sum(seconds for _, seconds in predicted)
    for number, stage in enumerate(stages, 1):
        if number < args.start:
            continue
        stage_estimate = sum(sec for step, sec in predicted if step.stage == stage)
        log(f"  forecast  stage {number}  {stage:34} "
            f"{timedelta(seconds=int(stage_estimate))}")
    log(f"  forecast  whole plan {timedelta(seconds=int(total_estimate))}, "
        f"ending around {(datetime.now() + timedelta(seconds=total_estimate)):%a %H:%M}")
    log("  (from measured seconds per candidate; the live rate replaces it per step)")

    for number, stage in enumerate(stages, 1):
        if number < args.start:
            log(f"SKIP stage {number}/{len(stages)}  {stage}")
            continue
        log(f"STAGE {number}/{len(stages)}  {stage}")
        stage_started = time.time()
        stage_steps = [s for s in steps if s.stage == stage]
        judge_step = next((s for s in stage_steps if s.kind == "verdicts"), None)

        for step in stage_steps:
            if step.kind == "verdicts":
                continue  # it runs alongside generation, below
            rest = todo[todo.index(step) + 1:]
            judge = judge_step if step.kind == "candidates" else None
            run_step(step, rates, rest, judge)

        # One more pass after generation, for whatever the last one missed.
        if judge_step is not None:
            judge_while(lambda: False, judge_step)

        log(f"STAGE {number}/{len(stages)} done in "
            f"{timedelta(seconds=int(time.time() - stage_started))}")

    log(f"PLAN finished in {timedelta(seconds=int(time.time() - started))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
