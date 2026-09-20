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
from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS

from results_store import CandidateResults, OneShotResults, VerificationResults

REPO = Path(__file__).resolve().parents[1]
MONKEYS = REPO / "llm_monkeys"
RESULTS = MONKEYS / "results"
PYTHON = str(REPO / ".venv" / "bin" / "python")
LOG = REPO / "logs" / "plan.log"

GEN_CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "8"))
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "32"))
JUDGE = os.getenv("MEDQA_JUDGE_MODEL", "gemini-3.8-flash")
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
    kind: str           # one-shot | candidates | verdicts
    target: int = 0     # candidates per question, for the generator

    @property
    def label(self) -> str:
        return f"{self.stage} / {self.name}"

    def command(self) -> list[str]:
        model = BASE_MODELS[self.base]
        if self.kind == "one-shot":
            return [PYTHON, str(REPO / "gdanschin_runtime" / "run_step1.py"),
                    "--base", self.base, "--max-tokens", str(model.max_tokens),
                    "--concurrency", str(GEN_CONCURRENCY)]
        if self.kind == "candidates":
            return [PYTHON, "-m", "inference.cli",
                    "--model", model.gateway_model, "--run-name", self.base,
                    "--difficult-questions", "difficult_questions.csv",
                    "--n-candidates", str(self.target),
                    "--concurrency", str(GEN_CONCURRENCY),
                    "--max-tokens", str(model.max_tokens)]
        # --judge-name only names the directory the verdicts go in. Without the
        # rest of the entry the verifier keeps its own defaults, so a judge
        # named for one setting would quietly have run at another.
        judge = JUDGE_MODELS[JUDGE]
        return [PYTHON, "-m", "verifier.cli",
                "--run-name", self.base, "--judge-name", JUDGE,
                "--model", judge.gateway_model,
                "--temperature", str(judge_temperature()),
                "--max-tokens", str(judge.max_tokens),
                "--concurrency", str(JUDGE_CONCURRENCY)]

    def done_and_total(self, questions: int) -> tuple[float, float, str]:
        """(done, total, unit) for the progress line."""
        if self.kind == "one-shot":
            store = OneShotResults(RESULTS, self.base)
            return len(store.read()), questions, "questions"
        candidates = CandidateResults(RESULTS, self.base)
        ids = candidates.questions()
        if self.kind == "candidates":
            stored = sum(candidates.candidate_count(q) for q in ids)
            return stored, questions * self.target, "candidates"
        judged = sum(1 for q in ids if (candidates.question_dir(q) / JUDGE).is_dir())
        return judged, questions, "questions judged"


def plan(k_first: int = 20, k_second: int = 50) -> list[Step]:
    """The stages, in the order they run."""
    steps: list[Step] = []
    for base in ("qwen3.8-27b-nr", "gpt-oss-120b"):
        stage = f"{base} k={k_first}"
        steps += [
            Step(stage, "single-step", base, "one-shot"),
            Step(stage, f"candidates k={k_first}", base, "candidates", k_first),
            Step(stage, "verdicts", base, "verdicts"),
        ]
    for base in ("gemma-4-26b", "qwen3.8-27b-nr", "gpt-oss-120b"):
        stage = f"{base} k={k_second}"
        steps += [
            Step(stage, f"candidates k={k_second}", base, "candidates", k_second),
            Step(stage, "verdicts", base, "verdicts"),
        ]
    return steps


def remaining_candidates(step: Step, questions: int) -> int:
    """How many candidates this step still has to generate."""
    if step.kind != "candidates":
        return 0
    store = CandidateResults(RESULTS, step.base)
    stored = {q: store.candidate_count(q) for q in store.questions()}
    return sum(max(step.target - stored.get(str(q), 0), 0)
               for q in _question_ids(questions))


def _question_ids(questions: int) -> list[str]:
    from inference._dataset import load_difficult_question_ids

    ids = load_difficult_question_ids(MONKEYS / "difficult_questions.csv")
    return list(ids)[:questions]


def _rate(base: str, rates: dict[str, float]) -> float:
    return rates.get(base, SECONDS_PER_CANDIDATE.get(
        base, DEFAULT_SECONDS_PER_CANDIDATE))


def forecast(steps: list[Step], questions: int,
             rates: dict[str, float]) -> list[tuple[Step, float]]:
    """Seconds each step still needs, walking the plan in order.

    Later stages are estimated against what the earlier ones will have
    produced, not against what is on disk now: taking a run from k=20 to k=50
    generates thirty candidates a question, not fifty, and a forecast that
    missed that would be out by most of a day.
    """
    projected: dict[str, int] = {}
    out: list[tuple[Step, float]] = []
    for step in steps:
        if step.kind == "verdicts":
            seconds = 0.0  # judged alongside generation, so it adds no wall clock
        elif step.kind == "one-shot":
            stored = len(OneShotResults(RESULTS, step.base).read())
            seconds = max(questions - stored, 0) * SECONDS_PER_ONE_SHOT_QUESTION
        else:
            have = projected.get(step.base)
            if have is None:
                have = stored_candidates(step.base)
            wanted = step.target * questions
            seconds = max(wanted - have, 0) * _rate(step.base, rates)
            projected[step.base] = max(have, wanted)
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

    return len(load_difficult_question_ids(MONKEYS / "difficult_questions.csv"))


def watch(step: Step, questions: int, started: float, stop: threading.Event,
          rates: dict[str, float], rest: list[Step] | None = None) -> None:
    """Log progress, what the rate so far says is left, and the plan's own eta."""
    first_done, _, _ = step.done_and_total(questions)
    while not stop.wait(REPORT_EVERY):
        done, total, unit = step.done_and_total(questions)
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
        plan_left = sum(sec for _, sec in forecast(rest or [], questions, rates))
        ahead = (f", then {timedelta(seconds=int(plan_left))} of plan left"
                 if plan_left else "")
        log(f"    {step.label}: {done:.0f}/{total:.0f} {unit}, "
            f"{made:.0f} this run in {timedelta(seconds=int(elapsed))} "
            f"({60 / per_unit if per_unit else 0:.1f}/min){eta}{ahead}")


def step_log(step: Step) -> Path:
    name = {"one-shot": "step1", "candidates": "step2", "verdicts": "step3"}[step.kind]
    return REPO / "logs" / f"{name}_{step.base}.log"


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


def stored_candidates(base: str) -> int:
    """How many candidates are on disk for a run, right now."""
    store = CandidateResults(RESULTS, base)
    return sum(store.candidate_count(q) for q in store.questions())


def judge_while(alive: Callable[[], bool], step: Step, questions: int) -> None:
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
        stored = stored_candidates(step.base)

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
            judged, total, _ = step.done_and_total(questions)
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


def run_step(step: Step, questions: int, rates: dict[str, float],
             rest: list[Step] | None = None, judge: Step | None = None) -> int:
    log(f"  START {step.label}"
        + (f"  (+ {judge.name} alongside)" if judge else ""))
    started = time.time()
    stop = threading.Event()
    watcher = threading.Thread(target=watch,
                               args=(step, questions, started, stop, rates, rest),
                               daemon=True)
    watcher.start()

    process = spawn(step)
    if judge is not None:
        judging = threading.Thread(
            target=judge_while,
            args=(lambda: process.poll() is None, judge, questions),
            daemon=True,
        )
        judging.start()
    code = process.wait()
    if judge is not None:
        judging.join()

    stop.set()
    watcher.join(timeout=5)
    done, total, unit = step.done_and_total(questions)
    log(f"  END   {step.label}: exit {code}, {timedelta(seconds=int(time.time() - started))}, "
        f"{done:.0f}/{total:.0f} {unit}  (log: {step_log(step).relative_to(REPO)})")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="stage number to start at (see --list)")
    parser.add_argument("--k-first", type=int, default=20)
    parser.add_argument("--k-second", type=int, default=50)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    steps = plan(args.k_first, args.k_second)
    stages: list[str] = []
    for step in steps:
        if step.stage not in stages:
            stages.append(step.stage)

    if args.list:
        for number, stage in enumerate(stages, 1):
            print(f"  {number}. {stage}")
            for step in steps:
                if step.stage == stage:
                    print(f"       {step.name}")
        return 0

    for base in {step.base for step in steps}:
        if base not in BASE_MODELS:
            print(f"unknown base model: {base}", file=sys.stderr)
            return 2
    if JUDGE not in JUDGE_MODELS:
        print(f"unknown judge: {JUDGE}", file=sys.stderr)
        return 2

    questions = difficult_total()
    rates: dict[str, float] = {}
    started = time.time()
    todo = [s for s in steps if stages.index(s.stage) + 1 >= args.start]

    log("=" * 72)
    judge_model = JUDGE_MODELS[JUDGE]
    log(f"PLAN starts at stage {args.start}/{len(stages)}, {questions} questions, "
        f"generation concurrency {GEN_CONCURRENCY}, judge {JUDGE} "
        f"({judge_model.gateway_model} at temperature {judge_temperature()}"
        + (" from MEDQA_JUDGE_TEMPERATURE" if JUDGE_TEMPERATURE is not None else "")
        + f") at {JUDGE_CONCURRENCY}, judging alongside generation every {JUDGE_CYCLE}s")
    predicted = forecast(todo, questions, rates)
    total_estimate = sum(seconds for _, seconds in predicted)
    for number, stage in enumerate(stages, 1):
        if number < args.start:
            continue
        stage_estimate = sum(sec for step, sec in predicted if step.stage == stage)
        log(f"  forecast  stage {number}  {stage:22} "
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
            run_step(step, questions, rates, rest, judge)

        # One more pass after generation, for whatever the last one missed.
        if judge_step is not None:
            judge_while(lambda: False, judge_step, questions)

        log(f"STAGE {number}/{len(stages)} done in "
            f"{timedelta(seconds=int(time.time() - stage_started))}")

    log(f"PLAN finished in {timedelta(seconds=int(time.time() - started))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
