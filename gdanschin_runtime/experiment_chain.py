"""Run the grounded-unframed-facts-cited-answer plan unattended, end to end.

    cd ~/Projects/med && setsid nohup python3 -m gdanschin_runtime.experiment_chain \
        > logs/experiment_chain.log 2>&1 < /dev/null &

Steps, in order, all on our own cards:

1-2. gemma-4-26b generates the experiment on med_qa and medbullets (k=50),
     then GLM judges both with the fact-only prompt.
3.   experiment_report decides whether the experiment succeeded.
3b.  general-facts-question-unaware-judge gets its medbullets run: gemma-4-26b
     with the background fact prompt and the reference answer prompt (k=50),
     judged fact-only by GLM, then reported.
4.   gpt-oss-20b-local's reference step 2 is regenerated from scratch through
     run_sweep's own machinery (free-form facts, no judge), medbullets first,
     with the first candidates checked for empty or JSON-shaped facts.
5-7. Only if step 3 succeeded: qwen3.6-27b-nr, gpt-oss-120b and gpt-oss-20b
     generate the experiment (k=50), then GLM judges all of them.
8.   Every local reference step 2 run is answered again from its stored facts
     with the cited-facts answer prompt, into
     results/experiments/reference-facts-cited-answer. No judge: the facts,
     and so the verdicts on them, are unchanged.

Resumable: generation tops up, judging skips judged questions, re-answering
skips stored candidates, and the one destructive step (deleting the old
gpt-oss-20b runs) is recorded in a state file so a restart never repeats it.
A failing step is logged and the chain moves on to the next.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

# gpt-oss-20b runs at k=100 through run_sweep, whose generator takes one
# question at a time: a concurrency of k is what keeps its replicas busy.
os.environ.setdefault("GEN_CONCURRENCY", "100")

from gdanschin_runtime.models import BASE_MODELS  # noqa: E402
from gdanschin_runtime.oss_sweep import MEDBULLETS, MEDQA, SWEEP, Target  # noqa: E402
from gdanschin_runtime.run_sweep import (  # noqa: E402
    MONKEYS, PYTHON, REPO, ensure_served, log, run as sweep_run, through_env)

EXPERIMENT = "grounded-unframed-facts-cited-answer"
EXP_DIR = f"results/experiments/{EXPERIMENT}"
REANSWER_DIR = "results/experiments/reference-facts-cited-answer"
# The general-facts experiment's own run; step 3b adds medbullets to it.
GENERAL_FACTS_DIR = "results/experiments/general-facts-question-unaware-judge"
GENERAL_FACTS_RUN = "gemma-4-26b-local-background-2"
JUDGE_NAME = "glm-5.3-flash-local-fact-only"
GLM = Target("glm-5.3-flash-local", "glm-5.3-flash", 0, False)
DATASETS = (MEDQA, MEDBULLETS)
DATASET_DIR = {MEDQA: "med_qa", MEDBULLETS: "medbullets"}
DIFFICULT = {MEDQA: "difficult_questions.csv", MEDBULLETS: "difficult_questions_mb.csv"}
K = 50
SHARDS = 4
STATE = MONKEYS / EXP_DIR / "chain_state.json"
ENV = {**os.environ, "PYTHONPATH": str(REPO)}

# (run name in models.py, entry in the serving catalogue, free-form facts)
EXPERIMENT_MODELS = (
    ("qwen3.6-27b-nr-local", "qwen3.6-27b", False),
    ("gpt-oss-120b-local", "gpt-oss-120b", False),
    ("gpt-oss-20b-local", "gpt-oss-20b", True),
)


# --- plumbing ---------------------------------------------------------------

def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def mark(key: str, value=True) -> None:
    current = state()
    current[key] = value
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(current, indent=1))


def serve(base: str, catalogue: str) -> bool:
    return ensure_served(Target(base, catalogue, K, False))


def call(command: list[str], logfile: Path) -> subprocess.Popen:
    handle = open(logfile, "a")
    return subprocess.Popen(through_env(command), cwd=MONKEYS, env=ENV,
                            stdout=handle, stderr=subprocess.STDOUT)


def questions(dataset: str) -> list[str]:
    lines = (MONKEYS / DIFFICULT[dataset]).read_text(encoding="utf-8-sig").splitlines()
    return [l.split(",")[0].strip() for l in lines[1:] if l.strip()]


def stored(results_dir: str, run: str, dataset: str) -> int:
    root = MONKEYS / results_dir / DATASET_DIR[dataset] / "facts-pipeline" / run
    return sum(1 for _ in root.glob("question_*/iteration_*.json"))


def judged(results_dir: str, run: str, dataset: str) -> int:
    root = MONKEYS / results_dir / DATASET_DIR[dataset] / "facts-pipeline" / run
    return sum(1 for _ in root.glob(f"question_*/{JUDGE_NAME}/result.json"))


# --- experiment: generate and judge -------------------------------------------

def generate_experiment(base: str, catalogue: str, free_form: bool, *,
                        exp_dir: str = EXP_DIR, run: str | None = None,
                        fact_prompt: str = "grounded-unframed",
                        answer_prompt: str = "cited-facts",
                        datasets: tuple[str, ...] = DATASETS) -> None:
    """Generate an experiment's candidates; `run` names the stored run and
    defaults to the model's own name."""
    run = run or base
    for dataset in datasets:
        want = len(questions(dataset)) * K
        for attempt in (1, 2):
            have = stored(exp_dir, run, dataset)
            if have >= want:
                log(f"  {run} on {DATASET_DIR[dataset]}: {have}/{want} candidates, complete")
                break
            if not serve(base, catalogue):
                log(f"  could not serve {catalogue}; {base} skipped")
                return
            log(f"  {run} on {DATASET_DIR[dataset]}: {have}/{want}, generating (attempt {attempt})")
            shard_dir = MONKEYS / "data" / "chain_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            ids = questions(dataset)
            procs = []
            for s in range(SHARDS):
                shard = shard_dir / f"{DATASET_DIR[dataset]}_{s}_of_{SHARDS}.csv"
                shard.write_text("question_id\n" + "\n".join(ids[s::SHARDS]) + "\n")
                command = [PYTHON, "-m", "inference.cli", "--results-dir", exp_dir,
                           "--model", BASE_MODELS[base].gateway_model, "--run-name", run,
                           "--fact-prompt", fact_prompt, "--answer-prompt", answer_prompt,
                           "--difficult-questions", str(shard), "--dataset", dataset,
                           "--n-candidates", str(K), "--concurrency", str(K),
                           "--max-tokens", "4096"]
                if free_form:
                    command.append("--free-form-facts")
                procs.append(call(command, REPO / "logs" / f"chain_step2_{run}_{DATASET_DIR[dataset]}.log"))
            for p in procs:
                p.wait()
        log(f"  {run} on {DATASET_DIR[dataset]}: {stored(exp_dir, run, dataset)}/{want} after generation")


def wait_for_judges(exp_dir: str, run: str) -> None:
    """Wait for judges already at work on this run, e.g. left running by an
    earlier chain, so that a second one does not judge the same questions."""
    pattern = f"verifier.cli --results-dir {exp_dir} --run-name {run} "
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        log(f"  a judge is already running on {run}; waiting for it")
        time.sleep(300)


def judge_experiment(bases: list[str], *, exp_dir: str = EXP_DIR,
                     datasets: tuple[str, ...] = DATASETS) -> None:
    for base in bases:
        wait_for_judges(exp_dir, base)
    if not serve(GLM.base, GLM.serve):
        log("  could not serve glm-5.3-flash; judging skipped")
        return
    for base in bases:
        procs = []
        for dataset in datasets:
            if not stored(exp_dir, base, dataset):
                continue
            log(f"  judging {base} on {DATASET_DIR[dataset]}")
            procs.append(call(
                [PYTHON, "-m", "verifier.cli", "--results-dir", exp_dir, "--run-name", base,
                 "--judge-name", JUDGE_NAME, "--judge-prompt", "fact-only",
                 "--model", "zai-org/GLM-5.3-Flash", "--temperature", "1.0",
                 "--max-tokens", "16384", "--request-timeout", "900", "--dataset", dataset,
                 "--difficult-questions", DIFFICULT[dataset], "--concurrency", "64"],
                REPO / "logs" / f"chain_step3_{base}_{DATASET_DIR[dataset]}.log"))
        for p in procs:
            p.wait()
        for dataset in datasets:
            log(f"  {base} on {DATASET_DIR[dataset]}: judged "
                f"{judged(exp_dir, base, dataset)}/{len(questions(dataset))}")


def report(base: str, experiment: str = EXPERIMENT) -> bool:
    out = MONKEYS / "results" / "experiments" / experiment / f"report_{base}"
    result = subprocess.run(
        through_env([PYTHON, "-m", "gdanschin_runtime.experiment_report",
                     "--experiment", experiment, "--run", base, "--judge", JUDGE_NAME,
                     "--out", str(out)]),
        cwd=REPO, env=ENV, capture_output=True, text=True)
    for line in (result.stdout or result.stderr).splitlines():
        log(f"    {line}")
    try:
        return bool(json.loads(Path(str(out) + ".json").read_text())["success"])
    except (OSError, KeyError, ValueError):
        log("  report failed; treating the experiment as not successful")
        return False


# --- step 4: regenerate gpt-oss-20b-local's reference step 2 -------------------

def facts_look_wrong(dataset: str, need: int = 60) -> str | None:
    """None if the first candidates look right, a reason if not, '' if too few yet."""
    root = MONKEYS / "results" / DATASET_DIR[dataset] / "facts-pipeline" / "gpt-oss-20b-local"
    files = list(root.glob("question_*/iteration_*.json"))
    if len(files) < need:
        return ""
    cands = [json.loads(f.read_text())["candidate"] for f in files[:need]]
    empty = sum(1 for c in cands if not c.get("facts"))
    shaped = sum(1 for c in cands for f in (c.get("facts") or [])
                 if f.strip().startswith(("{", "[")) or '":' in f)
    if empty or shaped:
        return f"{empty} empty fact lists and {shaped} JSON-shaped facts in the first {need}"
    return None


def fix_gpt_oss_20b() -> None:
    target = next(t for t in SWEEP if t.base == "gpt-oss-20b-local")
    if not target.free_form or target.judged:
        log("  oss_sweep no longer marks gpt-oss-20b-local free-form and unjudged; not touching it")
        return
    if not state().get("gpt_oss_20b_deleted"):
        for dataset in DATASETS:
            old = MONKEYS / "results" / DATASET_DIR[dataset] / "facts-pipeline" / "gpt-oss-20b-local"
            if old.exists():
                log(f"  deleting {old}")
                shutil.rmtree(old)
        mark("gpt_oss_20b_deleted")
    worker = threading.Thread(target=sweep_run, args=(target, 1, 1), daemon=True)
    worker.start()
    verdict = ""
    while worker.is_alive() and verdict == "":
        time.sleep(30)
        verdict = facts_look_wrong(MEDBULLETS)
    if verdict:
        log(f"  gpt-oss-20b-local looks wrong: {verdict}; stopping its generation")
        mark("gpt_oss_20b_failed", verdict)
        # run_sweep moves on to the next dataset when a generation ends, so
        # keep stopping them until its thread has nothing left to start.
        while worker.is_alive():
            subprocess.run(["pkill", "-f", "inference.cli.*--run-name gpt-oss-20b-local"])
            worker.join(timeout=10)
        return
    if verdict is None:
        log("  gpt-oss-20b-local: first medbullets candidates have facts, as sentences")
    worker.join()
    log("  gpt-oss-20b-local regenerated: "
        + ", ".join(f"{DATASET_DIR[d]} {stored('results', 'gpt-oss-20b-local', d)}" for d in DATASETS))


# --- step 8: answer again, with citations, from stored facts -------------------

def reanswer_all() -> None:
    for target in SWEEP:
        runs = [d for d in DATASETS if stored("results", target.base, d)]
        if not runs:
            continue
        if target.base == "gpt-oss-20b-local" and state().get("gpt_oss_20b_failed"):
            log("  gpt-oss-20b-local skipped: its regeneration failed")
            continue
        if not serve(target.base, target.serve):
            log(f"  could not serve {target.serve}; {target.base} skipped")
            continue
        for dataset in runs:
            log(f"  re-answering {target.base} on {DATASET_DIR[dataset]}: "
                f"{stored('results', target.base, dataset)} candidates")
            call([PYTHON, "-m", "inference.reanswer", "--source-results-dir", "results",
                  "--results-dir", REANSWER_DIR, "--run-name", target.base,
                  "--dataset", dataset, "--model", BASE_MODELS[target.base].gateway_model,
                  "--answer-prompt", "cited-facts", "--max-tokens", "4096",
                  "--concurrency", "128"],
                 REPO / "logs" / f"chain_step8_{target.base}_{DATASET_DIR[dataset]}.log").wait()
            log(f"  {target.base} on {DATASET_DIR[dataset]}: "
                f"{stored(REANSWER_DIR, target.base, dataset)} re-answered")


# --- the chain ------------------------------------------------------------------

def step(name: str, fn, *args):
    log(f"=== {name}")
    try:
        return fn(*args)
    except Exception as error:          # one step failing must not end the chain
        log(f"  {name} raised {type(error).__name__}: {error}")
        return None


def main() -> int:
    log(f"chain for {EXPERIMENT} starting; state {state()}")
    step("steps 1-2: gemma-4-26b generates", generate_experiment, "gemma-4-26b-local", "gemma-4-26b", False)
    step("steps 1-2: GLM judges gemma-4-26b", judge_experiment, ["gemma-4-26b-local"])
    success = step("step 3: report", report, "gemma-4-26b-local")
    mark("success", bool(success))
    log(f"  experiment {'succeeded' if success else 'did not succeed'}")
    step("step 3b: general-facts on medbullets, gemma-4-26b generates", generate_experiment,
         "gemma-4-26b-local", "gemma-4-26b", False,
         exp_dir=GENERAL_FACTS_DIR, run=GENERAL_FACTS_RUN, fact_prompt="background",
         answer_prompt="reference", datasets=(MEDBULLETS,))
    step("step 3b: GLM judges general-facts on medbullets", judge_experiment,
         [GENERAL_FACTS_RUN], exp_dir=GENERAL_FACTS_DIR, datasets=(MEDBULLETS,))
    step("step 3b: report", report, GENERAL_FACTS_RUN, "general-facts-question-unaware-judge")
    step("step 4: gpt-oss-20b-local reference step 2", fix_gpt_oss_20b)
    if success:
        for base, catalogue, free_form in EXPERIMENT_MODELS:
            step(f"steps 5-7: {base} generates", generate_experiment, base, catalogue, free_form)
        step("steps 5-7: GLM judges", judge_experiment, [b for b, _, _ in EXPERIMENT_MODELS])
    else:
        log("=== steps 5-7 skipped: the experiment did not succeed")
    step("step 8: re-answer reference runs with cited facts", reanswer_all)
    log("chain finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
