"""Run the grounded-unframed-facts-cited-answer plan unattended, end to end.

    cd ~/Projects/med && setsid nohup python3 -m gdanschin_runtime.experiment_chain \
        > logs/experiment_chain.log 2>&1 < /dev/null &

Steps, in order, all on our own cards:

1-2. gemma-4-26b generates the experiment on med_qa and medbullets (k=50),
     then GLM judges both with the fact-only prompt.
3.   experiment_report decides whether the experiment succeeded.
3a.  The step 1-2 candidates are copied, without verdicts, into
     grounded-unframed-facts-cited-answer-author-judge and judged by GLM with
     the author's prompt (question and options), then reported.
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
from typing import NamedTuple

# gpt-oss-20b runs at k=100 through run_sweep, whose generator takes one
# question at a time: a concurrency of k is what keeps its replicas busy.
os.environ.setdefault("GEN_CONCURRENCY", "100")

from gdanschin_runtime.models import BASE_MODELS  # noqa: E402
from gdanschin_runtime.oss_sweep import (  # noqa: E402
    MEDBULLETS, MEDQA, SWEEP, Target, candidates_command)
from gdanschin_runtime.run_sweep import (  # noqa: E402
    MONKEYS, PYTHON, REPO, ensure_served, log, run as sweep_run, through_env)

EXPERIMENT = "grounded-unframed-facts-cited-answer"
EXP_DIR = f"results/experiments/{EXPERIMENT}"
REANSWER_DIR = "results/experiments/reference-facts-cited-answer"
# Step 3a: the same grounded-unframed candidates judged with the author's
# prompt, which sees the question and options. Separates what the facts give
# from what seeing the question gives the judge.
AUTHOR_JUDGE_EXPERIMENT = "grounded-unframed-facts-cited-answer-author-judge"
AUTHOR_JUDGE_DIR = f"results/experiments/{AUTHOR_JUDGE_EXPERIMENT}"
AUTHOR_JUDGE_NAME = "glm-5.3-flash-local"
# The general-facts experiment's own run; step 3b adds medbullets to it.
GENERAL_FACTS_DIR = "results/experiments/general-facts-question-unaware-judge"
GENERAL_FACTS_RUN = "gemma-4-26b-local-background-2"
# The same general-facts candidates, answered again from their stored facts
# with the cited-facts prompt, so that a judge can be read by cited facts.
GENERAL_FACTS_CITED_DIR = "results/experiments/general-facts-cited-answer"
JUDGE_NAME = "glm-5.3-flash-local-fact-only"
# Step 3d: the same candidates, judged by a fact-only judge that rejects only
# clear errors, on med_qa before the gpt-oss fix.
CLEAR_ERRORS_JUDGE_NAME = "glm-5.3-flash-local-clear-errors-only"
GLM = Target("glm-5.3-flash-local", "glm-5.3-flash", 0, False)


class Judge(NamedTuple):
    """A served judge: its entry in BASE_MODELS, what to serve, how hard to
    drive it and how long it may think."""
    base: str
    serve: str
    concurrency: int
    max_tokens: int
    replicas: int

    def name(self, judge_prompt: str) -> str:
        # The authors' prompt keeps the bare name, as every stored judge has.
        return self.base if judge_prompt == "reference" else f"{self.base}-{judge_prompt}"


GLM_JUDGE = Judge("glm-5.3-flash-local", "glm-5.3-flash", 64, 16384, 1)
# Candidates to replace GLM for speed: four replicas each, so four times
# GLM's queue. Qwen judged the reference run at 8192 before; gpt-oss-120b
# gets the same.
GPT_OSS_JUDGE = Judge("gpt-oss-120b-local", "gpt-oss-120b", 256, 8192, 4)
QWEN_JUDGE = Judge("qwen3.6-35b-a3b-local", "qwen3.6-35b-a3b", 256, 8192, 4)
DATASETS = (MEDQA, MEDBULLETS)
DATASET_DIR = {MEDQA: "med_qa", MEDBULLETS: "medbullets"}
DIFFICULT = {MEDQA: "difficult_questions.csv", MEDBULLETS: "difficult_questions_mb.csv"}
K = 50
SHARDS = 8
# The workflow takes one question at a time, so a shard keeps K requests in
# flight only until its question's last candidates; four shards left
# gpt-oss-120b's replicas at ~26 requests each, and 200 more streams doubled
# its throughput. gpt-oss-20b is six times smaller and gets twice as many.
SHARDS_BY_BASE = {"gpt-oss-20b-local": 16}
# gpt-oss-20b reasons for up to ~3800 tokens before it writes its facts, so
# 4096 cut 2.5% of its fact lists off.
# qwen3.5-9b cut 3.4% of its grounded-unframed fact lists off at 4096.
# qwen3.5-4b gets the same as its larger sibling.
MAX_TOKENS_BY_BASE = {"gpt-oss-20b-local": 16384, "qwen3.5-9b-nr-local": 16384,
                      "qwen3.5-4b-nr-local": 16384}
# Requests the GLM judge serves at once, summed over all running judges.
# Measured on its single tp4 replica: 16 -> 706 tokens/s, 32 -> 1355, 64 ->
# 1824. Not higher: at 128 a request decodes at ~17 tokens/s, and the longest
# verdicts (16384 tokens) would outrun the 900 s request timeout.
JUDGE_CONCURRENCY = 64
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


def judged(results_dir: str, run: str, dataset: str, judge_name: str = JUDGE_NAME) -> int:
    root = MONKEYS / results_dir / DATASET_DIR[dataset] / "facts-pipeline" / run
    return sum(1 for _ in root.glob(f"question_*/{judge_name}/result.json"))


def copy_candidates(src_dir: str, dst_dir: str, run: str,
                    datasets: tuple[str, ...] = DATASETS) -> None:
    """Copy a run's candidates, without any judge's verdicts, into another
    experiment, so that it can be judged differently. Files already there are
    left alone."""
    for dataset in datasets:
        src = MONKEYS / src_dir / DATASET_DIR[dataset] / "facts-pipeline" / run
        dst = MONKEYS / dst_dir / DATASET_DIR[dataset] / "facts-pipeline" / run
        copied = 0
        for f in src.glob("question_*/iteration_*.json"):
            target = dst / f.parent.name / f.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied += 1
        if (src / "summary.json").is_file() and not (dst / "summary.json").exists():
            shutil.copy2(src / "summary.json", dst / "summary.json")
        log(f"  {run} on {DATASET_DIR[dataset]}: copied {copied}, "
            f"{stored(dst_dir, run, dataset)} in {dst_dir}")


# --- experiment: generate and judge -------------------------------------------

def generate_experiment(base: str, catalogue: str, free_form: bool, *,
                        exp_dir: str = EXP_DIR, run: str | None = None,
                        fact_prompt: str = "grounded-unframed",
                        answer_prompt: str = "cited-facts",
                        datasets: tuple[str, ...] = DATASETS,
                        shards: int | None = None, k: int = K) -> None:
    """Generate an experiment's candidates; `run` names the stored run and
    defaults to the model's own name."""
    run = run or base
    for dataset in datasets:
        want = len(questions(dataset)) * k
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
            shards = shards or SHARDS_BY_BASE.get(base, SHARDS)
            for s in range(shards):
                shard = shard_dir / f"{DATASET_DIR[dataset]}_{s}_of_{shards}.csv"
                shard.write_text("question_id\n" + "\n".join(ids[s::shards]) + "\n")
                command = [PYTHON, "-m", "inference.cli", "--results-dir", exp_dir,
                           "--model", BASE_MODELS[base].gateway_model, "--run-name", run,
                           "--fact-prompt", fact_prompt, "--answer-prompt", answer_prompt,
                           "--difficult-questions", str(shard), "--dataset", dataset,
                           "--n-candidates", str(k), "--concurrency", str(k),
                           "--max-tokens", str(MAX_TOKENS_BY_BASE.get(base, 4096))]
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
                     datasets: tuple[str, ...] = DATASETS,
                     judge_prompt: str = "fact-only",
                     judge_name: str = JUDGE_NAME,
                     judge: Judge = GLM_JUDGE,
                     stop_all: int = 1, stop_cited: int = 0,
                     concurrency: int | None = None) -> None:
    for base in bases:
        wait_for_judges(exp_dir, base)
    # A restarted chain passes every judging step again; swapping the cards
    # over to GLM for a run already judged would cost minutes for nothing.
    # Under a rule stricter than the first valid candidate a question counted
    # as judged may still want more, which only the verifier can tell.
    default_rule = stop_all <= 1 and stop_cited <= 0
    todo = {base: [d for d in datasets if stored(exp_dir, base, d)
                   and (not default_rule
                        or judged(exp_dir, base, d, judge_name) < len(questions(d)))]
            for base in bases}
    concurrency = concurrency or judge.concurrency
    if not any(todo.values()):
        log(f"  {', '.join(bases)}: already judged by {judge_name}")
        return
    if not serve(judge.base, judge.serve):
        log(f"  could not serve {judge.serve}; judging skipped")
        return
    for base in bases:
        # One dataset at a time, each with the whole of JUDGE_CONCURRENCY:
        # a single GLM replica serves every judge, so running them side by
        # side would only split the same throughput.
        for dataset in todo[base]:
            log(f"  judging {base} on {DATASET_DIR[dataset]} with {judge_name} "
                f"at concurrency {concurrency}, stopping at {stop_all} valid / {stop_cited} cited-valid")
            call(
                [PYTHON, "-m", "verifier.cli", "--results-dir", exp_dir, "--run-name", base,
                 "--judge-name", judge_name, "--judge-prompt", judge_prompt,
                 "--model", BASE_MODELS[judge.base].gateway_model, "--temperature", "1.0",
                 "--max-tokens", str(judge.max_tokens), "--request-timeout", "900",
                 "--dataset", dataset, "--difficult-questions", DIFFICULT[dataset],
                 "--concurrency", str(concurrency),
                 "--speculate-width", str(judge.replicas * 4),
                 "--stop-after-all-valid", str(stop_all),
                 "--stop-after-cited-valid", str(stop_cited),
                 # GLM runs on our own cards, so the extra requests cost
                 # nothing. It helps only once the tail leaves slots idle;
                 # twenty questions of long-reasoning checks still fill 64.
                 "--speculate-tail"],
                REPO / "logs" / f"chain_step3_{base}_{DATASET_DIR[dataset]}.log").wait()
        for dataset in datasets:
            log(f"  {base} on {DATASET_DIR[dataset]}: judged "
                f"{judged(exp_dir, base, dataset, judge_name)}/{len(questions(dataset))}")


def report(base: str, experiment: str = EXPERIMENT, judge_name: str = JUDGE_NAME) -> bool:
    # The experiment's own judge keeps the plain name; a second judge of the
    # same run gets its own file instead of overwriting that one.
    name = f"report_{base}" if judge_name in (JUDGE_NAME, AUTHOR_JUDGE_NAME) else f"report_{base}_{judge_name}"
    out = MONKEYS / "results" / "experiments" / experiment / name
    result = subprocess.run(
        through_env([PYTHON, "-m", "gdanschin_runtime.experiment_report",
                     "--experiment", experiment, "--run", base, "--judge", judge_name,
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

EMPTY_TOLERANCE = 0.05


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
    # Without a grammar gpt-oss-20b breaks its JSON now and then past what the
    # parser repairs, about one list in a hundred; the old run's fault was
    # most lists empty. Facts that are JSON are never fine.
    if empty > need * EMPTY_TOLERANCE or shaped:
        return f"{empty} empty fact lists and {shaped} JSON-shaped facts in the first {need}"
    return None


def regenerate_empty(base: str, catalogue: str, free_form: bool) -> None:
    """Generate again the candidates of a run that came out with no facts.

    Their answers were written from no facts at all. Done once, before any
    judge has seen the run: the store is write-once, so the candidates go and
    generate_experiment fills the gaps."""
    key = f"{base}_empty_regenerated"
    if state().get(key):
        log(f"  {base}: empty candidates already regenerated")
        return
    gone = 0
    for dataset in DATASETS:
        root = MONKEYS / EXP_DIR / DATASET_DIR[dataset] / "facts-pipeline" / base
        for f in root.glob("question_*/iteration_*.json"):
            if not json.loads(f.read_text())["candidate"].get("facts"):
                f.unlink()
                gone += 1
    log(f"  {base}: {gone} candidates without facts removed")
    generate_experiment(base, catalogue, free_form)
    mark(key)


def reanswer_general_facts() -> None:
    """General-facts answered again from the same facts, with FACTS USED."""
    if not any(stored(GENERAL_FACTS_CITED_DIR, GENERAL_FACTS_RUN, d) < stored(GENERAL_FACTS_DIR, GENERAL_FACTS_RUN, d)
               for d in DATASETS):
        log(f"  {GENERAL_FACTS_RUN}: already re-answered")
        return
    if not serve("gemma-4-26b-local", "gemma-4-26b"):
        log("  could not serve gemma-4-26b; re-answering skipped")
        return
    for dataset in DATASETS:
        log(f"  re-answering {GENERAL_FACTS_RUN} on {DATASET_DIR[dataset]} with cited-facts")
        call([PYTHON, "-m", "inference.reanswer", "--source-results-dir", GENERAL_FACTS_DIR,
              "--results-dir", GENERAL_FACTS_CITED_DIR, "--run-name", GENERAL_FACTS_RUN,
              "--dataset", dataset, "--model", BASE_MODELS["gemma-4-26b-local"].gateway_model,
              "--answer-prompt", "cited-facts", "--max-tokens", "4096", "--concurrency", "128"],
             REPO / "logs" / f"chain_reanswer_{GENERAL_FACTS_RUN}_{DATASET_DIR[dataset]}.log").wait()
        log(f"  {DATASET_DIR[dataset]}: {stored(GENERAL_FACTS_CITED_DIR, GENERAL_FACTS_RUN, dataset)}"
            f"/{stored(GENERAL_FACTS_DIR, GENERAL_FACTS_RUN, dataset)} re-answered")


def general_facts_judges() -> None:
    """The re-answered general-facts, judged with clear-errors-only: Qwen
    first, being quicker, then GLM. (Qwen also has the authors' prompt on
    them, from an earlier reading of the request.)"""
    for judge in (QWEN_JUDGE, GLM_JUDGE):
        prompt = "clear-errors-only"
        step(f"judge general-facts (cited) with {judge.name(prompt)}", judge_experiment,
             [GENERAL_FACTS_RUN], exp_dir=GENERAL_FACTS_CITED_DIR, judge_prompt=prompt,
             judge_name=judge.name(prompt), judge=judge)


def other_judges() -> None:
    """GLM's work again, by judges four times as fast to serve: can one of
    them stand in for it? The same candidates, the same prompts; whatever a
    judge has already judged is skipped."""
    for judge in (GPT_OSS_JUDGE, QWEN_JUDGE):
        for bases, exp_dir, prompt in (
                (["gemma-4-26b"], "results", "reference"),
                ([GENERAL_FACTS_RUN], GENERAL_FACTS_DIR, "fact-only"),
                (["gemma-4-26b-local"], EXP_DIR, "fact-only"),
                (["gemma-4-26b-local"], AUTHOR_JUDGE_DIR, "reference"),
                (["gemma-4-26b-local"], EXP_DIR, "clear-errors-only")):
            step(f"judge {bases[0]} in {exp_dir} with {judge.name(prompt)}", judge_experiment,
                 bases, exp_dir=exp_dir, judge_prompt=prompt,
                 judge_name=judge.name(prompt), judge=judge)


def retry_gpt_oss_20b() -> None:
    """Step 4 again, right after steps 5-7 generated with gpt-oss-20b, while it
    is still served. The first attempt stopped on facts the parser lost to a
    missing bracket; the parser now repairs them, but the store is write-once,
    so the candidates of that attempt go first."""
    if state().get("gpt_oss_20b_regenerated"):
        log("  gpt-oss-20b-local reference already regenerated")
        return
    if state().get("gpt_oss_20b_failed"):
        for dataset in DATASETS:
            old = MONKEYS / "results" / DATASET_DIR[dataset] / "facts-pipeline" / "gpt-oss-20b-local"
            if old.exists():
                log(f"  deleting the first attempt's {old}")
                shutil.rmtree(old)
        mark("gpt_oss_20b_failed", "")
    fix_gpt_oss_20b()
    if not state().get("gpt_oss_20b_failed"):
        mark("gpt_oss_20b_regenerated")


REFERENCE_SHARDS = 8


def sharded_reference(target: Target) -> None:
    """run_sweep's stage 2, the same command, split over question shards.

    One process takes a question at a time, and a question's candidates share
    their prompt prefix, so the router sends them all to one replica: two of
    gpt-oss-20b's four cards sat at 0%. Eight shards keep all four busy. What
    is stored already is skipped, as in one process."""
    if not ensure_served(target):
        return
    shard_dir = MONKEYS / "data" / "chain_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        want = target.k * len(questions(dataset))
        if stored("results", target.base, dataset) >= want:
            log(f"  {target.base} on {DATASET_DIR[dataset]}: complete")
            continue
        log(f"  stage 2 on {dataset}, {REFERENCE_SHARDS} shards")
        started = time.time()
        ids = questions(dataset)
        procs = []
        for i in range(REFERENCE_SHARDS):
            shard = shard_dir / f"reference_{DATASET_DIR[dataset]}_{i}_of_{REFERENCE_SHARDS}.csv"
            shard.write_text("question_id\n" + "\n".join(ids[i::REFERENCE_SHARDS]) + "\n")
            command = candidates_command(target, dataset, PYTHON)
            command[command.index("--difficult-questions") + 1] = str(shard)
            procs.append(call(command, REPO / "logs" / f"chain_step4_{target.base}_{DATASET_DIR[dataset]}.log"))
        for proc in procs:
            proc.wait()
        log(f"  stage 2 on {dataset} done in {(time.time() - started) / 60:.0f} min: "
            f"{stored('results', target.base, dataset)}/{want}")


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
    worker = threading.Thread(target=sharded_reference, args=(target,), daemon=True)
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

def step(name: str, fn, *args, **kwargs):
    log(f"=== {name}")
    try:
        return fn(*args, **kwargs)
    except Exception as error:          # one step failing must not end the chain
        log(f"  {name} raised {type(error).__name__}: {error}")
        return None


def main() -> int:
    log(f"chain for {EXPERIMENT} starting; state {state()}")
    step("steps 1-2: gemma-4-26b generates", generate_experiment, "gemma-4-26b-local", "gemma-4-26b", False)
    step("steps 1-2: GLM judges gemma-4-26b", judge_experiment, ["gemma-4-26b-local"])
    # Questions whose verdicts were unreadable or cut off were moved to
    # results/_malformed_verdicts; the judges above redo theirs, these two
    # runs are not otherwise judged by the chain.
    step("step 3c: re-judge repaired questions of general-facts on med_qa", judge_experiment,
         [GENERAL_FACTS_RUN], exp_dir=GENERAL_FACTS_DIR, datasets=(MEDQA,))
    step("step 3c: re-judge repaired questions of the reference run", judge_experiment,
         ["gemma-4-26b"], exp_dir="results", judge_prompt="reference",
         judge_name=AUTHOR_JUDGE_NAME)
    success = step("step 3: report", report, "gemma-4-26b-local")
    mark("success", bool(success))
    log(f"  experiment {'succeeded' if success else 'did not succeed'}")
    step("step 3a: copy grounded-unframed candidates for the author's judge", copy_candidates,
         EXP_DIR, AUTHOR_JUDGE_DIR, "gemma-4-26b-local")
    step("step 3a: GLM judges them with the author's prompt", judge_experiment,
         ["gemma-4-26b-local"], exp_dir=AUTHOR_JUDGE_DIR, judge_prompt="reference",
         judge_name=AUTHOR_JUDGE_NAME)
    step("step 3a: report", report, "gemma-4-26b-local", AUTHOR_JUDGE_EXPERIMENT, AUTHOR_JUDGE_NAME)
    step("step 3b: general-facts on medbullets, gemma-4-26b generates", generate_experiment,
         "gemma-4-26b-local", "gemma-4-26b", False,
         exp_dir=GENERAL_FACTS_DIR, run=GENERAL_FACTS_RUN, fact_prompt="background",
         answer_prompt="reference", datasets=(MEDBULLETS,))
    step("step 3b: GLM judges general-facts on medbullets", judge_experiment,
         [GENERAL_FACTS_RUN], exp_dir=GENERAL_FACTS_DIR, datasets=(MEDBULLETS,))
    step("step 3b: report", report, GENERAL_FACTS_RUN, "general-facts-question-unaware-judge")
    step("step 3d: GLM judges grounded-unframed on med_qa, clear errors only", judge_experiment,
         ["gemma-4-26b-local"], datasets=(MEDQA,), judge_prompt="clear-errors-only",
         judge_name=CLEAR_ERRORS_JUDGE_NAME)
    step("step 3d: report", report, "gemma-4-26b-local", EXPERIMENT, CLEAR_ERRORS_JUDGE_NAME)
    if state().get("gpt_oss_20b_failed"):
        log("=== step 4 deferred: it runs again after steps 5-7 generate with gpt-oss-20b")
    else:
        step("step 4: gpt-oss-20b-local reference step 2", fix_gpt_oss_20b)
    if success:
        for base, catalogue, free_form in EXPERIMENT_MODELS:
            step(f"steps 5-7: {base} generates", generate_experiment, base, catalogue, free_form)
        step("steps 5-7: gpt-oss-20b-local, candidates without facts again",
             regenerate_empty, "gpt-oss-20b-local", "gpt-oss-20b", True)
        other_judges()
        if state().get("gpt_oss_20b_failed"):
            step("step 4 (again): gpt-oss-20b-local reference step 2", retry_gpt_oss_20b)
        step("general-facts: answer again with FACTS USED", reanswer_general_facts)
        general_facts_judges()
        bases = [b for b, _, _ in EXPERIMENT_MODELS]
        step("steps 5-7: GLM judges", judge_experiment, bases)
        # The same candidates judged the authors' way too, beside step 3a's.
        for base in bases:
            step(f"steps 5-7: copy {base} candidates for the author's judge", copy_candidates,
                 EXP_DIR, AUTHOR_JUDGE_DIR, base)
        step("steps 5-7: GLM judges with the author's prompt", judge_experiment, bases,
             exp_dir=AUTHOR_JUDGE_DIR, judge_prompt="reference", judge_name=AUTHOR_JUDGE_NAME)
        for base in bases:
            step(f"steps 5-7: {base} report", report, base)
            step(f"steps 5-7: {base} report, author's judge", report, base,
                 AUTHOR_JUDGE_EXPERIMENT, AUTHOR_JUDGE_NAME)
    else:
        log("=== steps 5-7 skipped: the experiment did not succeed")
    step("step 8: re-answer reference runs with cited facts", reanswer_all)
    log("chain finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
