"""Three kinds of facts, three generators, three judges - on MedQA only.

    cd ~/Projects/med && setsid nohup python3 -m gdanschin_runtime.facts_matrix \\
        >> logs/facts_matrix.log 2>&1 < /dev/null &

Kinds of facts, all answered with the cited-facts prompt (FACTS USED), k=50:
    authors-facts       the authors' fact prompt
    general-facts       the background prompt
    grounded-unframed   the grounded-unframed prompt
Generators, served locally: gemma-4-26b, gpt-oss-120b, qwen3.6-27b (no
reasoning). Judged with the authors' prompt (authors-facts), fact-only
(general-facts), and the authors' prompt and clear-errors-only
(grounded-unframed), by:
    qwen3.6-35b-a3b on gemma and gpt-oss-120b, and gpt-oss-120b on qwen3.6-27b,
        until 5 candidates pass on their cited facts and 1 on all of them;
    GLM on all three, by the same rule.
A question runs out of candidates before either if it must.

Phases go by the model on the cards, so each is served once. Before each, a
short load test picks the concurrency past which throughput stops growing.
Everything is resumable: generation tops runs up, the verifier skips what a
judge has already settled and continues what it has not.
"""

from __future__ import annotations

import json
import re
import resource
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from gdanschin_runtime.experiment_chain import (
    AUTHOR_JUDGE_DIR, EXP_DIR, GENERAL_FACTS_CITED_DIR, GENERAL_FACTS_DIR,
    GENERAL_FACTS_RUN, GLM_JUDGE, GPT_OSS_JUDGE, K, MAX_TOKENS_BY_BASE, MEDQA, PYTHON,
    QWEN_JUDGE,
    REPO, BASE_MODELS, DATASET_DIR, MONKEYS, Judge, call, copy_candidates,
    generate_experiment, judge_experiment, judged, log, questions, serve, step, stored)
from gdanschin_runtime.question_ids import resolve

sys.path.insert(0, str(MONKEYS))
from inference._prompts import FACT_PROMPTS  # noqa: E402
from verifier._prompts import JUDGE_PROMPTS  # noqa: E402

DATASETS = (MEDQA,)
AUTHORS_DIR = "results/experiments/authors-facts-cited-answer"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
SERVER_LOGS = REPO / "gpu_serving" / "run"

# (base, what to serve)
GEMMA = ("gemma-4-26b-local", "gemma-4-26b")
GPT_OSS = ("gpt-oss-120b-local", "gpt-oss-120b")
QWEN27 = ("qwen3.6-27b-nr-local", "qwen3.6-27b")
GPT20 = ("gpt-oss-20b-local", "gpt-oss-20b")
QWEN9 = ("qwen3.5-9b-nr-local", "qwen3.5-9b")
GENERATORS = (GEMMA, GPT_OSS, QWEN27, GPT20, QWEN9)
# The two small generators keep k=100, as their reference runs were made.
K_BY_BASE = {GPT20[0]: 100, QWEN9[0]: 100}
# gpt-oss-20b answers an empty schema when held to one, so its facts are free-form.
FREE_FORM = {GPT20[0]}
# Generated afresh on the authors' fact prompt rather than answered again from
# the reference run: that run's facts were written under a 4096-token limit.
FRESH_AUTHORS_FACTS = {QWEN9[0]}


def k_of(base: str) -> int:
    return K_BY_BASE.get(base, K)

# Stopping rules: (all facts valid, cited facts valid)
DEEP = (1, 5)
FIRST = (1, 0)


def general_run(base: str) -> str:
    """gemma's general-facts run predates this matrix and kept its name."""
    return GENERAL_FACTS_RUN if base == GEMMA[0] else base


def judged_sets(base: str) -> list[tuple[str, str, str]]:
    """(results dir, run, judge prompt) for every set a generator is judged on.

    grounded-unframed (authors' prompt and clear-errors-only) was dropped from
    the judging on 2026-10-01; what had been judged of it stays on disk."""
    return [(AUTHORS_DIR, base, "reference"),
            (GENERAL_FACTS_CITED_DIR, general_run(base), "fact-only")]


# --- how hard to drive a model ------------------------------------------------

def server_load(catalogue: str, since: float, until: float) -> tuple[float, float]:
    """Mean requests running and tokens/s, summed over replicas, in a window."""
    pattern = re.compile(r"^\[(\S+ \S+) (\w+)\] Decode batch, #running-req: (\d+).*"
                         r"gen throughput \(token/s\): ([\d.]+)")
    per_rank: dict[str, list[tuple[int, float]]] = {}
    path = SERVER_LOGS / f"{catalogue}.log"
    for line in path.read_text(errors="replace").splitlines()[-20000:]:
        m = pattern.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if since <= t <= until:
            per_rank.setdefault(m.group(2), []).append((int(m.group(3)), float(m.group(4))))
    running = sum(sum(r for r, _ in v) / len(v) for v in per_rank.values())
    speed = sum(sum(t for _, t in v) / len(v) for v in per_rank.values())
    return running, speed


def sample_prompts(kind: str, judge_prompt: str = "fact-only") -> list[list[dict]]:
    """Real prompts from stored candidates: fact generation, or fact checks."""
    root = MONKEYS / EXP_DIR / "med_qa" / "facts-pipeline" / GEMMA[0]
    out = []
    for f in sorted(root.glob("question_*/iteration_0.json"))[:40]:
        record = json.loads(f.read_text())
        options = "\n".join(f"{k}. {v}" for k, v in sorted(record["options"].items()))
        if kind == "generate":
            spec = FACT_PROMPTS["grounded-unframed"]
            out.append([{"role": "system", "content": spec.system_instruction},
                        {"role": "user", "content": f"{spec.task}\n\nQuestion: {record['question']}\n\n"
                                                    f"Options:\n{options}\n\nGenerate atomic, verifiable "
                                                    f"statements as structured JSON conforming to the schema."}])
        else:
            spec = JUDGE_PROMPTS[judge_prompt]
            for fact in record["candidate"]["facts"][:5]:
                out.append([{"role": "system", "content": spec.system_instruction},
                            {"role": "user", "content": spec.format(record["question"], record["options"], fact)}])
    return out


# What earlier probes found, per served model and kind of work, so that a
# model is measured once. In cache/, which the laptop sync leaves alone.
THROUGHPUT = REPO / "cache" / "throughput.json"


def remembered(catalogue: str, kind: str) -> int | None:
    try:
        return json.loads(THROUGHPUT.read_text())[f"{catalogue}:{kind}"]["chosen"]
    except (OSError, KeyError, ValueError):
        return None


def remember(catalogue: str, kind: str, seen: list[tuple[int, float, float]], chosen: int) -> None:
    try:
        table = json.loads(THROUGHPUT.read_text())
    except (OSError, ValueError):
        table = {}
    table[f"{catalogue}:{kind}"] = {
        "chosen": chosen, "measured": time.strftime("%F %T"),
        "levels": [{"streams": lv, "running": round(r), "tokens_per_s": round(sp)} for lv, r, sp in seen]}
    THROUGHPUT.parent.mkdir(parents=True, exist_ok=True)
    THROUGHPUT.write_text(json.dumps(table, indent=1))


def probe(base: str, catalogue: str, kind: str, levels: tuple[int, ...],
          max_tokens: int, temperature: float) -> int:
    """The smallest load within 10% of the best throughput seen.

    Each level runs that many request loops for a warm-up and a measured
    window, read off the server's own log; the next level is tried only
    while the last one still bought more than 10%. A model measured before
    is not measured again."""
    known = remembered(catalogue, kind)
    if known:
        log(f"    {catalogue} {kind}: {known} in flight, as measured before")
        return known
    model = BASE_MODELS[base]
    extra = model.extra.get("extra_body", {}) if model.extra else {}
    prompts = sample_prompts(kind)
    seen: list[tuple[int, float, float]] = []
    for level in levels:
        stop = threading.Event()

        def loop(i: int) -> None:
            n = i
            while not stop.is_set():
                body = {"model": model.gateway_model, "messages": prompts[n % len(prompts)],
                        "max_tokens": max_tokens, "temperature": temperature, **extra}
                n += level
                try:
                    urllib.request.urlopen(urllib.request.Request(
                        ENDPOINT, json.dumps(body).encode(),
                        {"Content-Type": "application/json", "Authorization": "Bearer local"}),
                        timeout=900).read()
                except Exception:
                    time.sleep(1)

        threads = [threading.Thread(target=loop, args=(i,), daemon=True) for i in range(level)]
        for t in threads:
            t.start()
        time.sleep(75)
        start = time.time()
        time.sleep(45)
        running, speed = server_load(catalogue, start, time.time())
        stop.set()
        seen.append((level, running, speed))
        log(f"    probe {catalogue} {kind}: {level} streams -> {running:.0f} running, {speed:.0f} tok/s")
        for t in threads:          # let this level drain before measuring the next
            t.join(timeout=300)
        if len(seen) > 1 and seen[-1][2] < seen[-2][2] * 1.10:
            break
    best = max(s for _, _, s in seen)
    chosen = min(level for level, _, s in seen if s >= 0.9 * best)
    log(f"    probe {catalogue} {kind}: using {chosen}")
    remember(catalogue, kind, seen, chosen)
    return chosen


GEN_LEVELS = (200, 400, 800, 1200)
JUDGE_LEVELS = {1: (32, 64, 96, 128), 4: (128, 256, 512, 768)}


def drive_generation(base: str, catalogue: str) -> int:
    """Requests in flight for generation on this model."""
    if not serve(base, catalogue):
        raise RuntimeError(f"could not serve {catalogue}")
    return probe(base, catalogue, "generate", GEN_LEVELS, 4096, 0.8)


def drive_judge(judge: Judge) -> int:
    if not serve(judge.base, judge.serve):
        raise RuntimeError(f"could not serve {judge.serve}")
    return probe(judge.base, judge.serve, "judge", JUDGE_LEVELS[judge.replicas],
                 judge.max_tokens, 1.0)


# --- generation ---------------------------------------------------------------

def missing(exp_dir: str, run: str, k: int = K) -> bool:
    """Whether a run still lacks candidates on MedQA."""
    return any(stored(exp_dir, run, d) < k * len(questions(d)) for d in DATASETS)


def generate(base: str, catalogue: str, exp_dir: str, run: str, fact_prompt: str,
             in_flight: int) -> None:
    # A shard takes a question at a time and idles part of its k slots on each
    # question's last candidates: at in_flight/k shards qwen3.6-27b held 305
    # of 400. A third more shards makes up for it.
    k = k_of(base)
    generate_experiment(base, catalogue, base in FREE_FORM, exp_dir=exp_dir, run=run,
                        fact_prompt=fact_prompt, answer_prompt="cited-facts",
                        datasets=DATASETS, shards=max(1, round(1.3 * in_flight / k)), k=k)


def reanswer(base: str, source_dir: str, dest_dir: str, run: str, in_flight: int) -> None:
    """Answer a run's stored facts again with the cited-facts prompt."""
    for dataset in DATASETS:
        want = stored(source_dir, run, dataset)
        if stored(dest_dir, run, dataset) >= min(want, k_of(base) * len(questions(dataset))):
            log(f"  {run} on {DATASET_DIR[dataset]}: already answered with citations")
            continue
        log(f"  re-answering {run} on {DATASET_DIR[dataset]} at concurrency {in_flight}")
        call([PYTHON, "-m", "inference.reanswer", "--source-results-dir", source_dir,
              "--results-dir", dest_dir, "--run-name", run, "--dataset", dataset,
              "--model", BASE_MODELS[base].gateway_model, "--answer-prompt", "cited-facts",
              "--max-tokens", "4096", "--concurrency", str(in_flight)],
             REPO / "logs" / f"matrix_reanswer_{run}_{DATASET_DIR[dataset]}.log").wait()
        log(f"  {run}: {stored(dest_dir, run, dataset)}/{want} answered with citations")


# --- judging ------------------------------------------------------------------

def carry_verdicts(judge_name: str) -> None:
    """gemma's fact-only verdicts on general-facts, onto its re-answered copy.

    The facts are the same, and a fact-only judge never sees the answer, so
    only what the verdicts say about the answer changes: it is read off the
    re-answered candidates."""
    src_run = MONKEYS / GENERAL_FACTS_DIR / "med_qa" / "facts-pipeline" / GENERAL_FACTS_RUN
    dst_run = MONKEYS / GENERAL_FACTS_CITED_DIR / "med_qa" / "facts-pipeline" / GENERAL_FACTS_RUN
    carried = 0
    for src in sorted(src_run.glob(f"question_*/{judge_name}")):
        dst = dst_run / src.parent.name / judge_name
        if dst.exists():
            continue
        answers = {}
        for f in (dst_run / src.parent.name).glob("iteration_*.json"):
            cand = json.loads(f.read_text())["candidate"]
            answers[cand["candidate_index"]] = (cand.get("predicted_option"), bool(cand.get("is_correct")))
        tmp = dst.with_name(dst.name + ".incoming")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for f in src.glob("iteration_*.json"):
            record = json.loads(f.read_text())
            v = record["verification"]
            v["predicted_option"], v["is_correct"] = answers[v["candidate_index"]]
            (tmp / f.name).write_text(json.dumps(record, indent=2))
        result = json.loads((src / "result.json").read_text())
        chosen = result.get("selected_candidate_index")
        if chosen is not None:
            result["predicted_option"], result["is_correct"] = answers[chosen]
        (tmp / "result.json").write_text(json.dumps(result, indent=2))
        tmp.rename(dst)
        carried += 1
    log(f"  {judge_name}: verdicts of {carried} questions carried onto the re-answered general-facts")


def judge_pending(exp_dir: str, run: str, judge_name: str, rule: tuple[int, int]) -> bool:
    """Whether a judge still has work on a run, read off its stored verdicts.

    A question is settled once its verdicts meet the stopping rule or cover
    every candidate. Checked here so that a judge with nothing left to do is
    not put on the cards just for the verifier to find that out."""
    root = MONKEYS / exp_dir / "med_qa" / "facts-pipeline" / run
    dirs = {p.name.split("_", 1)[1]: p for p in root.glob("question_*") if p.is_dir()}
    ids = resolve(questions(MEDQA), set(dirs))
    if len(ids) < len(questions(MEDQA)) or not all(i in dirs for i in ids):
        return True
    for qid in ids:
        qdir = dirs[qid]
        jdir = qdir / judge_name
        if not (jdir / "result.json").is_file():
            return True
        verdicts = [json.loads(f.read_text())["verification"] for f in jdir.glob("iteration_*.json")]
        if len(verdicts) >= len(list(qdir.glob("iteration_*.json"))):
            continue
        n_all = n_cited = 0
        for v in verdicts:
            facts = v.get("fact_verifications") or []
            if not facts:
                continue
            n_all += bool(v.get("all_facts_correct"))
            cand = qdir / f"iteration_{v['candidate_index']}.json"
            cited = json.loads(cand.read_text())["candidate"].get("cited_facts") if cand.is_file() else None
            n_cited += (bool(v.get("all_facts_correct")) if cited is None else
                        all(facts[i]["verdict"] == 1 for i in cited if 0 <= i < len(facts)))
        if n_all < rule[0] or n_cited < rule[1]:
            return True
    return False


def judge_all(judge: Judge, bases: list[str], rule: tuple[int, int]) -> None:
    """Every set of these generators this judge has not settled yet; the
    judge is served, and its load measured, only if there is one."""
    judge_sets(judge, [s for base in bases for s in judged_sets(base)], rule,
               ", ".join(bases))


def judge_sets(judge: Judge, sets: list[tuple[str, str, str]], rule: tuple[int, int],
               what: str) -> None:
    todo = [(exp_dir, run, prompt) for exp_dir, run, prompt in sets
            if judge_pending(exp_dir, run, judge.name(prompt), rule)]
    if not todo:
        log(f"  {judge.base}: nothing left to judge on {what}")
        return
    concurrency = drive_judge(judge)
    for exp_dir, run, prompt in todo:
        step(f"{judge.name(prompt)} judges {run} in {exp_dir}", judge_experiment,
             [run], exp_dir=exp_dir, datasets=DATASETS, judge_prompt=prompt,
             judge_name=judge.name(prompt), judge=judge,
             stop_all=rule[0], stop_cited=rule[1], concurrency=concurrency)


def prepare_author_judge_copies() -> None:
    for base, _ in GENERATORS:
        copy_candidates(EXP_DIR, AUTHOR_JUDGE_DIR, base, datasets=DATASETS)


# --- the matrix ---------------------------------------------------------------

def phase_qwen27() -> None:
    if not missing(AUTHORS_DIR, QWEN27[0]) and not missing(GENERAL_FACTS_CITED_DIR, QWEN27[0]):
        log("  nothing left to generate on qwen3.6-27b")
        return
    in_flight = drive_generation(*QWEN27)
    reanswer(QWEN27[0], "results", AUTHORS_DIR, QWEN27[0], in_flight)
    generate(*QWEN27, GENERAL_FACTS_CITED_DIR, QWEN27[0], "background", in_flight)


def phase_gemma() -> None:
    if not missing(AUTHORS_DIR, GEMMA[0]):
        log("  nothing left to generate on gemma-4-26b")
        return
    in_flight = drive_generation(*GEMMA)
    generate(*GEMMA, AUTHORS_DIR, GEMMA[0], "reference", in_flight)


def phase_gpt_oss() -> None:
    if missing(AUTHORS_DIR, GPT_OSS[0]) or missing(GENERAL_FACTS_CITED_DIR, GPT_OSS[0]):
        in_flight = drive_generation(*GPT_OSS)
        generate(*GPT_OSS, AUTHORS_DIR, GPT_OSS[0], "reference", in_flight)
        generate(*GPT_OSS, GENERAL_FACTS_CITED_DIR, GPT_OSS[0], "background", in_flight)
    judge_all(GPT_OSS_JUDGE, [QWEN27[0]], DEEP)


def phase_qwen_judge() -> None:
    judge_all(QWEN_JUDGE, [GEMMA[0], GPT_OSS[0]], DEEP)


def phase_small_generators() -> None:
    """gpt-oss-20b and qwen3.5-9b on all three kinds of facts, k=100. gpt-oss-20b's
    authors' facts are its reference run answered again with citations;
    qwen3.5-9b's are generated afresh at its larger token limit."""
    for base, catalogue in (GPT20, QWEN9):
        k = k_of(base)
        wanted = [missing(AUTHORS_DIR, base, k), missing(GENERAL_FACTS_CITED_DIR, base, k),
                  missing(EXP_DIR, base, k)]
        if not any(wanted):
            log(f"  nothing left to generate on {catalogue}")
            continue
        in_flight = drive_generation(base, catalogue)
        if base in FRESH_AUTHORS_FACTS:
            generate(base, catalogue, AUTHORS_DIR, base, "reference", in_flight)
        else:
            reanswer(base, "results", AUTHORS_DIR, base, in_flight)
        generate(base, catalogue, GENERAL_FACTS_CITED_DIR, base, "background", in_flight)
        generate(base, catalogue, EXP_DIR, base, "grounded-unframed", in_flight)


def phase_small_judges() -> None:
    judge_all(QWEN_JUDGE, [GPT20[0]], DEEP)
    judge_all(GPT_OSS_JUDGE, [QWEN9[0]], DEEP)


# Step 1 drawn k times, for a majority vote to set beside step 2's: at the
# step 1 temperature (0.8) and at 1.0, each in a directory of its own.
ONE_SHOT_DIRS = {0.8: "results/experiments/one-shot-k",
                 1.0: "results/experiments/one-shot-k-temperature-1"}


def one_shot_missing(results_dir: str, run: str, attempts: int) -> bool:
    """Whether a difficult MedQA question still has fewer attempts than wanted."""
    root = MONKEYS / results_dir / "med_qa" / "single-step" / run
    files = {p.stem.split("_", 1)[1]: p for p in root.glob("question_*.json")}
    ids = resolve(questions(MEDQA), set(files))
    for qid in ids:
        if qid not in files:
            return True
        item = json.loads(files[qid].read_text())
        if sum(1 for a in item.get("attempts", []) if not a.get("error")) < attempts:
            return True
    return len(ids) < len(questions(MEDQA))


def phase_one_shot_k() -> None:
    """Each generator answers step 1's one-shot prompt k times on the
    difficult questions, k as in its step 2 runs, at both temperatures."""
    for base, catalogue in (GPT_OSS, GPT20, QWEN9, QWEN27, GEMMA):   # the served one first
        k = k_of(base)
        todo = [t for t, d in ONE_SHOT_DIRS.items() if one_shot_missing(d, base, k)]
        if not todo:
            log(f"  {base}: one-shot already at {k} attempts")
            continue
        in_flight = drive_generation(base, catalogue)
        for temperature in todo:
            results_dir = ONE_SHOT_DIRS[temperature]
            log(f"  {base}: one-shot x{k} at temperature {temperature}, {in_flight} in flight")
            call([PYTHON, "-m", "gdanschin_runtime.one_shot_many",
                  "--model", BASE_MODELS[base].gateway_model, "--run-name", base,
                  "--results-dir", results_dir, "--n-attempts", str(k),
                  "--temperature", str(temperature),
                  "--max-tokens", str(MAX_TOKENS_BY_BASE.get(base, 4096)),
                  "--concurrency", str(in_flight)],
                 REPO / "logs" / f"matrix_one_shot_{base}_t{temperature}.log").wait()
            log(f"  {base} at {temperature}: "
                f"{'complete' if not one_shot_missing(results_dir, base, k) else 'INCOMPLETE'}")


def phase_general_facts_authors_prompt() -> None:
    """general-facts judged with the authors' prompt, on two generators only:
    does the prompt that lifts authors-facts do anything for facts that do
    not argue about the options?"""
    judge_sets(QWEN_JUDGE, [(GENERAL_FACTS_CITED_DIR, general_run(GEMMA[0]), "reference"),
                            (GENERAL_FACTS_CITED_DIR, GPT_OSS[0], "reference")],
               DEEP, "general-facts of gemma-4-26b and gpt-oss-120b")


def phase_glm() -> None:
    # The same rule as the other judges, so that the vote among cited-valid
    # candidates compares across them; gemma's sets, judged by GLM to the
    # first valid candidate before, are continued rather than judged again.
    judge_all(GLM_JUDGE, [GEMMA[0], GPT_OSS[0], QWEN27[0]], DEEP)


def phase_glm_small() -> None:
    judge_all(GLM_JUDGE, [GPT20[0], QWEN9[0]], DEEP)


def phase_glm_general_facts_authors_prompt() -> None:
    """The phase 5c question asked of GLM too, last of all: on gemma, and on
    qwen3.6-27b, whose majority vote on general-facts (83%) no weaker judge
    could reach. gpt-oss-120b is left out: qwen's judge found nothing there."""
    judge_sets(GLM_JUDGE, [(GENERAL_FACTS_CITED_DIR, general_run(GEMMA[0]), "reference"),
                           (GENERAL_FACTS_CITED_DIR, QWEN27[0], "reference")],
               DEEP, "general-facts of gemma-4-26b and qwen3.6-27b")


def take_over(after: tuple[str, str, str] | None) -> None:
    """Let a matrix already running finish the set it is on, then stop it.

    `after` names that set as (results dir, run, judge name), given on the
    command line as --after DIR RUN JUDGE. Once every question of it is
    judged, the old process and its judge are stopped and this one carries
    on. Without it, a matrix already running is a mistake: this one exits."""
    others = [int(p) for p in subprocess.run(["pgrep", "-f", "gdanschin_runtime.facts_matrix"],
                                             capture_output=True, text=True).stdout.split()
              if int(p) != os.getpid()]
    def is_python(pid: int) -> bool:
        # The shell that launched a matrix carries its command line too; only
        # a process that is itself python counts.
        try:
            argv0 = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
        except OSError:
            return False
        return Path(argv0).name.startswith("python")
    others = [p for p in others if is_python(p)]
    if not others:
        return
    if after is None:
        log(f"  matrix {others} is already running and no --after was given; exiting")
        raise SystemExit(1)
    exp_dir, run, judge_name = after
    log(f"  waiting for matrix {others} to finish {judge_name} on {run} in {exp_dir}")
    while judged(exp_dir, run, MEDQA, judge_name) < len(questions(MEDQA)):
        time.sleep(60)
    for pid in others:
        subprocess.run(["kill", str(pid)])
    time.sleep(3)
    subprocess.run(["pkill", "-f", "verifier.cli"])
    time.sleep(5)
    log(f"  matrix {others} stopped; carrying on")


def raise_open_file_limit() -> None:
    """A probe or a judge at several hundred requests holds a socket for each;
    the box's soft limit of 1024 ended the first run mid-probe. Children
    inherit the raised limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    log(f"  open files allowed: {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}")


def main() -> int:
    log("facts matrix starting (MedQA only)")
    raise_open_file_limit()
    args = sys.argv[1:]
    take_over(tuple(args[args.index("--after") + 1:args.index("--after") + 4]) if "--after" in args else None)
    step("carry gemma's fact-only verdicts onto the re-answered general-facts",
         lambda: [carry_verdicts(j) for j in ("glm-5.3-flash-local-fact-only",
                                              "qwen3.6-35b-a3b-local-fact-only")])
    step("phase 1: qwen3.6-27b answers authors-facts with citations, generates general-facts",
         phase_qwen27)
    step("phase 2: gemma generates authors-facts", phase_gemma)
    step("copy grounded-unframed candidates for the authors' judge", prepare_author_judge_copies)
    step("phase 3: gpt-oss-120b generates authors- and general-facts, judges qwen3.6-27b",
         phase_gpt_oss)
    step("phase 4: qwen3.6-35b-a3b judges gemma and gpt-oss-120b", phase_qwen_judge)
    step("phase 5a: gpt-oss-20b and qwen3.5-9b generate all three kinds of facts (k=100)",
         phase_small_generators)
    step("copy grounded-unframed candidates for the authors' judge", prepare_author_judge_copies)
    step("phase 5b: qwen3.6-35b-a3b judges gpt-oss-20b, gpt-oss-120b judges qwen3.5-9b",
         phase_small_judges)
    step("phase 5b': step 1 one-shot k times on the difficult questions, at 0.8 and 1.0",
         phase_one_shot_k)
    step("phase 5c: qwen3.6-35b-a3b judges general-facts with the authors' prompt",
         phase_general_facts_authors_prompt)
    step("phase 6: GLM judges gemma, gpt-oss-120b and qwen3.6-27b", phase_glm)
    step("phase 7: GLM judges gpt-oss-20b and qwen3.5-9b", phase_glm_small)
    step("phase 8: GLM judges general-facts with the authors' prompt",
         phase_glm_general_facts_authors_prompt)
    log("facts matrix finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
