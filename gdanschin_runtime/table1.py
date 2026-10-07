"""Fill the paper's Table 1 ahead of the facts matrix, which is not urgent.

    cd ~/Projects/med && setsid nohup python3 -m gdanschin_runtime.table1 \\
        >> logs/table1.log 2>&1 < /dev/null &

1. The facts matrix is stopped. Questions its judge had not finished are
   judged again when it resumes.
2. general-facts on MedBullets for every Table 1 model, candidates only (no
   judge), for a majority vote: the background fact prompt, answered with
   cited facts, k as in each model's MedQA runs (100 for the small ones).
3. GLM, with the authors' prompt and the authors' rule (stop at the first
   candidate whose facts all pass), on the authors'-workflow candidates in
   results/ it has not judged yet: qwen3.5-4b on MedQA; gpt-oss-120b,
   gpt-oss-20b, qwen3.6-27b, qwen3.5-9b and qwen3.5-4b on MedBullets. The
   cheap sets go first, so most of the table fills early.
4. The facts matrix starts again and carries on where it was.

Resumable: generation tops runs up, the verifier skips settled questions.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from gdanschin_runtime.experiment_chain import (
    DATASET_DIR, GENERAL_FACTS_CITED_DIR, GLM_JUDGE, K, MEDBULLETS, MEDQA, REPO,
    generate_experiment, judge_experiment, log, questions, step, stored)
from gdanschin_runtime.facts_matrix import (
    FIRST, FREE_FORM, GEMMA, GPT20, GPT_OSS, K_BY_BASE, QWEN27, QWEN9, drive_generation,
    general_run, raise_open_file_limit, remembered)

QWEN4 = ("qwen3.5-4b-nr-local", "qwen3.5-4b")
TABLE_GENERATORS = (GEMMA, GPT_OSS, QWEN27, GPT20, QWEN9, QWEN4)
# qwen3.5-4b's reference runs drew 100 candidates, as the other small models'.
K_TABLE = {**K_BY_BASE, QWEN4[0]: 100}
# (dataset, run in results/) of the authors' workflow that GLM has not judged.
GLM_GAPS = ((MEDBULLETS, "qwen3.6-27b-nr-local"), (MEDBULLETS, "gpt-oss-120b"),
            (MEDBULLETS, "qwen3.5-9b-nr-local"), (MEDBULLETS, "gpt-oss-20b-local"),
            (MEDBULLETS, "qwen3.5-4b-nr-local"), (MEDQA, "qwen3.5-4b-nr-local"))


def stop_matrix() -> None:
    pids = [int(p) for p in subprocess.run(["pgrep", "-f", "gdanschin_runtime.facts_matrix"],
                                           capture_output=True, text=True).stdout.split()]
    pids = [p for p in pids if p != os.getpid()]
    if not pids:
        log("  no facts matrix running")
        return
    for pid in pids:
        subprocess.run(["kill", str(pid)])
    time.sleep(3)
    subprocess.run(["pkill", "-f", "verifier.cli"])
    time.sleep(5)
    log(f"  facts matrix {pids} and its judge stopped")


def general_facts_medbullets() -> None:
    for base, catalogue in TABLE_GENERATORS:
        k = K_TABLE.get(base, K)
        run = general_run(base)
        if stored(GENERAL_FACTS_CITED_DIR, run, MEDBULLETS) >= k * len(questions(MEDBULLETS)):
            log(f"  {run}: general-facts on medbullets already complete")
            continue
        in_flight = drive_generation(base, catalogue)
        # As in the matrix: a shard idles part of its k slots on each question's
        # last candidates, so a third more shards than in_flight/k.
        generate_experiment(base, catalogue, base in FREE_FORM, exp_dir=GENERAL_FACTS_CITED_DIR,
                            run=run, fact_prompt="background", answer_prompt="cited-facts",
                            datasets=(MEDBULLETS,), shards=max(1, round(1.3 * in_flight / k)), k=k)


def glm_gaps() -> None:
    concurrency = remembered(GLM_JUDGE.serve, "judge") or GLM_JUDGE.concurrency
    for dataset, run in GLM_GAPS:
        step(f"GLM with the authors' prompt judges {run} on {DATASET_DIR[dataset]}", judge_experiment,
             [run], exp_dir="results", datasets=(dataset,), judge_prompt="reference",
             judge_name=GLM_JUDGE.name("reference"), judge=GLM_JUDGE,
             stop_all=FIRST[0], stop_cited=FIRST[1], concurrency=concurrency)


def resume_matrix() -> None:
    with open(REPO / "logs" / "facts_matrix.log", "a") as out:
        subprocess.Popen([sys.executable, "-m", "gdanschin_runtime.facts_matrix"], cwd=REPO,
                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    log("  facts matrix started again; its log is logs/facts_matrix.log")


def main() -> int:
    log("table 1 starting")
    raise_open_file_limit()
    step("stop the facts matrix; it resumes after the table", stop_matrix)
    step("general-facts on medbullets, candidates only", general_facts_medbullets)
    step("GLM fills the table's gaps", glm_gaps)
    step("resume the facts matrix", resume_matrix)
    log("table 1 finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
