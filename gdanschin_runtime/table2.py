"""Fill the gaps of the paper's Table 2 (ablations on MedQA) once Table 1's
are filled.

    cd ~/Projects/med && setsid nohup python3 -m gdanschin_runtime.table2 \\
        >> logs/table2.log 2>&1 < /dev/null &

Waits for the Table 1 chain to finish. That chain starts the facts matrix
again on its way out; this one stops it, then:
1. GLM judges qwen3.5-9b's general-facts with the fact-only prompt, by the
   matrix's own rule, so the matrix finds that set done when it resumes;
2. qwen3.5-4b generates general-facts on MedQA (k=100), as the other models
   have them;
3. GLM judges those with the fact-only prompt, to the first candidate whose
   facts all pass, the rule Table 2 reads;
4. the facts matrix starts again.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from gdanschin_runtime.experiment_chain import (
    GENERAL_FACTS_CITED_DIR, GLM_JUDGE, MEDQA, generate_experiment, judge_experiment, log, step)
from gdanschin_runtime.facts_matrix import (
    DEEP, FIRST, QWEN9, drive_generation, raise_open_file_limit, remembered)
from gdanschin_runtime.table1 import K_TABLE, QWEN4, resume_matrix, stop_matrix


def table1_running() -> bool:
    pids = subprocess.run(["pgrep", "-f", "gdanschin_runtime.table1"],
                          capture_output=True, text=True).stdout.split()
    for pid in pids:
        # Its launching shell carries the same command line; only python counts.
        try:
            argv0 = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
        except OSError:
            continue
        if Path(argv0).name.startswith("python"):
            return True
    return False


def wait_for_table1() -> None:
    while table1_running():
        time.sleep(60)
    log("  the Table 1 chain has finished")


def judge_general_fact_only(base: str, rule: tuple[int, int]) -> None:
    concurrency = remembered(GLM_JUDGE.serve, "judge") or GLM_JUDGE.concurrency
    judge_experiment([base], exp_dir=GENERAL_FACTS_CITED_DIR, datasets=(MEDQA,),
                     judge_prompt="fact-only", judge_name=GLM_JUDGE.name("fact-only"),
                     judge=GLM_JUDGE, stop_all=rule[0], stop_cited=rule[1], concurrency=concurrency)


def generate_qwen4_general() -> None:
    k = K_TABLE[QWEN4[0]]
    in_flight = drive_generation(*QWEN4)
    generate_experiment(*QWEN4, False, exp_dir=GENERAL_FACTS_CITED_DIR, run=QWEN4[0],
                        fact_prompt="background", answer_prompt="cited-facts",
                        datasets=(MEDQA,), shards=max(1, round(1.3 * in_flight / k)), k=k)


def main() -> int:
    log("table 2 waiting for the Table 1 chain")
    raise_open_file_limit()
    wait_for_table1()
    step("stop the facts matrix; it resumes after the table", stop_matrix)
    step("GLM fact-only judges qwen3.5-9b's general-facts on MedQA (the matrix's rule)",
         judge_general_fact_only, QWEN9[0], DEEP)
    step("qwen3.5-4b generates general-facts on MedQA", generate_qwen4_general)
    step("GLM fact-only judges qwen3.5-4b's general-facts on MedQA (first valid)",
         judge_general_fact_only, QWEN4[0], FIRST)
    step("resume the facts matrix", resume_matrix)
    log("table 2 finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
