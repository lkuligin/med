"""The queue after the Table 2 chain: the last judgings for Table 2, the
facts GLM left unchecked on gemma, and fifty more candidates for the Table 1
models that drew fifty, so that Figure 1 draws pass@k up to k=100 for all six.
Order as set on 2026-10-10.

    cd ~/Projects/med && setsid nohup .venv/bin/python -m gdanschin_runtime.pass_at_100 \\
        >> logs/pass_at_100.log 2>&1 < /dev/null &

1. GLM finishes qwen3.5-4b's general-facts, the judging the Table 2 chain
   started: the step waits for a judge already at work, then settles what it
   left;
2. GLM judges qwen3.5-9b's general-facts by the matrix's rule;
3. GLM judges the facts its run on gemma-4-26b left unchecked after a wrong
   one, so that its agreement with Gemini counts every fact (complete_verdicts);
4. the facts matrix is stopped, if it runs, and the fifty more are drawn;
5. the facts matrix starts again.

Both general-facts judgings stop checking a candidate at its first wrong fact,
cited facts first (decided 2026-10-10, to save time): neither the first valid
candidate nor validity on the cited facts can change after it, and no
agreement between judges is measured on general facts.

The first fifty are the authors'-workflow runs in results/. They stay at fifty:
the tables, the judges and the files handed to Leo go on reading them as they
are. The new fifty go to an experiment of their own and are drawn the way the
first were - the authors' fact and answer prompts, temperature 0.8, the same
weights, here on our cards - at each run's token limit: 1024 for gemma on
MedQA, which the gateway ran at 1024, and 4096 otherwise. paper.tables pools
the two for pass@k only.
"""

from __future__ import annotations

from gdanschin_runtime.experiment_chain import (
    BASE_MODELS, DATASET_DIR, GLM_JUDGE, MEDBULLETS, MEDQA, PYTHON, REPO, call, generate_experiment,
    log, serve, step)
from gdanschin_runtime.facts_matrix import (
    DEEP, FIRST, FREE_FORM, GEMMA, GPT_OSS, QWEN27, QWEN9, drive_generation, raise_open_file_limit,
    remembered)
from gdanschin_runtime.table1 import QWEN4, resume_matrix, stop_matrix
from gdanschin_runtime.table2 import judge_general_fact_only

PASS_AT_100_DIR = "results/experiments/pass-at-100"
K_MORE = 50
# (base, what to serve), dataset, the token limit its first fifty were drawn at
TOP_UPS = ((GEMMA, MEDQA, 1024), (GEMMA, MEDBULLETS, 4096),
           (GPT_OSS, MEDQA, 4096), (GPT_OSS, MEDBULLETS, 4096),
           (QWEN27, MEDQA, 4096), (QWEN27, MEDBULLETS, 4096))


def top_up(model: tuple[str, str], dataset: str, max_tokens: int) -> None:
    base, catalogue = model
    in_flight = drive_generation(base, catalogue)
    # As in the matrix: a third more shards than in_flight/k keeps the slots filled.
    generate_experiment(base, catalogue, base in FREE_FORM, exp_dir=PASS_AT_100_DIR, run=base,
                        fact_prompt="reference", answer_prompt="reference", datasets=(dataset,),
                        shards=max(1, round(1.3 * in_flight / K_MORE)), k=K_MORE,
                        max_tokens=max_tokens)


def complete_gemma(dataset: str) -> None:
    """GLM, as it judged gemma-4-26b with the authors' prompt, judges the facts
    it left unchecked there."""
    run = "gemma-4-26b"
    concurrency = remembered(GLM_JUDGE.serve, "judge") or GLM_JUDGE.concurrency
    if not serve(GLM_JUDGE.base, GLM_JUDGE.serve):
        raise RuntimeError(f"could not serve {GLM_JUDGE.serve}")
    call([PYTHON, "-m", "gdanschin_runtime.complete_verdicts", "--results-dir", "results",
          "--run-name", run, "--judge-name", GLM_JUDGE.name("reference"), "--judge-prompt", "reference",
          "--model", BASE_MODELS[GLM_JUDGE.base].gateway_model, "--temperature", "1.0",
          "--max-tokens", str(GLM_JUDGE.max_tokens), "--request-timeout", "900",
          "--dataset", dataset, "--concurrency", str(concurrency)],
         REPO / "logs" / f"complete_verdicts_{run}_{DATASET_DIR[dataset]}.log").wait()


def main() -> int:
    log("pass@100 starting")
    raise_open_file_limit()
    # Before anything that stops the matrix: that kills every running judge.
    step("GLM fact-only finishes qwen3.5-4b's general-facts on MedQA (first valid)",
         judge_general_fact_only, QWEN4[0], FIRST, early_stop_facts=True)
    step("GLM fact-only judges qwen3.5-9b's general-facts on MedQA (the matrix's rule)",
         judge_general_fact_only, QWEN9[0], DEEP, early_stop_facts=True)
    for dataset in (MEDQA, MEDBULLETS):
        step(f"GLM judges the facts it left unchecked on gemma-4-26b, {dataset}", complete_gemma, dataset)
    step("stop the facts matrix; it resumes at the end", stop_matrix)
    for model, dataset, max_tokens in TOP_UPS:
        step(f"{model[0]} draws {K_MORE} more candidates on {dataset} at {max_tokens} tokens",
             top_up, model, dataset, max_tokens)
    step("resume the facts matrix", resume_matrix)
    log("pass@100 finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
