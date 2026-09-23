"""Drive the whole stage 2 and 3 sweep unattended, switching models itself.

    setsid nohup python3 -m gdanschin_runtime.run_sweep \
        > logs/sweep.log 2>&1 < /dev/null &

Nine models, hours each: nobody is going to sit through it, and a laptop
closing must not stop it. So this owns the whole sequence - serving a model,
generating its candidates over both datasets, judging the ones that are
judged, and moving on - and is started once, detached.

Resumable by construction rather than by bookkeeping. The generator tops each
question up to k rather than starting it over, the judge skips questions it
has already certified, and a model whose candidates are all present costs a
few seconds to confirm and skip. So a restart after a crash, a reboot or a
Ctrl-C loses at most the question in flight.

The table lives in oss_sweep, including which models are judged. Judging is
the half that costs money, and there is exactly one place that decides it.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from gdanschin_runtime.oss_sweep import (DATASETS, SWEEP, Target,
                                         candidates_command, verdicts_command)

REPO = Path(__file__).resolve().parents[1]
MONKEYS = REPO / "llm_monkeys"
PYTHON = str(REPO / ".venv" / "bin" / "python")
ENV = REPO / "gdanschin_runtime" / "env.sh"
SERVE = [sys.executable, "-m", "gpu_serving.cli"]
JUDGE_CYCLE = 300


def through_env(command: list[str]) -> list[str]:
    """Run a command with the runtime's environment applied.

    env.sh is what points model construction at our own endpoint; without it
    llm_monkeys builds a plain LiteLlm with the raw served name and litellm
    rejects it for having no provider. The shell scripts this driver replaced
    sourced it and it was easy to miss that a subprocess does not inherit
    what was never set. Sourced rather than copied into os.environ, so env.sh
    stays the one place that decides.
    """
    return ["bash", "-c", f'source "{ENV}" && exec "$@"', "_", *command]


def log(message: str) -> None:
    print(f"{time.strftime('%F %T')}  {message}", flush=True)


def serving_now() -> str | None:
    """Which model the box is serving, or None."""
    result = subprocess.run(SERVE + ["status"], cwd=REPO,
                            capture_output=True, text=True)
    first = result.stdout.splitlines()[:1]
    if not first or "->" not in first[0]:
        return None
    return first[0].split("->")[0].strip()


def ensure_served(target: Target) -> bool:
    """Serve this model, leaving it alone if it is already up.

    Checked rather than assumed: restarting a server that is already the
    right one would throw away a warm kernel cache and fifteen minutes on an
    FP8 model, and would kill a generation that is still running against it.
    """
    if serving_now() == target.serve:
        log(f"  already serving {target.serve}")
        return True
    stopped = subprocess.run(SERVE + ["down"], cwd=REPO,
                             capture_output=True, text=True)
    if stopped.returncode:
        # Not fatal, but worth saying: a stop that timed out leaves the old
        # server holding ports, and the next one dies on EADDRINUSE rather
        # than on anything that names the real cause.
        log(f"  down did not finish cleanly: {stopped.stderr.strip()[:200]}")
    log(f"  starting {target.serve}")
    result = subprocess.run(SERVE + ["up", target.serve], cwd=REPO,
                            capture_output=True, text=True)
    if result.returncode:
        log(f"  first attempt did not come up; stopping and retrying once")
        subprocess.run(SERVE + ["down"], cwd=REPO, capture_output=True, text=True)
        result = subprocess.run(SERVE + ["up", target.serve], cwd=REPO,
                                capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.strip():
            log(f"    {line.strip()}")
    if result.returncode:
        log(f"  FAILED to serve {target.serve}; skipping it")
        log(f"    {result.stderr.strip()[:400]}")
        return False
    return True


def judge_until(target: Target, generating: threading.Event) -> None:
    """Judge on a cycle while generation runs, then once more at the end.

    The judge works the external gateway and the generator our own cards, so
    the two cost each other nothing and there is no reason to wait for one
    before starting the other.
    """
    while not generating.is_set():
        for dataset in DATASETS:
            subprocess.run(through_env(verdicts_command(target, dataset, PYTHON)),
                           cwd=MONKEYS, capture_output=True, text=True)
        generating.wait(JUDGE_CYCLE)
    for dataset in DATASETS:
        subprocess.run(through_env(verdicts_command(target, dataset, PYTHON)),
                       cwd=MONKEYS, capture_output=True, text=True)
    log(f"  judging finished for {target.base}")


def generate(target: Target) -> None:
    for dataset in DATASETS:
        log(f"  stage 2 on {dataset}")
        started = time.time()
        subprocess.run(through_env(candidates_command(target, dataset, PYTHON)),
                       cwd=MONKEYS)
        log(f"  stage 2 on {dataset} done in {(time.time() - started) / 60:.0f} min")


def run(target: Target, index: int, total: int) -> threading.Thread | None:
    """Generate this model's candidates. Returns its judge thread, if any.

    The judge is deliberately not waited for. It works the external gateway
    and the generator our own cards, so holding the next model's generation
    until judging finishes leaves the cards idle for hours: on qwen3.5-4b at
    k=100, generation took five hours and judging seventeen. The thread is
    handed back so the sweep can join it at the end instead.
    """
    log(f"[{index}/{total}] {target.base}  k={target.k}  "
        f"{'judged' if target.judged else 'not judged'}")
    if not ensure_served(target):
        return None

    done = threading.Event()
    judge = None
    if target.judged:
        judge = threading.Thread(target=judge_until, args=(target, done),
                                 daemon=False)
        judge.start()

    generate(target)
    done.set()
    log(f"[{index}/{total}] {target.base} generated"
        + ("; judging continues alongside the next model" if judge else ""))
    return judge


def main() -> int:
    log(f"sweep of {len(SWEEP)} models starting")
    judges: list[threading.Thread] = []
    for index, target in enumerate(SWEEP, 1):
        try:
            judge = run(target, index, len(SWEEP))
            if judge:
                judges.append(judge)
        except Exception as error:      # one model failing must not end the sweep
            log(f"[{index}/{len(SWEEP)}] {target.base} raised "
                f"{type(error).__name__}: {error}")
    still = [j for j in judges if j.is_alive()]
    if still:
        log(f"generation done; waiting on {len(still)} judge(s) still running")
    for judge in judges:
        judge.join()
    log("sweep finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
