"""Work through the OSS-judge comparison unattended.

    setsid nohup python3 -m gdanschin_runtime.run_judges \
        > logs/judges.log 2>&1 < /dev/null &

Twenty-eight runs across four served models, hours each: the table lives in
judge_plan, this walks it. Started once, detached, so a laptop closing does
not stop it.

Resumable by construction rather than by bookkeeping, like the sweep: step 1
tops a question up to its attempt count, the verifier skips questions it has
already certified, and a run that is already complete costs seconds to
confirm. A restart loses at most the question in flight.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from gdanschin_runtime.judge_plan import (ATTEMPTS, JUDGE_TEMPERATURE,
                                          STEP1_TEMPERATURE, Step, plan)
from gdanschin_runtime.models import BASE_MODELS

REPO = Path(__file__).resolve().parents[1]
MONKEYS = REPO / "llm_monkeys"
PYTHON = str(REPO / ".venv" / "bin" / "python")
ENV = REPO / "gdanschin_runtime" / "env.sh"
SERVE = [sys.executable, "-m", "gpu_serving.cli"]
CONCURRENCY = 16


def log(message: str) -> None:
    print(f"{time.strftime('%F %T')}  {message}", flush=True)


def through_env(command: list[str]) -> list[str]:
    """Run a command with the runtime's environment applied.

    env.sh is what points model construction at our own endpoint; a subprocess
    inherits nothing that was never set, and without it llm_monkeys builds a
    plain LiteLlm with the raw served name and litellm rejects it for having
    no provider.
    """
    return ["bash", "-c", f'source "{ENV}" && exec "$@"', "_", *command]


def serving_now() -> str | None:
    """Which model the box is serving, or None."""
    result = subprocess.run(SERVE + ["status"], cwd=REPO,
                            capture_output=True, text=True)
    if result.returncode:
        return None
    first = result.stdout.strip().splitlines()[:1]
    return first[0].split(" -> ")[0].strip() if first else None


def ensure_served(step: Step) -> bool:
    """Serve what this step needs, leaving it alone if it is already up."""
    if serving_now() == step.serve:
        return True
    subprocess.run(SERVE + ["down"], cwd=REPO, capture_output=True, text=True)
    log(f"  starting {step.serve}")
    result = subprocess.run(SERVE + ["up", step.serve], cwd=REPO,
                            capture_output=True, text=True)
    if result.returncode:
        log("  did not come up; stopping and retrying once")
        subprocess.run(SERVE + ["down"], cwd=REPO, capture_output=True, text=True)
        result = subprocess.run(SERVE + ["up", step.serve], cwd=REPO,
                                capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.strip():
            log(f"    {line.strip()}")
    if result.returncode:
        log(f"  FAILED to serve {step.serve}: {result.stderr.strip()[:300]}")
    return not result.returncode


def command(step: Step) -> list[str]:
    """The command this step runs."""
    if step.kind == "step1":
        model = BASE_MODELS[step.base]
        return [PYTHON, "-m", "cli",
                "--model", model.gateway_model,
                "--run-name", step.base,
                "--dataset", step.dataset,
                "--n-attempts", str(ATTEMPTS),
                "--concurrency", str(CONCURRENCY),
                "--max-tokens", str(step.max_tokens),
                "--temperature", str(STEP1_TEMPERATURE)]
    judge = BASE_MODELS[step.judge]
    return [PYTHON, "-m", "verifier.cli",
            "--run-name", step.base,
            "--judge-name", step.judge,
            "--model", judge.gateway_model,
            "--dataset", step.dataset,
            "--temperature", str(JUDGE_TEMPERATURE),
            "--max-tokens", str(step.max_tokens),
            "--concurrency", str(CONCURRENCY),
            # Two things at once, and the second is the surprise. A candidate
            # passes only if every fact does, so checking the rest after the
            # first failure cannot change the verdict: measured over 262
            # candidates, 49% of the work was spent that way.
            #
            # And it is what fills the cards. Without it a candidate submits
            # all of its facts through one gather, taking every slot the
            # concurrency allows, while the other fifteen open questions wait
            # - so the server ran at sixteen requests, then trailed to one
            # waiting for that batch's slowest fact, over and over, averaging
            # 4.7. Facts sequential within a candidate, candidates sequential
            # within a question, sixteen questions open: one in-flight fact
            # per question, sixteen at all times.
            "--early-stop-facts",
            # Judge only the difficult questions. Most generators produced
            # candidates for exactly those, but the gateway-era runs carry the
            # whole split, and judging their simple questions spends cards on
            # answers every candidate already gets right.
            "--difficult-questions", str(MONKEYS / step.questions),
            # Both judges in this table run on our own cards, where a request
            # that turns out unnecessary costs time already paid for. Never
            # pass this for a judge billed per call: waiting is cheaper.
            "--speculate-tail"]


def run(step: Step, index: int, total: int) -> None:
    log(f"[{index}/{total}] {step.label}")
    if not ensure_served(step):
        log(f"[{index}/{total}] skipped: nothing serving")
        return
    started = time.time()
    subprocess.run(through_env(command(step)), cwd=MONKEYS)
    log(f"[{index}/{total}] done in {(time.time() - started) / 60:.0f} min")


def main() -> int:
    steps = plan()
    log(f"judge comparison: {len(steps)} runs")
    for index, step in enumerate(steps, 1):
        try:
            run(step, index, len(steps))
        except Exception as error:     # one run failing must not end the rest
            log(f"[{index}/{len(steps)}] raised "
                f"{type(error).__name__}: {error}")
    log("judge comparison finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
