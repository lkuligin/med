"""Where the stage 2 and 3 sweep has got to, on one screen.

    python3 -m gdanschin_runtime.sweep_progress

Reads the results on disk rather than any log, so it is right even after a
step was restarted, and costs nothing to run while the sweep is going.

Judged models are marked, and the judged column is left blank rather than
zeroed for the rest: a zero would read as "nothing judged yet" where the
truth is "this one is never judged", and that distinction is the expensive
one to get wrong.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

from gdanschin_runtime.oss_sweep import DATASETS, JUDGE, SWEEP
from results_store import dataset_dir_for

RESULTS = Path(__file__).resolve().parents[1] / "llm_monkeys" / "results"
DIFFICULT_COUNT = {"medbullets": 165, "med_qa": 483}


def counts(run: str, dataset_dir: str) -> tuple[int, int, float]:
    """(candidates stored, questions judged, seconds since the last write)."""
    root = RESULTS / dataset_dir / "facts-pipeline" / run
    if not root.is_dir():
        return 0, 0, 0.0
    candidates = judged = 0
    newest = 0.0
    for question in root.iterdir():
        if not question.is_dir():
            continue
        iterations = glob.glob(str(question / "iteration_*.json"))
        candidates += len(iterations)
        if (question / JUDGE).is_dir():
            judged += 1
        for path in iterations:
            newest = max(newest, os.path.getmtime(path))
    return candidates, judged, (time.time() - newest) if newest else 0.0


def main() -> int:
    print(f"  {'model':26} {'dataset':11} {'candidates':>18} {'judged':>12} "
          f"{'last write':>11}")
    shown = 0
    for target in SWEEP:
        for dataset in DATASETS:
            directory = dataset_dir_for(dataset)
            want = DIFFICULT_COUNT[directory] * target.k
            have, judged, idle = counts(target.base, directory)
            started = (RESULTS / directory / "facts-pipeline" / target.base).is_dir()
            if not started:
                # Shown as a blank row rather than hidden: "not started" and
                # "started and producing nothing" look identical otherwise,
                # and only one of them is a problem.
                continue
            share = have / want * 100 if want else 0
            judged_cell = (f"{judged}/{DIFFICULT_COUNT[directory]}"
                           if target.judged else "not judged")
            idle_cell = f"{idle / 60:.0f}m ago" if idle else "-"
            print(f"  {target.base:26} {directory:11} "
                  f"{have:8}/{want:<8} {share:3.0f}% {judged_cell:>12} "
                  f"{idle_cell:>11}")
            shown += 1
    if not shown:
        print("  nothing started yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
