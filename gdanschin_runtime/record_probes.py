"""Copy load-probe results from the facts-matrix log into cache/throughput.json.

    setsid nohup python3 -m gdanschin_runtime.record_probes >> logs/record_probes.log 2>&1 < /dev/null &

For a matrix started before facts_matrix remembered its own probes. Rereads
the log every few minutes and stops once the matrix has exited.
"""

from __future__ import annotations

import re
import subprocess
import time

from gdanschin_runtime.facts_matrix import REPO, remember, remembered

LOG = REPO / "logs" / "facts_matrix.log"
LEVEL = re.compile(r"probe (\S+) (generate|judge): (\d+) streams -> (\d+) running, (\d+) tok/s")
CHOSEN = re.compile(r"probe (\S+) (generate|judge): using (\d+)")


def record_once() -> None:
    seen: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for line in LOG.read_text(errors="replace").splitlines():
        if m := LEVEL.search(line):
            seen.setdefault((m[1], m[2]), []).append((int(m[3]), float(m[4]), float(m[5])))
        elif (m := CHOSEN.search(line)) and remembered(m[1], m[2]) is None:
            remember(m[1], m[2], seen.get((m[1], m[2]), []), int(m[3]))
            print(f"{time.strftime('%F %T')}  recorded {m[1]}:{m[2]} = {m[3]}", flush=True)
        if m := CHOSEN.search(line):
            seen.pop((m[1], m[2]), None)


def matrix_running() -> bool:
    return subprocess.run(["pgrep", "-f", "gdanschin_runtime.facts_matrix"],
                          capture_output=True).returncode == 0


def main() -> int:
    while True:
        record_once()
        if not matrix_running():
            return 0
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
