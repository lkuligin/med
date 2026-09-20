"""Move a finished run from single result files into the stored layout.

    python3 gdanschin_runtime/import_run.py \
        --run gemma-4-26b --judge gemini-3.8-flash \
        --step1 results/step1_gemma.json \
        --step2 results/step2.json \
        --step3 results/step3.json

For runs produced before results were stored per record. Reads only; the
originals are left alone, so a conversion that goes wrong costs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gdanschin_runtime import _bootstrap  # noqa: F401

from results_store import (
    DEFAULT_RESULTS_DIR,
    CandidateResults,
    OneShotResults,
    VerificationResults,
)


def _payload(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    results = data["results"] if isinstance(data, dict) and "results" in data else data
    summary = data.get("summary") if isinstance(data, dict) else None
    return {"summary": summary, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The same place every other tool here reads from. The store's own default
    # is relative to the working directory, which quietly puts a run wherever
    # the command happened to be typed.
    parser.add_argument("--results-dir",
                        default=str(_bootstrap.LLM_MONKEYS_ROOT / DEFAULT_RESULTS_DIR))
    parser.add_argument("--run", required=True, help="name to store the run under")
    parser.add_argument("--judge", help="name to store the verdicts under")
    parser.add_argument("--step1")
    parser.add_argument("--step2")
    parser.add_argument("--step3")
    args = parser.parse_args(argv)

    if args.step1:
        payload = _payload(args.step1)
        OneShotResults(args.results_dir, args.run).save(payload)
        print(f"  single-step: {len(payload['results'])} questions")

    if args.step2:
        payload = _payload(args.step2)
        CandidateResults(args.results_dir, args.run).save(payload)
        candidates = sum(len(q.get("candidates") or []) for q in payload["results"])
        print(f"  candidates:  {len(payload['results'])} questions, {candidates} candidates")

    if args.step3:
        if not args.judge:
            print("  --step3 needs --judge", file=sys.stderr)
            return 2
        payload = _payload(args.step3)
        VerificationResults(args.results_dir, args.run, args.judge).save(payload)
        judged = sum(
            len(q.get("candidate_verifications") or []) for q in payload["results"]
        )
        print(f"  verdicts:    {len(payload['results'])} questions, {judged} judged candidates")

    return 0


if __name__ == "__main__":
    sys.exit(main())
