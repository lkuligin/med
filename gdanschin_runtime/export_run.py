"""Assemble a stored run back into the single file the reference reads.

    python3 gdanschin_runtime/export_run.py --run gemma-4-26b
    python3 gdanschin_runtime/export_run.py --run gemma-4-26b --judge gemini-3.8-flash

Our runs are kept as a directory per run, one file per record, which is what
makes them resumable and safe to read while they are still going. Everything
outside this runtime - the reference's own analyzers included - expects one
JSON file per step, so this writes that file out of what is stored.

The export is a copy, not a move: the directory is left exactly as it was, and
running this twice costs nothing but the time to read.

For step 2 the per-question aggregates are recomputed rather than stored, and
meta_info and ground_truth_answer come from the dataset when a run predates
their being kept. Nothing else is derived: what the workflow wrote is what
comes back out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gdanschin_runtime import _bootstrap

from results_store import (
    CandidateResults,
    OneShotResults,
    VerificationResults,
    dataset_dir_for,
)

RESULTS = _bootstrap.LLM_MONKEYS_ROOT / "results"


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return path


def _dataset_fields(question_ids: list[str], summary: dict[str, Any] | None) -> dict:
    """meta_info and ground_truth_answer for runs stored before they were kept.

    Loaded only when something is actually missing, because it means reading
    the dataset, and a run stored today needs none of it.
    """
    from dataset import load_medqa_dataset

    summary = summary or {}
    questions = load_medqa_dataset(
        dataset_name=summary.get("dataset") or "bigbio/med_qa",
        config_name=summary.get("config") or "med_qa_en_source",
        split=summary.get("split") or "test",
    )
    wanted = set(question_ids)
    return {
        str(q.question_id): {"meta_info": q.meta_info, "ground_truth_answer": q.answer}
        for q in questions
        if str(q.question_id) in wanted
    }


def export_candidates(run: str, out: Path, dataset: str) -> Path | None:
    """Step 2, with the aggregates the single-file layout carries per question."""
    from inference._schemas import CandidateQuestionResult

    stored = CandidateResults(RESULTS, run, dataset).load()
    if stored is None:
        return None

    results = stored["results"]
    incomplete = [q["question_id"] for q in results if not q.get("ground_truth_answer")]
    from_dataset = _dataset_fields(incomplete, stored.get("summary")) if incomplete else {}
    if incomplete and not from_dataset:
        print(f"  warning: {len(incomplete)} questions have no meta_info or "
              f"ground_truth_answer and the dataset did not supply them")

    assembled = []
    for question in results:
        extra = from_dataset.get(str(question["question_id"]), {})
        question = {**question, **{k: v for k, v in extra.items() if v is not None}}
        # from_dict recomputes total_candidates, accuracy, the token totals and
        # the rest, which are per-question sums the directory layout does not
        # store because they are always derivable from the candidates.
        assembled.append(CandidateQuestionResult.from_dict(question).to_dict())

    return _write(out, {"summary": stored.get("summary"), "results": assembled})


def export_one_shot(run: str, out: Path, dataset: str) -> Path | None:
    """Step 1, which is stored whole and needs nothing added."""
    stored = OneShotResults(RESULTS, run, dataset).load()
    return None if stored is None else _write(out, stored)


def export_verdicts(run: str, judge: str, out: Path, dataset: str) -> Path | None:
    """Step 3, likewise stored whole, one file per judge."""
    stored = VerificationResults(RESULTS, run, judge, dataset).load()
    return None if stored is None else _write(out, stored)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="the run to assemble")
    parser.add_argument("--judge", default=None,
                        help="which judge's verdicts to export (default: every one)")
    parser.add_argument("--dataset", default=None,
                        help="which dataset's copy of the run (default: med_qa)")
    parser.add_argument("--out-dir", default="exports",
                        help="where the files go (default: exports/)")
    args = parser.parse_args(argv)

    run = args.run
    dataset = dataset_dir_for(args.dataset)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _bootstrap.LLM_MONKEYS_ROOT / out_dir

    written: list[Path] = []
    one_shot = export_one_shot(run, out_dir / f"results_one_shot_{run}.json", dataset)
    if one_shot:
        written.append(one_shot)
    candidates = export_candidates(
        run, out_dir / f"results_step2_{run}_candidates.json", dataset)
    if candidates:
        written.append(candidates)

    judges = ([args.judge] if args.judge
              else VerificationResults(RESULTS, run, "", dataset).judges())
    for judge in judges:
        path = export_verdicts(
            run, judge, out_dir / f"results_step3_{run}_{judge}.json", dataset)
        if path:
            written.append(path)

    if not written:
        print(f"nothing stored for {run} under {RESULTS / dataset}", file=sys.stderr)
        return 1
    for path in written:
        size = path.stat().st_size / 1_000_000
        print(f"  {path}  {size:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
