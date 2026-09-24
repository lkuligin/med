"""Assemble a stored run back into the single file the reference reads.

    python3 gdanschin_runtime/export_run.py --run gemma-4-26b
    python3 gdanschin_runtime/export_run.py --run gemma-4-26b --steps one-shot
    python3 gdanschin_runtime/export_run.py --run gemma-4-26b --judge gemini-3.8-flash

Exports are gzipped: a step 1 file is a record per question with the raw model
response in it, and these are handed over by upload, where the size is the cost.
Gzip is written directly, so the uncompressed file never exists on disk.

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
import gzip
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

# The reference puts the dataset in the file name and reads it back out of
# there - inference/run.py decides a run is MedBullets when "_mb" or
# "medbullets" appears in the path it was given. So an export has to carry the
# same mark, or the file we hand over is read as the wrong dataset; and two
# runs of one model on two datasets would land on the same name.
DATASET_SUFFIX = {"med_qa": "", "medbullets": "_mb"}

# One directory per step under exports/, so a handover of a single step is a
# directory rather than a pattern over file names.
STEP_DIR = {"one-shot": "one_shot", "candidates": "candidates", "verdicts": "verdicts"}


def _suffix(dataset: str) -> str:
    """What to append to an exported file name for this dataset."""
    return DATASET_SUFFIX.get(dataset, f"_{dataset}")


def _write(path: Path, payload: dict[str, Any]) -> Path:
    """Write the payload gzipped, atomically, at <path>.gz."""
    path = path.with_suffix(path.suffix + ".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".gz.tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
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


def _on_list(results: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    """The questions the difficult-questions list names, and no others.

    A run can hold more than the list: the list has been shortened since some
    of these runs were made, and nothing stored is ever thrown away. Those
    extra questions are the easy ones, so a file exporting them is not the
    measurement its name claims.

    Ids are compared with the padding stripped - MedBullets writes 001 in its
    list and the dataset hands the same question over as 1.
    """
    from inference._dataset import load_difficult_question_ids

    from gdanschin_runtime.fetch_dataset import KNOWN

    known = KNOWN.get(dataset)
    path = _bootstrap.LLM_MONKEYS_ROOT / known.difficult.name if known else None
    if path is None or not path.is_file():
        print(f"  warning: no difficult-questions list for {dataset}; "
              f"exporting every question the run holds")
        return results

    def bare(qid: Any) -> str:
        text = str(qid)
        return (text.lstrip("0") or "0") if text.isdigit() else text

    wanted = {bare(q) for q in load_difficult_question_ids(path)}
    return [q for q in results if bare(q["question_id"]) in wanted]


def export_candidates(run: str, out: Path, dataset: str,
                      difficult_only: bool = False) -> Path | None:
    """Step 2, with the aggregates the single-file layout carries per question."""
    from inference._schemas import CandidateQuestionResult

    stored = CandidateResults(RESULTS, run, dataset).load()
    if stored is None:
        return None

    results = stored["results"]
    if difficult_only:
        kept = _on_list(results, dataset)
        if len(kept) != len(results):
            print(f"  {len(kept)} of {len(results)} questions are on the "
                  f"difficult list; the rest are left out")
        results = kept
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


def export_verdicts(run: str, judge: str, out: Path, dataset: str,
                    difficult_only: bool = False) -> Path | None:
    """Step 3, likewise stored whole, one file per judge."""
    stored = VerificationResults(RESULTS, run, judge, dataset).load()
    if stored is None:
        return None
    if difficult_only:
        stored = {**stored, "results": _on_list(stored["results"], dataset)}
    return _write(out, stored)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="the run to assemble")
    parser.add_argument("--judge", default=None,
                        help="which judge's verdicts to export (default: every one)")
    parser.add_argument("--dataset", default=None,
                        help="which dataset's copy of the run (default: med_qa)")
    parser.add_argument("--out-dir", default="exports",
                        help="where the files go; each step lands in its own "
                             "subdirectory of it (default: exports/)")
    parser.add_argument("--difficult-only", action="store_true",
                        help="export only the questions the difficult-questions "
                             "list names, leaving out any others the run holds")
    parser.add_argument("--steps", nargs="+", default=["one-shot", "candidates", "verdicts"],
                        choices=["one-shot", "candidates", "verdicts"],
                        help="which steps to export (default: all of them)")
    args = parser.parse_args(argv)

    run = args.run
    dataset = dataset_dir_for(args.dataset)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _bootstrap.LLM_MONKEYS_ROOT / out_dir

    written: list[Path] = []
    mark = _suffix(dataset)
    if "one-shot" in args.steps:
        one_shot = export_one_shot(
            run,
            out_dir / STEP_DIR["one-shot"] / f"results_one_shot_{run}{mark}.json",
            dataset,
        )
        if one_shot:
            written.append(one_shot)
    if "candidates" in args.steps:
        candidates = export_candidates(
            run,
            out_dir / STEP_DIR["candidates"] / f"results_step2_{run}_candidates{mark}.json",
            dataset,
            difficult_only=args.difficult_only,
        )
        if candidates:
            written.append(candidates)
    if "verdicts" in args.steps:
        judges = ([args.judge] if args.judge
                  else VerificationResults(RESULTS, run, "", dataset).judges())
        for judge in judges:
            path = export_verdicts(
                run, judge,
                out_dir / STEP_DIR["verdicts"] / f"results_step3_{run}_{judge}{mark}.json",
                dataset,
                difficult_only=args.difficult_only,
            )
            if path:
                written.append(path)

    if not written:
        print(f"nothing stored for {run} under {RESULTS / dataset}", file=sys.stderr)
        return 1
    for path in written:
        size = path.stat().st_size / 1_000_000
        print(f"  {path}  {size:.1f} MB")
    return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
