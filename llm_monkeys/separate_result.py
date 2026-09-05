#!/usr/bin/env python3
"""Script to analyze multiple experiment results and separate all-correct question IDs.

Separates question IDs into a separate CSV that have been answered correctly
by ALL attempts by ALL models (i.e., in all experiments). Questions that failed
any attempt in any experiment are categorized as difficult.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


def natural_sort_key(s: str) -> list[int | str]:
    """Key for natural sorting of alphanumeric question IDs (e.g., '1', '2', '10')."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def _is_item_all_correct(item: dict[str, Any]) -> bool:
    if item.get("error"):
        return False

    raw_attempts = item.get("attempts")
    if raw_attempts is not None:
        if not raw_attempts:
            return False
        return all(
            isinstance(a, dict) and not a.get("error") and a.get("is_correct")
            for a in raw_attempts
        )

    if item.get("is_all_correct") is not None:
        return bool(item["is_all_correct"])

    if item.get("correct_attempts") is not None:
        total = int(item.get("total_attempts") or 0)
        correct = int(item.get("correct_attempts") or 0)
        return total > 0 and correct == total

    if item.get("is_correct") is not None:
        total = int(item.get("total_attempts", 1))
        return total > 0 and bool(item.get("is_correct"))

    return False


def load_results_file(
    file_path: str | Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load evaluation results JSON file, returning summary and question results list."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("summary"), data.get("results", [])
    if isinstance(data, list):
        return None, data
    raise ValueError(
        f"Expected dict or list at root of JSON in {path}, got {type(data).__name__}"
    )


def _analyze_file_results(file_path: str | Path) -> dict[str, Any]:
    summary, results = load_results_file(file_path)
    model_name = summary.get("model") if summary else None

    all_qids: set[str] = set()
    all_correct_ids: set[str] = set()

    for item in results:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("question_id"))
        all_qids.add(qid)
        if _is_item_all_correct(item):
            all_correct_ids.add(qid)

    return {
        "file_path": str(Path(file_path)),
        "model_name": model_name,
        "total_questions": len(results),
        "question_ids": all_qids,
        "all_correct_ids": all_correct_ids,
        "all_correct_count": len(all_correct_ids),
        "_raw_results": results,
    }


def separate_results(
    file_paths: Sequence[str | Path],
    output_csv: str | Path | None = None,
    difficult_csv: str | Path | None = None,
    include_metadata: bool = False,
    include_header: bool = True,
) -> dict[str, Any]:
    """Analyze multiple experiment results and separate questions answered correctly in all attempts across all models."""
    results: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, str]] = {}

    for fp in file_paths:
        analysis = _analyze_file_results(fp)
        results.append(analysis)

        for item in analysis["_raw_results"]:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id"))
            existing_meta = metadata_by_id.setdefault(
                qid,
                {"ground_truth": "", "ground_truth_answer": "", "question": ""},
            )
            for key in ("ground_truth", "ground_truth_answer", "question"):
                if not existing_meta[key] and item.get(key):
                    existing_meta[key] = str(item[key])

    if not results:
        all_evaluated_set: set[str] = set()
        all_correct_set: set[str] = set()
    else:
        all_evaluated_set = set.union(*(f["question_ids"] for f in results))
        all_correct_set = set.intersection(*(f["all_correct_ids"] for f in results))

    difficult_set = all_evaluated_set - all_correct_set
    all_correct_question_ids = sorted(all_correct_set, key=natural_sort_key)
    difficult_question_ids = sorted(difficult_set, key=natural_sort_key)

    def _write_csv(path: str | Path, qids: list[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if include_header:
                writer.writerow(
                    ["question_id", "ground_truth", "ground_truth_answer", "question"]
                    if include_metadata
                    else ["question_id"]
                )
            for qid in qids:
                if include_metadata:
                    meta = metadata_by_id.get(qid, {})
                    writer.writerow(
                        [
                            qid,
                            meta.get("ground_truth", ""),
                            meta.get("ground_truth_answer", ""),
                            meta.get("question", ""),
                        ]
                    )
                else:
                    writer.writerow([qid])

    if output_csv is not None:
        _write_csv(output_csv, all_correct_question_ids)

    if difficult_csv is not None:
        _write_csv(difficult_csv, difficult_question_ids)

    return {
        "files_analyzed": len(results),
        "total_unique_evaluated": len(all_evaluated_set),
        "all_correct_count": len(all_correct_question_ids),
        "all_correct_question_ids": all_correct_question_ids,
        "difficult_count": len(difficult_question_ids),
        "difficult_question_ids": difficult_question_ids,
        "output_csv": str(output_csv) if output_csv else None,
        "difficult_csv": str(difficult_csv) if difficult_csv else None,
        "files": [
            {
                "file_path": str(f["file_path"]),
                "model_name": f["model_name"],
                "total_questions": f["total_questions"],
                "all_correct_count": f["all_correct_count"],
                "all_correct_ids": sorted(f["all_correct_ids"], key=natural_sort_key),
            }
            for f in results
        ],
    }


def format_summary_report(stats: dict[str, Any]) -> str:
    """Format separation statistics into an informative terminal report."""
    lines = [
        "=" * 70,
        " SEPARATE RESULTS - MEDQA EXPERIMENT ANALYSIS",
        "=" * 70,
        f"Files analyzed: {stats.get('files_analyzed', 0)}",
        f"Total unique questions evaluated: {stats.get('total_unique_evaluated', 0)}",
        "",
        "Per-file summary:",
    ]
    for fa in stats.get("files", []):
        model_str = f" (Model: {fa.get('model_name')})" if fa.get("model_name") else ""
        lines.append(
            f"  • {fa.get('file_path')}{model_str}: "
            f"{fa.get('all_correct_count', 0)}/{fa.get('total_questions', 0)} all-correct questions"
        )
    lines.extend(
        [
            "",
            "-" * 70,
            f"Answered correctly by ALL attempts by ALL models: {stats.get('all_correct_count', 0)}",
            f"Difficult questions (< 100% correct across all models): {stats.get('difficult_count', 0)}",
            "-" * 70,
        ]
    )
    if stats.get("output_csv"):
        lines.append(f"Saved simple questions to:    {stats['output_csv']}")
    if stats.get("difficult_csv"):
        lines.append(f"Saved difficult questions to: {stats['difficult_csv']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for separating experiment results."""
    parser = argparse.ArgumentParser(
        description="Separate question IDs answered correctly across ALL attempts by ALL models into CSV.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to results JSON files to analyze",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="simple_questions.csv",
        help="Output CSV path for simple question IDs (default: simple_questions.csv)",
    )
    parser.add_argument(
        "-d",
        "--difficult-output",
        "--difficult-csv",
        dest="difficult_output",
        default=None,
        help="Optional output CSV path for difficult question IDs",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include question text and ground truth in output CSV",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write CSV header line",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis summary as JSON to stdout",
    )

    try:
        args = parser.parse_args(argv)
        stats = separate_results(
            file_paths=args.files,
            output_csv=args.output,
            difficult_csv=args.difficult_output,
            include_metadata=args.include_metadata,
            include_header=not args.no_header,
        )
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(format_summary_report(stats))
        return 0
    except (FileNotFoundError, ValueError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
