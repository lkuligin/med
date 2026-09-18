#!/usr/bin/env python3
"""Download the MedQA dataset into a machine-local cache.

Behaves identically on this laptop and on the GPU box. The cache defaults to
<repo>/cache/huggingface, which is gitignored and excluded from the mirror, so
the dataset is never synced (it is large and machine-local) and never deleted
by the mirror's --delete. Set HF_HOME to override, or pass --cache-dir.

    python3 gdanschin_runtime/fetch_dataset.py            # fetch + verify
    python3 gdanschin_runtime/fetch_dataset.py --verify-only

Verification is the point of the --verify step, not a formality. Throughout
llm_monkeys a question_id is the ROW INDEX within the split
(dataset.py: MedQAQuestion.from_dict(ds[i], question_id=str(i))), and
difficult_questions.csv is a list of those indices. If the copy on the Hub ever
changes row count or ordering, every id in that file silently points at a
different question and step 2 runs on the wrong set without any error.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run directly (python3 gdanschin_runtime/fetch_dataset.py) and the repo root is
# not on sys.path, so our own package is not importable. Put it there first.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gdanschin_runtime import _bootstrap  # noqa: E402,F401  (llm_monkeys onto sys.path)

DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "huggingface"
DIFFICULT_QUESTIONS = REPO_ROOT / "llm_monkeys" / "difficult_questions.csv"

# The full MedQA (USMLE, English) test split, as the committed
# difficult_questions.csv was built against it.
EXPECTED_TEST_ROWS = 1273


def read_difficult_ids(path: Path) -> list[int]:
    """Read question ids from the difficult-questions CSV, skipping the header."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    if rows and not rows[0].isdigit():
        rows = rows[1:]
    return [int(r) for r in rows if r.isdigit()]


def check_schema(record: dict) -> list[str]:
    """Check one raw record still parses into the shape the pipeline expects.

    Row count alone would not catch a schema change on the Hub: renamed fields
    or options arriving in a different shape would leave the count intact while
    every prompt silently loses its options.
    """
    # Imported here, not at module scope: llm_monkeys/dataset.py imports
    # datasets at import time, and we want our own missing-dependency message
    # rather than a traceback from three frames down.
    from dataset import MedQAQuestion

    problems: list[str] = []
    q = MedQAQuestion.from_dict(record, question_id="0")
    if not q.question:
        problems.append("sample record has an empty question field")
    if len(q.options) < 4:
        problems.append(f"sample record parsed {len(q.options)} options, expected >= 4")
    if not q.answer_idx:
        problems.append("sample record has an empty answer_idx")
    elif q.options and q.answer_idx not in q.options:
        problems.append(
            f"sample answer_idx {q.answer_idx!r} is not among the parsed "
            f"option keys {sorted(q.options)}"
        )
    return problems


def verify(split_rows: int, difficult_ids: list[int], split: str) -> list[str]:
    """Return a list of problems found; empty means the split looks as expected."""
    problems: list[str] = []

    if split == "test" and split_rows != EXPECTED_TEST_ROWS:
        problems.append(
            f"test split has {split_rows} rows, expected {EXPECTED_TEST_ROWS}. "
            "difficult_questions.csv indexes rows positionally, so its ids no "
            "longer identify the questions they were built from."
        )

    if difficult_ids:
        out_of_range = [i for i in difficult_ids if i >= split_rows]
        if out_of_range:
            problems.append(
                f"{len(out_of_range)} id(s) in {DIFFICULT_QUESTIONS.name} fall "
                f"outside the split (max index {split_rows - 1}), "
                f"e.g. {out_of_range[:5]}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="bigbio/med_qa")
    parser.add_argument("--dataset-config", default="med_qa_en_source")
    parser.add_argument(
        "--split",
        default="test",
        help="Split to fetch and verify (default: test, the one the pipeline uses)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Where to cache the download (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only check an already-cached copy",
    )
    args = parser.parse_args(argv)

    try:
        import datasets
    except ImportError:
        print(
            "the 'datasets' package is not installed.\n"
            f"  pip install -r {REPO_ROOT / 'llm_monkeys' / 'requirements.txt'}",
            file=sys.stderr,
        )
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset:   {args.dataset} ({args.dataset_config}, split={args.split})")
    print(f"cache dir: {cache_dir}")

    try:
        ds = datasets.load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            cache_dir=str(cache_dir),
            download_mode="reuse_cache_if_exists" if args.verify_only else None,
        )
    except Exception as exc:
        print(f"\nfailed to load the dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows = len(ds)
    print(f"rows:      {rows}")

    difficult_ids = read_difficult_ids(DIFFICULT_QUESTIONS)
    if difficult_ids:
        print(
            f"checked against {DIFFICULT_QUESTIONS.name}: "
            f"{len(difficult_ids)} ids, max {max(difficult_ids)}"
        )

    problems = verify(rows, difficult_ids, args.split)
    if rows:
        problems.extend(check_schema(ds[0]))
    if problems:
        print("\nVERIFICATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    print("verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
