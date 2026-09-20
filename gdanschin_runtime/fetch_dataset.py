#!/usr/bin/env python3
"""Download a dataset the pipeline runs on into a machine-local cache.

Behaves identically on this laptop and on the GPU box. The cache defaults to
<repo>/cache/huggingface, which is gitignored and excluded from the mirror, so
the dataset is never synced (it is large and machine-local) and never deleted
by the mirror's --delete. Set HF_HOME to override, or pass --cache-dir.

    python3 gdanschin_runtime/fetch_dataset.py                        # MedQA
    python3 gdanschin_runtime/fetch_dataset.py --dataset medbullets
    python3 gdanschin_runtime/fetch_dataset.py --verify-only

Verification is the point of the --verify step, not a formality, and what it
checks depends on the dataset. In MedQA a question_id is the ROW INDEX within
the split (dataset.py numbers them), so difficult_questions.csv is a list of
positions: if the copy on the Hub ever changes row count or ordering, every id
in that file silently points at a different question and step 2 runs on the
wrong set without any error. MedBullets carries its own ids, which survive a
reordering - there the check is that every difficult id is still present.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run directly (python3 gdanschin_runtime/fetch_dataset.py) and the repo root is
# not on sys.path, so our own package is not importable. Put it there first.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gdanschin_runtime import _bootstrap  # noqa: E402,F401  (llm_monkeys onto sys.path)

DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "huggingface"
MONKEYS = REPO_ROOT / "llm_monkeys"


class Known(NamedTuple):
    """A dataset the pipeline knows how to run on."""

    hf_id: str
    config: str | None
    split: str
    rows: int | None          # what the split held when this was written
    difficult: Path           # the list of questions step 2 works on
    ids_are_rows: bool        # whether a question id is its position in the split


# Two datasets, and two different meanings of a question id. MedQA has no id
# column, so llm_monkeys numbers the rows and difficult_questions.csv is a list
# of positions - which is why a change of row count or order there is
# dangerous rather than merely surprising. MedBullets carries its own ids, so
# they survive a reordering and the count check is only informative.
KNOWN: dict[str, Known] = {
    "med_qa": Known("bigbio/med_qa", "med_qa_en_source", "test", 1273,
                    MONKEYS / "difficult_questions.csv", True),
    "medbullets": Known("mkieffer/Medbullets", None, "op5_test", 308,
                        MONKEYS / "difficult_questions_mb.csv", False),
}


def read_difficult_ids(path: Path) -> list[str]:
    """Read question ids from the difficult-questions CSV, skipping the header.

    Kept as strings: MedBullets ids are zero-padded, and "001" is not 1 to
    anything that looks a question up by id.
    """
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    if rows and not rows[0].lstrip("0").isdigit():
        rows = rows[1:]
    return [r for r in rows if r]


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


def verify(known: Known, split_rows: int, difficult_ids: list[str],
           available_ids: set[str]) -> list[str]:
    """Return a list of problems found; empty means the split looks as expected."""
    problems: list[str] = []

    if known.rows is not None and split_rows != known.rows:
        problems.append(
            f"{known.split} split has {split_rows} rows, expected {known.rows}. "
            f"{known.difficult.name} " +
            ("indexes rows positionally, so its ids no longer identify the "
             "questions they were built from."
             if known.ids_are_rows else
             "may have been built against a different revision.")
        )

    missing = [i for i in difficult_ids if i not in available_ids]
    if missing:
        problems.append(
            f"{len(missing)} id(s) in {known.difficult.name} are not in the "
            f"split, e.g. {missing[:5]}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="med_qa",
                        help=f"which dataset to fetch: {', '.join(KNOWN)}, "
                             f"or a Hugging Face id")
    parser.add_argument("--dataset-config", default=None,
                        help="override the config the dataset is known by")
    parser.add_argument("--split", default=None,
                        help="override the split the pipeline uses")
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

    from results_store import dataset_dir_for

    known = KNOWN.get(dataset_dir_for(args.dataset))
    if known is None:
        print(f"unknown dataset: {args.dataset}   (known: {', '.join(KNOWN)})",
              file=sys.stderr)
        return 1
    if args.dataset_config is not None or args.split is not None:
        known = known._replace(
            config=args.dataset_config if args.dataset_config is not None else known.config,
            split=args.split if args.split is not None else known.split,
            rows=None,     # an override means the pinned count no longer applies
        )

    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset:   {known.hf_id} ({known.config}, split={known.split})")
    print(f"cache dir: {cache_dir}")

    try:
        ds = datasets.load_dataset(
            known.hf_id,
            known.config,
            split=known.split,
            cache_dir=str(cache_dir),
            download_mode="reuse_cache_if_exists" if args.verify_only else None,
        )
    except Exception as exc:
        print(f"\nfailed to load the dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows = len(ds)
    print(f"rows:      {rows}")

    # The ids as llm_monkeys will see them: a row position for MedQA, the
    # dataset's own id column for MedBullets. Comparing the difficult list
    # against anything else would check the wrong thing.
    available_ids = {
        str(i) if known.ids_are_rows
        else str(ds[i].get("idx") or ds[i].get("id") or ds[i].get("question_id") or i)
        for i in range(rows)
    }

    difficult_ids = read_difficult_ids(known.difficult)
    if difficult_ids:
        print(f"checked against {known.difficult.name}: {len(difficult_ids)} ids")
    else:
        print(f"no {known.difficult.name} yet: nothing to check ids against")

    problems = verify(known, rows, difficult_ids, available_ids)
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
