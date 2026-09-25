"""Did an experiment beat, or at least match, the reference pipeline?

    python -m gdanschin_runtime.experiment_report \
        --experiment grounded-unframed-facts-cited-answer --run gemma-4-26b-local \
        --judge glm-5.3-flash-local-fact-only

For each dataset, over the difficult questions, it compares:

- the experiment: mean candidate accuracy, majority vote, pass@k, and two
  judge selections - the first candidate whose facts all pass, and the first
  whose cited facts all pass (a candidate without a FACTS USED line counts
  all its facts);
- the reference: the same model's run with the authors' prompts in the main
  store, judged by a judge that sees the question and options;
- one pass: the first attempt of the model's step 1 run.

Each dataset is also split by whether its questions need an exhibit the text
does not include, from data/figure_labels_<dataset>.csv (labelled by an LLM):
questions that require a figure, and questions whose text is enough.

The verdict follows the rule agreed for this experiment: on at least one
dataset the better of the two judge selections is within MARGIN of the
reference judge, and above the experiment's own mean candidate accuracy - the
judge has to do better than picking a candidate at random. Reads stored
results only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from math import comb
from pathlib import Path

MARGIN = 0.04
ROOT = Path(__file__).resolve().parents[1] / "llm_monkeys"
DATASETS = {"med_qa": "difficult_questions.csv", "medbullets": "difficult_questions_mb.csv"}


def bare(qid) -> str:
    text = str(qid).strip()
    return (text.lstrip("0") or "0") if text.isdigit() else text


def difficult(dataset: str) -> list[str]:
    with open(ROOT / DATASETS[dataset], encoding="utf-8-sig") as handle:
        rows = [r[0].strip() for r in csv.reader(handle) if r and r[0].strip()]
    return [bare(r) for r in rows if r.lower() not in ("question_id", "id", "qid")]


def question_dirs(run_dir: Path) -> dict[str, Path]:
    return {bare(p.name.split("_", 1)[1]): p for p in run_dir.glob("question_*") if p.is_dir()}


def numbered(directory: Path) -> list[Path]:
    return sorted(directory.glob("iteration_*.json"),
                  key=lambda p: int(re.search(r"_(\d+)\.json$", p.name).group(1)))


def candidates(qdir: Path, k: int) -> list[dict]:
    return [json.loads(p.read_text()) for p in numbered(qdir)[:k]]


def verdicts(qdir: Path, judge: str, k: int) -> list[dict]:
    out = [json.loads(p.read_text())["verification"] for p in numbered(qdir / judge)]
    return sorted((v for v in out if v.get("candidate_index", 0) < k),
                  key=lambda v: v.get("candidate_index", 0))


def majority(cands: list[dict]) -> float:
    truth = cands[0]["ground_truth"]
    votes = collections.Counter(c["candidate"].get("predicted_option") for c in cands
                                if c["candidate"].get("predicted_option"))
    if not votes:
        return 0.0
    top = max(votes.values())
    winners = [o for o, n in votes.items() if n == top]
    return (truth in winners) / len(winners)


def pass_at(cands: list[dict], k: int) -> float:
    n = len(cands)
    c = sum(bool(x["candidate"].get("is_correct")) for x in cands)
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def first_valid(vs: list[dict], cited: dict[int, list[int] | None] | None) -> bool:
    def ok(v):
        if cited is None or cited.get(v["candidate_index"]) is None:
            return v.get("all_facts_correct")
        facts = v["fact_verifications"]
        return all(facts[i]["verdict"] == 1 for i in cited[v["candidate_index"]] if i < len(facts))
    chosen = next((v for v in vs if ok(v)), None)
    return bool(chosen and chosen.get("is_correct"))


def one_pass(dataset: str, run: str, qids: list[str]) -> float | None:
    """First-attempt accuracy of the model's step 1 run, re-parsed."""
    sys.path.insert(0, str(ROOT))
    from one_shot.parser import rescore_results
    records = [json.loads(p.read_text())
               for p in (ROOT / "results" / dataset / "single-step" / run).glob("question_*.json")]
    by_id = {bare(r["question_id"]): r for r in rescore_results(records)}
    got = [by_id[q] for q in qids if q in by_id]
    if not got:
        return None
    first = [min(r["attempts"], key=lambda a: a.get("attempt_index", 0)) for r in got]
    return sum(bool(a["is_correct"]) for a in first) / len(first)


def figure_labels(dataset: str) -> dict[str, str]:
    """question id -> label; empty when the dataset has not been labelled."""
    path = ROOT / "data" / f"figure_labels_{dataset}.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return {bare(r["question_id"]): r["label"] for r in csv.DictReader(handle)}


FIGURE_GROUPS = {
    "requires_figure": ("requires_figure",),
    "text_is_enough": ("no_figure", "mentions_figure_answerable"),
}


def evaluate(experiment: str, run: str, judge: str, reference_run: str,
             reference_judge: str, one_pass_run: str, k: int) -> dict:
    report = {}
    for dataset in DATASETS:
        exp_dirs = question_dirs(ROOT / "results" / "experiments" / experiment / dataset
                                 / "facts-pipeline" / run)
        ref_dirs = question_dirs(ROOT / "results" / dataset / "facts-pipeline" / reference_run)
        qids = [q for q in difficult(dataset)
                if q in exp_dirs and os.path.isfile(exp_dirs[q] / judge / "result.json")
                and q in ref_dirs and os.path.isfile(ref_dirs[q] / reference_judge / "result.json")]
        if not qids:
            report[dataset] = {"questions": 0}
            continue
        per_question = {}
        for q in qids:
            m = collections.Counter()
            cands = candidates(exp_dirs[q], k)
            cited = {c["candidate"]["candidate_index"]: c["candidate"].get("cited_facts") for c in cands}
            vs = verdicts(exp_dirs[q], judge, k)
            m["mean"] += sum(bool(c["candidate"].get("is_correct")) for c in cands) / len(cands)
            m["majority"] += majority(cands)
            m["pass@1"] += pass_at(cands, 1)
            m["pass@k"] += pass_at(cands, len(cands))
            m["judge_all"] += first_valid(vs, None)
            m["judge_cited"] += first_valid(vs, cited)
            ref = candidates(ref_dirs[q], k)
            m["ref_mean"] += sum(bool(c["candidate"].get("is_correct")) for c in ref) / len(ref)
            m["ref_majority"] += majority(ref)
            m["ref_judge"] += first_valid(verdicts(ref_dirs[q], reference_judge, k), None)
            per_question[q] = m

        def summarise(ids: list[str]) -> dict:
            total = collections.Counter()
            for q in ids:
                total.update(per_question[q])
            row = {key: value / len(ids) for key, value in total.items()}
            row["questions"] = len(ids)
            row["one_pass"] = one_pass(dataset, one_pass_run, ids)
            return row

        row = summarise(qids)
        best = max(row["judge_all"], row["judge_cited"])
        row["success"] = bool(best >= row["ref_judge"] - MARGIN and best > row["mean"])
        labels = figure_labels(dataset)
        if labels:
            row["by_figure"] = {
                group: summarise(ids)
                for group, members in FIGURE_GROUPS.items()
                if (ids := [q for q in qids if labels.get(q) in members])
            }
        report[dataset] = row
    # Success on at least one dataset is enough; a dataset with no judged
    # questions cannot be the one.
    report["success"] = any(report[d].get("questions") and report[d].get("success")
                            for d in DATASETS)
    return report


def markdown(report: dict, args) -> str:
    lines = [f"# {args.experiment}: {args.run} judged by {args.judge}", "",
             f"Reference: {args.reference_run} judged by {args.reference_judge}; "
             f"one pass: {args.one_pass_run}, first attempt; k={args.k}; "
             f"margin {int(MARGIN * 100)} pp.", ""]
    labels = [("questions", "Questions"), ("mean", "Mean candidate accuracy"),
              ("majority", "Majority vote"), ("pass@k", "pass@k"),
              ("judge_all", "Judge: first with all facts approved"),
              ("judge_cited", "Judge: first with cited facts approved"),
              ("ref_mean", "Reference: mean candidate accuracy"),
              ("ref_majority", "Reference: majority vote"),
              ("ref_judge", "Reference: judge"), ("one_pass", "One pass"),
              ("success", "Success")]
    datasets = [d for d in DATASETS if d in report]
    lines.append("| Metric | " + " | ".join(datasets) + " |")
    lines.append("|---|" + "---|" * len(datasets))
    for key, label in labels:
        cells = []
        for d in datasets:
            value = report[d].get(key)
            cells.append("-" if value is None else str(value) if isinstance(value, (bool, int))
                         else f"{100 * value:.1f}%")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", f"Overall success: {report['success']}"]

    split = [(d, g) for d in datasets for g in FIGURE_GROUPS
             if g in (report[d].get("by_figure") or {})]
    if split:
        lines += ["", "## By figure dependence", "",
                  "| Metric | " + " | ".join(f"{d}, {g}" for d, g in split) + " |",
                  "|---|" + "---|" * len(split)]
        for key, label in labels[:-1]:
            cells = []
            for d, g in split:
                value = report[d]["by_figure"][g].get(key)
                cells.append("-" if value is None else str(value) if isinstance(value, int)
                             else f"{100 * value:.1f}%")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--reference-run", default="gemma-4-26b")
    parser.add_argument("--reference-judge", default="glm-5.3-flash-local")
    parser.add_argument("--one-pass-run", default="gemma-4-26b-local")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--out", default=None, help="write the report here (.md and .json)")
    args = parser.parse_args(argv)
    report = evaluate(args.experiment, args.run, args.judge, args.reference_run,
                      args.reference_judge, args.one_pass_run, args.k)
    text = markdown(report, args)
    print(text)
    if args.out:
        Path(args.out + ".md").write_text(text)
        Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
