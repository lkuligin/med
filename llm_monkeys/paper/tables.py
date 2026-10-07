"""The numbers behind the paper's tables and Figure 1, read from stored runs.

    python -m paper.tables --markdown          (from llm_monkeys/, where results/ lives)

Reads only; writes one JSON (results/paper_tables.json by default) and, with
--markdown, prints the tables.

Every number comes from our own runs. Where the paper quotes the author's runs
instead (one shot and V1 of five models in Table 1), these are the values it
shows in parentheses when the two differ by 3 points or more.

Questions are the difficult lists, all of them ("all") or those that
label_figures.py did not label requires_figure ("no figure"). The rules, as in
the paper:

- one shot: the first of a model's three step 1 attempts, re-parsed with the
  current parser;
- V1, V2: the pipeline's own pick (result.json) with Gemini-3.8-Flash or
  GLM-5.3-Flash and the authors' judge prompt, i.e. the first candidate whose
  facts all pass; a question where none passes counts as wrong. Where GLM has
  not judged the run Gemini judged, V2 comes from our regenerated candidates
  (authors' prompts, answers citing their facts) and is marked so;
- majority vote: plurality over the candidates V2 picks from (V1's where there
  is no V2), a tie going to the option sampled first;
- general facts prompt: candidates written with the background fact prompt and
  answered citing their facts; their V2 is GLM with the fact-only judge prompt;
- p: the paper's one-sided McNemar test against one shot;
- avg_k, max_k: candidates a verifier checks before the first that passes, all
  k when none does;
- agreement, kappa: the two verifiers' verdicts on every fact both checked, on
  the same candidates (paper.irr).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from inference._dataset import load_difficult_question_ids
from label_figures import default_output, read_labels
from one_shot.parser import rescore_results
from paper.irr import agreement

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REGENERATED = RESULTS / "experiments" / "authors-facts-cited-answer"
GENERAL_FACTS = RESULTS / "experiments" / "general-facts-cited-answer"
# store directory -> (dataset name for label_figures, difficult list)
DATASETS = {"med_qa": ("bigbio/med_qa", "difficult_questions.csv"),
            "medbullets": ("mkieffer/Medbullets", "difficult_questions_mb.csv")}
GEMINI = "gemini-3.8-flash"
GLM = "glm-5.3-flash-local"
GLM_FACT_ONLY = "glm-5.3-flash-local-fact-only"


@dataclass(frozen=True)
class Slm:
    label: str
    run: str                  # the authors' workflow run, and the step 1 run of the same name
    k: int                    # candidates per question in it
    regenerated: str | None   # our regenerated run, where GLM judged it instead of `run`
    general: str              # the general facts prompt run


SLMS = (
    Slm("Gemma 4 26B", "gemma-4-26b", 50, None, "gemma-4-26b-local-background-2"),
    Slm("GPT-oss 120B", "gpt-oss-120b", 50, "gpt-oss-120b-local", "gpt-oss-120b-local"),
    Slm("GPT-oss 20B", "gpt-oss-20b-local", 100, "gpt-oss-20b-local", "gpt-oss-20b-local"),
    Slm("Qwen 3.6 27B", "qwen3.6-27b-nr-local", 50, "qwen3.6-27b-nr-local", "qwen3.6-27b-nr-local"),
    Slm("Qwen 3.5 9B", "qwen3.5-9b-nr-local", 100, "qwen3.5-9b-nr-local", "qwen3.5-9b-nr-local"),
    Slm("Qwen 3.5 4B", "qwen3.5-4b-nr-local", 100, None, "qwen3.5-4b-nr-local"),
)


# --- statistics ---------------------------------------------------------------

def vote(options: Sequence[str | None], truth: str | None) -> int:
    """1 if the plurality option is right; a tie goes to the option sampled first."""
    counts = collections.Counter(o for o in options if o)
    if not counts:
        return 0
    top = max(counts.values())
    return int(next(o for o in options if o and counts[o] == top) == truth)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct, as in [1]."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mcnemar(a: Sequence[int], b: Sequence[int]) -> dict[str, Any]:
    """The paper's one-sided McNemar test of B against A on paired 0/1 outcomes.

    improved: items B gets right and A wrong; degraded: the reverse.
    Z = |improved - degraded| / sqrt(improved + degraded) when the difference
    exceeds 1, else 0; p = erfc(Z / sqrt(2)) / 2. The absolute value makes p
    one-sided in whichever direction was observed, so `direction` says which.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired outcomes: {len(a)} and {len(b)}")
    improved = sum(1 for x, y in zip(a, b) if y and not x)
    degraded = sum(1 for x, y in zip(a, b) if x and not y)
    diff = abs(improved - degraded)
    z = diff / math.sqrt(improved + degraded) if diff > 1 else 0.0
    direction = "better" if improved > degraded else "worse" if degraded > improved else "same"
    return {"improved": improved, "degraded": degraded, "p": 0.5 * math.erfc(z / math.sqrt(2)),
            "direction": direction}


def first_valid_position(verdicts: dict[int, tuple[bool, tuple]], k: int) -> int:
    """Candidates a verifier checks, in sampling order, up to the first whose
    facts all pass; k when none does."""
    first = min((i for i, (all_valid, _) in verdicts.items() if all_valid), default=None)
    return k if first is None else first + 1


def paired_fact_verdicts(a: dict[int, tuple[bool, tuple]],
                         b: dict[int, tuple[bool, tuple]]) -> tuple[list, list]:
    """Two verifiers' verdicts on the facts both checked, in matching order.
    A fact either left unreadable is skipped, and so is a candidate whose fact
    lists differ in length."""
    left, right = [], []
    for i in sorted(set(a) & set(b)):
        fa, fb = a[i][1], b[i][1]
        if len(fa) != len(fb):
            continue
        for x, y in zip(fa, fb):
            if x is not None and y is not None:
                left.append(x)
                right.append(y)
    return left, right


# --- reading the store --------------------------------------------------------

def resolve(listed: Sequence[str], known: set[str]) -> list[str]:
    """Listed ids in the spelling `known` uses, by the rule of
    inference._dataset.load_difficult_questions: as written, without leading
    zeros, padded to three digits. Ids that match nothing are dropped."""
    out = []
    for qid in listed:
        for spelling in (qid, qid.lstrip("0") or "0", qid.zfill(3)):
            if spelling in known:
                out.append(spelling)
                break
    return out


def question_sets(dataset: str) -> dict[str, list[str]]:
    """The difficult questions of a dataset, all and without a required figure."""
    name, listing = DATASETS[dataset]
    known = {p.stem.split("_", 1)[1] for p in (RESULTS / dataset / "single-step" / GEMINI).glob("question_*.json")}
    ids = resolve(load_difficult_question_ids(ROOT / listing), known)
    labels = read_labels(default_output(name))
    return {"all": ids, "no figure": [q for q in ids if labels.get(q, ("",))[0] != "requires_figure"]}


def first_attempts(dataset: str, run: str) -> dict[str, int]:
    """Whether a model's first step 1 attempt is right, re-parsed."""
    records = [json.loads(p.read_text()) for p in (RESULTS / dataset / "single-step" / run).glob("question_*.json")]
    out = {}
    for record in rescore_results(records):
        first = min(record["attempts"], key=lambda a: a.get("attempt_index", 0))
        out[str(record["question_id"])] = int(bool(first.get("is_correct")))
    return out


@dataclass(frozen=True)
class Pool:
    truth: str | None
    options: tuple[str | None, ...]   # in sampling order
    correct: tuple[bool, ...]
    tokens: tuple[int, ...]           # output tokens, facts and answer


def _read_pool(args: tuple[Path, int]) -> Pool:
    qdir, k = args
    rows, truth = [], None
    for path in qdir.glob("iteration_*.json"):
        record = json.loads(path.read_text())
        cand = record["candidate"]
        truth = truth or record.get("ground_truth")
        tokens = cand.get("total_candidate_tokens") or (
            ((cand.get("fact_tokens") or {}).get("candidate_tokens") or 0)
            + ((cand.get("answer_tokens") or {}).get("candidate_tokens") or 0))
        rows.append((cand["candidate_index"], cand.get("predicted_option"), bool(cand.get("is_correct")), tokens))
    rows.sort(key=lambda r: r[0])
    rows = rows[:k]
    return Pool(truth, tuple(r[1] for r in rows), tuple(r[2] for r in rows), tuple(r[3] for r in rows))


def read_pools(executor: Executor, run_dir: Path, ids: list[str], k: int) -> dict[str, Pool] | None:
    """Each question's first k candidates; None unless every question has k."""
    if not run_dir.is_dir():
        return None
    pools = dict(zip(ids, executor.map(_read_pool, [(run_dir / f"question_{q}", k) for q in ids], chunksize=8)))
    return pools if all(len(p.options) == k for p in pools.values()) else None


def picks(run_dir: Path, judge: str, ids: list[str]) -> dict[str, int] | None:
    """Whether the pipeline's own pick is right; None unless every question is judged."""
    out = {}
    for q in ids:
        path = run_dir / f"question_{q}" / judge / "result.json"
        if not path.is_file():
            return None
        out[q] = int(json.loads(path.read_text()).get("is_correct") is True)
    return out


def _read_verdicts(judge_dir: Path) -> dict[int, tuple[bool, tuple]]:
    out = {}
    for path in judge_dir.glob("iteration_*.json"):
        v = json.loads(path.read_text())["verification"]
        facts = tuple(f.get("verdict") for f in v.get("fact_verifications") or [])
        out[v["candidate_index"]] = (bool(v.get("all_facts_correct")), facts)
    return out


def read_verdicts(executor: Executor, run_dir: Path, judge: str,
                  ids: list[str]) -> dict[str, dict[int, tuple[bool, tuple]]] | None:
    """Per question, candidate index -> (all facts pass, fact verdicts)."""
    if not all((run_dir / f"question_{q}" / judge / "result.json").is_file() for q in ids):
        return None
    return dict(zip(ids, executor.map(_read_verdicts, [run_dir / f"question_{q}" / judge for q in ids], chunksize=8)))


# --- the tables ---------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _scored(selection: dict[str, int], one_shot: dict[str, int], ids: list[str]) -> dict[str, Any]:
    picked = [selection[q] for q in ids]
    return {"accuracy": _mean(picked), **mcnemar([one_shot[q] for q in ids], picked)}


def compute(workers: int = 24) -> dict[str, Any]:
    report: dict[str, Any] = {"table1": {}, "table2": {}, "table3": {}, "coverage": {}, "reference": {}}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for dataset in DATASETS:
            sets = question_sets(dataset)
            ids = sets["all"]
            gemini = first_attempts(dataset, GEMINI)
            report["reference"][dataset] = {name: _mean([gemini[q] for q in qs]) for name, qs in sets.items()}
            for slm in SLMS:
                one = first_attempts(dataset, slm.run)
                run_dir = RESULTS / dataset / "facts-pipeline" / slm.run
                general_dir = GENERAL_FACTS / dataset / "facts-pipeline" / slm.general
                v1 = picks(run_dir, GEMINI, ids)
                v2, v2_dir = picks(run_dir, GLM, ids), run_dir
                if v2 is None and slm.regenerated:
                    regenerated_dir = REGENERATED / dataset / "facts-pipeline" / slm.regenerated
                    v2, v2_dir = picks(regenerated_dir, GLM, ids), regenerated_dir
                regenerated = v2 is not None and v2_dir != run_dir
                pools = read_pools(executor, run_dir, ids, slm.k)
                vote_pools = pools if v2 is None or not regenerated else read_pools(executor, v2_dir, ids, slm.k)
                general = read_pools(executor, general_dir, ids, slm.k)
                general_v2 = picks(general_dir, GLM_FACT_ONLY, ids)

                for name, qs in sets.items():
                    t1: dict[str, Any] = {"one_shot": _mean([one[q] for q in qs])}
                    if v1:
                        t1["v1"] = _scored(v1, one, qs)
                    if v2:
                        t1["v2"] = {**_scored(v2, one, qs), "regenerated": regenerated}
                    report["table1"].setdefault(name, {}).setdefault(dataset, {})[slm.label] = t1

                    t2: dict[str, Any] = {"one_shot": t1["one_shot"]}
                    if vote_pools:
                        votes = {q: vote(vote_pools[q].options, vote_pools[q].truth) for q in qs}
                        t2["majority_vote"] = {**_scored(votes, one, qs),
                                               "candidates": "regenerated" if regenerated else "run"}
                    if general:
                        t2["general_single"] = _mean([sum(general[q].correct) / slm.k for q in qs])
                        t2["general_majority_vote"] = _scored(
                            {q: vote(general[q].options, general[q].truth) for q in qs}, one, qs)
                        if general_v2:
                            t2["general_v2"] = _scored(general_v2, one, qs)
                    report["table2"].setdefault(name, {}).setdefault(dataset, {})[slm.label] = t2

                if pools:
                    report["coverage"].setdefault(dataset, {})[slm.label] = [
                        _mean([pass_at_k(slm.k, sum(pools[q].correct), k) for q in ids]) for k in range(1, slm.k + 1)]
                    t3: dict[str, Any] = {"tokens_per_candidate": statistics.mean(
                        t for q in ids for t in pools[q].tokens if t), "v2_regenerated": regenerated}
                    judged = {}
                    for key, (where, judge) in {"v1": (run_dir, GEMINI), "v2": (v2_dir, GLM)}.items():
                        verdicts = read_verdicts(executor, where, judge, ids)
                        if verdicts is None:
                            continue
                        judged[key] = (where, verdicts)
                        needed = [first_valid_position(verdicts[q], slm.k) for q in ids]
                        t3[key] = {"avg_k": _mean(needed), "max_k": max(needed),
                                   "none_passed": _mean([not any(ok for ok, _ in verdicts[q].values()) for q in ids])}
                    if "v1" in judged and "v2" in judged and judged["v1"][0] == judged["v2"][0]:
                        left, right = [], []
                        for q in ids:
                            a, b = paired_fact_verdicts(judged["v1"][1][q], judged["v2"][1][q])
                            left += a
                            right += b
                        irr = agreement(left, right)
                        t3["irr"] = {"facts": irr.n, "agreement": irr.observed, "kappa": irr.kappa}
                    report["table3"].setdefault(dataset, {})[slm.label] = t3
    return report


# --- printing -----------------------------------------------------------------

def _cell(result: dict[str, Any] | None, mark: str = "") -> str:
    if not result:
        return "—"
    p = result["p"]
    p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
    arrow = "↓" if result["direction"] == "worse" and p < 0.05 else ""
    return f"{result['accuracy'] * 100:.1f}%{mark}{arrow} ({'**' + p_text + '**' if p < 0.05 else p_text})"


def markdown(report: dict[str, Any]) -> str:
    """The tables as the paper lays them out, p in parentheses, bold below 0.05."""
    out = []
    for name, by_dataset in report["table1"].items():
        out += [f"Table 1 ({name})", "| SLM | one shot | V1 | V2 |", "|---|---|---|---|"]
        for dataset, rows in by_dataset.items():
            out.append(f"| *{dataset}* | | | |")
            for label, row in rows.items():
                dagger = "†" if row.get("v2", {}).get("regenerated") else ""
                out.append(f"| {label} | {row['one_shot'] * 100:.1f}% | {_cell(row.get('v1'))} | {_cell(row.get('v2'), dagger)} |")
        out.append("")
    for name, by_dataset in report["table2"].items():
        out += [f"Table 2 ({name})", "| SLM | one shot | majority vote | general facts | general facts, majority vote | general facts, V2 |",
                "|---|---|---|---|---|---|"]
        for dataset, rows in by_dataset.items():
            out.append(f"| *{dataset}* | | | | | |")
            for label, row in rows.items():
                single = f"{row['general_single'] * 100:.1f}%" if "general_single" in row else "—"
                out.append(f"| {label} | {row['one_shot'] * 100:.1f}% | {_cell(row.get('majority_vote'))} | {single} | "
                           f"{_cell(row.get('general_majority_vote'))} | {_cell(row.get('general_v2'))} |")
        out.append("")
    out += ["Table 3", "| SLM | tokens/cand | avg_k V1 | max_k V1 | avg_k V2 | max_k V2 | agreement | kappa |",
            "|---|---|---|---|---|---|---|---|"]
    for dataset, rows in report["table3"].items():
        out.append(f"| *{dataset}* | | | | | | | |")
        for label, row in rows.items():
            dagger = "†" if row.get("v2_regenerated") else ""
            v1, v2, irr = row.get("v1"), row.get("v2"), row.get("irr")
            cells = [f"{row['tokens_per_candidate']:,.0f}",
                     f"{v1['avg_k']:.1f}" if v1 else "—", str(v1["max_k"]) if v1 else "—",
                     f"{v2['avg_k']:.1f}{dagger}" if v2 else "—", f"{v2['max_k']}{dagger}" if v2 else "—",
                     f"{irr['agreement'] * 100:.1f}%" if irr else "—",
                     f"{irr['kappa']:.2f}" if irr and irr["kappa"] is not None else "—"]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
    reference = ", ".join(f"{d}: {v['all'] * 100:.1f}% (no figure {v['no figure'] * 100:.1f}%)"
                          for d, v in report["reference"].items())
    out += ["", f"Gemini-3.8-Flash one shot: {reference}"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "paper_tables.json")
    parser.add_argument("--markdown", action="store_true", help="also print the tables")
    parser.add_argument("--workers", type=int, default=24, help="processes reading the store")
    args = parser.parse_args(argv)
    report = compute(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.markdown:
        print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
