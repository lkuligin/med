"""The numbers behind the paper's tables and Figure 1, read from stored runs.

    python -m paper.tables --markdown          (from llm_monkeys/, where results/ lives)

Reads only; writes one JSON (results/paper_tables.json by default) and, with
--markdown, prints the tables as the paper lays them out.

Every table and the figure stand on the difficult questions that do not
require a figure: the difficult lists less the questions label_figures.py
labelled requires_figure, 459 for MedQA and 112 for Medbullets. The rules, as
in the paper:

- one shot: the first of a model's three step 1 attempts, re-parsed with the
  current parser. Leo's step 1 runs are these same runs;
- V1: Gemini-3.8-Flash with the authors' judge prompt picks the first
  candidate whose facts all pass; a question where none passes counts as
  wrong. Leo's runs on our candidates (imports/) for the three Qwen and
  GPT-oss 20B, ours for Gemma and GPT-oss 120B;
- V2: GLM-5.3-Flash by the same prompt and rule, on the same candidates;
- none accepted: questions where a verifier passed no candidate in full, so
  the pick counts as wrong;
- majority vote: plurality over those candidates, a tie going to the option
  sampled first;
- general facts prompt: candidates written with the background fact prompt and
  answered citing their facts; their V2 is GLM with the fact-only judge prompt;
- p: the paper's one-sided McNemar test against one shot;
- pass@k: over every candidate drawn for a question, 100 for each SLM; the
  models whose runs drew 50 have 50 more in PASS_AT_100, drawn the same way
  and pooled for this only;
- Table 3, avg_k and max_k: candidates a verifier checks up to the first that
  passes, all k when none does; agreement and kappa: the two verifiers'
  verdicts on every fact both judged, on the same candidates (paper.irr). It
  reads our own Gemini runs, which judged every fact; Leo's stop at a
  candidate's first wrong fact, which leaves too biased a set of facts to
  compare. So it covers the models with both runs of ours.
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
GENERAL_FACTS = RESULTS / "experiments" / "general-facts-cited-answer"
PASS_AT_100 = RESULTS / "experiments" / "pass-at-100"
LEO = ROOT / "imports"            # Leo's runs, in the single-file layout
# store directory -> (dataset name for label_figures, difficult list)
DATASETS = {"med_qa": ("bigbio/med_qa", "difficult_questions.csv"),
            "medbullets": ("mkieffer/Medbullets", "difficult_questions_mb.csv")}
GEMINI = "gemini-3.8-flash"
GLM = "glm-5.3-flash-local"
GLM_FACT_ONLY = "glm-5.3-flash-local-fact-only"


TITLES = {"med_qa": "MedQA", "medbullets": "Medbullets"}
MARK = {"med_qa": "", "medbullets": "_mb"}   # how Leo's file names tell the datasets apart


@dataclass(frozen=True)
class Slm:
    label: str
    run: str                    # the authors' workflow run, and the step 1 run of the same name
    k: int                      # candidates per question in it
    general: str                # the general facts prompt run
    leo_v1: str | None = None   # Leo's Gemini run on `run`'s candidates: LEO/<leo_v1><MARK>.json
    more: str | None = None     # 50 more candidates in PASS_AT_100, pooled for pass@k only


SLMS = (
    Slm("Gemma 4 26B", "gemma-4-26b", 50, "gemma-4-26b-local-background-2", more="gemma-4-26b-local"),
    Slm("GPT-oss 120B", "gpt-oss-120b", 50, "gpt-oss-120b-local", more="gpt-oss-120b-local"),
    Slm("GPT-oss 20B", "gpt-oss-20b-local", 100, "gpt-oss-20b-local",
        leo_v1="results_step3_gpt_oss-20b-local_verified"),
    Slm("Qwen 3.6 27B", "qwen3.6-27b-nr-local", 50, "qwen3.6-27b-nr-local",
        leo_v1="results_step3_qwen3.6-27b_verified", more="qwen3.6-27b-nr-local"),
    Slm("Qwen 3.5 9B", "qwen3.5-9b-nr-local", 100, "qwen3.5-9b-nr-local",
        leo_v1="results_step3_qwen3.5-9b-nr-local_gemini-3.8-flash"),
    Slm("Qwen 3.5 4B", "qwen3.5-4b-nr-local", 100, "qwen3.5-4b-nr-local",
        leo_v1="results_step3_qwen3.5-4b_verified"),
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


def coverage(correct: Sequence[Sequence[bool]]) -> list[float]:
    """Mean pass@k over questions, each question's candidates given as right
    or wrong, for k up to the fewest any question has."""
    return [_mean([pass_at_k(len(c), sum(c), k) for c in correct])
            for k in range(1, min(len(c) for c in correct) + 1)]


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


def questions(dataset: str) -> list[str]:
    """The difficult questions of a dataset that do not require a figure."""
    name, listing = DATASETS[dataset]
    known = {p.stem.split("_", 1)[1] for p in (RESULTS / dataset / "single-step" / GEMINI).glob("question_*.json")}
    ids = resolve(load_difficult_question_ids(ROOT / listing), known)
    labels = read_labels(default_output(name))
    return [q for q in ids if labels.get(q, ("",))[0] != "requires_figure"]


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


def outcomes(run_dir: Path, judge: str, ids: list[str]) -> dict[str, tuple[bool, int]] | None:
    """Per question, whether some candidate passed in full and whether the
    pipeline's own pick is right; None unless every question is judged."""
    out = {}
    for q in ids:
        path = run_dir / f"question_{q}" / judge / "result.json"
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        out[q] = (bool(result.get("found_valid_candidate")), int(result.get("is_correct") is True))
    return out


def picks(run_dir: Path, judge: str, ids: list[str]) -> dict[str, int] | None:
    """Whether the pipeline's own pick is right; None unless every question is judged."""
    judged = outcomes(run_dir, judge, ids)
    return None if judged is None else {q: right for q, (_, right) in judged.items()}


def leo_outcomes(stem: str, dataset: str, ids: list[str], leo: Path = LEO) -> dict[str, tuple[bool, int]]:
    """outcomes() of one of Leo's Gemini runs, from its single-file output;
    every question must be there."""
    data = json.loads((leo / f"{stem}{MARK[dataset]}.json").read_text())
    known, out = set(ids), {}
    for record in data["results"]:
        for q in resolve([str(record["question_id"])], known):
            out[q] = (bool(record.get("found_valid_candidate")), int(record.get("is_correct") is True))
    missing = [q for q in ids if q not in out]
    if missing:
        raise ValueError(f"{stem}{MARK[dataset]}: no verdict on {len(missing)} questions, e.g. {missing[:3]}")
    return out


def leo_picks(stem: str, dataset: str, ids: list[str], leo: Path = LEO) -> dict[str, int]:
    """Whether the pick of one of Leo's Gemini runs is right."""
    return {q: right for q, (_, right) in leo_outcomes(stem, dataset, ids, leo).items()}


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


def verifier_effort(executor: Executor, slm: Slm, run_dir: Path, pools: dict[str, Pool],
                    ids: list[str]) -> dict[str, Any]:
    """Table 3: output tokens per candidate, how far each verifier reads, and
    how far the two agree - from our own Gemini run, which judged every fact."""
    v1 = read_verdicts(executor, run_dir, GEMINI, ids)
    v2 = read_verdicts(executor, run_dir, GLM, ids)
    t3: dict[str, Any] = {"tokens_per_candidate": statistics.mean(
        t for q in ids for t in pools[q].tokens if t)}
    for key, verdicts in (("v1", v1), ("v2", v2)):
        if verdicts is not None:
            needed = [first_valid_position(verdicts[q], slm.k) for q in ids]
            t3[key] = {"avg_k": _mean(needed), "max_k": max(needed),
                       "none_passed": _mean([not any(ok for ok, _ in verdicts[q].values()) for q in ids])}
    if v1 is not None and v2 is not None:
        left, right = [], []
        for q in ids:
            a, b = paired_fact_verdicts(v1[q], v2[q])
            left += a
            right += b
        irr = agreement(left, right)
        t3["irr"] = {"facts": irr.n, "agreement": irr.observed, "kappa": irr.kappa}
    return t3


def compute(workers: int = 24) -> dict[str, Any]:
    report: dict[str, Any] = {"questions": {}, "reference": {}, "table1": {}, "table2": {},
                              "table3": {}, "coverage": {}}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for dataset in DATASETS:
            ids = questions(dataset)
            gemini = first_attempts(dataset, GEMINI)
            report["questions"][dataset] = len(ids)
            report["reference"][dataset] = _mean([gemini[q] for q in ids])
            for slm in SLMS:
                one = first_attempts(dataset, slm.run)
                run_dir = RESULTS / dataset / "facts-pipeline" / slm.run
                general_dir = GENERAL_FACTS / dataset / "facts-pipeline" / slm.general
                pools = read_pools(executor, run_dir, ids, slm.k)
                judged = {"v1": leo_outcomes(slm.leo_v1, dataset, ids) if slm.leo_v1
                          else outcomes(run_dir, GEMINI, ids),
                          "v2": outcomes(run_dir, GLM, ids)}
                general = read_pools(executor, general_dir, ids, slm.k)
                general_v2 = picks(general_dir, GLM_FACT_ONLY, ids)

                t1: dict[str, Any] = {"k": slm.k, "one_shot": _mean([one[q] for q in ids]),
                                      "v1_from": "Leo" if slm.leo_v1 else "ours"}
                for key, verdict in judged.items():
                    if verdict is not None:
                        t1[key] = {**_scored({q: right for q, (_, right) in verdict.items()}, one, ids),
                                   "none_accepted": sum(not found for found, _ in verdict.values())}
                report["table1"].setdefault(dataset, {})[slm.label] = t1

                t2: dict[str, Any] = {"one_shot": t1["one_shot"]}
                if pools:
                    t2["majority_vote"] = _scored({q: vote(pools[q].options, pools[q].truth) for q in ids}, one, ids)
                if general:
                    t2["general_single"] = _mean([sum(general[q].correct) / slm.k for q in ids])
                    t2["general_majority_vote"] = _scored(
                        {q: vote(general[q].options, general[q].truth) for q in ids}, one, ids)
                    if general_v2:
                        t2["general_v2"] = _scored(general_v2, one, ids)
                report["table2"].setdefault(dataset, {})[slm.label] = t2

                if pools:
                    more = (read_pools(executor, PASS_AT_100 / dataset / "facts-pipeline" / slm.more, ids, 50)
                            if slm.more else None)
                    report["coverage"].setdefault(dataset, {})[slm.label] = coverage(
                        [pools[q].correct + (more[q].correct if more else ()) for q in ids])
                    report["table3"].setdefault(dataset, {})[slm.label] = verifier_effort(
                        executor, slm, run_dir, pools, ids)
    return report


# --- printing -----------------------------------------------------------------

def _p(result: dict[str, Any]) -> str:
    p = result["p"]
    text = "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"**{text}**" if p < 0.05 else text


def _accuracy(result: dict[str, Any]) -> str:
    arrow = "↓" if result["direction"] == "worse" and result["p"] < 0.05 else ""
    return f"{result['accuracy'] * 100:.1f}%{arrow}"


def _cell(result: dict[str, Any] | None) -> str:
    return f"{_accuracy(result)} ({_p(result)})" if result else "—"


def markdown(report: dict[str, Any]) -> str:
    """The tables as the paper lays them out: p beside each result, bold below
    0.05; Table 2 on MedQA only; Table 3 only where the verifiers' agreement
    could be measured."""
    out = ["Table 1", "| SLM | k | % correct one shot | % correct sampling (V1) (p-value) | none accepted (V1) | "
           "% correct sampling (V2) (p-value) | none accepted (V2) |", "|---|---|---|---|---|---|---|"]
    for dataset, rows in report["table1"].items():
        out.append(f"| ***{TITLES[dataset]} dataset ({report['questions'][dataset]})*** | | | | | | |")
        for label, row in rows.items():
            v1, v2 = row.get("v1") or {}, row.get("v2") or {}
            out.append(f"| {label} | {row['k']} | {row['one_shot'] * 100:.1f}% | {_cell(row.get('v1'))} | "
                       f"{v1.get('none_accepted', '—')} | {_cell(row.get('v2'))} | {v2.get('none_accepted', '—')} |")
    out += ["", f"Table 2 ({TITLES['med_qa']}, {report['questions']['med_qa']} questions)",
            "| SLM | % correct one shot | % correct majority vote (p-value) | General facts prompt: % correct | "
            "General facts prompt: % correct majority vote (p-value) | "
            "General facts prompt: % correct sampling (V2) (p-value) |", "|---|---|---|---|---|---|"]
    for label, row in report["table2"]["med_qa"].items():
        single = f"{row['general_single'] * 100:.1f}%" if "general_single" in row else "—"
        out.append(f"| {label} | {row['one_shot'] * 100:.1f}% | {_cell(row.get('majority_vote'))} | {single} | "
                   f"{_cell(row.get('general_majority_vote'))} | {_cell(row.get('general_v2'))} |")
    out += ["", "Table 3", "| SLM | k | avg tokens/cand | avg_k (V1) | avg_k (V2) | agreement V1–V2 | IRR (Cohen's κ) |",
            "|---|---|---|---|---|---|---|"]
    k = {slm.label: slm.k for slm in SLMS}
    for dataset, rows in report["table3"].items():
        out.append(f"| ***{TITLES[dataset]} dataset ({report['questions'][dataset]})*** | | | | | | |")
        for label, row in rows.items():
            irr = row.get("irr")
            if not irr:
                continue
            kappa = f"{irr['kappa']:.2f}" if irr["kappa"] is not None else "—"
            out.append(f"| {label} | {k[label]} | {row['tokens_per_candidate']:,.0f} | {row['v1']['avg_k']:.1f} | "
                       f"{row['v2']['avg_k']:.1f} | {irr['agreement'] * 100:.1f}% | {kappa} |")
    reference = ", ".join(f"{TITLES[d]} {v * 100:.1f}%" for d, v in report["reference"].items())
    out += ["", f"Gemini-3.8-Flash one shot on the same questions: {reference}"]
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
