"""Look at a step 2 run by eye, from a notebook.

    from gdanschin_runtime.inspect_run import load, overview, question, candidate

    res = load()                 # newest results/*.json
    overview(res)                # one line per question
    question(res, 0)             # the question, and how the candidates voted
    candidate(res, 0, 3)         # one candidate's facts and reasoning in full

Everything prints rather than returns, because the point is reading it.
The loaders return plain lists and dicts if you want to compute instead.
"""

from __future__ import annotations

import json
import statistics
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

WIDTH = 100
RESULTS_DIR = Path(__file__).resolve().parents[1] / "llm_monkeys" / "results"


def load(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load a step 2 results file; by default the most recently written one.

    Reading a file a run is still appending to can catch it mid-write, so
    prefer a finished run or a copy.
    """
    if path is None:
        files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"no results files in {RESULTS_DIR}")
        path = files[-1]
        print(f"# {path.name}")
    data = json.loads(Path(path).read_text())
    return data["results"] if isinstance(data, dict) and "results" in data else data


def _by_id(results: list[dict], qid: int | str) -> dict:
    for q in results:
        if str(q["question_id"]) == str(qid):
            return q
    raise KeyError(f"question {qid} is not in this run")


def _ok(q: dict) -> list[dict]:
    return [c for c in q["candidates"] if not c.get("error")]


def overview(results: list[dict]) -> None:
    """One line per question: how the candidates did and how they voted."""
    print(f"  {'id':>5} {'truth':>5} {'correct':>9} {'vote':>5} {'share':>6} "
          f"{'facts':>6} {'empty':>6}")
    for q in results:
        cands = _ok(q)
        if not cands:
            continue
        votes = Counter(c["predicted_option"] for c in cands if c["predicted_option"])
        vote, count = votes.most_common(1)[0] if votes else ("-", 0)
        n_facts = [len(c["facts"]) for c in cands]
        correct = sum(bool(c["is_correct"]) for c in cands)
        flag = "" if vote == q.get("ground_truth") else "  <- vote wrong"
        print(f"  {q['question_id']:>5} {str(q.get('ground_truth')):>5} "
              f"{correct:>4}/{len(cands):<4} {str(vote):>5} {count/len(cands)*100:5.0f}% "
              f"{statistics.mean(n_facts):6.1f} {sum(1 for n in n_facts if n == 0):>6}{flag}")


def question(results: list[dict], qid: int | str, text: bool = True) -> None:
    """The question itself, then a line per candidate."""
    q = _by_id(results, qid)
    cands = _ok(q)
    print(f"question {q['question_id']}   ground truth: {q.get('ground_truth')}")
    if text and q.get("question"):
        print()
        print(textwrap.fill(q["question"], WIDTH))
        for key, opt in sorted((q.get("options") or {}).items()):
            mark = ">" if key == q.get("ground_truth") else " "
            print(f"  {mark} {key}. {opt}")
    print()
    print(f"  {'#':>3} {'said':>4} {'ok':>3} {'facts':>6} {'tokens':>7}")
    for c in cands:
        print(f"  {c['candidate_index']:>3} {str(c['predicted_option']):>4} "
              f"{'yes' if c['is_correct'] else 'no':>3} {len(c['facts']):>6} "
              f"{c['total_tokens']:>7}")


def candidate(results: list[dict], qid: int | str, index: int = 0,
              raw: bool = False) -> None:
    """One candidate in full: its facts, then the reasoning built on them."""
    q = _by_id(results, qid)
    match = [c for c in q["candidates"] if c["candidate_index"] == index]
    if not match:
        raise KeyError(f"question {qid} has no candidate {index}")
    c = match[0]

    print(f"question {q['question_id']}  candidate {index}  "
          f"said {c['predicted_option']}, truth {q.get('ground_truth')}, "
          f"{'correct' if c['is_correct'] else 'wrong'}")
    if c.get("error"):
        print(f"  error: {c['error']}")

    print(f"\n--- facts ({len(c['facts'])}) ---")
    if not c["facts"]:
        print("  none. The model returned an empty list; the answer below rests on nothing.")
    for i, f in enumerate(c["facts"], 1):
        print(textwrap.fill(f"{i}. {f}", WIDTH, subsequent_indent="   "))

    if raw:
        print("\n--- raw fact response ---")
        print(c.get("facts_raw_response") or "(empty)")

    print("\n--- reasoning ---")
    print(c.get("answer_raw_response") or "(empty)")


def facts_stats(results: list[dict]) -> None:
    """How many facts candidates produce, and how often none at all."""
    counts = [len(c["facts"]) for q in results for c in _ok(q)]
    if not counts:
        print("  no candidates")
        return
    empty = sum(1 for n in counts if n == 0)
    print(f"  candidates {len(counts)}   mean {statistics.mean(counts):.1f}   "
          f"median {statistics.median(counts)}   max {max(counts)}")
    print(f"  produced no facts at all: {empty} ({empty/len(counts)*100:.0f}%)")
    if empty < len(counts):
        rest = [n for n in counts if n]
        print(f"  among the rest: mean {statistics.mean(rest):.1f}, "
              f"median {statistics.median(rest)}")
    hist = Counter(min(n, 20) for n in counts)
    for k in sorted(hist):
        label = f"{k}" if k < 20 else "20+"
        print(f"    {label:>3} {'#' * round(hist[k] * 50 / len(counts)):<50} {hist[k]}")


def accuracy(results: list[dict]) -> None:
    """Selector baselines: what fact verification will have to beat."""
    per_q = []
    for q in results:
        cands = _ok(q)
        if not cands:
            continue
        votes = Counter(c["predicted_option"] for c in cands if c["predicted_option"])
        per_q.append({
            "share_correct": sum(bool(c["is_correct"]) for c in cands) / len(cands),
            "any": any(c["is_correct"] for c in cands),
            "vote_right": bool(votes) and votes.most_common(1)[0][0] == q.get("ground_truth"),
        })
    n = len(per_q)
    print(f"  questions {n}")
    print(f"    a single candidate, on average   {statistics.mean(p['share_correct'] for p in per_q)*100:5.1f}%")
    print(f"    majority vote                    {sum(p['vote_right'] for p in per_q)/n*100:5.1f}%")
    print(f"    at least one correct (ceiling)   {sum(p['any'] for p in per_q)/n*100:5.1f}%")


def _count_questions(path: Path) -> int | None:
    """Questions in a results file, tolerating one that is being written.

    A run saves after every question, so a plain json.load can land mid-write.
    Counting the question_id keys textually still gives the right answer then,
    which matters because that is exactly when you want to look.
    """
    if not path.is_file():
        return None
    text = path.read_text(errors="ignore")
    try:
        data = json.loads(text)
        results = data["results"] if isinstance(data, dict) and "results" in data else data
        return len(results)
    except json.JSONDecodeError:
        return text.count('"question_id"')


def progress(total: int = 483) -> None:
    """Where the pipeline has got to. Safe to re-run at any time."""
    import subprocess
    import time

    def alive(pattern: str) -> bool:
        return subprocess.run(["pgrep", "-f", pattern],
                              capture_output=True).returncode == 0

    step2 = RESULTS_DIR / "step2.json"
    step3 = RESULTS_DIR / "step3.json"
    generated = _count_questions(step2) or 0
    judged = _count_questions(step3) or 0

    gen_running = alive("[i]nference.cli")
    judge_running = alive("[v]erifier.cli") or alive("[s]napshot_results")

    def bar(done: int, width: int = 40) -> str:
        filled = round(done / total * width)
        return "#" * filled + "." * (width - filled)

    print(f"  {time.strftime('%H:%M:%S')}")
    print(f"  generation  {bar(generated)}  {generated:>3}/{total}  "
          f"{'running' if gen_running else 'stopped'}")
    print(f"  judging     {bar(judged)}  {judged:>3}/{total}  "
          f"{'running' if judge_running else 'idle'}")

    log = RESULTS_DIR.parents[1] / "logs" / "step2.log"
    if log.is_file():
        text = log.read_text(errors="ignore")
        exhausted = text.count("Exhausted")
        if exhausted:
            print(f"  WARNING: {exhausted} candidates gave up after all retries")


__all__ = ["load", "overview", "question", "candidate", "facts_stats", "accuracy",
           "progress"]
