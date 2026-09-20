"""Look at a step 2 run by eye, from a notebook.

    from gdanschin_runtime.inspect_run import load, overview, question, candidate

    res = load()                 # the stored run; name the model when there are several
    res = load("gpt-oss-120b")
    overview(res)                # one line per question
    question(res, 0)             # the question, and how the candidates voted
    candidate(res, 0, 3)         # one candidate's facts and reasoning in full
    accuracy(res)                # baselines and the verified metric
    step1()                      # the one-shot baseline on its own
    verified_curve()             # the method's accuracy as k grows

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

from gdanschin_runtime import _bootstrap  # noqa: F401

from results_store import (
    FACTS_PIPELINE,
    SINGLE_STEP,
    CandidateResults,
    OneShotResults,
    VerificationResults,
)

WIDTH = 100
RESULTS_DIR = _bootstrap.LLM_MONKEYS_ROOT / "results"


def _resolve_base(base: str | None, section: str = FACTS_PIPELINE) -> str:
    """The run to look at: the only one stored, unless told which."""
    if base:
        return base
    root = RESULTS_DIR / section
    found = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"no runs stored under {root}")
    raise ValueError(f"name the base model: {', '.join(found)}")


def _resolve_judge(base: str, judge: str | None) -> str | None:
    if judge:
        return judge
    found = VerificationResults(RESULTS_DIR, base, "").judges()
    if len(found) == 1:
        return found[0]
    if not found:
        return None
    raise ValueError(f"name the judge: {', '.join(found)}")


class Run(list):
    """A run's questions, remembering which run they came from.

    A plain list would do for reading, but then accuracy() cannot find the
    matching verdicts once more than one model has been run, and asking for the
    name twice in a notebook is how the wrong two halves get compared.
    """

    def __init__(self, results, base: str) -> None:
        super().__init__(results)
        self.base = base


def load(base: str | None = None, path: str | Path | None = None) -> list[dict[str, Any]]:
    """A stored step 2 run, assembled from its per-candidate files.

    Safe to call while the run is still going: each file is written once and
    never revised, so a read mid-run sees a prefix of the run, not a half
    record. Pass `path` instead to read a single-file result from elsewhere.
    """
    if path is not None:
        data = json.loads(Path(path).read_text())
        return data["results"] if isinstance(data, dict) and "results" in data else data
    base = _resolve_base(base)
    print(f"# {base}")
    stored = CandidateResults(RESULTS_DIR, base).load()
    return Run((stored or {"results": []})["results"], base)


def load_step1(base: str | None = None) -> dict[str, dict[str, Any]]:
    """The one-shot run for a model, keyed by question id."""
    return OneShotResults(RESULTS_DIR, _resolve_base(base, SINGLE_STEP)).read()


def load_verified(base: str | None = None,
                  judge: str | None = None) -> list[dict[str, Any]]:
    """The judge's verdicts for a stored run."""
    base = _resolve_base(base)
    judge = _resolve_judge(base, judge)
    if judge is None:
        return []
    stored = VerificationResults(RESULTS_DIR, base, judge).load()
    return (stored or {"results": []})["results"]


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


def _baselines(results: list[dict]) -> dict[str, float]:
    per_q = []
    for q in results:
        cands = _ok(q)
        if not cands:
            continue
        votes = Counter(c["predicted_option"] for c in cands if c["predicted_option"])
        per_q.append({
            "share": sum(bool(c["is_correct"]) for c in cands) / len(cands),
            "any": any(c["is_correct"] for c in cands),
            "vote": bool(votes) and votes.most_common(1)[0][0] == q.get("ground_truth"),
        })
    n = max(len(per_q), 1)
    return {
        "n": len(per_q),
        "single": statistics.mean(p["share"] for p in per_q) * 100 if per_q else 0.0,
        "vote": sum(p["vote"] for p in per_q) / n * 100,
        "unverified": sum(p["any"] for p in per_q) / n * 100,
    }


def _one_shot(base: str | None, qids) -> tuple[float, int] | None:
    """One-shot accuracy over the given questions, or None if none are stored.

    Averaged over attempts rather than taken from the record's is_correct, so
    that a run with several attempts per question is measured the same way as
    "single candidate, on average" below it. With one attempt the two agree.
    """
    if not base:
        return None
    records = OneShotResults(RESULTS_DIR, base).read()
    picked = [records[str(q)] for q in qids if str(q) in records]
    if not picked:
        return None
    rate = statistics.mean(
        (r.get("correct_attempts") or 0) / max(r.get("total_attempts") or 1, 1)
        for r in picked
    )
    return rate * 100, len(picked)


def _print_baselines(b: dict, indent: str = "    ",
                     one_shot: tuple[float, int] | None = None) -> None:
    if one_shot is not None:
        rate, covered = one_shot
        # Step 1 can cover fewer questions than step 2 has reached, and then it
        # is not the same measurement - say so rather than let the numbers sit
        # in one column as though they were comparable.
        note = "" if covered == b["n"] else f"   (only {covered} of {b['n']} questions)"
        print(f"{indent}one shot, no candidates        {rate:5.1f}%{note}")
    print(f"{indent}single candidate, on average    {b['single']:5.1f}%")
    print(f"{indent}majority vote                   {b['vote']:5.1f}%")
    print(f"{indent}unverified: any correct in k    {b['unverified']:5.1f}%   (ceiling for any selector)")


def accuracy(results: list[dict], verified: list[dict] | str | Path | None = None,
             base: str | None = None, judge: str | None = None) -> None:
    """Selector baselines, and the verified metric where step 3 has caught up.

    The two sections cover different question sets on purpose. Verification
    lags generation, so comparing the headline number against baselines
    computed over every generated question would flatter or punish it by
    whichever questions happen to be judged yet. The second section restricts
    both to the same questions.
    """
    base = base or getattr(results, "base", None)
    all_ids = [q["question_id"] for q in results]

    # Step 1 over everything it has answered, not only the questions step 2 has
    # reached. The per-section lines below are restricted to matching question
    # sets so they can be compared with each other; this one says how the model
    # does on its own, over the whole run.
    if base:
        records = OneShotResults(RESULTS_DIR, base).read()
        if records:
            whole = _one_shot(base, list(records))
            print(f"  STEP 1, WHOLE RUN ({whole[1]} questions answered once)")
            print(f"    one shot, no candidates       {whole[0]:5.1f}%")
            print()
    sizes = sorted(len(q.get("candidates") or []) for q in results)
    spread = (f"{sizes[0]} candidates each" if sizes and sizes[0] == sizes[-1]
              else f"{sizes[0]}-{sizes[-1]} candidates each")
    print(f"  ALL QUESTIONS THROUGH STEP 2 ({len(results)} questions, {spread})")
    _print_baselines(_baselines(results), one_shot=_one_shot(base, all_ids))

    if verified is None:
        verified = load_verified(base, judge)
        if not verified:
            print("\n  no step 3 results yet")
            return
    if isinstance(verified, (str, Path)):
        verified = load(path=verified)

    judged = {str(q["question_id"]): q for q in verified}
    both = [q for q in results if str(q["question_id"]) in judged]
    if not both:
        print("\n  no questions have been through both steps yet")
        return

    first_valid_right = found = hybrid_right = 0
    fell_back = fallback_right = 0
    for q in both:
        cv = judged[str(q["question_id"])].get("candidate_verifications", [])
        passing = next((c for c in cv if c.get("all_facts_correct")), None)
        if passing is not None:
            found += 1
            right = bool(passing.get("is_correct"))
            first_valid_right += right
            hybrid_right += right
        else:
            # Nothing passed verification. Rather than give up, fall back to the
            # majority of the candidates - it costs no extra judging, since they
            # were all generated anyway.
            fell_back += 1
            cands = _ok(q)
            votes = Counter(c["predicted_option"] for c in cands if c["predicted_option"])
            hit = bool(votes) and votes.most_common(1)[0][0] == q.get("ground_truth")
            fallback_right += hit
            hybrid_right += hit

    n = len(both)
    print(f"\n  QUESTIONS THROUGH BOTH STEPS ({n})")
    _print_baselines(_baselines(both),
                     one_shot=_one_shot(base, [q["question_id"] for q in both]))
    print(f"    first fully verified candidate  {first_valid_right / n * 100:5.1f}%   <- the metric")
    print(f"    verified, else majority vote    {hybrid_right / n * 100:5.1f}%   <- hybrid")
    print(f"    (a valid candidate was found for {found}/{n}; when found it was right "
          f"{first_valid_right / max(found, 1) * 100:.0f}% of the time)")
    if fell_back:
        print(f"    (the fallback fired {fell_back} times and the vote was right {fallback_right})")

    # A question judged before its last candidate existed, and still without a
    # pass, is not finished: the next judging pass continues it. Until then it
    # counts against the metric, which is why a run being watched live reads
    # worse than it is.
    open_questions = 0
    for q in both:
        cv = judged[str(q["question_id"])].get("candidate_verifications", [])
        if any(c.get("all_facts_correct") for c in cv):
            continue
        if len(cv) < len(q.get("candidates") or []):
            open_questions += 1
    if open_questions:
        print(f"    ({open_questions} still being judged; they count as wrong "
              f"until a later pass finds a valid candidate)")


def step1(base: str | None = None) -> None:
    """The one-shot baseline on its own: accuracy, and what went wrong."""
    base = _resolve_base(base, SINGLE_STEP)
    records = OneShotResults(RESULTS_DIR, base).read()
    if not records:
        print(f"  nothing stored for {base}")
        return
    rate, covered = _one_shot(base, list(records))
    attempts = sum(r.get("total_attempts") or 0 for r in records.values())
    unparsed = sum(1 for r in records.values() if not r.get("predicted_option"))
    errors = sum(1 for r in records.values() if r.get("error"))
    print(f"  {base}")
    print(f"  one shot, no candidates        {rate:5.1f}%   "
          f"({covered} questions, {attempts} attempts)")
    if unparsed:
        print(f"  answers that did not parse     {unparsed}")
    if errors:
        print(f"  questions that errored         {errors}")


def _difficult_total(default: int = 483) -> int:
    """How many questions a run covers, from the list the pipeline works on."""
    from inference._dataset import load_difficult_question_ids

    for name in ("difficult_questions.csv", "data/difficult_questions.candidate.csv"):
        path = _bootstrap.LLM_MONKEYS_ROOT / name
        if path.is_file():
            return len(load_difficult_question_ids(path))
    return default


def progress(base: str | None = None, judge: str | None = None,
             total: int | None = None) -> None:
    """Where one model's run has got to. Safe to re-run at any time.

    Everything is per run: the questions come from the difficult-questions
    list, and the target number of candidates from the run's own summary, so
    two models at different k each report against their own target rather than
    against whatever the last run happened to use.
    """
    import subprocess
    import time

    def alive(pattern: str) -> bool:
        return subprocess.run(["pgrep", "-f", pattern],
                              capture_output=True).returncode == 0

    base = _resolve_base(base)
    judge = _resolve_judge(base, judge)
    # Counting directories rather than parsing anything: this is meant to be
    # re-run every few seconds while a run is in flight.
    candidates = CandidateResults(RESULTS_DIR, base)
    questions = candidates.questions()
    counts = [candidates.candidate_count(q) for q in questions]
    # A question's directory appears with its first candidate, so counting
    # directories would report a question as done the moment it starts. Only
    # the ones that reached the target count are finished.
    summary = candidates.read_summary() or {}
    target = summary.get("n_candidates") or max(counts, default=0)
    generated = sum(1 for n in counts if n >= target) if target else 0
    if total is None:
        total = _difficult_total()
    judged = sum(1 for q in questions
                 if judge and (candidates.question_dir(q) / judge).is_dir())

    gen_running = alive("[i]nference.cli")
    judge_running = alive("[v]erifier.cli")

    def bar(done: int, width: int = 40) -> str:
        filled = round(done / total * width)
        return "#" * filled + "." * (width - filled)

    print(f"  {time.strftime('%H:%M:%S')}   {base}   k={target}"
          + (f"   judged by {judge}" if judge else ""))
    in_flight = len(questions) - generated
    print(f"  generation  {bar(generated)}  {generated:>3}/{total}  "
          f"{'running' if gen_running else 'stopped'}"
          + (f"   (+{in_flight} started)" if in_flight else ""))
    print(f"  judging     {bar(judged)}  {judged:>3}/{total}  "
          f"{'running' if judge_running else 'idle'}")
    if counts:
        print(f"  candidates  {sum(counts)} stored, {target} per finished question")

    log = RESULTS_DIR.parents[1] / "logs" / "step2.log"
    if log.is_file():
        text = log.read_text(errors="ignore")
        exhausted = text.count("Exhausted")
        if exhausted:
            print(f"  WARNING: {exhausted} candidates gave up after all retries")


def verified_curve(base: str | None = None, judge: str | None = None,
                   max_k: int | None = None, plot: bool = True):
    """The method's accuracy as a function of k: judge candidates in order,
    answer with the first whose facts all check out.

    Returns {k: {curve: value}} and, by default, plots it against the baselines
    and against what the judging costs.

    Unlike vote_curve, this cannot average over random subsets of the
    candidates. The verifier stops at the first candidate that passes, so for a
    question where candidate j passed, nothing after j was ever judged: a
    random subset containing candidate 17 has no verdict to read. Prefixes are
    the one ordering the stored verdicts answer exactly, so the curves here all
    use the first k candidates as generated, including the baselines, which
    keeps them comparable with each other if not with vote_curve.

    Ties in a majority vote are given fractional credit - two options level on
    2 votes each, one of them right, counts as half - which is the expectation
    of breaking the tie at random, without the noise of actually doing it.
    """
    base = _resolve_base(base)
    judge = _resolve_judge(base, judge)
    if judge is None:
        raise FileNotFoundError(f"no verdicts stored for {base}")

    candidates_store = CandidateResults(RESULTS_DIR, base)
    verdicts_store = VerificationResults(RESULTS_DIR, base, judge)

    questions = []
    for qid in candidates_store.questions():
        verdicts = verdicts_store.read_verdicts(qid)
        if not verdicts:
            continue  # step 3 has not reached this question yet
        records = candidates_store.read_candidates(qid)
        if not records:
            continue
        truth = records[0].get("ground_truth")
        cands = [r["candidate"] for r in records]
        judged = {v["verification"]["candidate_index"]: v["verification"]
                  for v in verdicts}
        passing = sorted(i for i, v in judged.items() if v.get("all_facts_correct"))
        questions.append({
            "picks": [c.get("predicted_option") for c in cands],
            "right": [bool(c.get("is_correct")) for c in cands],
            "truth": truth,
            # The position the method stops at, and whether it was right there.
            "first_pass": passing[0] if passing else None,
            "pass_right": (bool(judged[passing[0]].get("is_correct"))
                           if passing else False),
            "judged": len(judged),
        })

    if not questions:
        print(f"  no questions have been through both steps for {base}")
        return {}

    ks = list(range(1, (max_k or max(len(q["picks"]) for q in questions)) + 1))
    curves = {name: [] for name in
              ("verified", "verified, else vote", "majority vote", "any correct",
               "single candidate")}
    cost = {"judged by verifier": [], "generated": []}

    for k in ks:
        verified = hybrid = vote = anyright = judged_cost = single = 0.0
        for q in questions:
            take = min(k, len(q["picks"]))
            found = q["first_pass"] is not None and q["first_pass"] < take
            verified += q["pass_right"] if found else 0.0

            # Majority vote over the same prefix, ties shared out.
            counts = Counter(p for p in q["picks"][:take] if p)
            if counts:
                best = max(counts.values())
                winners = [opt for opt, c in counts.items() if c == best]
                vote_score = sum(w == q["truth"] for w in winners) / len(winners)
            else:
                vote_score = 0.0
            vote += vote_score
            hybrid += q["pass_right"] if found else vote_score
            anyright += any(q["right"][:take])
            # One candidate picked at random out of the first k, which is the
            # pipeline with no selector at all.
            single += sum(q["right"][:take]) / take

            # What the method spent: judging stops at the candidate it accepts,
            # and never goes past what was judged.
            judged_cost += (q["first_pass"] + 1 if found
                            else min(q["judged"], take))

        n = len(questions)
        curves["verified"].append(verified / n * 100)
        curves["verified, else vote"].append(hybrid / n * 100)
        curves["majority vote"].append(vote / n * 100)
        curves["any correct"].append(anyright / n * 100)
        curves["single candidate"].append(single / n * 100)
        cost["judged by verifier"].append(judged_cost / n)
        cost["generated"].append(float(k))

    one_shot = _one_shot(base, [q for q in candidates_store.questions()])

    if plot:
        import matplotlib.pyplot as plt

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax.plot(ks, curves["any correct"], "s--", color="#9467bd", alpha=0.7,
                label="any correct in k (ceiling)")
        ax.plot(ks, curves["verified, else vote"], "^-", color="#2ca02c",
                label="verified, else majority vote")
        ax.plot(ks, curves["verified"], "o-", color="#1f77b4",
                label="first fully verified candidate")
        ax.plot(ks, curves["majority vote"], "-", color="#ff7f0e",
                label="majority vote")
        ax.plot(ks, curves["single candidate"], "-", color="#8c564b", alpha=0.8,
                label="single candidate, on average")
        if one_shot is not None:
            ax.axhline(one_shot[0], ls=":", color="#888888",
                       label=f"one shot, no candidates ({one_shot[0]:.1f}%)")
        ax.set_ylabel("% of questions answered correctly")
        ax.set_title(f"{base}, judged by {judge}  ({len(questions)} questions)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

        ax2.plot(ks, cost["generated"], ":", color="#888888",
                 label="candidates generated")
        ax2.plot(ks, cost["judged by verifier"], "o-", color="#1f77b4",
                 label="candidates judged, on average")
        ax2.set_xlabel("candidates considered (k)")
        ax2.set_ylabel("per question")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return {k: {**{name: values[i] for name, values in curves.items()},
                "judged": cost["judged by verifier"][i]}
            for i, k in enumerate(ks)}


def monkeys_curve(base: str | None = None, judge: str | None = None,
                  max_k: int | None = None, plot: bool = True,
                  majority_vote: bool = False) -> dict[int, float]:
    """The method's accuracy against k, and nothing else but the baseline.

    One line: answer each question with the first of its k candidates whose
    facts all check out, counting a question with no such candidate as wrong.
    The red dashed line is the same model answering once, with no candidates
    and no verification, on the same questions.

    The dashed brown line is one candidate of the same pipeline picked at
    random - facts and an answer built on them, but no verification - which
    separates what the pipeline is worth from what selecting within it is
    worth.

    majority_vote=True adds the other selector for comparison: the answer most
    of the k candidates agree on. It is off by default because it reads all k
    candidates for every question, so it answers a different question from the
    one this chart is about - what the method buys over answering once.

    Returns {k: accuracy}, the verified curve, whatever is drawn.
    verified_curve() is the same measurement with every selector and the
    judging cost alongside it.
    """
    rows = verified_curve(base, judge, max_k, plot=False)
    curve = {k: row["verified"] for k, row in rows.items()}
    single = {k: row["single candidate"] for k, row in rows.items()}
    vote = {k: row["majority vote"] for k, row in rows.items()}
    if not curve:
        return {}

    base = _resolve_base(base)
    judge = _resolve_judge(base, judge)
    candidates = CandidateResults(RESULTS_DIR, base)
    one_shot = _one_shot(base, candidates.questions())
    counted = sum(1 for q in candidates.questions()
                  if (candidates.question_dir(q) / judge).is_dir())

    if plot:
        import matplotlib.pyplot as plt

        ks = sorted(curve)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ks, [curve[k] for k in ks], "o-", color="#1f77b4",
                label="first verified")
        if majority_vote:
            ax.plot(ks, [vote[k] for k in ks], "s-", color="#ff7f0e", alpha=0.9,
                    label=f"majority vote ({vote[ks[-1]]:.1f}%)")
        ax.plot(ks, [single[k] for k in ks], "--", color="#8c564b", alpha=0.8,
                label=f"single candidate, on average ({single[ks[-1]]:.1f}%)")
        if one_shot is not None:
            ax.axhline(one_shot[0], ls="--", color="red",
                       label=f"single shot baseline ({one_shot[0]:.1f}%)")
        ax.set_xlabel("candidates considered (k)")
        ax.set_ylabel("% of questions answered correctly")
        ax.set_title(f"{base}, judged by {judge}  ({counted} questions)")
        ax.set_xticks([k for k in ks if k % 2 == 0 or k == 1])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    return curve


def vote_curve(results: list[dict], max_k: int = 20, trials: int = 40,
               seed: int = 0, plot: bool = True):
    """Majority-vote accuracy as a function of k, with single-candidate and
    pass@k for context. Returns {k: {curve: value}}.

    Each k is averaged over `trials` random subsets of the candidates rather
    than taking the first k. The candidates are i.i.d. draws, so any subset is
    as valid as another, and averaging removes the accident of ordering - with
    one ordering the curve jitters by several points and invites reading
    meaning into noise.

    Ties are broken at random, which is what a real tie means: with k=2 and two
    different answers there is nothing to choose between them, and always
    taking the first would quietly favour whichever candidate was generated
    earlier.
    """
    import random

    rng = random.Random(seed)
    ks = list(range(1, max_k + 1))
    vote_hits = {k: 0.0 for k in ks}
    any_hits = {k: 0.0 for k in ks}
    single = 0.0
    questions = 0

    for q in results:
        cands = [c for c in _ok(q) if c.get("predicted_option")]
        if not cands:
            continue
        questions += 1
        truth = q.get("ground_truth")
        picks = [c["predicted_option"] for c in cands]
        right = [p == truth for p in picks]
        single += sum(right) / len(right)

        for k in ks:
            # A question may have fewer usable candidates than k, when some
            # answers did not parse. Use everything it has rather than skipping
            # it: dropping the question instead made the curve fall at k=20,
            # which pass@k cannot do.
            take = min(k, len(picks))
            v = a = 0
            for _ in range(trials):
                sample = rng.sample(range(len(picks)), take)
                counts = Counter(picks[i] for i in sample)
                best = max(counts.values())
                winners = [opt for opt, c in counts.items() if c == best]
                v += rng.choice(winners) == truth
                a += any(right[i] for i in sample)
            vote_hits[k] += v / trials
            any_hits[k] += a / trials

    n = max(questions, 1)
    curves = {
        "majority vote": [vote_hits[k] / n * 100 for k in ks],
        "any correct (pass@k)": [any_hits[k] / n * 100 for k in ks],
        "single candidate": [single / n * 100] * len(ks),
    }

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ks, curves["majority vote"], "o-", label="majority vote", color="#1f77b4")
        ax.plot(ks, curves["any correct (pass@k)"], "s--", label="any correct (ceiling)",
                color="#9467bd", alpha=0.7)
        ax.plot(ks, curves["single candidate"], ":", label="single candidate",
                color="#888888")
        ax.set_xlabel("candidates considered (k)")
        ax.set_ylabel("% of questions answered correctly")
        ax.set_title(f"Majority vote vs k  ({questions} questions, "
                     f"averaged over {trials} random subsets)")
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {k: {name: values[i] for name, values in curves.items()}
            for i, k in enumerate(ks)}


__all__ = ["load", "load_step1", "load_verified", "overview", "question",
           "candidate", "facts_stats", "step1", "accuracy", "progress",
           "monkeys_curve", "verified_curve", "vote_curve"]
