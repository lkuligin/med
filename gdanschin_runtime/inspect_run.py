"""Look at a step 2 run by eye, from a notebook.

    from gdanschin_runtime.inspect_run import load, overview, question, candidate

    res = load()                 # the stored run; name the model when there are several
    res = load("gpt-oss-120b")
    overview(res)                # one line per question
    question(res, 0)             # the question, and how the candidates voted
    candidate(res, 0, 3)         # one candidate's facts and reasoning in full
    accuracy(res)                # baselines and the verified metric
    step1()                      # the one-shot baseline on its own
    compare_step1()              # every model's one-shot run, side by side
    catalogue()                  # every model and judge, and what is stored
    verified_curve()             # the method's accuracy as k grows

Everything prints rather than returns, because the point is reading it.
The loaders return plain lists and dicts if you want to compute instead.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

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

# Which dataset's runs a call reads when it does not say. Kept in the
# environment rather than in a module variable on purpose: a notebook running
# %autoreload re-executes this file whenever it changes, which puts a module
# variable back to its default - so use_dataset() would appear to work and the
# next call would quietly read med_qa again. The environment survives that.
DATASET_ENV = "MEDQA_INSPECT_DATASET"
DEFAULT_DATASET = "med_qa"


def current_dataset() -> str:
    """The dataset in force for calls that name none."""
    return os.environ.get(DATASET_ENV) or DEFAULT_DATASET


def use_dataset(name: str) -> None:
    """Read the runs stored for this dataset from here on.

        use_dataset('medbullets')

    Naming a dataset in a call always wins over this. catalogue() lists every
    dataset whatever it is set to.
    """
    from results_store import dataset_dir_for

    os.environ[DATASET_ENV] = dataset_dir_for(name)
    print(f"# reading runs stored under {current_dataset()}")


def datasets() -> list[str]:
    """Datasets that have runs stored."""
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())


def _ds(dataset: str | None) -> str:
    """The dataset a call should read: the one it names, else the one in force.

    Every entry point takes it, rather than reading a module-level default
    only, because %autoreload puts a module variable back to its default
    whenever this file changes - which looks like a call quietly reading the
    wrong dataset.

    Deliberately reads the environment itself rather than calling
    current_dataset(): a notebook that has had this file reloaded under it can
    hold a patched copy of this function next to a namespace that predates any
    name it would call, and the failure is a NameError in the middle of
    something unrelated. Depending on nothing but os keeps that impossible.
    """
    return dataset or os.environ.get("MEDQA_INSPECT_DATASET") or "med_qa"


def _resolve_base(base: str | None, section: str = FACTS_PIPELINE,
                  dataset: str | None = None) -> str:
    """The run to look at: the only one stored, unless told which."""
    if base:
        return base
    root = RESULTS_DIR / _ds(dataset) / section
    found = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"no runs stored under {root}")
    raise ValueError(f"name the base model: {', '.join(found)}")


def _resolve_run(base: str | None, dataset: str | None = None) -> str:
    """The run to watch, named under either step's results.

    _resolve_base looks in one section, which is right for reading a finished
    run but wrong for watching one: a plan starts with step 1, so for the first
    half hour the only directory the run has is under single-step, and
    resolving against the candidates alone would refuse to report on it.
    """
    if base:
        return base
    found: set[str] = set()
    for section in (FACTS_PIPELINE, SINGLE_STEP):
        root = RESULTS_DIR / _ds(dataset) / section
        if root.is_dir():
            found |= {d.name for d in root.iterdir() if d.is_dir()}
    if len(found) == 1:
        return found.pop()
    if not found:
        raise FileNotFoundError(f"no runs stored under {RESULTS_DIR}")
    raise ValueError(f"name the base model: {', '.join(sorted(found))}")


def _resolve_judge(base: str, judge: str | None,
                   dataset: str | None = None) -> str | None:
    if judge:
        return judge
    found = VerificationResults(RESULTS_DIR, base, "", _ds(dataset)).judges()
    if len(found) == 1:
        return found[0]
    if not found:
        return None
    raise ValueError(f"name the judge: {', '.join(found)}"
                     "   (catalogue() lists what each one is)")


class Run(list):
    """A run's questions, remembering which run they came from.

    A plain list would do for reading, but then accuracy() cannot find the
    matching verdicts once more than one model has been run, and asking for the
    name twice in a notebook is how the wrong two halves get compared.
    """

    def __init__(self, results, base: str, judge: str | None = None,
                 dataset: str | None = None) -> None:
        super().__init__(results)
        self.base = base
        self.judge = judge
        self.dataset = dataset


def load(base: str | None = None, judge: str | None = None,
         path: str | Path | None = None,
         dataset: str | None = None) -> list[dict[str, Any]]:
    """A stored step 2 run, assembled from its per-candidate files.

    Safe to call while the run is still going: each file is written once and
    never revised, so a read mid-run sees a prefix of the run, not a half
    record. Pass `path` instead to read a single-file result from elsewhere.

    Naming the judge here saves naming it again in every call that reads
    verdicts: the run remembers it, and accuracy() and the curves use it unless
    they are given another. With one judge stored it is found on its own.
    """
    if path is not None:
        data = json.loads(Path(path).read_text())
        return data["results"] if isinstance(data, dict) and "results" in data else data
    dataset = _ds(dataset)
    base = _resolve_base(base, dataset=dataset)
    stored = CandidateResults(RESULTS_DIR, base, dataset).load()
    held = (stored or {"results": []})["results"]
    results = _on_list(held, dataset)
    scope = ("" if len(results) == len(held)
             else f"   {len(results)} of {len(held)} stored questions are on the list")
    print(f"# {base}" + (f"   in {dataset}" if dataset != "med_qa" else "")
          + (f"   judged by {judge}" if judge else "") + scope)
    return Run(results, base, judge, dataset)


def _questions_on_list(store, dataset: str | None = None) -> list[str]:
    """The questions a run holds that the difficult-questions list names.

    Every reader of a step 2 or step 3 run goes through this, so that a chart
    cannot end up drawing a baseline over one set of questions and a curve
    over another.
    """
    listed = _difficult_ids(dataset)
    if listed is None:
        return store.questions()
    ids = {_norm_id(q) for q in listed}
    return [q for q in store.questions() if _norm_id(q) in ids]


def _on_list(results: list[dict[str, Any]],
             dataset: str | None = None) -> list[dict[str, Any]]:
    """The questions of a run that the difficult-questions list names.

    A run can hold more than the list names: a list can be shortened between
    runs, and nothing stored is ever thrown away. The questions it drops are
    the easy ones by definition, so leaving them in flatters every number the
    pipeline reports - seventeen points of it on medbullets - and makes the
    result incomparable with anything measured on the list.

    Steps 2 and 3 are what this covers. Step 1 answers the whole split and is
    reported over the whole split, here as everywhere else.
    """
    listed = _difficult_ids(dataset)
    if listed is None:
        return results
    ids = {_norm_id(q) for q in listed}
    return [q for q in results if _norm_id(q["question_id"]) in ids]


def load_step1(base: str | None = None,
               dataset: str | None = None) -> dict[str, dict[str, Any]]:
    """The one-shot run for a model, keyed by question id."""
    dataset = _ds(dataset)
    return OneShotResults(
        RESULTS_DIR, _resolve_base(base, SINGLE_STEP, dataset), dataset).read()


def load_verified(base: str | None = None, judge: str | None = None,
                  dataset: str | None = None) -> list[dict[str, Any]]:
    """The judge's verdicts for a stored run."""
    dataset = _ds(dataset)
    base = _resolve_base(base, dataset=dataset)
    judge = _resolve_judge(base, judge, dataset)
    if judge is None:
        return []
    stored = VerificationResults(RESULTS_DIR, base, judge, dataset).load()
    return (stored or {"results": []})["results"]


def judges(base: str | None = None, dataset: str | None = None) -> list[str]:
    """Judges that have verdicts stored for a run."""
    dataset = _ds(dataset)
    return VerificationResults(
        RESULTS_DIR, _resolve_base(base, dataset=dataset), "", dataset).judges()


def stored_runs(dataset: str | None = None) -> dict[str, dict[str, Any]]:
    """What is on disk, per run name: step 1, step 2 and who judged it.

    Keyed by run name rather than by model, because a name is what every
    function here takes - and because a run name need not be a model name: a
    model measured twice under different settings is two runs.
    """
    dataset = _ds(dataset)
    found: dict[str, dict[str, Any]] = {}
    one_shot_root = RESULTS_DIR / dataset / SINGLE_STEP
    if one_shot_root.is_dir():
        for directory in sorted(one_shot_root.iterdir()):
            if directory.is_dir():
                found.setdefault(directory.name, {})["one_shot"] = sum(
                    1 for _ in directory.glob("question_*.json"))
    candidates_root = RESULTS_DIR / dataset / FACTS_PIPELINE
    if candidates_root.is_dir():
        for directory in sorted(candidates_root.iterdir()):
            if not directory.is_dir():
                continue
            store = CandidateResults(RESULTS_DIR, directory.name, dataset)
            questions = store.questions()
            row = found.setdefault(directory.name, {})
            row["questions"] = len(questions)
            row["candidates"] = sum(store.candidate_count(q) for q in questions)
            row["k"] = (store.read_summary() or {}).get("n_candidates")
            row["judges"] = {
                judge: sum(1 for q in questions
                           if (store.question_dir(q) / judge).is_dir())
                for judge in VerificationResults(RESULTS_DIR, directory.name, "", dataset).judges()
            }
    return found


def catalogue() -> None:
    """Everything a call here can be pointed at: the registry, and what is stored.

    Two different lists, on purpose. models.py says what may be run; the
    results directory says what has been. A name in the first and not the
    second has nothing to read yet; a name in the second and not the first is
    usually a run kept under a name of its own.
    """
    from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS

    for name in datasets():
        runs = stored_runs(name)
        mark = "  <- in force" if name == current_dataset() else ""
        print(f"DATASET {name}{mark}: {', '.join(runs) if runs else 'nothing stored'}")
    print("  use_dataset(<name>) to read another\n")

    stored = stored_runs()

    def summarise(name: str) -> str:
        row = stored.get(name)
        if not row:
            return "nothing stored"
        parts = []
        if row.get("one_shot"):
            parts.append(f"step 1: {row['one_shot']} questions")
        if row.get("questions"):
            k = f" at k={row['k']}" if row.get("k") else ""
            parts.append(f"step 2: {row['questions']} questions, "
                         f"{row['candidates']} candidates{k}")
        for judge, judged in (row.get("judges") or {}).items():
            parts.append(f"{judge}: {judged} judged")
        return "   ".join(parts) or "nothing stored"

    print("BASE MODELS (models.py)      gateway model              tokens")
    for m in BASE_MODELS.values():
        print(f"  {m.name:<24} {m.gateway_model:<26} {m.max_tokens:>6}")
        print(f"      {summarise(m.name)}")

    print("\nJUDGES (models.py)           gateway model              tokens    t")
    judged_by: dict[str, list[str]] = {}
    for row_name, row in stored.items():
        for judge, count in (row.get("judges") or {}).items():
            judged_by.setdefault(judge, []).append(f"{row_name} ({count})")
    judged_by_all = list(judged_by)
    for j in JUDGE_MODELS.values():
        print(f"  {j.name:<24} {j.gateway_model:<26} {j.max_tokens:>6} {j.temperature:>4}")
        where = judged_by.pop(j.name, None)
        print(f"      verdicts for: {', '.join(where) if where else 'nothing stored'}")

    loose = [name for name in stored
             if name not in BASE_MODELS and name not in JUDGE_MODELS]
    if loose:
        print("\nSTORED UNDER A NAME OF ITS OWN (not in models.py)")
        for name in loose:
            print(f"  {name:<24} {summarise(name)}")
    if judged_by:
        print("\nJUDGES NOT IN models.py")
        for judge, where in judged_by.items():
            print(f"  {judge:<24} judged {', '.join(where)}")

    first_judge = next(iter(judged_by_all), None) or "the judge above"
    print(f"\n  pass any of these as base= or judge=, e.g. "
          f"accuracy(load('gemma-4-26b'), judge='{first_judge}')")


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
        # The same three as counts. "On average" counts a question by the share
        # of its candidates that were right, so its total is fractional.
        "single_hits": sum(p["share"] for p in per_q),
        "vote_hits": float(sum(p["vote"] for p in per_q)),
        "any_hits": float(sum(p["any"] for p in per_q)),
    }


def _norm_id(qid: Any) -> str:
    """A question id in one form, whatever store or list it came from.

    The one-shot store keys a question by the id the dataset hands it, while
    the per-candidate store and the difficult-questions list carry the id
    medbullets writes, which is padded to three digits. Compared as strings,
    '1' and '001' are different questions, so every medbullets question below
    100 - two thirds of the list - dropped out of the one-shot baseline, and
    what was left still read as a plausible number.
    """
    text = str(qid)
    return (text.lstrip("0") or "0") if text.isdigit() else text


class OneShot(NamedTuple):
    """Step 1 measured two ways, because the two answer different questions.

    `attempt0` is the first attempt and nothing else, which is what the
    author's "Single-Shot Attempt 0" means and what a baseline against the
    pipeline should be: one call, one answer.

    `averaged` is the share of attempts that were right, which is the
    like-for-like partner of "on average" over candidates. With one attempt
    per question the two are the same number.
    """

    averaged: float
    attempt0: float
    covered: int


def _one_shot(base: str | None, qids, dataset: str | None = None) -> OneShot | None:
    """One-shot accuracy over the given questions, or None if none are stored."""
    if not base:
        return None
    records = OneShotResults(RESULTS_DIR, base, _ds(dataset)).read()
    by_id = {_norm_id(k): v for k, v in records.items()}
    picked = [by_id[_norm_id(q)] for q in qids if _norm_id(q) in by_id]
    if not picked:
        return None
    averaged = statistics.mean(
        (r.get("correct_attempts") or 0) / max(r.get("total_attempts") or 1, 1)
        for r in picked
    )
    first = []
    for r in picked:
        attempts = r.get("attempts") or []
        opening = next((a for a in attempts if a.get("attempt_index") == 0), None)
        # A record written before attempts were kept one by one only has the
        # tally, and then the first attempt cannot be told from the rest.
        first.append(bool(opening.get("is_correct")) if opening
                     else bool(r.get("is_correct")))
    return OneShot(averaged * 100,
                   sum(first) / len(first) * 100,
                   len(picked))


# The names the author's analyzer prints, so that a number of ours can be put
# beside a number of his without anyone having to work out which is which.
STEP_1 = "Step 1 Baseline (Single-Shot Attempt 0)"
STEP_2_AVERAGE = "Step 2 Facts extraction Pipeline (On Average)"
STEP_2_VOTE = "Step 2 Facts extraction Pipeline (Majority vote)"
STEP_3 = "Step 3 Verifier (First Valid Candidate)"
STEP_3_HYBRID = "Step 3 Verifier (First Valid, else Majority vote)"
CEILING = "Unverified Baseline (Generator Pass@k)"
# A frontier model answering once, on the same questions. This is what the
# method competes with: fifty candidates and a judge are worth their cost only
# if they beat one call to this, so it belongs beside every score rather than
# in a comparison run by hand now and then.
REFERENCE = os.environ.get("MEDQA_REFERENCE_RUN", "gemini-3.8-flash")
LABEL = 51


def reference_label() -> str:
    return f"Reference: {REFERENCE} (Single-Shot Attempt 0)"


def _score(hits: float, n: int) -> str:
    """A rate written as the percentage and the count it came from.

    A percentage on its own hides how much is behind it: 46.1% over 165
    questions and 46.1% over 12 are the same string and not the same claim.
    """
    if not n:
        return "  n/a"
    count = f"{hits:.0f}" if abs(hits - round(hits)) < 0.05 else f"{hits:.1f}"
    return f"{hits / n * 100:5.1f}% ({count}/{n})"


def _line(label: str, hits: float, n: int, indent: str = "    ",
          note: str = "") -> None:
    print(f"{indent}• {label:<{LABEL}}: {_score(hits, n)}{note}")


def _step1_run(base: str | None, dataset: str | None = None,
               prefer_full: bool = False) -> str | None:
    """The run that holds step 1 for a model, or None if none does.

    A run over the whole split is stored apart from a run over the difficult
    questions, under <name>-full, because the same model at two coverages
    under one name would leave a directory nothing could describe. The cost is
    that a step 2 run named for the model has no step 1 of its own, and the
    baseline it should be compared against sits next door.
    """
    if not base:
        return None
    # Two callers want different runs, so the caller says which.
    #
    # A baseline for the pipeline wants the run of the same coverage: the
    # difficult-list run under the model's own name, which answers exactly
    # the questions steps 2 and 3 worked on.
    #
    # The WHOLE SPLIT block wants the split. Reading the difficult-list run
    # there printed "WHOLE SPLIT, 483 questions" for a split of 1273 - the
    # block's own heading contradicting its number.
    order = (f"{base}-full", base) if prefer_full else (base, f"{base}-full")
    for name in order:
        if OneShotResults(RESULTS_DIR, name, _ds(dataset)).read():
            return name
    return None


def _reference(qids, dataset: str | None = None,
               base: str | None = None) -> OneShot | None:
    """The reference model over the same questions, or None.

    None when the reference is what is being measured, when it has no run
    stored for this dataset, or when it did not answer these questions - a
    reference that covers half of them would be a different measurement
    printed in the same column.
    """
    if not REFERENCE or base == REFERENCE:
        return None
    measured = _one_shot(REFERENCE, qids, dataset)
    if measured is None or measured.covered < len(list(qids)):
        return None
    return measured


def _print_baselines(b: dict, indent: str = "    ",
                     one_shot: OneShot | None = None,
                     tail: tuple = (),
                     reference: OneShot | None = None) -> None:
    """The selectors in one block, weakest first, with the bound underneath.

    `tail` is whatever step 3 has to add, printed with the selectors rather
    than in a block of its own: the numbers are all over the same questions,
    and reading them down one column is the whole point.
    """
    if one_shot is not None:
        covered = one_shot.covered
        # Step 1 can cover fewer questions than step 2 has reached, and then it
        # is not the same measurement - say so rather than let the numbers sit
        # in one column as though they were comparable.
        note = "" if covered == b["n"] else f"   (only {covered} of {b['n']} questions)"
        _line(STEP_1, one_shot.attempt0 * covered / 100, covered, indent, note)
    _line(STEP_2_AVERAGE, b["single_hits"], b["n"], indent)
    _line(STEP_2_VOTE, b["vote_hits"], b["n"], indent)
    for label, hits, note in tail:
        _line(label, hits, b["n"], indent, note)
    _line(CEILING, b["any_hits"], b["n"], indent)
    if reference is not None:
        _line(reference_label(),
              reference.attempt0 * reference.covered / 100, reference.covered,
              indent)


def accuracy(results: list[dict], verified: list[dict] | str | Path | None = None,
             base: str | None = None, judge: str | None = None,
             dataset: str | None = None) -> None:
    """Selector baselines, and the verified metric where step 3 has caught up.

    The two sections cover different question sets on purpose. Verification
    lags generation, so comparing the headline number against baselines
    computed over every generated question would flatter or punish it by
    whichever questions happen to be judged yet. The second section restricts
    both to the same questions.
    """
    base = base or getattr(results, "base", None)
    judge = judge or getattr(results, "judge", None)
    dataset = _ds(dataset or getattr(results, "dataset", None))
    all_ids = [q["question_id"] for q in results]
    step1 = _step1_run(base, dataset)
    whole_run = _step1_run(base, dataset, prefer_full=True)

    # Step 1 over everything it has answered, not only the questions step 2 has
    # reached. The per-section lines below are restricted to matching question
    # sets so they can be compared with each other; this one says how the model
    # does on its own, over the whole run.
    # Say why the block is missing rather than leave it out in silence: the two
    # reasons look identical on screen and are fixed differently.
    if not base:
        print("  WHOLE SPLIT not shown: this result does not carry a run name."
              "\n    load() attaches one; a plain list of records does not."
              " Pass base='<run>'.\n")
    elif not whole_run:
        print(f"  WHOLE SPLIT not shown: no step 1 run stored as {base!r} or "
              f"{base + '-full'!r} in {dataset}.\n    one_shot_runs"
              f"('{dataset}') lists the ones there are.\n")
    else:
        records = OneShotResults(RESULTS_DIR, whole_run, dataset).read()
        whole = _one_shot(whole_run, list(records), dataset)
        named = "" if whole_run == base else f"   (run {whole_run})"
        print(f"  WHOLE SPLIT, {whole.covered} questions{named}")
        _line(STEP_1, whole.attempt0 * whole.covered / 100, whole.covered)
        print()
    sizes = sorted(len(q.get("candidates") or []) for q in results)
    spread = (f"{sizes[0]} candidates each" if sizes and sizes[0] == sizes[-1]
              else f"{sizes[0]}-{sizes[-1]} candidates each")

    if verified is None:
        verified = load_verified(base, judge, dataset)
        if not verified:
            print(f"  THROUGH STEP 2, {len(results)} questions, {spread}")
            _print_baselines(_baselines(results),
                             one_shot=_one_shot(step1, all_ids, dataset),
                             reference=_reference(all_ids, dataset, base))
            print("\n  no step 3 results yet")
            return
    if isinstance(verified, (str, Path)):
        verified = load(path=verified)

    judged = {_norm_id(q["question_id"]): q for q in verified}
    both = [q for q in results if _norm_id(q["question_id"]) in judged]
    if not both:
        print("\n  no questions have been through both steps yet")
        return
    # Step 3 lags step 2 while a run is in flight, and then the two cover
    # different questions and need blocks of their own. Once it has caught up
    # there is one set of questions and one block.
    if len(both) < len(results):
        print(f"  THROUGH STEP 2 ONLY, {len(results)} questions, {spread}")
        _print_baselines(_baselines(results),
                         one_shot=_one_shot(step1, all_ids, dataset),
                         reference=_reference(all_ids, dataset, base))
        print()

    first_valid_right = found = hybrid_right = 0
    fell_back = fallback_right = 0
    for q in both:
        cv = judged[_norm_id(q["question_id"])].get("candidate_verifications", [])
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
    print(f"  THROUGH BOTH STEPS, {n} questions, {spread}")
    fallback = (f"   (fell back {fell_back}x, right {fallback_right})"
                if fell_back else "")
    both_ids = [q["question_id"] for q in both]
    _print_baselines(
        _baselines(both),
        one_shot=_one_shot(step1, both_ids, dataset),
        reference=_reference(both_ids, dataset, base),
        tail=((STEP_3, first_valid_right, "   <- the metric"),
              (STEP_3_HYBRID, hybrid_right, fallback),
              ("Valid Coverage", found,
               f"   (right {first_valid_right / max(found, 1) * 100:.0f}% when found)")))

    # A question judged before its last candidate existed, and still without a
    # pass, is not finished: the next judging pass continues it. Until then it
    # counts against the metric, which is why a run being watched live reads
    # worse than it is.
    open_questions = 0
    for q in both:
        cv = judged[_norm_id(q["question_id"])].get("candidate_verifications", [])
        if any(c.get("all_facts_correct") for c in cv):
            continue
        if len(cv) < len(q.get("candidates") or []):
            open_questions += 1
    if open_questions:
        print(f"    ({open_questions} still being judged; they count as wrong "
              f"until a later pass finds a valid candidate)")


def step1(base: str | None = None, dataset: str | None = None) -> None:
    """The one-shot baseline on its own: accuracy, and what went wrong."""
    dataset = _ds(dataset)
    base = _resolve_base(base, SINGLE_STEP, dataset)
    records = OneShotResults(RESULTS_DIR, base, dataset).read()
    if not records:
        print(f"  nothing stored for {base}")
        return
    measured = _one_shot(base, list(records), dataset)
    attempts = sum(r.get("total_attempts") or 0 for r in records.values())
    unparsed = sum(1 for r in records.values() if not r.get("predicted_option"))
    errors = sum(1 for r in records.values() if r.get("error"))
    print(f"  {base}")
    _line(STEP_1, measured.attempt0 * measured.covered / 100, measured.covered,
          indent="  ", note=f"   ({attempts} attempts)")
    if unparsed:
        print(f"  answers that did not parse     {unparsed}")
    if errors:
        print(f"  questions that errored         {errors}")


def one_shot_runs(dataset: str | None = None) -> list[str]:
    """Every model that has a step 1 run stored."""
    root = RESULTS_DIR / _ds(dataset) / SINGLE_STEP
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def _one_shot_run(base: str, dataset: str | None = None) -> dict[str, Any]:
    """One model's step 1 run, reduced to what a comparison needs.

    The per-question score is the share of that question's attempts that were
    right, not the record's is_correct, so a run with several attempts per
    question is measured the same way as a run with one. With one attempt the
    two agree, and the score is 0 or 1.
    """
    store = OneShotResults(RESULTS_DIR, base, _ds(dataset))
    records = store.read()
    summary = store.read_summary() or {}
    scores, tokens, seconds = {}, [], []
    unparsed = errors = 0
    for qid, r in records.items():
        scores[str(qid)] = ((r.get("correct_attempts") or 0)
                            / max(r.get("total_attempts") or 1, 1))
        if r.get("total_tokens"):
            tokens.append(r["total_tokens"])
        if r.get("latency_seconds"):
            seconds.append(r["latency_seconds"])
        unparsed += not r.get("predicted_option")
        errors += bool(r.get("error"))
    return {
        "base": base,
        "scores": scores,
        "stored": len(scores),
        "attempts": summary.get("n_attempts") or 1,
        "unparsed": unparsed,
        "errors": errors,
        "tokens": statistics.mean(tokens) if tokens else 0.0,
        # Wall clock over the questions, not the mean of the records'
        # latency_seconds: that field is the time from the start of the run to
        # when a question finished, so its mean is half the run's length
        # whatever the model did, and reads as a per-question cost.
        "seconds": ((summary.get("total_time_seconds") or 0.0) / len(scores)
                    if scores and summary.get("total_time_seconds")
                    else statistics.mean(seconds) if seconds else 0.0),
        "accuracy_all": statistics.mean(scores.values()) * 100 if scores else 0.0,
    }


def _wilson(hits: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval for an accuracy, Wilson's rather than the textbook one.

    On 483 questions the textbook interval is close enough, but it misbehaves
    near 0 and 100% and can reach past them, which is how a model that never
    missed ends up reported as somewhere between 98 and 102% correct.

    With several attempts per question the interval is approximate: the unit is
    the question, and a question is a fractional hit rather than a coin flip.
    """
    if not n:
        return 0.0, 0.0
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(centre - half, 0.0) * 100, min(centre + half, 1.0) * 100


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided p for a run of wins and losses being a coin.

    Only the questions the two runs disagreed on carry information about which
    model is better - the ones they both got right, or both got wrong, say
    nothing - so the test is over those alone. With one attempt per question
    this is exactly McNemar's exact test; with several, "won" means the model
    got more of its attempts right on that question, which is the same test on
    a coarser event.
    """
    n = wins + losses
    if not n:
        return 1.0
    fewer = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(fewer + 1)) / 2 ** n
    return min(2 * tail, 1.0)


def compare_step1(runs: list[str] | None = None, plot: bool = True,
                  full: bool = False,
                  dataset: str | None = None) -> dict[str, dict[str, Any]]:
    """Compare the one-shot runs of several models, on the questions they share.

        compare_step1()                              # every stored run
        compare_step1(["gemma-4-26b", "gpt-oss-120b"])

    Restricted to the questions every run answered, because that is the only
    thing a comparison can mean: one run stopping 40 questions early would
    otherwise be credited or blamed for whichever questions it missed. Each
    run's accuracy over everything it did store is in the returned dict as
    accuracy_all, and printed too when it differs by more than a point.

    Prints a table, then each pair head to head: the questions one got and the
    other missed, and whether that gap is more than noise. Pass plot=False for
    the numbers alone, full=True to score each run on all its own questions
    instead of the shared ones.

    Returns {model: {accuracy, low, high, n, ...}} for computing on.
    """
    dataset = _ds(dataset)
    runs = runs or one_shot_runs(dataset)
    if not runs:
        raise FileNotFoundError(
            f"no step 1 runs stored under {RESULTS_DIR / dataset / SINGLE_STEP}")
    measured = [_one_shot_run(base, dataset) for base in runs]
    missing = [m["base"] for m in measured if not m["stored"]]
    if missing:
        print(f"  nothing stored for {', '.join(missing)}")
        measured = [m for m in measured if m["stored"]]
    if not measured:
        return {}

    shared = set.intersection(*(set(m["scores"]) for m in measured))
    if not shared and not full:
        print("  these runs have no questions in common")
        return {}
    for m in measured:
        ids = sorted(m["scores"], key=lambda q: int(q) if q.isdigit() else 0)
        if not full:
            ids = [q for q in ids if q in shared]
        m["ids"] = ids
        m["hits"] = sum(m["scores"][q] for q in ids)
        m["n"] = len(ids)
        m["accuracy"] = m["hits"] / m["n"] * 100 if m["n"] else 0.0
        m["low"], m["high"] = _wilson(m["hits"], m["n"])
    measured.sort(key=lambda m: m["accuracy"], reverse=True)

    if full:
        print("  ONE SHOT, each run on its own questions")
    elif len(measured) == 1:
        print(f"  ONE SHOT, {measured[0]['base']} on its {len(shared)} questions")
    else:
        print(f"  ONE SHOT, the {len(shared)} questions all "
              f"{len(measured)} runs answered")
        stored = ", ".join(f"{m['base']} {m['stored']}" for m in measured)
        if any(m["stored"] != len(shared) for m in measured):
            print(f"  (stored: {stored})")
    if len({m["attempts"] for m in measured}) > 1:
        attempts = ", ".join(f"{m['base']} x{m['attempts']}" for m in measured)
        print(f"  (attempts per question differ: {attempts})")

    width = max(len(m["base"]) for m in measured)
    print(f"\n  {'model':<{width}}  {'accuracy':>8}  {'95% CI':>13}  {'n':>4}  "
          f"{'unparsed':>8}  {'errors':>6}  {'tokens/q':>8}  {'wall s/q':>8}")
    for m in measured:
        note = ""
        if not full and abs(m["accuracy_all"] - m["accuracy"]) > 1:
            note = f"   (all {m['stored']}: {m['accuracy_all']:.1f}%)"
        print(f"  {m['base']:<{width}}  {m['accuracy']:7.1f}%  "
              f"{m['low']:5.1f} - {m['high']:5.1f}  {m['n']:>4}  "
              f"{m['unparsed']:>8}  {m['errors']:>6}  "
              f"{m['tokens']:>8.0f}  {m['seconds']:>8.2f}{note}")

    if len(measured) > 1 and shared:
        print(f"\n  HEAD TO HEAD on the shared questions")
        for i, a in enumerate(measured):
            for b in measured[i + 1:]:
                wins = sum(1 for q in shared if a["scores"][q] > b["scores"][q])
                losses = sum(1 for q in shared if a["scores"][q] < b["scores"][q])
                p = _sign_test(wins, losses)
                verdict = "clear" if p < 0.05 else "within noise"
                print(f"  {a['base']:<{width}} vs {b['base']:<{width}}  "
                      f"{a['accuracy'] - b['accuracy']:+5.1f} pts   "
                      f"won {wins:>3}, lost {losses:>3}, tied {len(shared) - wins - losses:>3}   "
                      f"p={p:.3f}  {verdict}")

        # What is left to win, and what the models already agree on: a question
        # every run gets right is not where the next point comes from.
        every = sum(1 for q in shared if all(m["scores"][q] == 1 for m in measured))
        none = sum(1 for q in shared if all(m["scores"][q] == 0 for m in measured))
        print(f"\n  every run right  {every:>4}   "
              f"no run right  {none:>4}   they differ  {len(shared) - every - none:>4}")

    if plot:
        import matplotlib.pyplot as plt

        names = [m["base"] for m in measured][::-1]
        values = [m["accuracy"] for m in measured][::-1]
        lows = [m["accuracy"] - m["low"] for m in measured][::-1]
        highs = [m["high"] - m["accuracy"] for m in measured][::-1]
        fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 2))
        ax.barh(names, values, xerr=[lows, highs], color="#1f77b4",
                error_kw={"ecolor": "#333333", "capsize": 4})
        for y, value in enumerate(values):
            ax.text(value + 0.4, y, f"{value:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("% of questions answered correctly, one shot")
        ax.set_title(f"One shot, {measured[0]['n']} questions"
                     + ("" if full else ", the same for every model"))
        ax.set_xlim(0, max(m["high"] for m in measured) + 6)
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

    return {m["base"]: {k: v for k, v in m.items() if k not in ("scores", "ids")}
            for m in measured}


def _difficult_ids(dataset: str | None = None) -> list[str] | None:
    """The questions a run covers, or None when the dataset has no list yet.

    The ids rather than a count, because a run may hold questions the current
    list no longer names - a list can be shortened between runs, and nothing
    stored is ever thrown away - and only the ids tell the two apart.

    None rather than a guess: a caller that has no list should fall back to
    what the run itself holds. Falling back to another dataset's number is how
    a bar ends up reading 483/308.
    """
    from inference._dataset import load_difficult_question_ids

    from gdanschin_runtime.fetch_dataset import KNOWN

    dataset = _ds(dataset)
    known = KNOWN.get(dataset)
    names = ([known.difficult.name] if known else []) + [
        "data/difficult_questions.candidate.csv"]
    for name in names:
        path = _bootstrap.LLM_MONKEYS_ROOT / name
        if path.is_file():
            return [str(q) for q in load_difficult_question_ids(path)]
    return None


def _split_size(dataset: str | None = None, default: int | None = None) -> int:
    """How many questions the dataset's split holds - what step 1 covers."""
    from gdanschin_runtime.fetch_dataset import KNOWN

    known = KNOWN.get(_ds(dataset))
    if known and known.rows:
        return known.rows
    if default is not None:
        return default
    raise FileNotFoundError(f"no known split size for {_ds(dataset)}")


def _one_shot_stored(base: str, dataset: str | None = None) -> int:
    """How many questions step 1 has answered, counted without parsing them.

    progress() is meant to be re-run every few seconds while a run is in
    flight, and reading all 483 records to learn how many there are would be
    the slowest thing in it. A record is written once, whole, so its existence
    is enough.
    """
    dataset = _ds(dataset)

    # The whole-split run first, because this row is measured against the
    # split. A model can have both: "<base>" from a run over the difficult
    # list and "<base>-full" over everything. Reading the list run here
    # reports its 483 answers against the split's 1273 - the one-against-the
    # -other mistake the comment further down warns about - while the split
    # run answers all 1273 and is what the row is asking about.
    for name in (f"{base}-full", base):
        directory = OneShotResults(RESULTS_DIR, name, dataset).directory
        if directory.is_dir():
            return sum(1 for _ in directory.glob("question_*.json"))
    return 0


def _progress_numbers(base: str | None, judge: str | None,
                      total: int | None, dataset: str | None) -> dict[str, Any]:
    """Where a run has got to, as numbers. Read by progress() and by watch().

    Counting files and directories rather than parsing anything: this is meant
    to be asked every few seconds while a run is in flight.
    """
    import subprocess

    def alive(pattern: str) -> bool:
        return subprocess.run(["pgrep", "-f", pattern],
                              capture_output=True).returncode == 0

    dataset = _ds(dataset)
    base = _resolve_run(base, dataset)
    judge = _resolve_judge(base, judge, dataset)
    answered = _one_shot_stored(base, dataset)
    # Same order as the count above, so the attempt count describes the run
    # the row is reporting rather than the other one.
    summary = (OneShotResults(RESULTS_DIR, f"{base}-full", dataset).read_summary()
               or OneShotResults(RESULTS_DIR, base, dataset).read_summary()
               or {})
    attempts = summary.get("n_attempts") or 1
    candidates = CandidateResults(RESULTS_DIR, base, dataset)
    questions = candidates.questions()
    # Steps 2 and 3 both cover the list the run was given, and are both counted
    # against it, so the bars are read down the page as one run. The run may
    # hold more than the list - a list can be shortened between runs, and
    # nothing stored is ever thrown away - and those questions are left out of
    # every count rather than inflating one of them: counting what the run
    # holds against the list it is working now is how a finished step comes to
    # report 308/165, and counting either against what is in the store is worse
    # still, because that denominator grows while generation runs.
    listed = None if total is not None else _difficult_ids(dataset)
    wanted = questions if listed is None else listed
    on_list = [candidates.candidate_count(q) for q in wanted]
    # A question's directory appears with its first candidate, so counting
    # directories would report a question as done the moment it starts. Only
    # the ones that reached the target count are finished.
    summary = candidates.read_summary() or {}
    target = summary.get("n_candidates") or max(on_list, default=0)
    # Two denominators, because the steps cover different things: step 1
    # answers every question in the split, while steps 2 and 3 work through the
    # list they were given. Reporting one against the other is how a bar reads
    # 300/135.
    pipeline_total = total if total is not None else len(wanted)
    if total is None:
        from gdanschin_runtime.fetch_dataset import KNOWN

        known = KNOWN.get(dataset)
        total = known.rows if known and known.rows else pipeline_total
    # Scoped to this run, not to the step: a plan runs the models one after
    # another, and an unscoped check reports that step 1 is running when it is
    # running for somebody else. Every step carries the run name on its command
    # line, so the name is what tells one model's work from another's.
    return {
        "base": base, "judge": judge, "dataset": dataset,
        "answered": answered, "attempts": attempts,
        "generated": sum(1 for n in on_list if n >= target) if target else 0,
        "started": sum(1 for n in on_list if n),
        "judged": sum(1 for q in wanted
                      if judge and (candidates.question_dir(q) / judge).is_dir()),
        "stored": sum(on_list), "target": target,
        "total": total, "pipeline_total": pipeline_total,
        "one_shot_running": alive(f"[r]un_step1.py.*--base {base}"
                                  f"|[-]m cli.*--run-name {base}"),
        "gen_running": alive(f"[i]nference.cli.*--run-name {base}"),
        "judge_running": alive(f"[v]erifier.cli.*--run-name {base}"),
    }


def progress(base: str | None = None, judge: str | None = None,
             total: int | None = None, dataset: str | None = None) -> None:
    """Where one model's run has got to, step by step. Safe to re-run at any time.

    One line per step - the one-shot answers of step 1, the candidates of step
    2, the verdicts of step 3 - all against the same question list, so the
    three bars are read down the page as one run.

    Everything else is per run too: the questions come from the
    difficult-questions list, and the target number of candidates from the
    run's own summary, so two models at different k each report against their
    own target rather than against whatever the last run happened to use.
    """
    import time

    n = _progress_numbers(base, judge, total, dataset)

    def bar(done: int, out_of: int, width: int = 40) -> str:
        filled = round(done / out_of * width) if out_of else 0
        return "#" * filled + "." * (width - filled)

    print(f"  {time.strftime('%H:%M:%S')}   {n['base']}   {n['dataset']}"
          + (f"   k={n['target']}" if n["target"] else "")
          + (f"   judged by {n['judge']}" if n["judge"] else ""))
    print(f"  one shot    {bar(n['answered'], n['total'])}  "
          f"{n['answered']:>3}/{n['total']}  "
          f"{'running' if n['one_shot_running'] else 'idle'}"
          + (f"   ({n['attempts']} attempts each)" if n["attempts"] > 1 else ""))
    in_flight = n["started"] - n["generated"]
    print(f"  generation  {bar(n['generated'], n['pipeline_total'])}  "
          f"{n['generated']:>3}/{n['pipeline_total']}  "
          f"{'running' if n['gen_running'] else 'stopped'}"
          + (f"   (+{in_flight} started)" if in_flight else ""))
    print(f"  judging     {bar(n['judged'], n['pipeline_total'])}  "
          f"{n['judged']:>3}/{n['pipeline_total']}  "
          f"{'running' if n['judge_running'] else 'idle'}")
    if n["stored"]:
        print(f"  candidates  {n['stored']} stored, {n['target']} per finished question")

    log = RESULTS_DIR.parents[1] / "logs" / "step2.log"
    if log.is_file():
        text = log.read_text(errors="ignore")
        exhausted = text.count("Exhausted")
        if exhausted:
            print(f"  WARNING: {exhausted} candidates gave up after all retries")


def watch(base: str | None = None, judge: str | None = None,
          dataset: str | None = None, every: int = 20,
          total: int | None = None) -> None:
    """Three live bars for a run, refreshing until it is done or you interrupt.

        watch('gemma-4-26b', dataset='medbullets')

    Bars rather than a printed snapshot, because the interesting part of a long
    run is the rate: tqdm keeps it, and the estimate that comes with it, from
    what the counts do between refreshes. The work itself is happening in
    another process - all this does is read what it has written so far.
    """
    import time

    from tqdm.auto import tqdm

    n = _progress_numbers(base, judge, total, dataset)
    print(f"{n['base']} · {n['dataset']}"
          + (f" · k={n['target']}" if n["target"] else "")
          + (f" · judged by {n['judge']}" if n["judge"] else ""))
    steps = [("one shot", "answered", "total"),
             ("generation", "generated", "pipeline_total"),
             ("judging", "judged", "pipeline_total")]
    # The percentage belongs on the right, with the counts: by default tqdm
    # puts it in the left label together with the description, so the label is
    # a different length on every row and no fixed width fits it.
    bar_format = "{desc} {bar} {percentage:3.0f}%  {n_fmt}/{total_fmt}{postfix}"
    width = max(len(name) for name, _, _ in steps)
    bars = [tqdm(total=n[cap], initial=n[key], desc=name.ljust(width),
                 unit="q", leave=True, bar_format=bar_format)
            for name, key, cap in steps]
    # A notebook bar is three widgets in a row - label, bar, text - and the row
    # shares its width between them, so the bar shrinks by however long the
    # text on its right happens to be. Three steps with three different texts
    # therefore draw three different bars. Fixing the first two widths makes
    # the three rows line up whatever the text says.
    for barred in bars:
        container = getattr(barred, "container", None)
        if container is None:
            continue                        # a terminal bar, nothing to lay out
        label, bar_widget = container.children[0], container.children[1]
        label.layout.width = "7em"
        bar_widget.layout.width = "22em"
        bar_widget.layout.flex = "0 0 auto"
    running = {"one shot": "one_shot_running", "generation": "gen_running",
               "judging": "judge_running"}

    try:
        while True:
            for barred, (name, key, cap) in zip(bars, steps):
                barred.total = n[cap]
                barred.n = n[key]
                barred.set_postfix_str(
                    ("running" if n[running[name]] else "idle")
                    + (f", {n['stored']} candidates" if name == "generation" else ""),
                    refresh=False)
                barred.refresh()
            done = all(n[key] >= n[cap] for _, key, cap in steps)
            if done or not any(n[flag] for flag in running.values()):
                break
            time.sleep(every)
            n = _progress_numbers(base, judge, total, dataset)
    except KeyboardInterrupt:
        pass
    finally:
        for barred in bars:
            barred.close()


def verified_curve(base: str | None = None, judge: str | None = None,
                   max_k: int | None = None, plot: bool = True,
                   dataset: str | None = None):
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
    dataset = _ds(dataset)
    base = _resolve_base(base, dataset=dataset)
    judge = _resolve_judge(base, judge, dataset)
    if judge is None:
        raise FileNotFoundError(f"no verdicts stored for {base}")

    candidates_store = CandidateResults(RESULTS_DIR, base, dataset)
    verdicts_store = VerificationResults(RESULTS_DIR, base, judge, dataset)

    # The list the run works, not everything the run happens to hold.
    on_list = _questions_on_list(candidates_store, dataset)
    questions = []
    for qid in on_list:
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
            "id": qid,
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

    one_shot = _one_shot(_step1_run(base, dataset), on_list, dataset)

    if plot:
        import matplotlib.pyplot as plt

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        n = len(questions)

        def legend(name: str, at_last_k: float) -> str:
            """A line's name and where it ends up, as a rate and a count."""
            return f"{name}: {_score(at_last_k * n / 100, n).strip()}"

        ax.plot(ks, curves["any correct"], "s--", color="#9467bd", alpha=0.7,
                label=legend(CEILING, curves["any correct"][-1]))
        ax.plot(ks, curves["verified, else vote"], "^-", color="#2ca02c",
                label=legend(STEP_3_HYBRID, curves["verified, else vote"][-1]))
        ax.plot(ks, curves["verified"], "o-", color="#1f77b4",
                label=legend(STEP_3, curves["verified"][-1]))
        ax.plot(ks, curves["majority vote"], "-", color="#ff7f0e",
                label=legend(STEP_2_VOTE, curves["majority vote"][-1]))
        ax.plot(ks, curves["single candidate"], "-", color="#8c564b", alpha=0.8,
                label=legend(STEP_2_AVERAGE, curves["single candidate"][-1]))
        if one_shot is not None:
            ax.axhline(one_shot.attempt0, ls=":", color="#888888",
                       label=legend(STEP_1, one_shot.attempt0))
        reference = _reference([q["id"] for q in questions], dataset, base)
        if reference is not None:
            ax.axhline(reference.attempt0, ls="-.", color="#d62728", alpha=0.8,
                       label=legend(reference_label(), reference.attempt0))
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
                "judged": cost["judged by verifier"][i],
                "questions": len(questions)}
            for i, k in enumerate(ks)}


def _tick_stride(count: int) -> int:
    """A stride that leaves the x axis readable however far k goes.

    Labelling every other k is fine to twenty and unreadable at a hundred,
    where fifty labels overlap into a smear. Picked from round numbers so the
    ticks land somewhere a reader expects them.
    """
    for stride in (1, 2, 5, 10, 20, 25, 50):
        if count / stride <= 14:
            return stride
    return 100


def monkeys_curve(base: str | None = None, judge: str | None = None,
                  max_k: int | None = None, plot: bool = True,
                  majority_vote: bool = False,
                  dataset: str | None = None) -> dict[int, float]:
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
    dataset = _ds(dataset)
    rows = verified_curve(base, judge, max_k, plot=False, dataset=dataset)
    curve = {k: row["verified"] for k, row in rows.items()}
    single = {k: row["single candidate"] for k, row in rows.items()}
    vote = {k: row["majority vote"] for k, row in rows.items()}
    if not curve:
        return {}

    base = _resolve_base(base, dataset=dataset)
    judge = _resolve_judge(base, judge, dataset)
    candidates = CandidateResults(RESULTS_DIR, base, dataset)
    on_list = _questions_on_list(candidates, dataset)
    one_shot = _one_shot(_step1_run(base, dataset), on_list, dataset)
    counted = sum(1 for q in on_list
                  if (candidates.question_dir(q) / judge).is_dir())

    if plot:
        import matplotlib.pyplot as plt

        ks = sorted(curve)
        fig, ax = plt.subplots(figsize=(9, 5))
        n = rows[ks[-1]]["questions"]

        def legend(name: str, at_last_k: float) -> str:
            return f"{name}: {_score(at_last_k * n / 100, n).strip()}"

        # Markers shrink as the curve lengthens: at k=100 full-size ones
        # merge into a band and hide the line they are meant to mark.
        size = 5 if len(ks) <= 30 else 2.5
        ax.plot(ks, [curve[k] for k in ks], "o-", color="#1f77b4",
                markersize=size, label=legend(STEP_3, curve[ks[-1]]))
        if majority_vote:
            ax.plot(ks, [vote[k] for k in ks], "s-", color="#ff7f0e", alpha=0.9,
                    markersize=size, label=legend(STEP_2_VOTE, vote[ks[-1]]))
        ax.plot(ks, [single[k] for k in ks], "--", color="#8c564b", alpha=0.8,
                label=legend(STEP_2_AVERAGE, single[ks[-1]]))
        if one_shot is not None:
            ax.axhline(one_shot.attempt0, ls="--", color="#888888",
                       label=legend(STEP_1, one_shot.attempt0))
        ax.set_xlabel("candidates considered (k)")
        ax.set_ylabel("% of questions answered correctly")
        ax.set_title(f"{base}, judged by {judge}  ({counted} questions)")
        stride = _tick_stride(len(ks))
        ax.set_xticks([k for k in ks
                       if k % stride == 0 or k == ks[0] or k == ks[-1]])
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


__all__ = ["use_dataset", "current_dataset", "datasets", "load", "load_step1", "load_verified", "overview", "question",
           "candidate", "facts_stats", "step1", "compare_step1", "one_shot_runs",
           "catalogue", "judges", "stored_runs", "watch",
           "accuracy", "progress", "monkeys_curve", "verified_curve", "vote_curve"]
