"""Carry a judge's verdicts onto another copy of a run that holds the same facts.

    cd ~/Projects/med && python3 -m gdanschin_runtime.carry_verdicts \\
        --source results/experiments/authors-facts-cited-answer --target results \\
        --run gpt-oss-20b-local --judge glm-5.3-flash-local --first-valid

A judge reads the question, the options and one fact, never the answer. So
when two copies of a run hold the same facts candidate by candidate, as when
one was answered again from the other's facts, a verdict on one is a verdict
on the other. Only what the verdicts say about the answer changes, and that is
read off the target's own candidates. A candidate whose facts differ stops the
carry before anything of its question is written.

--first-valid keeps the verdicts up to the first candidate whose facts all
pass, where the authors' rule stops the judge, so the target reads as a run
judged by that rule; without it every verdict is carried. A question the
target already has verdicts for is left as it is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from gdanschin_runtime import _bootstrap

from results_store import VerificationResults

TOKENS = ("total_tokens", "total_prompt_tokens", "total_candidate_tokens")


def outcome(source: dict, verdicts: list[dict]) -> dict:
    """The question-level record the verifier writes, recomputed for these
    verdicts: the first candidate whose facts all pass is the one selected."""
    valid = [v for v in verdicts if v.get("all_facts_correct")]
    first = valid[0] if valid else {}
    first_right = next((v for v in valid if v.get("is_correct")), {})
    return {
        **source,
        **{key: sum(v.get(key) or 0 for v in verdicts) for key in TOKENS},
        "candidates_evaluated": len(verdicts),
        "found_valid_candidate": bool(valid),
        "selected_candidate_index": first.get("candidate_index"),
        "attempt_number": first.get("attempt_number"),
        "predicted_option": first.get("predicted_option"),
        "is_correct": first.get("is_correct"),
        "num_correct_candidates": len(valid),
        "correct_candidates_right_answers": sum(bool(v.get("is_correct")) for v in valid),
        "correct_candidates_wrong_answers": sum(not v.get("is_correct") for v in valid),
        "first_correct_candidate_pos_assumptions_only": first.get("attempt_number"),
        "first_correct_candidate_pos_all_and_answer": first_right.get("attempt_number"),
    }


def carry_question(src: Path, dst: Path, first_valid: bool) -> None:
    """The verdicts in judge directory src, answered by the candidates beside
    dst, written to dst whole or not at all."""
    candidates = {}
    for path in dst.parent.glob("iteration_*.json"):
        candidate = json.loads(path.read_text())["candidate"]
        candidates[candidate["candidate_index"]] = candidate
    records = sorted(((path.name, json.loads(path.read_text())) for path in src.glob("iteration_*.json")),
                     key=lambda item: item[1]["verification"]["candidate_index"])
    if first_valid:
        first = next((n for n, (_, record) in enumerate(records)
                      if record["verification"].get("all_facts_correct")), None)
        if first is not None:
            records = records[:first + 1]
    for _, record in records:
        verdict = record["verification"]
        candidate = candidates[verdict["candidate_index"]]
        if verdict.get("facts") != candidate.get("facts"):
            raise ValueError(f"{dst.parent}: candidate {verdict['candidate_index']} holds "
                             f"other facts than the ones judged in {src}")
        verdict["predicted_option"] = candidate.get("predicted_option")
        verdict["is_correct"] = bool(candidate.get("is_correct"))

    source = json.loads((src / "result.json").read_text())
    verdicts = [record["verification"] for _, record in records]
    result = outcome(source, verdicts)
    if len(verdicts) < source.get("candidates_evaluated", len(verdicts)):
        # A judge that stopped here would have spent only these candidates' time.
        result["total_latency_seconds"] = round(
            sum(v.get("total_latency_seconds") or 0 for v in verdicts), 4)

    incoming = dst.with_name(dst.name + ".incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir()
    for name, record in records:
        (incoming / name).write_text(json.dumps(record, indent=2))
    (incoming / "result.json").write_text(json.dumps(result, indent=2))
    incoming.rename(dst)


def resummarise(target: VerificationResults, stored: dict, **extra) -> None:
    """Rewrite target's run summary over every question it holds verdicts for,
    keeping from stored what the recomputation does not produce."""
    # Here rather than at the top: the schemas need pydantic, which only the
    # project venv has, and facts_matrix imports this module under any python.
    from verifier._schemas import QuestionVerificationResult, VerifierWorkflowSummary

    fresh = VerifierWorkflowSummary.from_results(
        [QuestionVerificationResult.from_dict(q) for q in target.load()["results"]],
        model=stored.get("model", ""),
        run_name=target.run_name,
        judge_name=target.judge_name,
        total_time_seconds=stored.get("total_time_seconds") or 0.0,
        temperature=stored.get("temperature"),
    )
    target.write_summary({**stored, **fresh.to_dict(), **extra})


def write_summary(source: VerificationResults, target: VerificationResults,
                  first_valid: bool) -> None:
    """The target's run summary, saying where its verdicts came from."""
    resummarise(target, source.read_summary() or {},
                created_at=datetime.now(timezone.utc).isoformat(),
                carried_from=str(source.directory),
                carried_verdicts="up to the first valid candidate" if first_valid else "all")


def carry(source: Path, target: Path, run: str, judge: str, dataset: str = "med_qa",
          first_valid: bool = False) -> int:
    """Carry judge's verdicts on run from the store source to the store target;
    returns how many questions were carried."""
    src_run = VerificationResults(source, run, judge, dataset)
    dst_run = VerificationResults(target, run, judge, dataset)
    carried = 0
    for question_id in src_run.candidates.questions():
        src, dst = src_run.judge_dir(question_id), dst_run.judge_dir(question_id)
        if (src / "result.json").is_file() and dst.parent.is_dir() and not dst.exists():
            carry_question(src, dst, first_valid)
            carried += 1
    if carried:
        write_summary(src_run, dst_run, first_valid)
    return carried


def _store(path: str) -> Path:
    store = Path(path)
    return store if store.is_absolute() else _bootstrap.LLM_MONKEYS_ROOT / store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True,
                        help="the store holding the verdicts, relative to llm_monkeys/")
    parser.add_argument("--target", required=True,
                        help="the store to carry them into, relative to llm_monkeys/")
    parser.add_argument("--run", required=True, help="the run, named alike in both stores")
    parser.add_argument("--judge", required=True, help="whose verdicts to carry")
    parser.add_argument("--dataset", default="med_qa")
    parser.add_argument("--first-valid", action="store_true",
                        help="stop at the first candidate whose facts all pass")
    args = parser.parse_args(argv)
    carried = carry(_store(args.source), _store(args.target), args.run, args.judge,
                    args.dataset, args.first_valid)
    print(f"{args.run}: {args.judge}'s verdicts on {carried} questions carried into {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
