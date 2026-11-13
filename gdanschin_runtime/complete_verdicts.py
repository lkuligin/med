"""Judge the facts a judge left unchecked after a candidate's first wrong one.

    cd ~/Projects/med/llm_monkeys && source ../gdanschin_runtime/env.sh && \\
        ../.venv/bin/python -m gdanschin_runtime.complete_verdicts \\
        --results-dir results --run-name gemma-4-26b --judge-name glm-5.3-flash-local \\
        --judge-prompt reference --model zai-org/GLM-5.3-Flash --temperature 1.0 \\
        --max-tokens 16384 --request-timeout 900 --dataset bigbio/med_qa --concurrency 64

A judge run with early_stop_facts stops checking a candidate at its first wrong
fact. Which candidate passes does not depend on it - the rest cannot pass - but
agreement between two judges is counted fact by fact, on candidates both
judged in full, and a run stopped that way leaves mostly the candidates whose
facts all pass to compare. GLM's run on gemma-4-26b (2026-09-24) stopped so on
1,365 candidates.

The judge reads the question, the options and one fact, each fact on its own,
so the facts after the stop can be judged now as they would have been then:
the same judge and settings, given as verifier.cli takes them. The new verdicts
are appended in order; the earlier ones and every question's outcome stay as
they are. Resumable: a candidate judged in full is left alone.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from gdanschin_runtime import _bootstrap  # noqa: F401

from gdanschin_runtime.carry_verdicts import resummarise
from verifier.cli import build_config, parse_args
from verifier.workflow import VerifierWorkflow

# candidate verdict total -> the token count each fact verdict adds to it
TOKENS = {"total_tokens": "total_tokens", "total_prompt_tokens": "prompt_tokens",
          "total_candidate_tokens": "candidate_tokens"}


def cut_short(verification: dict) -> bool:
    """Whether checking stopped at a wrong fact before the last one, in order."""
    checked = verification.get("fact_verifications") or []
    return (0 < len(checked) < len(verification.get("facts") or [])
            and checked[-1].get("verdict") == 0
            and all(f.get("fact_index") is None for f in checked))


async def complete(workflow: VerifierWorkflow, path: Path, semaphore: asyncio.Semaphore) -> int:
    """Judge one candidate's unchecked facts and append the verdicts; returns
    how many were added."""
    record = json.loads(path.read_text())
    verification = record["verification"]
    facts, checked = verification["facts"], verification["fact_verifications"]
    question = json.loads((path.parent / "result.json").read_text())
    added = await asyncio.gather(*[
        workflow.verify_single_fact(
            question_text=question["question"], options=question["options"], fact=facts[i],
            session_id=f"complete_{path.parent.parent.name}_c{verification['candidate_index']}"
                       f"_f{i}_{uuid.uuid4().hex[:8]}",
            user_id=f"user_{path.parent.parent.name}", semaphore=semaphore)
        for i in range(len(checked), len(facts))])
    verification["fact_verifications"] = checked + [fv.to_dict() for fv in added]
    for total, field in TOKENS.items():
        verification[total] = (verification.get(total) or 0) + sum(getattr(fv.tokens, field) for fv in added)
    verification["facts_judged_after_stop"] = len(added)
    incoming = path.with_name(path.name + ".incoming")
    incoming.write_text(json.dumps(record, indent=2))
    incoming.replace(path)
    return len(added)


async def amain(argv: list[str] | None = None) -> int:
    config = build_config(parse_args(argv))
    config.validate()
    workflow = VerifierWorkflow(config=config)
    store = workflow.store
    paths = [path for question_id in store.candidates.questions()
             for path in store.judge_dir(question_id).glob("iteration_*.json")
             if cut_short(json.loads(path.read_text())["verification"])]
    semaphore = asyncio.Semaphore(config.concurrency)
    added = await asyncio.gather(*[complete(workflow, path, semaphore) for path in paths])
    if paths:
        resummarise(store, store.read_summary() or {}, facts_judged_after_stop=sum(added))
    print(f"{config.run_name}: {len(paths)} candidates judged in full, {sum(added)} verdicts added")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
