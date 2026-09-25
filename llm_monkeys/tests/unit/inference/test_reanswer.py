"""Unit tests for re-answering stored facts with another answer prompt."""

from __future__ import annotations

import asyncio

from config import CandidateInferenceConfig
from inference._schemas import CandidateQuestionResult, CandidateResult, StepTokenUsage
from inference.reanswer import ReanswerWorkflow
from results_store import CandidateResults


def _candidate(index: int) -> CandidateResult:
    return CandidateResult(
        candidate_index=index, facts=["fact one", "fact two", "fact three"],
        facts_raw_response="raw facts", answer_raw_response="FINAL ANSWER: A",
        predicted_option="A", is_correct=False, fact_latency_seconds=1.0,
        answer_latency_seconds=1.0, total_latency_seconds=2.0,
        fact_tokens=StepTokenUsage(prompt_tokens=10, candidate_tokens=5, total_tokens=15),
        answer_tokens=StepTokenUsage(prompt_tokens=20, candidate_tokens=5, total_tokens=25),
        total_prompt_tokens=30, total_candidate_tokens=10, total_tokens=40,
    )


def _store_source(root) -> None:
    q = CandidateQuestionResult(
        question_id="7", meta_info=None, question="Q?", options={"A": "a", "B": "b"},
        ground_truth="B", ground_truth_answer="b", candidates=[_candidate(0), _candidate(1)],
    )
    q.update_aggregates()
    CandidateResults(str(root), "run", "med_qa").save({"summary": None, "results": [q.to_dict()]})


def test_reanswer_keeps_facts_replaces_answer_and_resumes(tmp_path):
    _store_source(tmp_path / "src")
    config = CandidateInferenceConfig(
        results_dir=str(tmp_path / "dst"), run_name="run", answer_prompt="cited-facts",
        dataset_name="bigbio/med_qa",
    )
    workflow = ReanswerWorkflow(config)
    prompts = []

    async def fake_step(runner, prompt, session_id, user_id):
        prompts.append(prompt)
        return "Reasoning.\nFACTS USED: 1, 3\nFINAL ANSWER: B", StepTokenUsage(prompt_tokens=7, candidate_tokens=3, total_tokens=10), 0.5, None

    workflow._invoke_step = fake_step
    assert asyncio.run(workflow.reanswer(str(tmp_path / "src"))) == 2
    assert "[1] fact one" in prompts[0]

    out = CandidateResults(str(tmp_path / "dst"), "run", "med_qa").load()
    cand = out["results"][0]["candidates"][0]
    assert cand["facts"] == ["fact one", "fact two", "fact three"]
    assert cand["predicted_option"] == "B" and cand["is_correct"] is True
    assert cand["cited_facts"] == [0, 2]
    assert cand["total_tokens"] == 15 + 10
    # The source is left as it was.
    src = CandidateResults(str(tmp_path / "src"), "run", "med_qa").load()
    assert src["results"][0]["candidates"][0]["predicted_option"] == "A"
    # A second run finds everything done.
    assert asyncio.run(ReanswerWorkflow(config).reanswer(str(tmp_path / "src"))) == 0
