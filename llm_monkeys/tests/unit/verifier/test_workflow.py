"""Unit tests for verifier.workflow module."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import VerifierConfig
from verifier._dataset import Step2CandidateData, Step2QuestionData
from verifier.workflow import VerifierWorkflow


def make_mock_event(text: str, usage: Any = None) -> MagicMock:
    part = MagicMock(text=text, thought=False)
    return MagicMock(content=MagicMock(parts=[part]), usage_metadata=usage)


@pytest.mark.asyncio
async def test_workflow_verify_single_fact_success():
    mock_runner = MagicMock()
    mock_usage = MagicMock(
        prompt_token_count=120,
        candidates_token_count=35,
        total_token_count=155,
        cached_content_token_count=0,
        thoughts_token_count=0,
    )

    async def mock_run(**kwargs):
        yield make_mock_event(
            '{"is_correct": 1, "rationale": "Fact is clinically accurate."}',
            mock_usage,
        )

    mock_runner.run_async = mock_run

    config = VerifierConfig()
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)

    semaphore = asyncio.Semaphore(1)
    fv = await workflow.verify_single_fact(
        question_text="A 30-year-old female presents with fever...",
        options={"A": "Opt A", "B": "Opt B"},
        fact="Penicillin binds to penicillin-binding proteins.",
        session_id="test_sess",
        user_id="test_user",
        semaphore=semaphore,
    )

    assert fv.verdict == 1
    assert fv.is_correct is True
    assert "clinically accurate" in fv.rationale
    assert fv.tokens.total_tokens == 155
    assert fv.error is None


@pytest.mark.asyncio
async def test_workflow_verify_single_fact_failure():
    mock_runner = MagicMock()

    async def mock_run(**kwargs):
        raise RuntimeError("Service unavailable")
        yield

    mock_runner.run_async = mock_run

    config = VerifierConfig(max_retries=0, rate_limit_max_retries=0)
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)

    semaphore = asyncio.Semaphore(1)
    fv = await workflow.verify_single_fact(
        question_text="Q text",
        options={"A": "A"},
        fact="False statement",
        session_id="test_sess",
        user_id="test_user",
        semaphore=semaphore,
    )

    assert fv.verdict == 0
    assert fv.is_correct is False
    assert "Service unavailable" in fv.error


@pytest.mark.asyncio
async def test_workflow_candidate_evaluation_early_stops_on_first_valid():
    """Verify that candidate evaluation early stops immediately once a candidate has ALL facts correct."""
    mock_runner = MagicMock()

    # Candidate 0 (attempt 1): facts ["Fact A0_1", "Fact A0_2"]. Fact A0_2 fails.
    # Candidate 1 (attempt 2): facts ["Fact A1_1", "Fact A1_2"]. All pass.
    # Candidate 2 (attempt 3): facts ["Fact A2_1"]. Pass.
    # With early_stop_candidates=True (default), Candidate 2 must NEVER be evaluated!
    call_log: list[str] = []

    async def mock_run(new_message=None, **kwargs):
        prompt_text = ""
        if new_message and new_message.parts:
            prompt_text = new_message.parts[0].text

        call_log.append(prompt_text)

        if "Fact A0_2" in prompt_text:
            yield make_mock_event(
                '{"is_correct": 0, "rationale": "Fact A0_2 is false."}'
            )
        else:
            yield make_mock_event('{"is_correct": 1, "rationale": "Accurate."}')

    mock_runner.run_async = mock_run

    config = VerifierConfig()  # early_stop_candidates=True by default
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)

    q = Step2QuestionData(
        question_id="test_q1",
        question="Which medication is first-line?",
        options={"A": "Med A", "B": "Med B", "C": "Med C"},
        ground_truth="B",
        candidates=[
            Step2CandidateData(
                candidate_index=0,
                facts=["Fact A0_1", "Fact A0_2"],
                predicted_option="A",
                is_correct=False,
            ),
            Step2CandidateData(
                candidate_index=1,
                facts=["Fact A1_1", "Fact A1_2"],
                predicted_option="A",
                is_correct=False,
            ),
            Step2CandidateData(
                candidate_index=2,
                facts=["Fact A2_1"],
                predicted_option="B",
                is_correct=True,
            ),
        ],
    )

    semaphore = asyncio.Semaphore(2)
    q_res = await workflow.verify_question(q, semaphore)

    assert q_res.question_id == "test_q1"
    assert q_res.found_valid_candidate is True
    # Stopped at candidate 1 (attempt 2); candidate 2 was never evaluated!
    assert q_res.candidates_evaluated == 2
    assert len(q_res.candidate_verifications) == 2
    assert not any("Fact A2_1" in call for call in call_log)

    # Candidate 1 is selected
    assert q_res.attempt_number == 2
    assert q_res.selected_candidate_index == 1
    assert q_res.predicted_option == "A"
    assert q_res.is_correct is False

    # Candidate 0 failed all_facts_correct
    assert q_res.candidate_verifications[0].all_facts_correct is False
    # Candidate 1 passed all_facts_correct
    assert q_res.candidate_verifications[1].all_facts_correct is True

    assert q_res.num_correct_candidates == 1
    assert q_res.correct_candidates_right_answers == 0
    assert q_res.correct_candidates_wrong_answers == 1


@pytest.mark.asyncio
async def test_workflow_candidate_evaluation_evaluates_all_candidates():
    """Verify that all candidates are evaluated when early_stop_candidates is False."""
    mock_runner = MagicMock()

    call_log: list[str] = []

    async def mock_run(new_message=None, **kwargs):
        prompt_text = ""
        if new_message and new_message.parts:
            prompt_text = new_message.parts[0].text

        call_log.append(prompt_text)

        if "Fact A0_2" in prompt_text:
            yield make_mock_event(
                '{"is_correct": 0, "rationale": "Fact A0_2 is false."}'
            )
        else:
            yield make_mock_event('{"is_correct": 1, "rationale": "Accurate."}')

    mock_runner.run_async = mock_run

    config = VerifierConfig(early_stop_candidates=False)
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)

    q = Step2QuestionData(
        question_id="test_q1",
        question="Which medication is first-line?",
        options={"A": "Med A", "B": "Med B", "C": "Med C"},
        ground_truth="B",
        candidates=[
            Step2CandidateData(
                candidate_index=0,
                facts=["Fact A0_1", "Fact A0_2"],
                predicted_option="A",
                is_correct=False,
            ),
            Step2CandidateData(
                candidate_index=1,
                facts=["Fact A1_1", "Fact A1_2"],
                predicted_option="A",
                is_correct=False,
            ),
            Step2CandidateData(
                candidate_index=2,
                facts=["Fact A2_1"],
                predicted_option="B",
                is_correct=True,
            ),
        ],
    )

    semaphore = asyncio.Semaphore(2)
    q_res = await workflow.verify_question(q, semaphore)

    assert q_res.question_id == "test_q1"
    assert q_res.found_valid_candidate is True
    # Evaluated ALL 3 candidates!
    assert q_res.candidates_evaluated == 3
    assert len(q_res.candidate_verifications) == 3

    # Candidate 1 is first valid by assumptions only
    assert q_res.attempt_number == 2
    assert q_res.selected_candidate_index == 1
    assert q_res.predicted_option == "A"
    assert q_res.is_correct is False

    # Candidate 0 failed all_facts_correct
    assert q_res.candidate_verifications[0].all_facts_correct is False
    # Candidate 1 passed all_facts_correct
    assert q_res.candidate_verifications[1].all_facts_correct is True
    # Candidate 2 passed all_facts_correct
    assert q_res.candidate_verifications[2].all_facts_correct is True

    # New metrics
    assert q_res.num_correct_candidates == 2
    assert q_res.correct_candidates_right_answers == 1  # candidate 2
    assert q_res.correct_candidates_wrong_answers == 1  # candidate 1
    assert q_res.first_correct_candidate_pos_assumptions_only == 2  # candidate 1
    assert q_res.first_correct_candidate_pos_all_and_answer == 3  # candidate 2


@pytest.mark.asyncio
async def test_workflow_no_valid_candidate_found():
    """Verify behavior when no candidates pass all-fact verification."""
    mock_runner = MagicMock()

    async def mock_run(**kwargs):
        # All facts fail
        yield make_mock_event('{"is_correct": 0, "rationale": "Incorrect."}')

    mock_runner.run_async = mock_run

    config = VerifierConfig()
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)

    q = Step2QuestionData(
        question_id="q_fail",
        question="Question?",
        options={"A": "A"},
        ground_truth="A",
        candidates=[
            Step2CandidateData(
                candidate_index=0, facts=["F0"], predicted_option="A", is_correct=True
            ),
            Step2CandidateData(
                candidate_index=1, facts=["F1"], predicted_option="A", is_correct=True
            ),
        ],
    )

    semaphore = asyncio.Semaphore(1)
    q_res = await workflow.verify_question(q, semaphore)

    assert q_res.found_valid_candidate is False
    assert q_res.selected_candidate_index is None
    assert q_res.attempt_number is None
    assert q_res.predicted_option is None
    assert q_res.is_correct is None
    assert q_res.candidates_evaluated == 2


@pytest.mark.asyncio
async def test_workflow_run_with_resume_and_file_io(tmp_path: Path):
    """Test full workflow execution, persistence, and resuming."""
    mock_runner = MagicMock()

    async def mock_run(**kwargs):
        yield make_mock_event('{"is_correct": 1, "rationale": "True."}')

    mock_runner.run_async = mock_run

    config = VerifierConfig(
        results_dir=str(tmp_path),
        run_name="test-run",
        judge_name="test-judge",
        save_every_n_questions=1,
    )
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)

    questions = [
        Step2QuestionData(
            question_id="Q1",
            question="Q1 text",
            options={"A": "A"},
            ground_truth="A",
            candidates=[Step2CandidateData(0, ["Fact 1"], "A", True)],
        ),
        Step2QuestionData(
            question_id="Q2",
            question="Q2 text",
            options={"B": "B"},
            ground_truth="B",
            candidates=[Step2CandidateData(0, ["Fact 2"], "B", True)],
        ),
    ]

    summary, results = await workflow.run(questions=questions)

    assert summary.total_questions == 2
    assert summary.questions_with_valid_candidate == 2
    assert summary.correct_answers == 2
    assert len(results) == 2
    # Read back from the store and check structure
    saved = workflow.store.load()
    assert saved is not None
    assert "summary" in saved
    assert len(saved["results"]) == 2
    assert workflow.store.verdict_path("Q1", 0).is_file()

    # Second run should resume and skip already processed questions
    workflow_resume = VerifierWorkflow(config=config, verifier_runner=mock_runner)
    summary2, results2 = await workflow_resume.run(questions=questions)
    assert summary2.total_questions == 2
    assert len(results2) == 2


@pytest.mark.asyncio
async def test_workflow_continues_a_question_that_found_no_valid_candidate(tmp_path):
    """A question judged to the end without a pass resumes at its new candidates."""
    judged_facts = []

    async def mock_run(**kwargs):
        # The prompt carries the fact under verification; record what is judged.
        judged_facts.append(kwargs)
        yield make_mock_event('{"is_correct": 0, "rationale": "No."}')

    mock_runner = MagicMock()
    mock_runner.run_async = mock_run

    config = VerifierConfig(
        results_dir=str(tmp_path),
        run_name="test-run",
        judge_name="test-judge",
        save_every_n_questions=1,
    )
    question = Step2QuestionData(
        question_id="Q1",
        question="Q1 text",
        options={"A": "A"},
        ground_truth="A",
        candidates=[
            Step2CandidateData(0, ["Fact 0"], "A", True),
            Step2CandidateData(1, ["Fact 1"], "A", True),
        ],
    )

    first = VerifierWorkflow(config=config, verifier_runner=mock_runner)
    summary, results = await first.run(questions=[question])
    assert results[0].found_valid_candidate is False
    assert results[0].candidates_evaluated == 2
    judged_first_time = len(judged_facts)
    assert judged_first_time == 2

    # Two more candidates arrive for the same question
    question.candidates += [
        Step2CandidateData(2, ["Fact 2"], "A", True),
        Step2CandidateData(3, ["Fact 3"], "A", True),
    ]

    second = VerifierWorkflow(config=config, verifier_runner=mock_runner)
    summary2, results2 = await second.run(questions=[question])

    # Only the new ones cost anything, and the verdicts cover all four
    assert len(judged_facts) - judged_first_time == 2
    assert results2[0].candidates_evaluated == 4
    assert [cv.candidate_index for cv in results2[0].candidate_verifications] == [
        0,
        1,
        2,
        3,
    ]
    assert [cv.attempt_number for cv in results2[0].candidate_verifications] == [
        1,
        2,
        3,
        4,
    ]
    assert second.store.verdict_path("Q1", 3).is_file()

    # A question that did find a valid candidate is left alone
    third = VerifierWorkflow(config=config, verifier_runner=mock_runner)
    third.store.save(
        {
            "summary": None,
            "results": [
                {
                    **results2[0].to_dict(),
                    "question_id": "Q2",
                    "found_valid_candidate": True,
                }
            ],
        }
    )
    before = len(judged_facts)
    question2 = Step2QuestionData(
        question_id="Q2",
        question="Q2 text",
        options={"A": "A"},
        ground_truth="A",
        candidates=question.candidates + [Step2CandidateData(4, ["Fact 4"], "A", True)],
    )
    await third.run(questions=[question2])
    assert len(judged_facts) == before


def test_keep_listed_filters_to_the_list_whatever_the_padding():
    """A run can hold candidates for every question while only the difficult
    ones are to be judged; medbullets writes '001' in one place and '1' in
    another, and both name the same question."""
    from verifier._dataset import Step2QuestionData
    from verifier.workflow import keep_listed

    questions = [Step2QuestionData(question_id=q, question="?") for q in ("001", "002", "010", "0")]
    kept = keep_listed(questions, ["1", "010", "0"])

    assert [q.question_id for q in kept] == ["001", "010", "0"]


@pytest.mark.asyncio
async def test_questions_are_judged_side_by_side_even_with_early_stop_facts(tmp_path: Path):
    """With early_stop_facts a question has one request in flight at a time, so
    questions judged one after another left every other slot of the semaphore
    idle: concurrency 8 behaved as 1. The questions have to overlap."""
    import asyncio

    in_flight = peak = 0

    async def mock_run(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        yield make_mock_event('{"is_correct": 1, "rationale": "True."}')

    mock_runner = MagicMock()
    mock_runner.run_async = mock_run
    config = VerifierConfig(results_dir=str(tmp_path), run_name="test-run",
                            judge_name="test-judge", concurrency=4, early_stop_facts=True)
    workflow = VerifierWorkflow(config=config, verifier_runner=mock_runner)
    questions = [
        Step2QuestionData(question_id=f"Q{i}", question="?", options={"A": "A"}, ground_truth="A",
                          candidates=[Step2CandidateData(0, ["f1", "f2"], "A", True)])
        for i in range(8)
    ]

    summary, results = await workflow.run(questions=questions)

    assert peak == 4
    assert [r.question_id for r in results] == [f"Q{i}" for i in range(8)]
    assert summary.questions_with_valid_candidate == 8


def test_spread_is_one_while_questions_fill_the_slots():
    """The schedule that discards nothing: as many open questions as slots,
    one candidate each, one fact each. Widening before the questions run out
    would judge candidates that the first valid one makes unnecessary."""
    from verifier.workflow import _Spread

    spread = _Spread(16)
    for spread.active in (16, 17, 32):
        assert spread.width() == 1


def test_spread_opens_up_only_as_questions_run_out():
    """Eleven of sixteen slots would otherwise idle to the end of a run."""
    from verifier.workflow import _Spread

    spread = _Spread(16, cap=16)
    spread.active = 8
    assert spread.width() == 2
    spread.active = 3
    assert spread.width() == 5
    spread.active = 1
    assert spread.width() == 16


def test_spread_will_not_go_wider_than_its_cap():
    """A question's candidates share a prompt prefix, so cache-aware routing
    sends them to one replica and the width is that replica's queue depth. A
    deep queue of long generations is what the per-check timeout cancels, so
    how wide is safe is a property of the judge, not of the arithmetic."""
    from verifier.workflow import _Spread

    narrow = _Spread(16, cap=4)
    narrow.active = 1
    assert narrow.width() == 4, "sixteen slots free, but four is the limit"

    wide = _Spread(16, cap=16)
    wide.active = 1
    assert wide.width() == 16

    # The cap never forces more than the slots allow.
    both = _Spread(4, cap=16)
    both.active = 1
    assert both.width() == 4


def test_spread_never_returns_less_than_one():
    """A question always gets to judge something, whatever the arithmetic:
    zero would stall the run, and active is decremented by the questions
    themselves so it can be seen at zero."""
    from verifier.workflow import _Spread

    spread = _Spread(0)
    assert spread.width() == 1
    assert _Spread(16, cap=16).width() == 16
    spread = _Spread(4)
    spread.active = 100
    assert spread.width() == 1
    assert _Spread(16, cap=0).width() == 1


@pytest.mark.asyncio
async def test_facts_after_the_first_wrong_one_are_not_checked(tmp_path: Path):
    """A candidate passes only if every fact does, so a check after the first
    failure cannot change the verdict. Measured over a real run, half the
    work went that way."""
    asked: list[str] = []

    async def mock_run(new_message=None, **kwargs):
        text = new_message.parts[0].text if new_message and new_message.parts else ""
        for name in ("fact-1", "fact-2", "fact-3", "fact-4"):
            if f'"{name}"' in text:
                asked.append(name)
                break
        verdict = 0 if '"fact-2"' in text else 1
        yield make_mock_event(f'{{"is_correct": {verdict}, "rationale": "."}}')

    runner = MagicMock()
    runner.run_async = mock_run
    config = VerifierConfig(results_dir=str(tmp_path), run_name="r", judge_name="j",
                            concurrency=4, early_stop_facts=True)
    workflow = VerifierWorkflow(config=config, verifier_runner=runner)
    question = Step2QuestionData(
        question_id="Q1", question="?", options={"A": "A"}, ground_truth="A",
        candidates=[Step2CandidateData(
            0, ["fact-1", "fact-2", "fact-3", "fact-4"], "A", True)])

    summary, results = await workflow.run(questions=[question])

    assert asked == ["fact-1", "fact-2"], "fact-3 and fact-4 cannot change it"
    assert results[0].candidate_verifications[0].all_facts_correct is False
    assert summary.questions_with_valid_candidate == 0


@pytest.mark.asyncio
async def test_a_candidate_with_no_facts_does_not_stall_the_question(tmp_path: Path):
    """Step 2 can produce a candidate whose fact extraction returned nothing.
    It cannot pass, and it must not stop the question from reaching one that
    can."""
    async def mock_run(new_message=None, **kwargs):
        yield make_mock_event('{"is_correct": 1, "rationale": "."}')

    runner = MagicMock()
    runner.run_async = mock_run
    config = VerifierConfig(results_dir=str(tmp_path), run_name="r", judge_name="j",
                            concurrency=4, early_stop_facts=True)
    workflow = VerifierWorkflow(config=config, verifier_runner=runner)
    question = Step2QuestionData(
        question_id="Q1", question="?", options={"A": "A"}, ground_truth="A",
        candidates=[Step2CandidateData(0, [], "A", True),
                    Step2CandidateData(1, ["fact-1"], "A", True)])

    summary, results = await workflow.run(questions=[question])

    verdicts = results[0].candidate_verifications
    assert verdicts[0].all_facts_correct is False
    assert verdicts[1].all_facts_correct is True
    assert summary.questions_with_valid_candidate == 1


@pytest.mark.asyncio
async def test_one_fact_per_question_is_in_flight_with_early_stop(tmp_path: Path):
    """The invariant the schedule rests on: a question contributes exactly one
    request at a time, so N open questions fill N slots and no more."""
    import asyncio

    in_flight: dict[str, int] = {}
    worst = 0

    async def mock_run(new_message=None, **kwargs):
        nonlocal worst
        text = new_message.parts[0].text if new_message and new_message.parts else ""
        qid = "Q1" if "first question" in text else "Q2"
        in_flight[qid] = in_flight.get(qid, 0) + 1
        worst = max(worst, max(in_flight.values()))
        await asyncio.sleep(0.01)
        in_flight[qid] -= 1
        yield make_mock_event('{"is_correct": 1, "rationale": "."}')

    runner = MagicMock()
    runner.run_async = mock_run
    config = VerifierConfig(results_dir=str(tmp_path), run_name="r", judge_name="j",
                            concurrency=8, early_stop_facts=True)
    workflow = VerifierWorkflow(config=config, verifier_runner=runner)
    questions = [
        Step2QuestionData(question_id="Q1", question="the first question",
                          options={"A": "A"}, ground_truth="A",
                          candidates=[Step2CandidateData(0, ["a", "b", "c"], "A", True)]),
        Step2QuestionData(question_id="Q2", question="the second question",
                          options={"A": "A"}, ground_truth="A",
                          candidates=[Step2CandidateData(0, ["a", "b", "c"], "A", True)]),
    ]

    await workflow.run(questions=questions)

    assert worst == 1, "a question must never have two facts in flight"


@pytest.mark.asyncio
async def test_the_tail_is_not_speculated_on_unless_asked(tmp_path: Path):
    """The extra candidates are extra requests. On a judge that charges per
    call that is money spent to finish a few minutes sooner, so the default
    has to be to wait instead."""
    import asyncio

    in_flight = peak = 0

    async def mock_run(new_message=None, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        # Nothing passes, so every candidate of the question is reached.
        yield make_mock_event('{"is_correct": 0, "rationale": "."}')

    runner = MagicMock()
    runner.run_async = mock_run
    question = Step2QuestionData(
        question_id="Q1", question="?", options={"A": "A"}, ground_truth="A",
        candidates=[Step2CandidateData(i, ["f1"], "A", True) for i in range(8)])

    # One question, eight slots: without speculation seven stay idle.
    config = VerifierConfig(results_dir=str(tmp_path), run_name="r", judge_name="j",
                            concurrency=8, early_stop_facts=True)
    await VerifierWorkflow(config=config, verifier_runner=runner).run(
        questions=[question])
    assert peak == 1, "one question, one candidate, one request"

    in_flight = peak = 0
    config = VerifierConfig(results_dir=str(tmp_path), run_name="r2", judge_name="j",
                            concurrency=8, early_stop_facts=True,
                            speculate_tail=True, speculate_width=8)
    await VerifierWorkflow(config=config, verifier_runner=runner).run(
        questions=[question])
    assert peak == 8, "asked for, so the idle slots are filled"
