"""Unit tests for verifier._schemas module."""

from __future__ import annotations

import pytest

from inference._schemas import StepTokenUsage
from verifier._schemas import (
    CandidateVerificationResult,
    FactTruthfulness,
    FactVerification,
    FactVerificationResult,
    QuestionVerificationResult,
    VerifierWorkflowSummary,
)


def test_fact_truthfulness_enum():
    assert FactTruthfulness.YES == "YES"
    assert FactTruthfulness.NO == "NO"
    assert str(FactTruthfulness.YES) == "YES"
    assert str(FactTruthfulness.NO) == "NO"


def test_fact_verification_pydantic_schema():
    # Direct enum strings
    v_yes = FactVerification(is_correct="YES", rationale="Accurate mechanism.")
    assert v_yes.is_correct == FactTruthfulness.YES
    assert v_yes.rationale == "Accurate mechanism."

    v_no = FactVerification(is_correct="NO", rationale="False statement.")
    assert v_no.is_correct == FactTruthfulness.NO

    # Coercions from integer 1/0, boolean, and lowercase strings
    assert FactVerification(is_correct=1).is_correct == FactTruthfulness.YES
    assert FactVerification(is_correct=0).is_correct == FactTruthfulness.NO
    assert FactVerification(is_correct=True).is_correct == FactTruthfulness.YES
    assert FactVerification(is_correct=False).is_correct == FactTruthfulness.NO
    assert FactVerification(is_correct="yes").is_correct == FactTruthfulness.YES
    assert FactVerification(is_correct="no").is_correct == FactTruthfulness.NO

    # JSON Schema enforcement
    schema = FactVerification.model_json_schema()
    assert schema["$defs"]["FactTruthfulness"]["enum"] == ["YES", "NO"]

    # Rejection of invalid values
    with pytest.raises(Exception):
        FactVerification(is_correct=2)
    with pytest.raises(Exception):
        FactVerification(is_correct="MAYBE")
    with pytest.raises(Exception):
        FactVerification(is_correct="uncertain")


def test_fact_verification_result_serialization():
    tokens = StepTokenUsage(prompt_tokens=50, candidate_tokens=20, total_tokens=70)
    fv = FactVerificationResult(
        fact="Penicillin inhibits cell wall synthesis.",
        verdict=1,
        is_correct=True,
        rationale="Correct bactericidal mechanism.",
        raw_response='{"is_correct": 1}',
        latency_seconds=1.23,
        tokens=tokens,
    )

    data = fv.to_dict()
    assert data["fact"] == "Penicillin inhibits cell wall synthesis."
    assert data["verdict"] == 1
    assert data["is_correct"] is True
    assert data["tokens"]["total_tokens"] == 70

    restored = FactVerificationResult.from_dict(data)
    assert restored.fact == fv.fact
    assert restored.verdict == fv.verdict
    assert restored.is_correct == fv.is_correct
    assert restored.tokens.total_tokens == 70
    assert fv.truthfulness == FactTruthfulness.YES
    assert restored.truthfulness == FactTruthfulness.YES

    fv_false = FactVerificationResult(
        fact="Penicillin is a macrolide.",
        verdict=0,
        is_correct=False,
    )
    assert fv_false.truthfulness == FactTruthfulness.NO


def test_candidate_verification_result_serialization():
    fv1 = FactVerificationResult(
        fact="Fact 1",
        verdict=1,
        is_correct=True,
        tokens=StepTokenUsage(prompt_tokens=10, candidate_tokens=5, total_tokens=15),
    )
    fv2 = FactVerificationResult(
        fact="Fact 2",
        verdict=1,
        is_correct=True,
        tokens=StepTokenUsage(prompt_tokens=10, candidate_tokens=5, total_tokens=15),
    )

    cv = CandidateVerificationResult(
        candidate_index=0,
        attempt_number=1,
        facts=["Fact 1", "Fact 2"],
        fact_verifications=[fv1, fv2],
        all_facts_correct=True,
        predicted_option="A",
        ground_truth="A",
        is_correct=True,
        total_latency_seconds=2.5,
        total_tokens=30,
        total_prompt_tokens=20,
        total_candidate_tokens=10,
    )

    d = cv.to_dict()
    assert d["all_facts_correct"] is True
    assert len(d["fact_verifications"]) == 2

    restored = CandidateVerificationResult.from_dict(d)
    assert restored.candidate_index == 0
    assert restored.attempt_number == 1
    assert restored.all_facts_correct is True
    assert restored.is_correct is True
    assert len(restored.fact_verifications) == 2


def test_question_verification_result_serialization():
    cv = CandidateVerificationResult(
        candidate_index=1,
        attempt_number=2,
        facts=["Fact 1"],
        fact_verifications=[],
        all_facts_correct=True,
        predicted_option="C",
        ground_truth="C",
        is_correct=True,
    )
    qv = QuestionVerificationResult(
        question_id="42",
        meta_info="step1",
        question="What is the treatment?",
        options={"A": "Drug A", "C": "Drug C"},
        ground_truth="C",
        ground_truth_answer="Drug C",
        candidates_evaluated=2,
        found_valid_candidate=True,
        selected_candidate_index=1,
        attempt_number=2,
        predicted_option="C",
        is_correct=True,
        candidate_verifications=[cv],
        total_latency_seconds=5.0,
        total_tokens=100,
        total_prompt_tokens=70,
        total_candidate_tokens=30,
    )

    d = qv.to_dict()
    assert d["question_id"] == "42"
    assert d["found_valid_candidate"] is True
    assert d["attempt_number"] == 2
    assert d["is_correct"] is True

    restored = QuestionVerificationResult.from_dict(d)
    assert restored.question_id == "42"
    assert restored.attempt_number == 2
    assert restored.is_correct is True
    assert len(restored.candidate_verifications) == 1


def test_verifier_workflow_summary_from_results():
    fv_pass = FactVerificationResult(
        fact="F1", verdict=1, is_correct=True, tokens=StepTokenUsage(10, 5, 15)
    )
    fv_fail = FactVerificationResult(
        fact="F2", verdict=0, is_correct=False, tokens=StepTokenUsage(10, 5, 15)
    )

    # Q1: Passed on attempt 2, correct answer
    cv1_fail = CandidateVerificationResult(
        candidate_index=0,
        attempt_number=1,
        facts=["F2"],
        fact_verifications=[fv_fail],
        all_facts_correct=False,
        predicted_option="A",
        ground_truth="B",
        is_correct=False,
    )
    cv1_pass = CandidateVerificationResult(
        candidate_index=1,
        attempt_number=2,
        facts=["F1"],
        fact_verifications=[fv_pass],
        all_facts_correct=True,
        predicted_option="B",
        ground_truth="B",
        is_correct=True,
    )
    q1 = QuestionVerificationResult(
        question_id="1",
        meta_info=None,
        question="Q1?",
        options={"A": "Opt A", "B": "Opt B"},
        ground_truth="B",
        ground_truth_answer="Opt B",
        candidates_evaluated=2,
        found_valid_candidate=True,
        selected_candidate_index=1,
        attempt_number=2,
        predicted_option="B",
        is_correct=True,
        candidate_verifications=[cv1_fail, cv1_pass],
        total_tokens=30,
        total_prompt_tokens=20,
        total_candidate_tokens=10,
    )

    # Q2: Found valid candidate on attempt 1, but answer incorrect
    cv2_pass = CandidateVerificationResult(
        candidate_index=0,
        attempt_number=1,
        facts=["F1"],
        fact_verifications=[fv_pass],
        all_facts_correct=True,
        predicted_option="A",
        ground_truth="C",
        is_correct=False,
    )
    q2 = QuestionVerificationResult(
        question_id="2",
        meta_info=None,
        question="Q2?",
        options={"A": "Opt A", "C": "Opt C"},
        ground_truth="C",
        ground_truth_answer="Opt C",
        candidates_evaluated=1,
        found_valid_candidate=True,
        selected_candidate_index=0,
        attempt_number=1,
        predicted_option="A",
        is_correct=False,
        candidate_verifications=[cv2_pass],
        total_tokens=15,
        total_prompt_tokens=10,
        total_candidate_tokens=5,
    )

    # Q3: No valid candidate found
    cv3_fail = CandidateVerificationResult(
        candidate_index=0,
        attempt_number=1,
        facts=["F2"],
        fact_verifications=[fv_fail],
        all_facts_correct=False,
        predicted_option="A",
        ground_truth="A",
        is_correct=True,
    )
    q3 = QuestionVerificationResult(
        question_id="3",
        meta_info=None,
        question="Q3?",
        options={"A": "Opt A"},
        ground_truth="A",
        ground_truth_answer="Opt A",
        candidates_evaluated=1,
        found_valid_candidate=False,
        selected_candidate_index=None,
        attempt_number=None,
        predicted_option=None,
        is_correct=None,
        candidate_verifications=[cv3_fail],
        total_tokens=15,
        total_prompt_tokens=10,
        total_candidate_tokens=5,
    )

    summary = VerifierWorkflowSummary.from_results(
        results=[q1, q2, q3],
        model="vertex_ai/gemini-3-flash-preview",
        run_name="test-run",
        judge_name="test-judge",
        total_time_seconds=12.5,
    )

    assert summary.total_questions == 3
    assert summary.completed_questions == 3
    assert summary.failed_questions == 0
    assert summary.questions_with_valid_candidate == 2
    assert summary.correct_answers == 1  # only Q1 had correct answer among valid
    assert summary.accuracy == 0.5  # 1/2
    assert summary.overall_accuracy == round(1 / 3, 4)
    assert summary.average_attempts_to_valid == 1.5  # (2 + 1) / 2
    assert summary.total_candidates_evaluated == 4
    assert summary.total_facts_verified == 4
    assert summary.total_facts_correct == 2
    assert summary.total_facts_incorrect == 2
    assert summary.total_tokens == 60
    assert summary.total_time_seconds == 12.5
