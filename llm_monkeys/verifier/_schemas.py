"""Data models and schemas for the MedQA fact verification workflow (Step 3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from inference._schemas import StepTokenUsage


class FactVerification(BaseModel):
    """Structured output for binary fact verification by LLM-as-a-judge."""

    is_correct: int = Field(
        description="Binary classification: 1 if the medical fact is factually and clinically correct, 0 if incorrect or false.",
        ge=0,
        le=1,
    )
    rationale: str = Field(
        default="",
        description="Brief clinical reasoning or explanation for the verdict (1 or 0).",
    )


@dataclass
class FactVerificationResult:
    """Detailed verification result for a single atomic medical fact."""

    fact: str
    verdict: int  # 0 or 1
    is_correct: bool  # True if verdict == 1
    rationale: str = ""
    raw_response: str = ""
    latency_seconds: float = 0.0
    tokens: StepTokenUsage = field(default_factory=StepTokenUsage)
    error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert FactVerificationResult to a dictionary."""
        res = asdict(self)
        res["tokens"] = self.tokens.to_dict()
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactVerificationResult:
        """Construct FactVerificationResult from dictionary."""
        tokens_data = data.get("tokens") or {}
        valid_token_fields = {
            k: v for k, v in tokens_data.items() if k in StepTokenUsage.__annotations__
        }
        return cls(
            fact=str(data.get("fact", "")),
            verdict=int(data.get("verdict", 0)),
            is_correct=bool(data.get("is_correct", False)),
            rationale=str(data.get("rationale", "")),
            raw_response=str(data.get("raw_response", "")),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            tokens=StepTokenUsage(**valid_token_fields),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class CandidateVerificationResult:
    """Detailed verification result for a candidate's complete set of facts."""

    candidate_index: int
    attempt_number: int  # 1-indexed attempt number
    facts: list[str]
    fact_verifications: list[FactVerificationResult]
    all_facts_correct: bool
    predicted_option: str | None
    ground_truth: str
    is_correct: bool  # whether candidate's answer matches ground_truth
    total_latency_seconds: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_candidate_tokens: int = 0
    error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateVerificationResult to a dictionary."""
        res = asdict(self)
        res["fact_verifications"] = [fv.to_dict() for fv in self.fact_verifications]
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateVerificationResult:
        """Construct CandidateVerificationResult from dictionary."""
        fvs = [
            FactVerificationResult.from_dict(fv)
            for fv in data.get("fact_verifications") or []
        ]
        return cls(
            candidate_index=int(data.get("candidate_index", 0)),
            attempt_number=int(data.get("attempt_number", 1)),
            facts=list(data.get("facts") or []),
            fact_verifications=fvs,
            all_facts_correct=bool(data.get("all_facts_correct", False)),
            predicted_option=data.get("predicted_option"),
            ground_truth=str(data.get("ground_truth", "")),
            is_correct=bool(data.get("is_correct", False)),
            total_latency_seconds=float(data.get("total_latency_seconds", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
            total_candidate_tokens=int(data.get("total_candidate_tokens", 0)),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class QuestionVerificationResult:
    """Verification outcome for a single question across evaluated candidates."""

    question_id: str
    meta_info: str | None
    question: str
    options: dict[str, str]
    ground_truth: str
    ground_truth_answer: str | None
    candidates_evaluated: int
    found_valid_candidate: bool
    selected_candidate_index: int | None
    attempt_number: (
        int | None
    )  # 1-indexed attempt number of the first candidate with all facts correct
    predicted_option: str | None
    is_correct: (
        bool | None
    )  # whether selected candidate's answer is correct against ground truth
    candidate_verifications: list[CandidateVerificationResult] = field(
        default_factory=list
    )
    total_latency_seconds: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_candidate_tokens: int = 0
    num_correct_candidates: int = 0
    correct_candidates_right_answers: int = 0
    correct_candidates_wrong_answers: int = 0
    first_correct_candidate_pos_assumptions_only: int | None = None
    first_correct_candidate_pos_all_and_answer: int | None = None
    error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert QuestionVerificationResult to a dictionary."""
        res = asdict(self)
        res["candidate_verifications"] = [
            cv.to_dict() for cv in self.candidate_verifications
        ]
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionVerificationResult:
        """Construct QuestionVerificationResult from dictionary."""
        cvs = [
            CandidateVerificationResult.from_dict(cv)
            for cv in data.get("candidate_verifications") or []
        ]
        return cls(
            question_id=str(data.get("question_id", "")),
            meta_info=data.get("meta_info"),
            question=str(data.get("question", "")),
            options=dict(data.get("options") or {}),
            ground_truth=str(data.get("ground_truth", "")),
            ground_truth_answer=data.get("ground_truth_answer"),
            candidates_evaluated=int(data.get("candidates_evaluated", 0)),
            found_valid_candidate=bool(data.get("found_valid_candidate", False)),
            selected_candidate_index=data.get("selected_candidate_index"),
            attempt_number=data.get("attempt_number"),
            predicted_option=data.get("predicted_option"),
            is_correct=data.get("is_correct"),
            candidate_verifications=cvs,
            total_latency_seconds=float(data.get("total_latency_seconds", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
            total_candidate_tokens=int(data.get("total_candidate_tokens", 0)),
            num_correct_candidates=int(data.get("num_correct_candidates", 0)),
            correct_candidates_right_answers=int(
                data.get("correct_candidates_right_answers", 0)
            ),
            correct_candidates_wrong_answers=int(
                data.get("correct_candidates_wrong_answers", 0)
            ),
            first_correct_candidate_pos_assumptions_only=data.get(
                "first_correct_candidate_pos_assumptions_only"
            ),
            first_correct_candidate_pos_all_and_answer=data.get(
                "first_correct_candidate_pos_all_and_answer"
            ),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class VerifierWorkflowSummary:
    """Aggregate statistics and metrics for the verification workflow run."""

    model: str
    input_filepath: str
    output_filepath: str
    total_questions: int
    completed_questions: int
    failed_questions: int
    questions_with_valid_candidate: int
    correct_answers: int
    accuracy: float
    overall_accuracy: float
    average_attempts_to_valid: float
    total_candidates_evaluated: int
    total_facts_verified: int
    total_facts_correct: int
    total_facts_incorrect: int
    total_time_seconds: float
    average_question_latency_seconds: float
    total_tokens: int
    total_prompt_tokens: int
    total_candidate_tokens: int
    total_correct_candidates: int = 0
    total_correct_candidates_right_answers: int = 0
    total_correct_candidates_wrong_answers: int = 0
    first_positions_assumptions_only: dict[str, int | None] = field(
        default_factory=dict
    )
    first_positions_all_and_answer: dict[str, int | None] = field(default_factory=dict)
    avg_position_assumptions_only: float = 0.0
    avg_position_all_and_answer: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)

    @classmethod
    def from_results(
        cls,
        results: list[QuestionVerificationResult],
        model: str,
        input_filepath: str,
        output_filepath: str,
        total_time_seconds: float,
    ) -> VerifierWorkflowSummary:
        """Compute aggregate summary from question verification results."""
        total_questions = len(results)
        completed_questions = sum(1 for r in results if not r.error)
        failed_questions = sum(1 for r in results if r.error)
        questions_with_valid = sum(1 for r in results if r.found_valid_candidate)
        correct_answers = sum(
            1 for r in results if r.found_valid_candidate and r.is_correct is True
        )

        accuracy = (
            round(correct_answers / questions_with_valid, 4)
            if questions_with_valid > 0
            else 0.0
        )
        overall_accuracy = (
            round(correct_answers / total_questions, 4) if total_questions > 0 else 0.0
        )

        attempts = [
            r.attempt_number
            for r in results
            if r.found_valid_candidate and r.attempt_number is not None
        ]
        avg_attempts = round(sum(attempts) / len(attempts), 2) if attempts else 0.0

        total_candidates_eval = sum(r.candidates_evaluated for r in results)
        total_facts = sum(
            len(cv.fact_verifications)
            for r in results
            for cv in r.candidate_verifications
        )
        total_facts_correct = sum(
            1
            for r in results
            for cv in r.candidate_verifications
            for fv in cv.fact_verifications
            if fv.verdict == 1
        )
        total_facts_incorrect = sum(
            1
            for r in results
            for cv in r.candidate_verifications
            for fv in cv.fact_verifications
            if fv.verdict == 0
        )

        total_tokens = sum(r.total_tokens for r in results)
        total_prompt_tokens = sum(r.total_prompt_tokens for r in results)
        total_candidate_tokens = sum(r.total_candidate_tokens for r in results)
        avg_latency = (
            round(total_time_seconds / total_questions, 4)
            if total_questions > 0
            else 0.0
        )

        total_correct_cands = sum(r.num_correct_candidates for r in results)
        total_right_ans = sum(r.correct_candidates_right_answers for r in results)
        total_wrong_ans = sum(r.correct_candidates_wrong_answers for r in results)

        first_positions_assumptions_only = {
            r.question_id: r.first_correct_candidate_pos_assumptions_only
            for r in results
        }
        first_positions_all_and_answer = {
            r.question_id: r.first_correct_candidate_pos_all_and_answer for r in results
        }

        valid_positions_assump = [
            pos for pos in first_positions_assumptions_only.values() if pos is not None
        ]
        avg_pos_assump = (
            round(sum(valid_positions_assump) / len(valid_positions_assump), 2)
            if valid_positions_assump
            else 0.0
        )

        valid_positions_all_and_ans = [
            pos for pos in first_positions_all_and_answer.values() if pos is not None
        ]
        avg_pos_all_and_ans = (
            round(
                sum(valid_positions_all_and_ans) / len(valid_positions_all_and_ans),
                2,
            )
            if valid_positions_all_and_ans
            else 0.0
        )

        return cls(
            model=model,
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            total_questions=total_questions,
            completed_questions=completed_questions,
            failed_questions=failed_questions,
            questions_with_valid_candidate=questions_with_valid,
            correct_answers=correct_answers,
            accuracy=accuracy,
            overall_accuracy=overall_accuracy,
            average_attempts_to_valid=avg_attempts,
            total_candidates_evaluated=total_candidates_eval,
            total_facts_verified=total_facts,
            total_facts_correct=total_facts_correct,
            total_facts_incorrect=total_facts_incorrect,
            total_time_seconds=round(total_time_seconds, 2),
            average_question_latency_seconds=avg_latency,
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_candidate_tokens=total_candidate_tokens,
            total_correct_candidates=total_correct_cands,
            total_correct_candidates_right_answers=total_right_ans,
            total_correct_candidates_wrong_answers=total_wrong_ans,
            first_positions_assumptions_only=first_positions_assumptions_only,
            first_positions_all_and_answer=first_positions_all_and_answer,
            avg_position_assumptions_only=avg_pos_assump,
            avg_position_all_and_answer=avg_pos_all_and_ans,
        )


__all__ = [
    "FactVerification",
    "FactVerificationResult",
    "CandidateVerificationResult",
    "QuestionVerificationResult",
    "VerifierWorkflowSummary",
]
