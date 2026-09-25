"""Data models and schemas for candidate inference workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field


class MedicalFacts(BaseModel):
    """Structured output representing atomic verifiable medical facts relevant to answering a question."""

    facts: list[str] = Field(
        description=(
            "A list of atomic, verifiable medical facts, clinical principles, "
            "pathophysiological mechanisms, or pharmacological rules relevant to answering the question."
        )
    )

    def to_formatted_bullet_points(self) -> str:
        """Format facts into bullet points."""
        if not self.facts:
            return "No specific facts extracted."
        return "\n".join(f"- {f.strip()}" for f in self.facts if f.strip())


@dataclass
class StepTokenUsage:
    """Token usage counters for a single inference step."""

    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    thoughts_tokens: int = 0

    @classmethod
    def from_usage_metadata(
        cls, usage: types.GenerateContentResponseUsageMetadata | None
    ) -> StepTokenUsage:
        """Construct StepTokenUsage from ADK / Gemini usage metadata."""
        if usage is None:
            return cls()

        def _get_int(val: Any) -> int:
            if isinstance(val, (int, float)):
                return int(val)
            return 0

        prompt = _get_int(getattr(usage, "prompt_token_count", 0))
        candidate = _get_int(getattr(usage, "candidates_token_count", 0))
        total = _get_int(getattr(usage, "total_token_count", 0))
        cached = _get_int(getattr(usage, "cached_content_token_count", 0))

        thoughts_val = getattr(usage, "thoughts_token_count", None)
        if thoughts_val is None:
            thoughts_val = getattr(usage, "reasoning_tokens", 0)
        thoughts = _get_int(thoughts_val)

        return cls(
            prompt_tokens=prompt,
            candidate_tokens=candidate,
            total_tokens=total,
            cached_tokens=cached,
            thoughts_tokens=thoughts,
        )

    def to_dict(self) -> dict[str, int]:
        """Convert token usage to dictionary."""
        return asdict(self)


@dataclass
class CandidateResult:
    """Detailed result for a single candidate reasoning path."""

    candidate_index: int
    facts: list[str]
    facts_raw_response: str
    answer_raw_response: str
    predicted_option: str | None
    is_correct: bool
    fact_latency_seconds: float
    answer_latency_seconds: float
    total_latency_seconds: float
    fact_tokens: StepTokenUsage
    answer_tokens: StepTokenUsage
    total_prompt_tokens: int
    total_candidate_tokens: int
    total_tokens: int
    error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # 0-based indices into facts that the answer says it relies on; None when
    # the answer prompt does not ask for them or the answer did not say.
    cited_facts: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateResult to dictionary."""
        res = asdict(self)
        res["fact_tokens"] = self.fact_tokens.to_dict()
        res["answer_tokens"] = self.answer_tokens.to_dict()
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateResult:
        """Construct CandidateResult from dictionary representation."""
        fact_tokens_data = data.get("fact_tokens") or {}
        answer_tokens_data = data.get("answer_tokens") or {}
        return cls(
            cited_facts=data.get("cited_facts"),
            candidate_index=int(data.get("candidate_index", 0)),
            facts=list(data.get("facts") or []),
            facts_raw_response=str(data.get("facts_raw_response", "")),
            answer_raw_response=str(data.get("answer_raw_response", "")),
            predicted_option=data.get("predicted_option"),
            is_correct=bool(data.get("is_correct", False)),
            fact_latency_seconds=float(data.get("fact_latency_seconds", 0.0)),
            answer_latency_seconds=float(data.get("answer_latency_seconds", 0.0)),
            total_latency_seconds=float(data.get("total_latency_seconds", 0.0)),
            fact_tokens=StepTokenUsage(**fact_tokens_data)
            if isinstance(fact_tokens_data, dict)
            else StepTokenUsage(),
            answer_tokens=StepTokenUsage(**answer_tokens_data)
            if isinstance(answer_tokens_data, dict)
            else StepTokenUsage(),
            total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
            total_candidate_tokens=int(data.get("total_candidate_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            error=data.get("error"),
            timestamp=str(data.get("timestamp", "")),
        )


@dataclass
class CandidateQuestionResult:
    """Aggregate result for a question evaluated over N candidates."""

    question_id: str
    meta_info: str | None
    question: str
    options: dict[str, str]
    ground_truth: str
    ground_truth_answer: str
    candidates: list[CandidateResult] = field(default_factory=list)
    total_candidates: int = 0
    successful_candidates: int = 0
    failed_candidates: int = 0
    correct_candidates: int = 0
    accuracy: float = 0.0
    total_latency_seconds: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_candidate_tokens: int = 0
    error: str | None = None

    def update_aggregates(self) -> None:
        """Compute summary counters over candidates."""
        self.total_candidates = len(self.candidates)
        self.successful_candidates = sum(
            1 for c in self.candidates if not c.error and c.predicted_option is not None
        )
        self.failed_candidates = self.total_candidates - self.successful_candidates
        self.correct_candidates = sum(1 for c in self.candidates if c.is_correct)
        self.accuracy = (
            round(self.correct_candidates / self.total_candidates, 4)
            if self.total_candidates > 0
            else 0.0
        )
        self.total_latency_seconds = round(
            sum(c.total_latency_seconds for c in self.candidates), 4
        )
        self.total_tokens = sum(c.total_tokens for c in self.candidates)
        self.total_prompt_tokens = sum(c.total_prompt_tokens for c in self.candidates)
        self.total_candidate_tokens = sum(
            c.total_candidate_tokens for c in self.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateQuestionResult to dictionary."""
        self.update_aggregates()
        return {
            "question_id": self.question_id,
            "meta_info": self.meta_info,
            "question": self.question,
            "options": self.options,
            "ground_truth": self.ground_truth,
            "ground_truth_answer": self.ground_truth_answer,
            "total_candidates": self.total_candidates,
            "successful_candidates": self.successful_candidates,
            "failed_candidates": self.failed_candidates,
            "correct_candidates": self.correct_candidates,
            "accuracy": self.accuracy,
            "total_latency_seconds": self.total_latency_seconds,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_candidate_tokens": self.total_candidate_tokens,
            "error": self.error,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateQuestionResult:
        """Construct CandidateQuestionResult from dictionary representation."""
        candidates = [
            CandidateResult.from_dict(c) for c in (data.get("candidates") or [])
        ]
        res = cls(
            question_id=str(data.get("question_id", "")),
            meta_info=data.get("meta_info"),
            question=str(data.get("question", "")),
            options=dict(data.get("options") or {}),
            ground_truth=str(data.get("ground_truth", "")),
            ground_truth_answer=str(data.get("ground_truth_answer", "")),
            candidates=candidates,
            total_candidates=int(data.get("total_candidates", len(candidates))),
            successful_candidates=int(data.get("successful_candidates", 0)),
            failed_candidates=int(data.get("failed_candidates", 0)),
            correct_candidates=int(data.get("correct_candidates", 0)),
            accuracy=float(data.get("accuracy", 0.0)),
            total_latency_seconds=float(data.get("total_latency_seconds", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
            total_candidate_tokens=int(data.get("total_candidate_tokens", 0)),
            error=data.get("error"),
        )
        res.update_aggregates()
        return res


@dataclass
class CandidateWorkflowSummary:
    """Summary statistics for the candidate generation workflow."""

    model: str
    dataset: str
    config: str | None
    split: str
    n_candidates: int
    total_questions: int
    completed_questions: int
    failed_questions: int
    total_candidates_generated: int
    total_successful_candidates: int
    total_failed_candidates: int
    total_correct_candidates: int
    overall_accuracy: float
    total_time_seconds: float
    average_question_latency_seconds: float
    average_candidate_latency_seconds: float
    total_tokens: int
    total_prompt_tokens: int
    total_candidate_tokens: int
    created_at: str
    total_thoughts_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateWorkflowSummary to dictionary."""
        return asdict(self)

    @classmethod
    def from_results(
        cls,
        results: list[CandidateQuestionResult],
        model: str,
        dataset: str,
        config: str | None,
        split: str,
        n_candidates: int,
        total_time_seconds: float,
    ) -> CandidateWorkflowSummary:
        """Compute aggregate workflow summary statistics."""
        total_q = len(results)
        completed_q = sum(1 for r in results if not r.error)
        failed_q = total_q - completed_q

        all_candidates = [c for r in results for c in r.candidates]
        total_c = len(all_candidates)
        successful_c = sum(1 for c in all_candidates if not c.error)
        failed_c = total_c - successful_c
        correct_c = sum(1 for c in all_candidates if c.is_correct)
        overall_acc = round(correct_c / total_c, 4) if total_c > 0 else 0.0

        avg_q_latency = (
            round(sum(r.total_latency_seconds for r in results) / total_q, 4)
            if total_q > 0
            else 0.0
        )
        avg_c_latency = (
            round(sum(c.total_latency_seconds for c in all_candidates) / total_c, 4)
            if total_c > 0
            else 0.0
        )

        total_tokens = sum(c.total_tokens for c in all_candidates)
        total_prompt = sum(c.total_prompt_tokens for c in all_candidates)
        total_candidate = sum(c.total_candidate_tokens for c in all_candidates)
        total_thoughts = sum(
            c.fact_tokens.thoughts_tokens + c.answer_tokens.thoughts_tokens
            for c in all_candidates
        )

        return cls(
            model=model,
            dataset=dataset,
            config=config,
            split=split,
            n_candidates=n_candidates,
            total_questions=total_q,
            completed_questions=completed_q,
            failed_questions=failed_q,
            total_candidates_generated=total_c,
            total_successful_candidates=successful_c,
            total_failed_candidates=failed_c,
            total_correct_candidates=correct_c,
            overall_accuracy=overall_acc,
            total_time_seconds=round(total_time_seconds, 2),
            average_question_latency_seconds=avg_q_latency,
            average_candidate_latency_seconds=avg_c_latency,
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt,
            total_candidate_tokens=total_candidate,
            total_thoughts_tokens=total_thoughts,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
