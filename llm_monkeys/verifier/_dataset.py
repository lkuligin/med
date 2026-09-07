"""Dataset loading and data parsing utilities for Step 2 candidate results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _DictSerializable:
    """Mixin providing dictionary conversion for dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass instance to a dictionary."""
        return asdict(self)


@dataclass
class Step2CandidateData(_DictSerializable):
    """Representation of a single candidate reasoning path from Step 2."""

    candidate_index: int = 0
    facts: list[str] = field(default_factory=list)
    predicted_option: str | None = None
    is_correct: bool = False
    answer_raw_response: str = ""
    facts_raw_response: str = ""
    total_latency_seconds: float = 0.0
    total_tokens: int = 0
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step2CandidateData:
        """Construct Step2CandidateData from a dictionary."""
        return cls(
            candidate_index=int(data.get("candidate_index", 0)),
            facts=list(data.get("facts") or []),
            predicted_option=data.get("predicted_option"),
            is_correct=bool(data.get("is_correct", False)),
            answer_raw_response=str(data.get("answer_raw_response", "")),
            facts_raw_response=str(data.get("facts_raw_response", "")),
            total_latency_seconds=float(data.get("total_latency_seconds", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            error=data.get("error"),
        )


@dataclass
class Step2QuestionData(_DictSerializable):
    """Representation of a question and its candidate pool from Step 2."""

    question_id: str
    question: str
    options: dict[str, str] = field(default_factory=dict)
    ground_truth: str = ""
    candidates: list[Step2CandidateData] = field(default_factory=list)
    meta_info: str | None = None
    ground_truth_answer: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step2QuestionData:
        """Construct Step2QuestionData from a dictionary."""
        return cls(
            question_id=str(data.get("question_id", "")),
            question=str(data.get("question", "")),
            options=dict(data.get("options") or {}),
            ground_truth=str(
                data.get("ground_truth")
                or data.get("answer_idx")
                or data.get("answer")
                or ""
            ),
            candidates=[
                Step2CandidateData.from_dict(c)
                for c in data.get("candidates") or []
                if isinstance(c, dict)
            ],
            meta_info=data.get("meta_info"),
            ground_truth_answer=data.get("ground_truth_answer"),
        )


def load_step2_results(
    filepath: str | Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[Step2QuestionData]:
    """Load Step 2 JSON candidate output.

    Handles both wrapped format `{"summary": ..., "results": [...]}` and bare list `[...]`.

    Args:
        filepath: Path to Step 2 results JSON file.
        limit: Optional maximum number of questions to load.
        offset: Number of questions to skip from start.

    Returns:
        List of Step2QuestionData objects.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Step 2 results file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict) and "results" in data:
        raw_items = data["results"]
    elif isinstance(data, dict):
        raise ValueError(
            f"Expected 'results' key in JSON dict at {path}, found keys: {list(data.keys())}"
        )
    else:
        raise ValueError(
            f"Unexpected JSON format at {path}: expected dict or list, got {type(data)}"
        )

    start = max(0, offset)
    stop = start + limit if limit is not None and limit > 0 else None
    selected = [
        Step2QuestionData.from_dict(item)
        for item in raw_items
        if isinstance(item, dict)
    ][start:stop]

    logger.info(
        "Loaded %d questions from %s (offset=%d, limit=%s)",
        len(selected),
        path,
        offset,
        limit,
    )
    return selected


__all__ = [
    "Step2CandidateData",
    "Step2QuestionData",
    "load_step2_results",
]
