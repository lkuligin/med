"""MedQA dataset loading and prompt formatting utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import datasets

logger = logging.getLogger(__name__)


@dataclass
class MedQAQuestion:
    """Represents a single question from the MedQA dataset."""

    question_id: str
    question: str
    options: dict[str, str]  # e.g., {"A": "Option text", "B": "Option text", ...}
    answer_idx: str  # e.g., "A", "B", "C", "D", "E"
    answer: str  # Ground truth answer text
    meta_info: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], question_id: str | None = None) -> MedQAQuestion:
        """Parse raw dataset record into MedQAQuestion."""
        raw_options = data.get("options")
        if isinstance(raw_options, list):
            options = {
                opt["key"].strip().upper(): str(opt.get("value", "")).strip()
                for opt in raw_options
                if isinstance(opt, dict) and "key" in opt
            }
        elif isinstance(raw_options, dict):
            options = {k.strip().upper(): str(v).strip() for k, v in raw_options.items()}
        else:
            options = {}

        qid = str(
            question_id
            if question_id is not None
            else data.get("id") or data.get("question_id") or "0"
        )

        return cls(
            question_id=qid,
            question=data.get("question", "").strip(),
            options=options,
            answer_idx=str(data.get("answer_idx", "")).strip().upper(),
            answer=str(data.get("answer", "")).strip(),
            meta_info=data.get("meta_info"),
        )

    def format_options(self) -> str:
        """Format options into alphabetical key-value lines."""
        return "\n".join(f"{k}. {v}" for k, v in sorted(self.options.items()))


def format_one_shot_prompt(question: MedQAQuestion) -> str:
    """Format a MedQA question into a standardized one-shot prompt."""
    return (
        "The following is a multiple-choice medical examination question. "
        "Select the single best option letter (e.g., A, B, C, D, or E) and provide a concise medical explanation.\n\n"
        f"Question: {question.question}\n"
        "Options:\n"
        f"{question.format_options()}\n"
        "Provide concise clinical reasoning evaluating the options, and conclude your response on a new line with:\n"
        "FINAL ANSWER: [Option Letter]"
    )


def load_medqa_dataset(
    dataset_name: str = "bigbio/med_qa",
    config_name: str = "med_qa_en_source",
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
) -> list[MedQAQuestion]:
    """Load questions from MedQA dataset via Hugging Face datasets library."""
    logger.info("Loading MedQA dataset: %s (config: %s, split: %s)", dataset_name, config_name, split)
    ds = datasets.load_dataset(dataset_name, config_name, split=split)

    start = max(0, offset)
    end = len(ds) if limit is None else min(start + limit, len(ds))

    questions = [
        MedQAQuestion.from_dict(ds[i], question_id=str(i))
        for i in range(start, end)
    ]
    logger.info(
        "Loaded %d questions (offset=%d, limit=%s, total_in_split=%d)",
        len(questions),
        offset,
        limit,
        len(ds),
    )
    return questions

