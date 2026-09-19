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
    def from_dict(
        cls, data: dict[str, Any], question_id: str | None = None
    ) -> MedQAQuestion:
        """Parse raw dataset record into MedQAQuestion."""
        raw_options = data.get("options")
        if isinstance(raw_options, list):
            options = {
                opt["key"].strip().upper(): str(opt.get("value", "")).strip()
                for opt in raw_options
                if isinstance(opt, dict) and "key" in opt
            }
        elif isinstance(raw_options, dict):
            options = {
                k.strip().upper(): str(v).strip() for k, v in raw_options.items()
            }
        elif any(k in data for k in ("opa", "opb", "opc", "opd", "ope")):
            options = {
                k[2].upper(): str(data[k]).strip()
                for k in ("opa", "opb", "opc", "opd", "ope")
                if k in data and data[k] and str(data[k]).strip().lower() != "nan"
            }
        else:
            options = {}

        qid = str(
            question_id
            if question_id is not None
            else data.get("id") or data.get("question_id") or data.get("idx") or "0"
        )

        raw_answer_idx = data.get("answer_idx")
        raw_answer = data.get("answer")

        if raw_answer_idx:
            answer_idx = str(raw_answer_idx).strip().upper()
            answer = str(raw_answer or "").strip()
            if not answer and answer_idx in options:
                answer = options[answer_idx]
        elif raw_answer:
            cleaned_ans = str(raw_answer).strip()
            if cleaned_ans.upper() in options or cleaned_ans.upper() in (
                "A",
                "B",
                "C",
                "D",
                "E",
            ):
                answer_idx = cleaned_ans.upper()
                answer = options.get(answer_idx, cleaned_ans)
            else:
                answer = cleaned_ans
                matched_idx = next(
                    (
                        k
                        for k, v in options.items()
                        if v.strip().lower() == cleaned_ans.strip().lower()
                    ),
                    "",
                )
                answer_idx = matched_idx
        else:
            answer_idx = ""
            answer = ""

        meta_info = data.get("meta_info") or data.get("explanation")

        return cls(
            question_id=qid,
            question=data.get("question", "").strip(),
            options=options,
            answer_idx=answer_idx,
            answer=answer,
            meta_info=meta_info,
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
    config_name: str | None = "med_qa_en_source",
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
) -> list[MedQAQuestion]:
    """Load questions from MedQA or MedBullets dataset via Hugging Face datasets library."""
    is_medbullets = "medbullets" in dataset_name.lower()
    if is_medbullets and config_name in ("med_qa_en_source", "default"):
        config_name = None

    logger.info(
        "Loading dataset: %s (config: %s, split: %s)", dataset_name, config_name, split
    )

    load_kwargs: dict[str, Any] = {"split": split}
    if config_name is not None:
        load_kwargs["name"] = config_name

    try:
        ds = datasets.load_dataset(dataset_name, **load_kwargs)
    except Exception as exc:
        if "401" in str(exc) or "Unauthorized" in str(exc):
            ds = datasets.load_dataset(dataset_name, **load_kwargs, token=False)
        else:
            raise

    start = max(0, offset)
    end = len(ds) if limit is None else min(start + limit, len(ds))

    questions = [
        MedQAQuestion.from_dict(
            ds[i],
            question_id=str(
                ds[i].get("idx") or ds[i].get("id") or ds[i].get("question_id") or i
            ),
        )
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
