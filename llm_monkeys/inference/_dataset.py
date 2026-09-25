"""Dataset utilities for loading difficult questions and formatting candidate prompts."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from dataset import MedQAQuestion, load_medqa_dataset
from inference._prompts import (
    ANSWER_PROMPTS,
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_FACT_PROMPT,
    FACT_PROMPTS,
)

logger = logging.getLogger(__name__)


def natural_sort_key(s: str) -> list[int | str]:
    """Sort key for natural alphanumeric ordering (e.g., '1', '2', '10')."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def load_difficult_question_ids(
    csv_path: str | Path = "difficult_questions.csv",
) -> list[str]:
    """Read difficult question IDs from CSV file."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Difficult questions CSV file not found: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        rows = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    if rows and rows[0].lower() in ("question_id", "id", "qid"):
        rows = rows[1:]

    logger.info("Loaded %d question IDs from %s", len(rows), path)
    return rows


def load_difficult_questions(
    csv_path: str | Path = "difficult_questions.csv",
    dataset_name: str = "bigbio/med_qa",
    config_name: str | None = "med_qa_en_source",
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
) -> list[MedQAQuestion]:
    """Load MedQA questions filtered to only those present in difficult_questions.csv."""
    difficult_ids = load_difficult_question_ids(csv_path)
    if not difficult_ids:
        logger.warning("No difficult question IDs found in %s", csv_path)
        return []

    all_questions = {
        q.question_id: q
        for q in load_medqa_dataset(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
        )
    }

    matched = []
    for qid in difficult_ids:
        if qid in all_questions:
            matched.append(all_questions[qid])
        elif qid.lstrip("0") in all_questions:
            matched.append(all_questions[qid.lstrip("0")])
        elif qid.zfill(3) in all_questions:
            matched.append(all_questions[qid.zfill(3)])
    start = max(0, offset)
    stop = start + limit if limit is not None else None
    selected = matched[start:stop]

    logger.info(
        "Selected %d difficult questions (offset=%d, limit=%s, total_difficult=%d)",
        len(selected),
        offset,
        limit,
        len(matched),
    )
    return selected


def format_fact_generation_prompt(
    question: MedQAQuestion,
    fact_prompt: str = DEFAULT_FACT_PROMPT,
) -> str:
    """Format prompt for Step 1: generating atomic verifiable medical facts.

    `fact_prompt` names an entry of FACT_PROMPTS; the default is the reference
    wording every stored run was generated with.
    """
    return (
        f"{FACT_PROMPTS[fact_prompt].task}\n\n"
        f"Question: {question.question}\n\n"
        f"Options:\n{question.format_options()}\n\n"
        "Generate atomic, verifiable statements as structured JSON conforming to the schema."
    )


def format_answer_generation_prompt(
    question: MedQAQuestion,
    facts: list[str],
    answer_prompt: str = DEFAULT_ANSWER_PROMPT,
) -> str:
    """Format prompt for Step 2: reasoning and generating final answer from facts.

    `answer_prompt` names an entry of ANSWER_PROMPTS; numbered prompts show the
    facts as [1], [2], ... so the answer can refer to them.
    """
    spec = ANSWER_PROMPTS[answer_prompt]
    if not facts:
        facts_block = "No additional facts provided."
    elif spec.numbered_facts:
        facts_block = "\n".join(f"[{i}] {f}" for i, f in enumerate(facts, start=1))
    else:
        facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        "The following is a multiple-choice medical examination question.\n\n"
        f"Question: {question.question}\n\n"
        f"Options:\n{question.format_options()}\n\n"
        f"Relevant Medical Facts:\n{facts_block}\n\n"
        f"{spec.instructions}"
    )
