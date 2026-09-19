"""Dataset utilities for loading difficult questions and formatting candidate prompts."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from dataset import MedQAQuestion, load_medqa_dataset

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


def format_fact_generation_prompt(question: MedQAQuestion) -> str:
    """Format prompt for Step 1: generating atomic verifiable medical facts."""
    return (
        "Analyze the following multiple-choice medical examination question and options. "
        "Extract and generate all key atomic, verifiable medical facts, clinical principles, "
        "pathophysiological mechanisms, and pharmacological properties relevant to solving this question accurately.\n\n"
        f"Question: {question.question}\n\n"
        f"Options:\n{question.format_options()}\n\n"
        "Generate atomic, verifiable statements as structured JSON conforming to the schema."
    )


def format_answer_generation_prompt(
    question: MedQAQuestion,
    facts: list[str],
) -> str:
    """Format prompt for Step 2: reasoning and generating final answer from facts."""
    facts_block = (
        "\n".join(f"- {f}" for f in facts) if facts else "No additional facts provided."
    )
    return (
        "The following is a multiple-choice medical examination question.\n\n"
        f"Question: {question.question}\n\n"
        f"Options:\n{question.format_options()}\n\n"
        f"Relevant Medical Facts:\n{facts_block}\n\n"
        "Based on these clinical facts, reason through the scenario and determine the single best option. "
        "Conclude your response on a new line with:\n"
        "FINAL ANSWER: [Option Letter]"
    )
