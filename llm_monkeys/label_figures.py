#!/usr/bin/env python3
"""Label every question of a split by whether answering it needs an exhibit.

MedQA and MedBullets questions sometimes point at an ECG, a radiograph or a
photograph that the text does not contain. A text-only model cannot answer
those however well it reasons, so a run's accuracy means something different on
them, and reporting the two together hides it.

    python3 label_figures.py --dataset bigbio/med_qa
    python3 label_figures.py --dataset mkieffer/Medbullets --output figure_labels_mb.csv

The judge is asked once per question and answers with one of three labels:

    requires_figure             the exhibit holds what the options turn on
    mentions_figure_answerable  an exhibit is mentioned, the text still decides
    no_figure                   no exhibit is referred to

Output is a CSV of question_id,label,reason beside difficult_questions.csv, and
is read back on the next run: labels already in it are kept, so a stopped run
resumes and a finished one costs nothing to repeat.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from config import (
    DEFAULT_VERIFIER_MODEL,
    BaseInferenceConfig,
    is_medbullets_dataset,
    register_litellm_model_pricing,
    resolve_dataset_name,
)
from dataset import MedQAQuestion, load_medqa_dataset
from inference.workflow import _calculate_backoff, is_rate_limit_error
from model_factory import build_model

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

LABELS = ("requires_figure", "mentions_figure_answerable", "no_figure")

SYSTEM_INSTRUCTION = (
    "You are an expert physician screening medical licensing examination "
    "questions for a text-only model that cannot see images. Decide whether a "
    "question can be answered from its text alone, and answer with one of the "
    "three labels you are given and a one-sentence reason."
)

PROMPT = """You are screening medical licensing exam questions for a text-only model that cannot see images.
Decide whether the question can be answered from its text alone.

Labels:
- requires_figure: the question refers to an image, photograph, ECG, radiograph, scan, graph, table or other exhibit that is not included in the text, and the information needed to choose among the options is in that exhibit.
- mentions_figure_answerable: an exhibit is mentioned, but the text and options alone give enough to choose the answer.
- no_figure: no exhibit is referred to.

Question:
{question}

Options:
{options}

Reply with JSON only: {{"label": "<one of the three labels>", "reason": "<one sentence>"}}"""


class FigureLabel(BaseModel):
    """Structured output for one question's screening."""

    label: str = Field(description="One of: " + ", ".join(LABELS))
    reason: str = Field(default="", description="One sentence explaining the label.")


class FigureLabelConfig(BaseInferenceConfig):
    """How the screening runs.

    Temperature is the judge's, not the generator's: this is a judgement about
    a question, made by the same model the pipeline verifies facts with, and
    Gemini 3 is meant to be sampled at 1.0.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", DEFAULT_VERIFIER_MODEL)
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("max_tokens", 16384)
        kwargs.setdefault("concurrency", 8)
        super().__init__(**kwargs)


def format_prompt(question: MedQAQuestion) -> str:
    """The screening prompt for one question, options included.

    The options matter: "an exhibit is mentioned but the text still decides"
    can only be told apart from "the exhibit decides" by reading what there is
    to choose between.
    """
    options = "\n".join(f"{k}. {v}" for k, v in sorted((question.options or {}).items()))
    return PROMPT.format(question=question.question, options=options)


def parse_label(raw_response: str) -> tuple[str, str]:
    """Pull the label and reason out of a reply, or ("", "") if it has neither.

    The reply is asked for as JSON and usually is, but a model may wrap it in
    prose or a code fence, so the first JSON object in the text is taken.
    """
    match = re.search(r"\{.*\}", raw_response or "", re.S)
    if not match:
        return "", ""
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", ""
    label = str(data.get("label", "")).strip()
    return (label if label in LABELS else ""), str(data.get("reason", "")).strip()


def create_label_agent(config: FigureLabelConfig | None = None) -> Agent:
    """An ADK agent that screens one question."""
    cfg = config or FigureLabelConfig()
    register_litellm_model_pricing()
    return Agent(
        name="medqa_figure_screener",
        model=build_model(cfg),
        instruction=SYSTEM_INSTRUCTION,
        output_schema=FigureLabel,
        generate_content_config=types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            response_mime_type="application/json",
        ),
    )


def read_labels(path: str | Path) -> dict[str, tuple[str, str]]:
    """What has already been labelled, keyed by the id the dataset spells.

    Rows whose label is not one of the three are dropped rather than kept, so
    a run that errored on a question tries it again instead of freezing the
    error into the file.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        return {
            row["question_id"].strip(): (row["label"], row.get("reason", ""))
            for row in csv.DictReader(handle)
            if row.get("label") in LABELS
        }


def write_labels(path: str | Path, questions: Iterable[MedQAQuestion],
                 labels: dict[str, tuple[str, str]]) -> None:
    """Write the file in dataset order, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "label", "reason"])
        for question in questions:
            qid = str(question.question_id)
            if qid in labels:
                writer.writerow([qid, *labels[qid]])
    temp.replace(path)


class FigureLabelWorkflow:
    """Screen a split, one model call per question."""

    def __init__(self, config: FigureLabelConfig | None = None,
                 runner: Runner | None = None) -> None:
        self.config = config or FigureLabelConfig()
        self.session_service = InMemorySessionService()
        self.runner = runner or Runner(
            agent=create_label_agent(self.config),
            session_service=self.session_service,
            app_name="medqa_figure_app",
        )

    async def _ask(self, question: MedQAQuestion) -> str:
        """One call, retried with backoff while the gateway says it is busy."""
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=format_prompt(question))]
        )
        for attempt in range(self.config.rate_limit_max_retries + 1):
            session_id = f"figlab_{question.question_id}_{int(time.time() * 1000)}_{attempt}"
            try:
                await self.session_service.create_session(
                    app_name=self.runner.app_name, user_id="medqa_figure_user",
                    session_id=session_id,
                )
                text = ""
                async for event in self.runner.run_async(
                    user_id="medqa_figure_user", session_id=session_id,
                    new_message=content,
                ):
                    for part in getattr(getattr(event, "content", None), "parts", []) or []:
                        if getattr(part, "text", None) and not getattr(part, "thought", False):
                            text += part.text
                return text
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt == self.config.rate_limit_max_retries:
                    raise
                await asyncio.sleep(_calculate_backoff(
                    attempt, self.config.base_retry_delay, self.config.max_retry_delay))
        return ""

    async def label(self, question: MedQAQuestion) -> tuple[str, str]:
        """This question's label, or ("", reason) when the model never gave one."""
        for attempt in range(self.config.max_retries + 1):
            try:
                label, reason = parse_label(await self._ask(question))
                if label:
                    return label, reason
                logger.warning("Question %s: no usable label in the reply (attempt %d)",
                               question.question_id, attempt + 1)
            except Exception as exc:
                logger.warning("Question %s failed (attempt %d): %s",
                               question.question_id, attempt + 1, str(exc)[:200])
                if attempt == self.config.max_retries:
                    return "", str(exc)[:200]
        return "", "no label in the reply"

    async def run(self, questions: list[MedQAQuestion],
                  on_done: Any = None) -> dict[str, tuple[str, str]]:
        """Label every question given, with at most `concurrency` calls in flight."""
        semaphore = asyncio.Semaphore(self.config.concurrency)
        labels: dict[str, tuple[str, str]] = {}

        async def _one(question: MedQAQuestion) -> None:
            async with semaphore:
                label, reason = await self.label(question)
                if label:
                    labels[str(question.question_id)] = (label, reason)
                if on_done:
                    on_done(question, label, reason, len(labels))

        await asyncio.gather(*(_one(q) for q in questions))
        return labels


def default_output(dataset_name: str) -> Path:
    """Where a split's labels live, named as the difficult lists are."""
    suffix = "_mb" if is_medbullets_dataset(dataset_name) else ""
    return _SCRIPT_DIR / f"figure_labels{suffix}.csv"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="bigbio/med_qa",
                        help="HuggingFace dataset name (default: bigbio/med_qa)")
    parser.add_argument("--dataset-config", default=None,
                        help="Dataset configuration (default: the dataset's own)")
    parser.add_argument("--split", default=None,
                        help="Dataset split (default: the dataset's own)")
    parser.add_argument("--model", default=DEFAULT_VERIFIER_MODEL,
                        help=f"Model that labels (default: {DEFAULT_VERIFIER_MODEL})")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Maximum output tokens per question (default: 16384)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Maximum concurrent requests (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Label at most this many unlabelled questions")
    parser.add_argument("--output", default=None,
                        help="CSV to write (default: figure_labels[_mb].csv beside this script)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Write the CSV after every N questions (default: 25)")
    parser.add_argument("--log-level", default="INFO")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    dataset_name = resolve_dataset_name(args.dataset)
    medbullets = is_medbullets_dataset(dataset_name)
    questions = load_medqa_dataset(
        dataset_name=dataset_name,
        config_name=args.dataset_config if args.dataset_config is not None
        else (None if medbullets else "med_qa_en_source"),
        split=args.split or ("op5_test" if medbullets else "test"),
    )
    path = Path(args.output) if args.output else default_output(dataset_name)

    labels = read_labels(path)
    todo = [q for q in questions if str(q.question_id) not in labels]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{dataset_name}: {len(questions)} questions, {len(labels)} already labelled, "
          f"{len(todo)} to do -> {path}", flush=True)
    if not todo:
        return 0

    config = FigureLabelConfig(model_name=args.model, temperature=args.temperature,
                              max_tokens=args.max_tokens, concurrency=args.concurrency)
    workflow = FigureLabelWorkflow(config=config)

    def _progress(question: MedQAQuestion, label: str, reason: str, done: int) -> None:
        if label:
            labels[str(question.question_id)] = (label, reason)
        # Written as it goes: a split takes a model call per question, and
        # holding half an hour of them in memory means a stopped run has
        # nothing to resume from.
        if done and done % args.save_every == 0:
            write_labels(path, questions, labels)
            print(f"  {done}/{len(todo)} labelled", flush=True)

    labels.update(await workflow.run(todo, on_done=_progress))
    write_labels(path, questions, labels)

    counts: dict[str, int] = {}
    for label, _ in labels.values():
        counts[label] = counts.get(label, 0) + 1
    missing = len(questions) - len(labels)
    print(f"{dataset_name}: {len(labels)} labelled {counts}"
          + (f", {missing} still unlabelled" if missing else ""), flush=True)
    return 0 if not missing else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(create_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
