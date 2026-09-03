"""Workflow module for running MedQA one-shot inference using Google ADK."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import InferenceConfig
from dataset import MedQAQuestion, format_one_shot_prompt, load_medqa_dataset

from .agent import create_medqa_agent, create_runner
from .parser import evaluate_prediction, extract_predicted_option

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageAccumulator:
    """Accumulates token usage across multiple inference attempts."""

    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    had_usage: bool = False

    def add(self, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        """Add usage metadata from a response event if present."""
        if usage is None:
            return
        self.had_usage = True
        self.prompt_tokens += usage.prompt_token_count or 0
        self.candidate_tokens += usage.candidates_token_count or 0
        self.total_tokens += usage.total_token_count or 0
        self.cached_tokens += usage.cached_content_token_count or 0


@dataclass
class AttemptResult:
    """Detailed result for a single inference attempt on a question."""

    attempt_index: int
    predicted_option: str | None
    is_correct: bool
    raw_response: str
    latency_seconds: float
    prompt_tokens: int | None = None
    candidate_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None
    parse_retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert attempt result to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptResult:
        """Construct AttemptResult from a dictionary representation."""
        return cls(
            attempt_index=int(data.get("attempt_index", 0)),
            predicted_option=data.get("predicted_option"),
            is_correct=bool(data.get("is_correct", False)),
            raw_response=str(data.get("raw_response", "")),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            prompt_tokens=data.get("prompt_tokens"),
            candidate_tokens=data.get("candidate_tokens"),
            total_tokens=data.get("total_tokens"),
            cached_tokens=data.get("cached_tokens"),
            error=data.get("error"),
            parse_retries=int(data.get("parse_retries", 0)),
        )


@dataclass
class InferenceItemResult:
    """Detailed result for a question evaluation across multiple inference attempts."""

    question_id: str
    meta_info: str | None
    question: str
    options: dict[str, str]
    ground_truth: str
    ground_truth_answer: str
    prompt: str
    attempts: list[AttemptResult] = field(default_factory=list)
    predicted_option: str | None = None
    is_correct: bool = False
    is_all_correct: bool = False
    correct_attempts: int = 0
    total_attempts: int = 0
    raw_response: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int | None = None
    candidate_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None
    parse_retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        d = asdict(self)
        d["attempts"] = [
            a.to_dict() if isinstance(a, AttemptResult) else a for a in self.attempts
        ]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceItemResult:
        """Construct InferenceItemResult from a dictionary representation."""
        raw_attempts = data.get("attempts")
        attempts: list[AttemptResult] = []
        if isinstance(raw_attempts, list) and raw_attempts:
            for item in raw_attempts:
                if isinstance(item, dict):
                    attempts.append(AttemptResult.from_dict(item))
                elif isinstance(item, AttemptResult):
                    attempts.append(item)
        elif "raw_response" in data or "predicted_option" in data:
            # Reconstruct single attempt from legacy/top-level fields if attempts list missing
            attempts.append(
                AttemptResult(
                    attempt_index=0,
                    predicted_option=data.get("predicted_option"),
                    is_correct=bool(data.get("is_correct", False)),
                    raw_response=str(data.get("raw_response", "")),
                    latency_seconds=float(data.get("latency_seconds", 0.0)),
                    prompt_tokens=data.get("prompt_tokens"),
                    candidate_tokens=data.get("candidate_tokens"),
                    total_tokens=data.get("total_tokens"),
                    cached_tokens=data.get("cached_tokens"),
                    error=data.get("error"),
                    parse_retries=int(data.get("parse_retries", 0)),
                )
            )

        correct_attempts = (
            data.get("correct_attempts")
            if data.get("correct_attempts") is not None
            else sum(1 for a in attempts if a.is_correct)
        )
        total_attempts = (
            data.get("total_attempts")
            if data.get("total_attempts") is not None
            else len(attempts)
        )
        is_all_correct = (
            bool(data.get("is_all_correct"))
            if data.get("is_all_correct") is not None
            else (correct_attempts == total_attempts and total_attempts > 0)
        )
        is_correct = (
            bool(data.get("is_correct"))
            if data.get("is_correct") is not None
            else is_all_correct
        )

        predicted_option = data.get("predicted_option")
        if predicted_option is None and attempts:
            predicted_option = attempts[0].predicted_option

        raw_response = str(data.get("raw_response", ""))
        if not raw_response and attempts:
            raw_response = attempts[0].raw_response

        return cls(
            question_id=str(data.get("question_id", "")),
            meta_info=data.get("meta_info"),
            question=str(data.get("question", "")),
            options=dict(data.get("options") or {}),
            ground_truth=str(data.get("ground_truth", "")),
            ground_truth_answer=str(data.get("ground_truth_answer", "")),
            prompt=str(data.get("prompt", "")),
            attempts=attempts,
            predicted_option=predicted_option,
            is_correct=is_correct,
            is_all_correct=is_all_correct,
            correct_attempts=int(correct_attempts),
            total_attempts=int(total_attempts),
            raw_response=raw_response,
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            prompt_tokens=data.get("prompt_tokens"),
            candidate_tokens=data.get("candidate_tokens"),
            total_tokens=data.get("total_tokens"),
            cached_tokens=data.get("cached_tokens"),
            error=data.get("error"),
            parse_retries=int(data.get("parse_retries", 0)),
        )


@dataclass
class WorkflowSummary:
    """Aggregate statistics and metadata for the inference workflow."""

    model: str
    dataset: str
    config: str
    split: str
    n_attempts: int
    total_questions: int
    completed: int
    failed: int
    correct: int
    accuracy: float
    simple_questions: int
    difficult_questions: int
    total_time_seconds: float
    average_latency_seconds: float
    total_tokens: int
    total_prompt_tokens: int
    total_candidate_tokens: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_results(
        cls,
        results: list[InferenceItemResult],
        model: str,
        dataset: str,
        config: str,
        split: str,
        n_attempts: int,
        total_time_seconds: float,
    ) -> WorkflowSummary:
        """Construct WorkflowSummary by computing aggregate statistics over results."""
        total = len(results)
        completed = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)
        simple_count = sum(1 for r in results if r.is_all_correct)
        difficult_count = total - simple_count
        correct = simple_count
        avg_latency = (
            sum(r.latency_seconds for r in results) / total if total else 0.0
        )

        return cls(
            model=model,
            dataset=dataset,
            config=config,
            split=split,
            n_attempts=n_attempts,
            total_questions=total,
            completed=completed,
            failed=failed,
            correct=correct,
            accuracy=round(correct / total, 4) if total else 0.0,
            simple_questions=simple_count,
            difficult_questions=difficult_count,
            total_time_seconds=round(total_time_seconds, 2),
            average_latency_seconds=round(avg_latency, 4),
            total_tokens=sum(r.total_tokens or 0 for r in results),
            total_prompt_tokens=sum(r.prompt_tokens or 0 for r in results),
            total_candidate_tokens=sum(r.candidate_tokens or 0 for r in results),
            created_at=datetime.now(timezone.utc).isoformat(),
        )


class OneShotInferenceWorkflow:
    """Orchestrates multi-attempt one-shot inference across MedQA questions using ADK and LiteLLM."""

    def __init__(
        self,
        config: InferenceConfig | None = None,
        agent: Agent | None = None,
        runner: Runner | None = None,
        session_service: InMemorySessionService | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.session_service = session_service or InMemorySessionService()
        self.agent = agent or create_medqa_agent(self.config)
        self.runner = runner or create_runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="medqa_workflow_app",
        )

    async def _invoke_agent_with_retry(
        self,
        prompt_text: str,
        session_id: str,
        user_id: str = "med_eval_user",
    ) -> tuple[str, types.GenerateContentResponseUsageMetadata | None]:
        """Invoke ADK runner with exponential backoff on transient or rate-limit errors."""
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=prompt_text)]
        )

        for attempt in range(self.config.max_retries + 1):
            try:
                try:
                    await self.session_service.create_session(
                        app_name="medqa_workflow_app",
                        user_id=user_id,
                        session_id=session_id,
                    )
                except Exception:
                    pass

                output_parts: list[str] = []
                usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

                async for event in self.runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=content,
                ):
                    if event.content and event.content.parts:
                        output_parts.extend(
                            part.text for part in event.content.parts if part.text
                        )
                    if event.usage_metadata:
                        usage_metadata = event.usage_metadata

                return "".join(output_parts), usage_metadata

            except Exception as exc:
                if attempt >= self.config.max_retries:
                    raise

                backoff = min(
                    self.config.max_retry_delay,
                    (self.config.base_retry_delay * (2**attempt))
                    + random.uniform(0.5, 2.0),
                )
                logger.warning(
                    "Attempt %d/%d failed for session %s: %s. Backing off for %.2fs...",
                    attempt + 1,
                    self.config.max_retries,
                    session_id,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("Failed to invoke agent after retries.")

    async def run_single_attempt(
        self,
        question: MedQAQuestion,
        question_idx: int,
        attempt_idx: int,
        semaphore: asyncio.Semaphore,
        max_parse_retries: int | None = None,
    ) -> AttemptResult:
        """Execute a single inference attempt for a question under concurrency limit."""
        if max_parse_retries is None:
            max_parse_retries = getattr(self.config, "max_parse_retries", 3)

        async with semaphore:
            prompt_text = format_one_shot_prompt(question)
            start_time = time.perf_counter()

            raw_response = ""
            predicted_option: str | None = None
            error_msg: str | None = None
            attempts_made = 0
            usage_acc = TokenUsageAccumulator()

            for parse_attempt in range(max_parse_retries + 1):
                attempts_made += 1
                session_id = (
                    f"medqa_sess_{question.question_id}_q{question_idx}_a{attempt_idx}_p{parse_attempt}_{time.time_ns()}"
                )
                try:
                    curr_response, usage = await self._invoke_agent_with_retry(
                        prompt_text=prompt_text,
                        session_id=session_id,
                    )
                    raw_response = curr_response
                    error_msg = None
                    usage_acc.add(usage)

                    predicted_option = extract_predicted_option(
                        raw_response, question.options
                    )
                    if predicted_option is not None:
                        break

                    if parse_attempt < max_parse_retries:
                        logger.warning(
                            "Failed to parse predicted option for question_id=%s attempt %d on parse retry %d/%d (response: %r). Retrying...",
                            question.question_id,
                            attempt_idx,
                            parse_attempt + 1,
                            max_parse_retries + 1,
                            raw_response[:100] if raw_response else "",
                        )
                except Exception as exc:
                    logger.error(
                        "Inference failed for question_id=%s attempt %d on parse retry %d/%d: %s",
                        question.question_id,
                        attempt_idx,
                        parse_attempt + 1,
                        max_parse_retries + 1,
                        exc,
                    )
                    error_msg = str(exc)
                    break

            latency = round(time.perf_counter() - start_time, 4)
            is_correct = (
                evaluate_prediction(predicted_option, question.answer_idx)
                if predicted_option is not None
                else False
            )
            retries_used = max(0, attempts_made - 1)

            return AttemptResult(
                attempt_index=attempt_idx,
                predicted_option=predicted_option,
                is_correct=is_correct,
                raw_response=raw_response,
                latency_seconds=latency,
                prompt_tokens=usage_acc.prompt_tokens if usage_acc.had_usage else None,
                candidate_tokens=usage_acc.candidate_tokens if usage_acc.had_usage else None,
                total_tokens=usage_acc.total_tokens if usage_acc.had_usage else None,
                cached_tokens=usage_acc.cached_tokens if usage_acc.had_usage else None,
                error=error_msg,
                parse_retries=retries_used,
            )

    async def run_single_question(
        self,
        question: MedQAQuestion,
        index: int,
        semaphore: asyncio.Semaphore,
        n_attempts: int | None = None,
        max_parse_retries: int | None = None,
    ) -> InferenceItemResult:
        """Execute one-shot inference across n_attempts for a single question."""
        if n_attempts is None:
            n_attempts = getattr(self.config, "n_attempts", 3)
        if max_parse_retries is None:
            max_parse_retries = getattr(self.config, "max_parse_retries", 3)

        prompt_text = format_one_shot_prompt(question)
        start_time = time.perf_counter()

        attempt_tasks = [
            self.run_single_attempt(
                question=question,
                question_idx=index,
                attempt_idx=attempt_idx,
                semaphore=semaphore,
                max_parse_retries=max_parse_retries,
            )
            for attempt_idx in range(n_attempts)
        ]
        attempts: list[AttemptResult] = list(await asyncio.gather(*attempt_tasks))

        total_latency = round(time.perf_counter() - start_time, 4)
        has_usage = any(a.prompt_tokens is not None for a in attempts)
        prompt_tokens = (
            sum(a.prompt_tokens or 0 for a in attempts) if has_usage else None
        )
        candidate_tokens = (
            sum(a.candidate_tokens or 0 for a in attempts) if has_usage else None
        )
        total_tokens = (
            sum(a.total_tokens or 0 for a in attempts) if has_usage else None
        )
        cached_tokens = (
            sum(a.cached_tokens or 0 for a in attempts) if has_usage else None
        )
        total_parse_retries = sum(a.parse_retries for a in attempts)

        correct_attempts = sum(1 for a in attempts if a.is_correct)
        is_all_correct = (
            correct_attempts == len(attempts) if attempts else False
        )
        error_msg = next((a.error for a in attempts if a.error), None)

        primary_predicted_option = (
            attempts[0].predicted_option if attempts else None
        )
        primary_raw_response = attempts[0].raw_response if attempts else ""

        return InferenceItemResult(
            question_id=question.question_id,
            meta_info=question.meta_info,
            question=question.question,
            options=question.options,
            ground_truth=question.answer_idx,
            ground_truth_answer=question.answer,
            prompt=prompt_text,
            attempts=attempts,
            predicted_option=primary_predicted_option,
            is_correct=is_all_correct,
            is_all_correct=is_all_correct,
            correct_attempts=correct_attempts,
            total_attempts=len(attempts),
            raw_response=primary_raw_response,
            latency_seconds=total_latency,
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            error=error_msg,
            parse_retries=total_parse_retries,
        )

    def _load_existing_results(
        self, output_filepath: str | Path | None
    ) -> dict[str, InferenceItemResult]:
        """Load already processed results from destination file if it exists."""
        if not output_filepath:
            return {}

        path = Path(output_filepath)
        if not path.exists() or not path.is_file():
            return {}

        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                return {}

            data = json.loads(content)
            results_data: list[dict[str, Any]] = []

            if isinstance(data, dict) and "results" in data:
                if isinstance(data["results"], list):
                    results_data = data["results"]
            elif isinstance(data, list):
                results_data = data

            existing_results: dict[str, InferenceItemResult] = {}
            for item in results_data:
                if isinstance(item, dict) and "question_id" in item:
                    item_res = InferenceItemResult.from_dict(item)
                    existing_results[str(item_res.question_id)] = item_res

            logger.info(
                "Loaded %d existing processed results from destination file: %s",
                len(existing_results),
                output_filepath,
            )
            return existing_results
        except Exception as exc:
            logger.warning(
                "Failed to read existing results from destination file %s: %s. Proceeding with fresh run.",
                output_filepath,
                exc,
            )
            return {}

    def _save_to_json(
        self,
        output_filepath: str,
        summary: WorkflowSummary,
        results: list[InferenceItemResult],
    ) -> None:
        """Write workflow summary and detailed results with attempts to JSON file atomically."""
        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results],
        }
        temp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp_path.replace(out_path)
        logger.info("Saved %d results to %s", len(results), output_filepath)

    def _build_summary(
        self,
        results: list[InferenceItemResult],
        total_time: float,
    ) -> WorkflowSummary:
        """Compute aggregate statistics and build WorkflowSummary."""
        return WorkflowSummary.from_results(
            results=results,
            model=self.config.model_name,
            dataset=self.config.dataset_name,
            config=self.config.dataset_config,
            split=self.config.dataset_split,
            n_attempts=self.config.n_attempts,
            total_time_seconds=total_time,
        )

    def _save_checkpoint(
        self,
        results_map: dict[str, InferenceItemResult],
        elapsed_time: float,
    ) -> None:
        """Save a snapshot of currently available results to the destination file."""
        if not self.config.output_filepath:
            return
        saved_results = list(results_map.values())
        summary = self._build_summary(saved_results, elapsed_time)
        self._save_to_json(self.config.output_filepath, summary, saved_results)

    async def run(
        self,
        questions: list[MedQAQuestion] | None = None,
        progress_callback: Callable[[int, int, InferenceItemResult], None]
        | None = None,
    ) -> tuple[WorkflowSummary, list[InferenceItemResult]]:
        """Run multi-attempt one-shot inference on the dataset and return summary + results."""
        if questions is None:
            questions = load_medqa_dataset(
                dataset_name=self.config.dataset_name,
                config_name=self.config.dataset_config,
                split=self.config.dataset_split,
                limit=self.config.limit,
                offset=self.config.offset,
            )

        total_questions = len(questions)
        logger.info(
            "Starting MedQA One-Shot Inference Workflow (total=%d, model=%s, n_attempts=%d, concurrency=%d)",
            total_questions,
            self.config.model_name,
            self.config.n_attempts,
            self.config.concurrency,
        )

        existing_results_map: dict[str, InferenceItemResult] = (
            self._load_existing_results(self.config.output_filepath)
            if self.config.output_filepath
            else {}
        )
        if existing_results_map:
            skipped_count = sum(
                1 for q in questions if str(q.question_id) in existing_results_map
            )
            logger.info(
                "Destination file '%s' exists: %d/%d questions in this run already processed and will be skipped.",
                self.config.output_filepath,
                skipped_count,
                total_questions,
            )

        semaphore = asyncio.Semaphore(self.config.concurrency)
        save_lock = asyncio.Lock()
        workflow_start = time.perf_counter()
        completed_count = 0
        in_progress_results: dict[str, InferenceItemResult] = dict(existing_results_map)

        async def _worker(idx: int, q: MedQAQuestion) -> InferenceItemResult:
            nonlocal completed_count
            qid = str(q.question_id)
            if qid in existing_results_map:
                res = existing_results_map[qid]
                logger.debug(
                    "Skipping question_id=%s (already processed in destination file)",
                    qid,
                )
            else:
                res = await self.run_single_question(
                    question=q,
                    index=idx,
                    semaphore=semaphore,
                    n_attempts=self.config.n_attempts,
                    max_parse_retries=self.config.max_parse_retries,
                )

            in_progress_results[qid] = res
            completed_count += 1

            if self.config.output_filepath and (
                completed_count % max(1, self.config.save_every_n) == 0
            ):
                async with save_lock:
                    self._save_checkpoint(
                        in_progress_results,
                        time.perf_counter() - workflow_start,
                    )

            if progress_callback:
                progress_callback(completed_count, total_questions, res)
            elif (
                completed_count % max(1, self.config.save_every_n) == 0
                or completed_count == total_questions
            ):
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Latest Q%s: True=%s Correct=%d/%d (All Correct=%s)",
                    completed_count,
                    total_questions,
                    (completed_count / total_questions) * 100,
                    res.question_id,
                    res.ground_truth,
                    res.correct_attempts,
                    res.total_attempts,
                    res.is_all_correct,
                )
            return res

        results = list(
            await asyncio.gather(*[_worker(i, q) for i, q in enumerate(questions)])
        )
        total_time = time.perf_counter() - workflow_start
        summary = self._build_summary(results, total_time)

        if self.config.output_filepath:
            for r in results:
                in_progress_results[str(r.question_id)] = r
            self._save_checkpoint(in_progress_results, total_time)

        logger.info(
            "Workflow finished in %.2fs. All Correct (Simple): %d/%d (%.2f%%)",
            total_time,
            summary.correct,
            total_questions,
            summary.accuracy * 100,
        )
        return summary, results
