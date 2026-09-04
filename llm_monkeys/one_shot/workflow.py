"""Workflow module for running MedQA one-shot inference using Google ADK."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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

from config import InferenceConfig, resolve_model_name
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
    thoughts_tokens: int = 0
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
        self.thoughts_tokens += (
            getattr(usage, "thoughts_token_count", 0)
            or getattr(usage, "reasoning_tokens", 0)
            or 0
        )


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
    thoughts_tokens: int | None = None
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
            thoughts_tokens=data.get("thoughts_tokens"),
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
    thoughts_tokens: int | None = None
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
                    thoughts_tokens=data.get("thoughts_tokens"),
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
            thoughts_tokens=data.get("thoughts_tokens"),
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
    total_thoughts_tokens: int = 0

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
        completed = sum(
            1 for r in results if not r.error and not any(a.error for a in r.attempts)
        )
        failed = sum(1 for r in results if r.error or any(a.error for a in r.attempts))
        simple_count = sum(
            1
            for r in results
            if r.is_all_correct and not r.error and not any(a.error for a in r.attempts)
        )
        difficult_count = total - simple_count
        correct = simple_count
        avg_latency = sum(r.latency_seconds for r in results) / total if total else 0.0

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
            total_thoughts_tokens=sum(r.thoughts_tokens or 0 for r in results),
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception represents a 429 rate limit or RESOURCE_EXHAUSTED error."""
    exc_name = type(exc).__name__
    if any(
        term in exc_name
        for term in ("RateLimit", "ResourceExhausted", "TooManyRequests")
    ):
        return True

    status_code = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "http_status", None)
    )
    if status_code in (429, "429", "RESOURCE_EXHAUSTED"):
        return True

    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("error", {})
        if isinstance(err, dict):
            if err.get("code") == 429 or err.get("status") == "RESOURCE_EXHAUSTED":
                return True

    msg = str(exc)
    msg_lower = msg.lower()
    patterns = (
        "429",
        "resource_exhausted",
        "throttled",
        "too many concurrent requests",
        "rate limit",
        "rate_limit",
        "quota exceeded",
    )
    return any(pattern in msg_lower for pattern in patterns)


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
        self.config.model_name = resolve_model_name(self.config.model_name)

        os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
        if self.config.project_id:
            os.environ.setdefault("VERTEXAI_PROJECT", self.config.project_id)
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.config.project_id)
        if self.config.location:
            os.environ.setdefault("VERTEXAI_LOCATION", self.config.location)

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

        rate_limit_max_retries = getattr(self.config, "rate_limit_max_retries", 10)
        max_retries = getattr(self.config, "max_retries", 5)
        total_retry_limit = max(max_retries, rate_limit_max_retries)

        for attempt in range(total_retry_limit + 1):
            curr_session_id = (
                f"{session_id}_try{attempt}" if attempt > 0 else session_id
            )
            try:
                try:
                    await self.session_service.create_session(
                        app_name="medqa_workflow_app",
                        user_id=user_id,
                        session_id=curr_session_id,
                    )
                except Exception:
                    pass

                output_parts: list[str] = []
                thought_parts: list[str] = []
                usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

                async for event in self.runner.run_async(
                    user_id=user_id,
                    session_id=curr_session_id,
                    new_message=content,
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                if getattr(part, "thought", False):
                                    thought_parts.append(part.text)
                                else:
                                    output_parts.append(part.text)
                    if event.usage_metadata:
                        usage_metadata = event.usage_metadata

                raw_text = "".join(output_parts)
                # If model only emitted thoughts and no non-thought text, fall back to thought content
                if not raw_text.strip() and thought_parts:
                    raw_text = "".join(thought_parts)

                return raw_text, usage_metadata

            except Exception as exc:
                is_rate_limit = is_rate_limit_error(exc)
                allowed_retries = (
                    rate_limit_max_retries if is_rate_limit else max_retries
                )

                if attempt >= allowed_retries:
                    logger.error(
                        "Exhausted all %d retries for session %s (is_rate_limit=%s): %s",
                        allowed_retries,
                        curr_session_id,
                        is_rate_limit,
                        exc,
                    )
                    raise

                retry_after = getattr(exc, "retry_after", None) or getattr(
                    exc, "retry_delay", None
                )
                if (
                    retry_after is not None
                    and isinstance(retry_after, (int, float))
                    and retry_after > 0
                ):
                    backoff = min(
                        self.config.max_retry_delay,
                        float(retry_after) + random.uniform(0.5, 2.0),
                    )
                elif is_rate_limit:
                    # Exponential backoff with higher jitter for rate limits to prevent thundering herd
                    backoff = min(
                        self.config.max_retry_delay,
                        (self.config.base_retry_delay * (1.5**attempt))
                        + random.uniform(1.0, 5.0),
                    )
                else:
                    backoff = min(
                        self.config.max_retry_delay,
                        (self.config.base_retry_delay * (2**attempt))
                        + random.uniform(0.5, 2.0),
                    )

                logger.warning(
                    "%sAttempt %d/%d failed for session %s: %s. Backing off for %.2fs...",
                    "[429 / Rate Limit] " if is_rate_limit else "",
                    attempt + 1,
                    allowed_retries,
                    curr_session_id,
                    exc,
                    backoff,
                    exc_info=not is_rate_limit,
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
                session_id = f"medqa_sess_{question.question_id}_q{question_idx}_a{attempt_idx}_p{parse_attempt}_{time.time_ns()}"
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
                        exc_info=True,
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
                candidate_tokens=usage_acc.candidate_tokens
                if usage_acc.had_usage
                else None,
                total_tokens=usage_acc.total_tokens if usage_acc.had_usage else None,
                cached_tokens=usage_acc.cached_tokens if usage_acc.had_usage else None,
                thoughts_tokens=usage_acc.thoughts_tokens
                if (usage_acc.had_usage and usage_acc.thoughts_tokens > 0)
                else None,
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
        total_tokens = sum(a.total_tokens or 0 for a in attempts) if has_usage else None
        cached_tokens = (
            sum(a.cached_tokens or 0 for a in attempts) if has_usage else None
        )
        has_thoughts = any(a.thoughts_tokens is not None for a in attempts)
        thoughts_tokens = (
            sum(a.thoughts_tokens or 0 for a in attempts) if has_thoughts else None
        )
        total_parse_retries = sum(a.parse_retries for a in attempts)

        correct_attempts = sum(1 for a in attempts if a.is_correct)
        is_all_correct = correct_attempts == len(attempts) if attempts else False
        error_msg = next((a.error for a in attempts if a.error), None)

        primary_predicted_option = attempts[0].predicted_option if attempts else None
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
            thoughts_tokens=thoughts_tokens,
            error=error_msg,
            parse_retries=total_parse_retries,
        )

    def _is_valid_result(self, res: InferenceItemResult | None) -> bool:
        """Check if an existing result is complete and free of errors."""
        if res is None:
            return False
        if res.error:
            return False
        if any(a.error for a in res.attempts):
            return False
        expected_attempts = getattr(self.config, "n_attempts", 3)
        effective_attempts = max(len(res.attempts), res.total_attempts)
        if expected_attempts > 0 and effective_attempts < expected_attempts:
            return False
        if not res.attempts and not res.predicted_option and not res.raw_response:
            return False
        return True

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
                exc_info=True,
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
            valid_cached_count = sum(
                1
                for q in questions
                if str(q.question_id) in existing_results_map
                and self._is_valid_result(existing_results_map[str(q.question_id)])
            )
            failed_cached_count = sum(
                1
                for q in questions
                if str(q.question_id) in existing_results_map
                and not self._is_valid_result(existing_results_map[str(q.question_id)])
            )
            logger.info(
                "Destination file '%s' exists: %d/%d questions in this run already processed and will be skipped, "
                "%d questions failed previously and will be re-processed.",
                self.config.output_filepath,
                valid_cached_count,
                total_questions,
                failed_cached_count,
            )

        semaphore = asyncio.Semaphore(self.config.concurrency)
        save_lock = asyncio.Lock()
        workflow_start = time.perf_counter()
        completed_count = 0
        in_progress_results: dict[str, InferenceItemResult] = dict(existing_results_map)

        async def _worker(idx: int, q: MedQAQuestion) -> InferenceItemResult:
            nonlocal completed_count
            qid = str(q.question_id)
            if qid in existing_results_map and self._is_valid_result(
                existing_results_map[qid]
            ):
                res = existing_results_map[qid]
                logger.debug(
                    "Skipping question_id=%s (already successfully processed in destination file)",
                    qid,
                )
            else:
                if qid in existing_results_map:
                    logger.info(
                        "Re-processing question_id=%s (previously failed with error: %s)",
                        qid,
                        existing_results_map[qid].error or "attempt error",
                    )
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
            if len(in_progress_results) == len(results):
                summary = self._build_summary(results, total_time)
            else:
                summary = self._build_summary(
                    list(in_progress_results.values()), total_time
                )

        logger.info(
            "Workflow finished in %.2fs. All Correct (Simple): %d/%d (%.2f%%)",
            total_time,
            summary.correct,
            summary.total_questions,
            summary.accuracy * 100,
        )
        return summary, results
