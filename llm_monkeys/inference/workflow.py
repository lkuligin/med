"""Workflow module for candidate-based MedQA reasoning using Google ADK."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import CandidateInferenceConfig
from dataset import MedQAQuestion
from inference._dataset import (
    format_answer_generation_prompt,
    format_fact_generation_prompt,
    load_difficult_questions,
)
from inference._schemas import (
    CandidateQuestionResult,
    CandidateResult,
    CandidateWorkflowSummary,
    StepTokenUsage,
)
from inference.agent import (
    create_answer_generation_agent,
    create_fact_generation_agent,
    create_runner,
)
from results_store import build_store
from inference.parser import (
    evaluate_prediction,
    extract_predicted_option,
    parse_medical_facts,
)

logger = logging.getLogger(__name__)


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
    if isinstance(response, dict) and isinstance(response.get("error"), dict):
        if (
            response["error"].get("code") == 429
            or response["error"].get("status") == "RESOURCE_EXHAUSTED"
        ):
            return True

    msg_lower = str(exc).lower()
    return any(
        p in msg_lower
        for p in (
            "429",
            "resource_exhausted",
            "throttled",
            "too many concurrent requests",
            "rate limit",
            "rate_limit",
            "quota exceeded",
            "request queue is full",
        )
    )


def _calculate_backoff(
    exc: Exception,
    attempt: int,
    is_rate_limit: bool,
    config: CandidateInferenceConfig,
) -> float:
    """Calculate exponential backoff duration with jitter."""
    retry_after = getattr(exc, "retry_after", None) or getattr(exc, "retry_delay", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        delay = float(retry_after) + random.uniform(0.5, 2.0)
    elif is_rate_limit:
        delay = (config.base_retry_delay * (1.5**attempt)) + random.uniform(1.0, 5.0)
    else:
        delay = (config.base_retry_delay * (2**attempt)) + random.uniform(0.5, 2.0)
    return min(config.max_retry_delay, delay)


class CandidateInferenceWorkflow:
    """Orchestrates candidate-based two-step reasoning inference for complex MedQA questions."""

    def __init__(
        self,
        config: CandidateInferenceConfig | None = None,
        fact_agent: Agent | None = None,
        fact_runner: Runner | None = None,
        answer_agent: Agent | None = None,
        answer_runner: Runner | None = None,
        session_service: InMemorySessionService | None = None,
    ) -> None:
        self.config = config or CandidateInferenceConfig()
        self.store = build_store(self.config)
        self._save_lock = threading.Lock()

        os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
        if self.config.project_id:
            os.environ.setdefault("VERTEXAI_PROJECT", self.config.project_id)
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.config.project_id)
        if self.config.location:
            os.environ.setdefault("VERTEXAI_LOCATION", self.config.location)

        self.session_service = session_service or InMemorySessionService()

        self.fact_agent = fact_agent or create_fact_generation_agent(self.config)
        self.fact_runner = fact_runner or create_runner(
            self.fact_agent,
            session_service=self.session_service,
            app_name="medqa_fact_gen_app",
        )

        self.answer_agent = answer_agent or create_answer_generation_agent(self.config)
        self.answer_runner = answer_runner or create_runner(
            self.answer_agent,
            session_service=self.session_service,
            app_name="medqa_answer_gen_app",
        )

    async def _run_agent_with_retry(
        self,
        runner: Runner,
        prompt_text: str,
        session_id: str,
        user_id: str,
        max_retries: int | None = None,
        rate_limit_max_retries: int | None = None,
    ) -> tuple[str, types.GenerateContentResponseUsageMetadata | None]:
        """Invoke an ADK runner with retry and exponential backoff."""
        max_retries = self.config.max_retries if max_retries is None else max_retries
        rate_limit_max_retries = (
            self.config.rate_limit_max_retries
            if rate_limit_max_retries is None
            else rate_limit_max_retries
        )

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)],
        )

        for attempt in range(rate_limit_max_retries + 1):
            curr_session = f"{session_id}_try{attempt}"
            try:
                try:
                    await self.session_service.create_session(
                        app_name=runner.app_name,
                        user_id=user_id,
                        session_id=curr_session,
                    )
                except Exception:
                    pass

                output_parts: list[str] = []
                thought_parts: list[str] = []
                usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=curr_session,
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

                raw_text = "".join(output_parts) or "".join(thought_parts)
                return raw_text, usage_metadata

            except Exception as exc:
                is_rate_limit = is_rate_limit_error(exc)
                allowed_retries = (
                    rate_limit_max_retries if is_rate_limit else max_retries
                )

                if attempt >= allowed_retries:
                    logger.error(
                        "Exhausted %d retries for session %s (rate_limit=%s): %s",
                        allowed_retries,
                        curr_session,
                        is_rate_limit,
                        exc,
                    )
                    raise

                backoff = _calculate_backoff(exc, attempt, is_rate_limit, self.config)
                logger.warning(
                    "%sAttempt %d/%d failed for session %s: %s. Backing off for %.2fs...",
                    "[Rate Limit] " if is_rate_limit else "",
                    attempt + 1,
                    allowed_retries,
                    curr_session,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("Failed to invoke agent after retries.")

    async def _invoke_step(
        self,
        runner: Runner,
        prompt: str,
        session_id: str,
        user_id: str,
    ) -> tuple[str, StepTokenUsage, float, str | None]:
        """Execute a single agent step with timing, usage extraction, and error handling."""
        t0 = time.perf_counter()
        raw_text = ""
        usage_meta = None
        error = None
        try:
            raw_text, usage_meta = await self._run_agent_with_retry(
                runner=runner,
                prompt_text=prompt,
                session_id=session_id,
                user_id=user_id,
            )
        except Exception as e:
            error = str(e)

        latency = round(time.perf_counter() - t0, 4)
        tokens = StepTokenUsage.from_usage_metadata(usage_meta)
        return raw_text, tokens, latency, error

    async def run_single_candidate(
        self,
        question: MedQAQuestion,
        candidate_idx: int,
        semaphore: asyncio.Semaphore,
    ) -> CandidateResult:
        """Run a two-step reasoning candidate under concurrency limit."""
        async with semaphore:
            timestamp = datetime.now(timezone.utc).isoformat()
            user_id = f"user_q{question.question_id}"
            base_session = (
                f"q{question.question_id}_c{candidate_idx}_{int(time.time() * 1000)}"
            )

            # Step 1: Generate atomic medical facts
            raw_facts, fact_tokens, fact_latency, fact_error = await self._invoke_step(
                runner=self.fact_runner,
                prompt=format_fact_generation_prompt(question),
                session_id=f"fact_{base_session}",
                user_id=user_id,
            )
            if fact_error:
                logger.error(
                    "Fact generation failed for Q%s candidate %d: %s",
                    question.question_id,
                    candidate_idx,
                    fact_error,
                )
                return CandidateResult(
                    candidate_index=candidate_idx,
                    facts=[],
                    facts_raw_response=raw_facts,
                    answer_raw_response="",
                    predicted_option=None,
                    is_correct=False,
                    fact_latency_seconds=fact_latency,
                    answer_latency_seconds=0.0,
                    total_latency_seconds=fact_latency,
                    fact_tokens=fact_tokens,
                    answer_tokens=StepTokenUsage(),
                    total_prompt_tokens=fact_tokens.prompt_tokens,
                    total_candidate_tokens=fact_tokens.candidate_tokens,
                    total_tokens=fact_tokens.total_tokens,
                    error=f"Fact generation error: {fact_error}",
                    timestamp=timestamp,
                )

            parsed_facts = parse_medical_facts(raw_facts)

            # Step 2: Generate final answer based on facts
            raw_ans, ans_tokens, ans_latency, ans_error = await self._invoke_step(
                runner=self.answer_runner,
                prompt=format_answer_generation_prompt(question, parsed_facts),
                session_id=f"ans_{base_session}",
                user_id=user_id,
            )
            if ans_error:
                logger.error(
                    "Answer generation failed for Q%s candidate %d: %s",
                    question.question_id,
                    candidate_idx,
                    ans_error,
                )

            predicted_opt = (
                extract_predicted_option(raw_ans, question.options)
                if not ans_error
                else None
            )
            is_correct = (
                evaluate_prediction(predicted_opt, question.answer_idx)
                if predicted_opt
                else False
            )

            return CandidateResult(
                candidate_index=candidate_idx,
                facts=parsed_facts,
                facts_raw_response=raw_facts,
                answer_raw_response=raw_ans,
                predicted_option=predicted_opt,
                is_correct=is_correct,
                fact_latency_seconds=fact_latency,
                answer_latency_seconds=ans_latency,
                total_latency_seconds=round(fact_latency + ans_latency, 4),
                fact_tokens=fact_tokens,
                answer_tokens=ans_tokens,
                total_prompt_tokens=fact_tokens.prompt_tokens
                + ans_tokens.prompt_tokens,
                total_candidate_tokens=fact_tokens.candidate_tokens
                + ans_tokens.candidate_tokens,
                total_tokens=fact_tokens.total_tokens + ans_tokens.total_tokens,
                error=f"Answer generation error: {ans_error}" if ans_error else None,
                timestamp=timestamp,
            )

    async def run_question_candidates(
        self,
        question: MedQAQuestion,
        n_candidates: int,
        semaphore: asyncio.Semaphore,
        existing_question_result: CandidateQuestionResult | None = None,
        on_candidate_complete: (
            Callable[[CandidateResult, CandidateQuestionResult], None] | None
        ) = None,
    ) -> CandidateQuestionResult:
        """Generate N candidate reasoning paths for a single question, reusing existing valid candidates."""
        candidates_by_index: dict[int, CandidateResult] = {}
        if existing_question_result:
            used_indices: set[int] = set()
            for c in existing_question_result.candidates:
                if not c.error:
                    idx = c.candidate_index
                    if idx is None or idx < 0 or idx in used_indices:
                        idx = 0
                        while idx in used_indices:
                            idx += 1
                        c.candidate_index = idx
                    used_indices.add(idx)
                    candidates_by_index[idx] = c

        needed_indices: list[int] = []
        next_idx = 0
        while len(candidates_by_index) + len(needed_indices) < n_candidates:
            if next_idx not in candidates_by_index:
                needed_indices.append(next_idx)
            next_idx += 1

        question_res = CandidateQuestionResult(
            question_id=str(question.question_id),
            meta_info=question.meta_info,
            question=question.question,
            options=question.options,
            ground_truth=question.answer_idx,
            ground_truth_answer=question.answer,
            candidates=[
                candidates_by_index[i] for i in sorted(candidates_by_index.keys())
            ],
        )
        question_res.update_aggregates()

        if not needed_indices:
            logger.info(
                "Question %s already has all %d candidates completed (valid=%d). Skipping.",
                question.question_id,
                n_candidates,
                len(candidates_by_index),
            )
            return question_res

        logger.info(
            "Generating %d missing candidates for Question %s (existing_valid=%d, target=%d)...",
            len(needed_indices),
            question.question_id,
            len(candidates_by_index),
            n_candidates,
        )

        async def _run_candidate_task(idx: int) -> CandidateResult:
            cand = await self.run_single_candidate(question, idx, semaphore)
            candidates_by_index[idx] = cand
            question_res.candidates = [
                candidates_by_index[i] for i in sorted(candidates_by_index.keys())
            ]
            question_res.update_aggregates()
            if on_candidate_complete:
                try:
                    on_candidate_complete(cand, question_res)
                except Exception as cb_err:
                    logger.warning(
                        "Error in on_candidate_complete callback: %s", cb_err
                    )
            return cand

        tasks = [_run_candidate_task(idx) for idx in needed_indices]
        await asyncio.gather(*tasks)

        question_res.candidates = [
            candidates_by_index[i] for i in sorted(candidates_by_index.keys())
        ]
        question_res.update_aggregates()
        return question_res

    def load_existing_results(self) -> dict[str, CandidateQuestionResult]:
        """Load previously stored results for this run, keyed by question id."""
        try:
            data = self.store.load()
            if data is None:
                return {}

            existing: dict[str, CandidateQuestionResult] = {}
            for item in data["results"]:
                if isinstance(item, dict) and "question_id" in item:
                    q_res = CandidateQuestionResult.from_dict(item)
                    existing[str(q_res.question_id)] = q_res

            logger.info(
                "Loaded %d existing question results from %s", len(existing), self.store
            )
            return existing
        except Exception as e:
            logger.warning(
                "Could not read existing results from %s: %s", self.store, e
            )
            return {}

    def save_results(
        self,
        results: list[CandidateQuestionResult],
        summary: CandidateWorkflowSummary | None = None,
    ) -> None:
        """Persist results and summary, one file per candidate.

        Records already on disk are left untouched, so a save costs only the
        candidates generated since the last one however long the run gets.
        """
        payload: dict[str, Any] = {
            "summary": summary.to_dict() if summary else None,
            "results": [r.to_dict() for r in results],
        }
        with self._save_lock:
            try:
                self.store.save(payload)
                logger.debug("Successfully saved %d results to %s", len(results), self.store)
            except Exception as e:
                logger.error("Failed to save results to %s: %s", self.store, e)
                raise

    async def run(
        self,
        questions: list[MedQAQuestion] | None = None,
    ) -> tuple[CandidateWorkflowSummary, list[CandidateQuestionResult]]:
        """Run candidate generation workflow on difficult questions."""
        start_time = time.perf_counter()

        if questions is None:
            questions = load_difficult_questions(
                csv_path=self.config.difficult_questions_path,
                dataset_name=self.config.dataset_name,
                config_name=self.config.dataset_config,
                split=self.config.dataset_split,
                limit=self.config.limit,
                offset=self.config.offset,
            )

        if not questions:
            logger.warning("No questions to evaluate.")
            empty_summary = CandidateWorkflowSummary.from_results(
                results=[],
                model=self.config.model_name,
                dataset=self.config.dataset_name,
                config=self.config.dataset_config,
                split=self.config.dataset_split,
                n_candidates=self.config.n_candidates,
                total_time_seconds=0.0,
            )
            return empty_summary, []

        existing_results = self.load_existing_results()
        all_results_map: dict[str, CandidateQuestionResult] = dict(existing_results)

        # Overview of what exists in destination file
        completed_count = 0
        partial_count = 0
        new_count = 0
        for q in questions:
            qid = str(q.question_id)
            if qid in existing_results:
                valid_count = sum(
                    1 for c in existing_results[qid].candidates if not c.error
                )
                if valid_count >= self.config.n_candidates:
                    completed_count += 1
                else:
                    partial_count += 1
            else:
                new_count += 1

        logger.info(
            "Starting CandidateInferenceWorkflow: %d questions in run, target N=%d candidates each, concurrency=%d, model=%s",
            len(questions),
            self.config.n_candidates,
            self.config.concurrency,
            self.config.model_name,
        )
        if existing_results:
            logger.info(
                "Results already stored in '%s' (%d total stored questions). "
                "Current run breakdown: %d already completed (will skip), "
                "%d need additional candidates, %d are completely new.",
                self.store,
                len(existing_results),
                completed_count,
                partial_count,
                new_count,
            )

        results: list[CandidateQuestionResult] = []
        semaphore = asyncio.Semaphore(self.config.concurrency)
        candidate_save_counter = 0

        def _on_candidate_done(
            _cand: CandidateResult, _qres: CandidateQuestionResult
        ) -> None:
            nonlocal candidate_save_counter
            candidate_save_counter += 1
            all_results_map[str(_qres.question_id)] = _qres
            if (
                self.config.save_every_n_candidates > 0
                and candidate_save_counter % self.config.save_every_n_candidates == 0
            ):
                checkpoint_summary = CandidateWorkflowSummary.from_results(
                    results=list(all_results_map.values()),
                    model=self.config.model_name,
                    dataset=self.config.dataset_name,
                    config=self.config.dataset_config,
                    split=self.config.dataset_split,
                    n_candidates=self.config.n_candidates,
                    total_time_seconds=round(time.perf_counter() - start_time, 2),
                )
                self.save_results(
                    list(all_results_map.values()), checkpoint_summary
                )

        for q_idx, question in enumerate(questions):
            qid = str(question.question_id)
            existing_q = existing_results.get(qid)

            # Fast path check: if question already has enough valid candidates, skip completely
            if existing_q is not None:
                valid_count = sum(1 for c in existing_q.candidates if not c.error)
                if valid_count >= self.config.n_candidates:
                    logger.info(
                        "Question %d/%d (ID: %s) already has %d/%d candidates completed. Skipping.",
                        q_idx + 1,
                        len(questions),
                        qid,
                        valid_count,
                        self.config.n_candidates,
                    )
                    existing_q.update_aggregates()
                    results.append(existing_q)
                    all_results_map[qid] = existing_q
                    continue

            logger.info(
                "Processing Question %d/%d (ID: %s)...",
                q_idx + 1,
                len(questions),
                qid,
            )

            q_res = await self.run_question_candidates(
                question=question,
                n_candidates=self.config.n_candidates,
                semaphore=semaphore,
                existing_question_result=existing_q,
                on_candidate_complete=_on_candidate_done,
            )
            results.append(q_res)
            all_results_map[qid] = q_res

            if (
                self.config.save_every_n_questions > 0
                and len(results) % self.config.save_every_n_questions == 0
            ):
                checkpoint_summary = CandidateWorkflowSummary.from_results(
                    results=list(all_results_map.values()),
                    model=self.config.model_name,
                    dataset=self.config.dataset_name,
                    config=self.config.dataset_config,
                    split=self.config.dataset_split,
                    n_candidates=self.config.n_candidates,
                    total_time_seconds=round(time.perf_counter() - start_time, 2),
                )
                self.save_results(
                    list(all_results_map.values()), checkpoint_summary
                )

        total_time = round(time.perf_counter() - start_time, 2)
        all_results_list = list(all_results_map.values())
        summary = CandidateWorkflowSummary.from_results(
            results=all_results_list,
            model=self.config.model_name,
            dataset=self.config.dataset_name,
            config=self.config.dataset_config,
            split=self.config.dataset_split,
            n_candidates=self.config.n_candidates,
            total_time_seconds=total_time,
        )

        self.save_results(all_results_list, summary)
        logger.info(
            "Workflow complete: %d total questions saved (%d in current run), %d total candidates, overall accuracy: %.2f%%, time: %.2fs",
            summary.total_questions,
            len(results),
            summary.total_candidates_generated,
            summary.overall_accuracy * 100,
            summary.total_time_seconds,
        )

        return summary, results


__all__ = [
    "CandidateInferenceWorkflow",
    "is_rate_limit_error",
]
