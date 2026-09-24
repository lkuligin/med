"""Workflow orchestration module for MedQA fact verification using Google ADK."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import VerifierConfig
from results_store import build_store, stored_results
from inference._dataset import load_difficult_question_ids
from inference._schemas import StepTokenUsage
from inference.parser import evaluate_prediction
from inference.workflow import _calculate_backoff, is_rate_limit_error
from verifier._dataset import Step2CandidateData, Step2QuestionData, load_step2_results
from verifier._prompts import format_fact_verification_prompt
from verifier._schemas import (
    CandidateVerificationResult,
    FactVerificationResult,
    QuestionVerificationResult,
    VerifierWorkflowSummary,
)
from verifier.agent import create_fact_verifier_agent, create_runner
from verifier.parser import parse_fact_verification

logger = logging.getLogger(__name__)


def keep_listed(
    questions: list[Step2QuestionData], question_ids: list[str]
) -> list[Step2QuestionData]:
    """The questions whose id is on the list, in their original order.

    Ids are compared without leading zeros: medbullets pads them to three
    digits in some places and not in others, and '1' and '001' are one question.
    """
    def norm(qid: object) -> str:
        return str(qid).strip().lstrip("0") or "0"

    wanted = {norm(q) for q in question_ids}
    kept = [q for q in questions if norm(q.question_id) in wanted]
    logger.info("Kept %d of %d questions on the list", len(kept), len(questions))
    return kept


class _Spread:
    """How many candidates one question may judge at once.

    One, while there are at least as many open questions as request slots:
    that is the schedule that keeps every slot filled with a distinct
    question and throws nothing away, since a question stops at its first
    valid candidate.

    More only once questions run out. With three questions left and sixteen
    slots, eleven of them would otherwise idle to the end of the run. Then
    each of the three may judge five candidates side by side - spread evenly
    rather than fourteen piled on one, so the work that turns out to have
    been unnecessary is a little from each rather than a lot from one.
    """

    def __init__(self, slots: int) -> None:
        self.slots = max(1, slots)
        self.active = 0

    def width(self) -> int:
        return max(1, self.slots // max(1, self.active))


class VerifierWorkflow:
    """Orchestrates fact verification of candidate reasoning paths from Step 2."""

    def __init__(
        self,
        config: VerifierConfig | None = None,
        verifier_agent: Agent | None = None,
        verifier_runner: Runner | None = None,
        session_service: InMemorySessionService | None = None,
    ) -> None:
        self.config = config or VerifierConfig()
        # The store also carries what step 3 reads: whichever layout is in
        # use, the candidates being judged are reached as store.candidates.
        self.store = build_store(self.config)

        os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
        if self.config.project_id:
            os.environ.setdefault("VERTEXAI_PROJECT", self.config.project_id)
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.config.project_id)
        if self.config.location:
            os.environ.setdefault("VERTEXAI_LOCATION", self.config.location)

        self.session_service = session_service or InMemorySessionService()
        self.verifier_agent = verifier_agent or create_fact_verifier_agent(self.config)
        self.verifier_runner = verifier_runner or create_runner(
            self.verifier_agent,
            session_service=self.session_service,
            app_name="medqa_verifier_app",
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
        """Invoke an ADK runner with retry and exponential backoff on errors and rate limits."""
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

                async with asyncio.timeout(120.0):
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
                        "Exhausted %d retries for verifier session %s (rate_limit=%s): %s",
                        allowed_retries,
                        curr_session,
                        is_rate_limit,
                        exc,
                    )
                    raise

                backoff = _calculate_backoff(exc, attempt, is_rate_limit, self.config)
                logger.warning(
                    "%sAttempt %d/%d failed for verifier session %s: %s. Backing off for %.2fs...",
                    "[Rate Limit] " if is_rate_limit else "",
                    attempt + 1,
                    allowed_retries,
                    curr_session,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("Failed to invoke verifier agent after retries.")

    async def verify_single_fact(
        self,
        question_text: str,
        options: dict[str, str],
        fact: str,
        session_id: str,
        user_id: str,
        semaphore: asyncio.Semaphore,
    ) -> FactVerificationResult:
        """Verify a single atomic medical fact with the verifier LLM-as-a-judge under semaphore."""
        prompt = format_fact_verification_prompt(question_text, options, fact)
        t0 = time.perf_counter()
        raw_text = ""
        usage_meta = None
        error = None

        async with semaphore:
            try:
                raw_text, usage_meta = await self._run_agent_with_retry(
                    runner=self.verifier_runner,
                    prompt_text=prompt,
                    session_id=session_id,
                    user_id=user_id,
                )
            except Exception as exc:
                error = str(exc)
                logger.error("Error verifying fact: %s - %s", fact[:60], exc)

        latency = round(time.perf_counter() - t0, 4)
        tokens = StepTokenUsage.from_usage_metadata(usage_meta)

        if error:
            verdict, rationale = 0, f"Verification failed: {error}"
        else:
            verdict, rationale = parse_fact_verification(raw_text)

        return FactVerificationResult(
            fact=fact,
            verdict=verdict,
            is_correct=(verdict == 1 and error is None),
            rationale=rationale,
            raw_response=raw_text,
            latency_seconds=latency,
            tokens=tokens,
            error=error,
        )

    async def verify_candidate(
        self,
        question: Step2QuestionData,
        candidate: Step2CandidateData,
        attempt_number: int,
        semaphore: asyncio.Semaphore,
    ) -> CandidateVerificationResult:
        """Verify all atomic facts for a single candidate reasoning path."""
        t0 = time.perf_counter()
        facts = candidate.facts or []
        is_ans_correct = evaluate_prediction(
            candidate.predicted_option, question.ground_truth
        )

        if not facts:
            return CandidateVerificationResult(
                candidate_index=candidate.candidate_index,
                attempt_number=attempt_number,
                facts=[],
                fact_verifications=[],
                all_facts_correct=False,
                predicted_option=candidate.predicted_option,
                ground_truth=question.ground_truth,
                is_correct=is_ans_correct,
                total_latency_seconds=round(
                time.perf_counter()
                - t0
                + sum(cv.total_latency_seconds for cv in prior),
                4,
            ),
                total_tokens=0,
                total_prompt_tokens=0,
                total_candidate_tokens=0,
                error="Candidate produced no facts to verify",
            )

        user_id = f"user_q{question.question_id}"
        base_session = (
            f"q{question.question_id}_c{candidate.candidate_index}_a{attempt_number}"
        )

        async def _verify_fact(f_idx: int, fact_text: str) -> FactVerificationResult:
            sess_id = f"vf_{base_session}_f{f_idx}_{uuid.uuid4().hex[:8]}"
            return await self.verify_single_fact(
                question_text=question.question,
                options=question.options,
                fact=fact_text,
                session_id=sess_id,
                user_id=user_id,
                semaphore=semaphore,
            )

        fact_verifications: list[FactVerificationResult] = []
        if getattr(self.config, "early_stop_facts", False):
            for f_idx, fact in enumerate(facts):
                fv = await _verify_fact(f_idx, fact)
                fact_verifications.append(fv)
                if not fv.is_correct:
                    break
        else:
            fact_verifications = await asyncio.gather(
                *[_verify_fact(i, f) for i, f in enumerate(facts)]
            )

        all_correct = len(fact_verifications) == len(facts) and all(
            fv.verdict == 1 for fv in fact_verifications
        )

        return CandidateVerificationResult(
            candidate_index=candidate.candidate_index,
            attempt_number=attempt_number,
            facts=facts,
            fact_verifications=fact_verifications,
            all_facts_correct=all_correct,
            predicted_option=candidate.predicted_option,
            ground_truth=question.ground_truth,
            is_correct=is_ans_correct,
            total_latency_seconds=round(time.perf_counter() - t0, 4),
            total_tokens=sum(fv.tokens.total_tokens for fv in fact_verifications),
            total_prompt_tokens=sum(
                fv.tokens.prompt_tokens for fv in fact_verifications
            ),
            total_candidate_tokens=sum(
                fv.tokens.candidate_tokens for fv in fact_verifications
            ),
            error=None,
        )

    async def verify_question(
        self,
        question: Step2QuestionData,
        semaphore: asyncio.Semaphore,
        on_candidate_complete: (
            Callable[[CandidateVerificationResult, int, int], None] | None
        ) = None,
        prior: list[CandidateVerificationResult] | None = None,
        spread: _Spread | None = None,
    ) -> QuestionVerificationResult:
        """Evaluate candidates for a question, verifying their atomic facts and computing curve metrics.

        Args:
            question: The question and its candidates.
            semaphore: Limits concurrent judge requests.
            on_candidate_complete: Called after each candidate is judged.
            prior: Verdicts this question already has, from an earlier run that
                reached the end of its candidates without finding a valid one.
                Those candidates are not judged again; the metrics below are
                computed over the old verdicts and the new ones together.
        """
        t0 = time.perf_counter()
        candidates = question.candidates or []
        max_cands = self.config.max_candidates_per_question
        if max_cands and max_cands > 0:
            candidates = candidates[:max_cands]

        prior = list(prior or [])
        already_judged = {cv.candidate_index for cv in prior}
        candidates = [c for c in candidates if c.candidate_index not in already_judged]

        total_candidates = len(prior) + len(candidates)
        logger.info(
            "Starting verification for Question %s (%d candidates available, "
            "%d already judged, ground_truth=%s)",
            question.question_id,
            total_candidates,
            len(prior),
            question.ground_truth,
        )

        completed_count = len(prior)

        async def _verify_candidate_task(
            attempt_number: int,
            candidate: Step2CandidateData,
        ) -> CandidateVerificationResult:
            nonlocal completed_count
            try:
                cand_res = await self.verify_candidate(
                    question=question,
                    candidate=candidate,
                    attempt_number=attempt_number,
                    semaphore=semaphore,
                )
            except Exception as e:
                logger.error(
                    "Unexpected error verifying candidate %d (attempt #%d) for question %s: %s",
                    candidate.candidate_index,
                    attempt_number,
                    question.question_id,
                    e,
                )
                cand_res = CandidateVerificationResult(
                    candidate_index=candidate.candidate_index,
                    attempt_number=attempt_number,
                    facts=candidate.facts or [],
                    fact_verifications=[],
                    all_facts_correct=False,
                    predicted_option=candidate.predicted_option,
                    ground_truth=question.ground_truth,
                    is_correct=evaluate_prediction(
                        candidate.predicted_option, question.ground_truth
                    ),
                    total_latency_seconds=0.0,
                    total_tokens=0,
                    total_prompt_tokens=0,
                    total_candidate_tokens=0,
                    error=str(e),
                )

            completed_count += 1
            if cand_res.all_facts_correct:
                logger.info(
                    "Question %s [%d/%d candidates]: Candidate %d (attempt #%d) ALL facts correct. "
                    "Predicted option: %s, Ground truth: %s. Answer correct: %s",
                    question.question_id,
                    completed_count,
                    total_candidates,
                    candidate.candidate_index,
                    attempt_number,
                    candidate.predicted_option,
                    question.ground_truth,
                    cand_res.is_correct,
                )
            else:
                correct_count = sum(
                    1 for fv in cand_res.fact_verifications if fv.verdict == 1
                )
                logger.info(
                    "Question %s [%d/%d candidates]: Candidate %d (attempt #%d) had incorrect facts (%d/%d correct).",
                    question.question_id,
                    completed_count,
                    total_candidates,
                    candidate.candidate_index,
                    attempt_number,
                    correct_count,
                    len(cand_res.fact_verifications),
                )

            if on_candidate_complete:
                try:
                    on_candidate_complete(cand_res, completed_count, total_candidates)
                except Exception as cb_err:
                    logger.warning("on_candidate_complete callback error: %s", cb_err)

            return cand_res

        candidate_verifications: list[CandidateVerificationResult] = list(prior)
        if getattr(self.config, "early_stop_candidates", True):
            position = 0
            found = False
            while position < len(candidates) and not found:
                # One candidate at a time while questions still fill the
                # slots; several once they do not, so the tail of a run does
                # not leave most of the concurrency idle. See _Spread.
                width = spread.width() if spread is not None else 1
                batch = candidates[position:position + width]
                judged = await asyncio.gather(*[
                    _verify_candidate_task(len(prior) + position + offset + 1,
                                           candidate)
                    for offset, candidate in enumerate(batch)
                ])
                for cand_res in judged:
                    candidate_verifications.append(cand_res)
                    if cand_res.all_facts_correct and not found:
                        found = True
                        logger.info(
                            "Question %s: Early stopping at attempt #%d "
                            "(candidate %d); found first candidate with ALL "
                            "statements correct.",
                            question.question_id,
                            cand_res.attempt_number,
                            cand_res.candidate_index,
                        )
                position += len(batch)
        else:
            candidate_verifications += list(
                await asyncio.gather(
                    *[
                        _verify_candidate_task(attempt_number, candidate)
                        for attempt_number, candidate in enumerate(
                            candidates, start=len(prior) + 1
                        )
                    ]
                )
            )

        correct_candidates = [
            cv for cv in candidate_verifications if cv.all_facts_correct
        ]
        num_correct_candidates = len(correct_candidates)
        right_answers_count = sum(1 for cv in correct_candidates if cv.is_correct)
        wrong_answers_count = sum(1 for cv in correct_candidates if not cv.is_correct)

        first_assumptions_only = next(
            (cv for cv in candidate_verifications if cv.all_facts_correct),
            None,
        )
        first_all_and_answer = next(
            (
                cv
                for cv in candidate_verifications
                if cv.all_facts_correct and cv.is_correct
            ),
            None,
        )

        pos_assumptions_only = (
            first_assumptions_only.attempt_number if first_assumptions_only else None
        )
        cand_idx_assumptions_only = (
            first_assumptions_only.candidate_index if first_assumptions_only else None
        )

        pos_all_and_answer = (
            first_all_and_answer.attempt_number if first_all_and_answer else None
        )
        cand_idx_all_and_answer = (
            first_all_and_answer.candidate_index if first_all_and_answer else None
        )

        if first_assumptions_only is not None:
            found_valid = True
            selected_cand_idx = first_assumptions_only.candidate_index
            selected_attempt = first_assumptions_only.attempt_number
            selected_pred_opt = first_assumptions_only.predicted_option
            is_answer_correct = first_assumptions_only.is_correct
        else:
            found_valid = False
            selected_cand_idx = None
            selected_attempt = None
            selected_pred_opt = None
            is_answer_correct = None

        pos_assump_str = (
            f"attempt #{pos_assumptions_only} (candidate_index={cand_idx_assumptions_only})"
            if pos_assumptions_only is not None
            else "None"
        )
        pos_all_ans_str = (
            f"attempt #{pos_all_and_answer} (candidate_index={cand_idx_all_and_answer})"
            if pos_all_and_answer is not None
            else "None"
        )
        logger.info(
            "Question %s verification complete: evaluated %d candidate(s). "
            "# of correct candidates: %d (for correct candidates: %d right answers, %d wrong answers). "
            "Position of first correct candidate (only first assumptions): %s. "
            "Position of first correct candidate (all assumptions and answer): %s.",
            question.question_id,
            len(candidate_verifications),
            num_correct_candidates,
            right_answers_count,
            wrong_answers_count,
            pos_assump_str,
            pos_all_ans_str,
        )

        if not found_valid:
            logger.warning(
                "Question %s: Checked all %d candidates; NO candidate had all facts correct.",
                question.question_id,
                len(candidate_verifications),
            )

        return QuestionVerificationResult(
            question_id=question.question_id,
            meta_info=question.meta_info,
            question=question.question,
            options=question.options,
            ground_truth=question.ground_truth,
            ground_truth_answer=question.ground_truth_answer,
            candidates_evaluated=len(candidate_verifications),
            found_valid_candidate=found_valid,
            selected_candidate_index=selected_cand_idx,
            attempt_number=selected_attempt,
            predicted_option=selected_pred_opt,
            is_correct=is_answer_correct,
            candidate_verifications=candidate_verifications,
            total_latency_seconds=round(time.perf_counter() - t0, 4),
            total_tokens=sum(cv.total_tokens for cv in candidate_verifications),
            total_prompt_tokens=sum(
                cv.total_prompt_tokens for cv in candidate_verifications
            ),
            total_candidate_tokens=sum(
                cv.total_candidate_tokens for cv in candidate_verifications
            ),
            num_correct_candidates=num_correct_candidates,
            correct_candidates_right_answers=right_answers_count,
            correct_candidates_wrong_answers=wrong_answers_count,
            first_correct_candidate_pos_assumptions_only=pos_assumptions_only,
            first_correct_candidate_pos_all_and_answer=pos_all_and_answer,
            error=None,
        )

    def load_existing_results(self) -> dict[str, QuestionVerificationResult]:
        """Load this judge's existing verdicts for resumption."""
        try:
            data = self.store.load()
            if data is None:
                return {}

            loaded = {
                item["question_id"]: QuestionVerificationResult.from_dict(item)
                for item in stored_results(data)
                if isinstance(item, dict) and "question_id" in item
            }
            logger.info("Resumed %d verified questions from %s", len(loaded), self.store)
            return loaded
        except Exception as e:
            logger.warning(
                "Failed to load existing results from %s: %s. Starting fresh.",
                self.store,
                e,
            )
            return {}

    def save_results(
        self,
        results: list[QuestionVerificationResult],
        summary: VerifierWorkflowSummary | None = None,
    ) -> None:
        """Persist verification results and summary, one file per verdict."""
        payload: dict[str, Any] = {
            "summary": summary.to_dict() if summary else None,
            "results": [r.to_dict() for r in results],
        }
        try:
            self.store.save(payload)
            logger.debug(
                "Successfully saved %d verification results to %s",
                len(results),
                self.store,
            )
        except Exception as e:
            logger.error("Failed to save results to %s: %s", self.store, e)
            raise

    async def run(
        self,
        questions: list[Step2QuestionData] | None = None,
        on_question_complete: (
            Callable[
                [QuestionVerificationResult, list[QuestionVerificationResult]], None
            ]
            | None
        ) = None,
        on_candidate_complete: (
            Callable[[CandidateVerificationResult, int, int], None] | None
        ) = None,
    ) -> tuple[VerifierWorkflowSummary, list[QuestionVerificationResult]]:
        """Execute the Step 3 verification workflow across questions."""
        start_time = time.perf_counter()

        if questions is None:
            questions = load_step2_results(
                store=self.store.candidates,
                limit=self.config.limit,
                offset=self.config.offset,
            )
        if self.config.difficult_questions:
            questions = keep_listed(
                questions, load_difficult_question_ids(self.config.difficult_questions)
            )

        if not questions:
            logger.warning("No questions found for verification.")
            empty_summary = VerifierWorkflowSummary.from_results(
                results=[],
                model=self.config.resolved_model_name,
                run_name=self.store.run_name,
                judge_name=self.store.judge_name,
                temperature=self.config.temperature,
                total_time_seconds=0.0,
            )
            return empty_summary, []

        logger.info(
            "Starting VerifierWorkflow: %d questions, concurrency=%d, model=%s",
            len(questions),
            self.config.concurrency,
            self.config.resolved_model_name,
        )

        existing_results = self.load_existing_results()
        results: list[QuestionVerificationResult] = []
        semaphore = asyncio.Semaphore(self.config.concurrency)
        save_lock = asyncio.Lock()
        # As many questions open as requests allowed: enough to fill the
        # semaphore when facts are sequential, and a stopped run loses the
        # partial work of only that many questions, not of every one.
        open_questions = asyncio.Semaphore(self.config.concurrency)
        # How wide a question may go. One while questions fill the slots;
        # wider once they run out, so a run's tail does not leave most of the
        # concurrency idle waiting for the last few questions.
        # None unless asked for: without it a question judges one candidate at
        # a time to the end of the run, which leaves slots idle over the last
        # few questions but sends no request that the verdict does not need.
        spread = (_Spread(self.config.concurrency)
                  if getattr(self.config, "speculate_tail", False) else None)

        # Questions run side by side and the semaphore alone bounds requests in
        # flight. Within a question the candidates are sequential by design (the
        # search stops at the first valid one), and with early_stop_facts so are
        # its facts - one question at a time would then keep a single request in
        # flight whatever the concurrency says.
        async def _verify(q_idx: int, question: Step2QuestionData) -> None:
            previous = existing_results.get(question.question_id)
            prior: list[CandidateVerificationResult] = []
            if previous is not None:
                judged = {cv.candidate_index for cv in previous.candidate_verifications}
                unjudged = [
                    c for c in (question.candidates or [])
                    if c.candidate_index not in judged
                ]
                # Nothing to add when a valid candidate was already found: the
                # run stops at it, so later candidates would never be reached.
                if previous.found_valid_candidate or not unjudged:
                    logger.info(
                        "Question %s already verified in previous run, skipping.",
                        question.question_id,
                    )
                    results.append(previous)
                    return
                prior = list(previous.candidate_verifications)
                logger.info(
                    "Question %s: %d candidates judged before without a valid one, "
                    "%d new candidates to judge.",
                    question.question_id,
                    len(prior),
                    len(unjudged),
                )

            logger.info(
                "Processing Question %d/%d (ID: %s)...",
                q_idx + 1,
                len(questions),
                question.question_id,
            )

            if spread is not None:
                spread.active += 1
            try:
                q_res = await self.verify_question(
                    question=question,
                    semaphore=semaphore,
                    on_candidate_complete=on_candidate_complete,
                    prior=prior,
                    spread=spread,
                )
            finally:
                if spread is not None:
                    spread.active -= 1
            async with save_lock:
                results.append(q_res)

                if on_question_complete:
                    try:
                        on_question_complete(q_res, results)
                    except Exception as e:
                        logger.warning("on_question_complete callback error: %s", e)

                if (
                    self.config.save_every_n_questions > 0
                    and len(results) % self.config.save_every_n_questions == 0
                ):
                    interim_summary = VerifierWorkflowSummary.from_results(
                        results=results,
                        model=self.config.resolved_model_name,
                        run_name=self.store.run_name,
                        judge_name=self.store.judge_name,
                        temperature=self.config.temperature,
                        total_time_seconds=round(time.perf_counter() - start_time, 2),
                    )
                    self.save_results(results=results, summary=interim_summary)

        async def _bounded(q_idx: int, question: Step2QuestionData) -> None:
            async with open_questions:
                await _verify(q_idx, question)

        await asyncio.gather(*[_bounded(i, q) for i, q in enumerate(questions)])
        order = {q.question_id: i for i, q in enumerate(questions)}
        results.sort(key=lambda r: order.get(r.question_id, len(order)))

        total_time = time.perf_counter() - start_time
        summary = VerifierWorkflowSummary.from_results(
            results=results,
            model=self.config.resolved_model_name,
            run_name=self.store.run_name,
            judge_name=self.store.judge_name,
            temperature=self.config.temperature,
            total_time_seconds=total_time,
        )

        self.save_results(results=results, summary=summary)

        logger.info(
            "VerifierWorkflow completed: %d questions processed, %d valid candidates found, "
            "%d correct answers (Accuracy: %.2f%%) in %.2fs. "
            "Total # of correct candidates: %d (for correct candidates: %d right answers, %d wrong answers). "
            "First correct candidate positions (only first assumptions): %s (avg: %.2f). "
            "First correct candidate positions (all assumptions and answer): %s (avg: %.2f).",
            summary.total_questions,
            summary.questions_with_valid_candidate,
            summary.correct_answers,
            summary.accuracy * 100,
            summary.total_time_seconds,
            summary.total_correct_candidates,
            summary.total_correct_candidates_right_answers,
            summary.total_correct_candidates_wrong_answers,
            summary.first_positions_assumptions_only,
            summary.avg_position_assumptions_only,
            summary.first_positions_all_and_answer,
            summary.avg_position_all_and_answer,
        )
        return summary, results


__all__ = [
    "VerifierWorkflow",
]
