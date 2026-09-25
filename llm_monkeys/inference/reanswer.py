"""Answer again from facts already generated, with another answer prompt.

    python -m inference.reanswer --source-results-dir results \
        --results-dir results/experiments/reference-facts-cited-answer \
        --run-name qwen3.5-9b-nr-local --dataset mkieffer/Medbullets \
        --model Qwen/Qwen3.5-9B --answer-prompt cited-facts --max-tokens 4096

Every candidate keeps the facts, fact tokens and fact latency it was generated
with; only the answer step runs, and its outputs replace the stored ones. The
source run is read and never written. Candidates already present in the
destination are skipped, so a stopped run resumes where it left off.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys

from config import CandidateInferenceConfig, resolve_dataset_name
from dataset import MedQAQuestion
from inference._dataset import format_answer_generation_prompt
from inference._prompts import ANSWER_PROMPTS
from inference._schemas import (
    CandidateQuestionResult,
    CandidateResult,
    CandidateWorkflowSummary,
)
from inference.parser import (
    evaluate_prediction,
    extract_cited_facts,
    extract_predicted_option,
)
from inference.workflow import CandidateInferenceWorkflow
from results_store import CandidateResults, dataset_dir_for

logger = logging.getLogger(__name__)


class ReanswerWorkflow(CandidateInferenceWorkflow):
    """The step 2 workflow with the fact step replaced by stored facts."""

    async def reanswer_candidate(
        self, question: MedQAQuestion, source: CandidateResult
    ) -> CandidateResult:
        prompt = format_answer_generation_prompt(
            question, source.facts, self.config.answer_prompt
        )
        raw, tokens, latency, error = await self._invoke_step(
            runner=self.answer_runner,
            prompt=prompt,
            session_id=f"reans_q{question.question_id}_c{source.candidate_index}",
            user_id=f"user_q{question.question_id}",
        )
        predicted = extract_predicted_option(raw, question.options) if not error else None
        cited = (
            extract_cited_facts(raw, len(source.facts))
            if ANSWER_PROMPTS[self.config.answer_prompt].numbered_facts and not error
            else None
        )
        return dataclasses.replace(
            source,
            answer_raw_response=raw,
            predicted_option=predicted,
            is_correct=evaluate_prediction(predicted, question.answer_idx) if predicted else False,
            answer_latency_seconds=latency,
            total_latency_seconds=round(source.fact_latency_seconds + latency, 4),
            answer_tokens=tokens,
            total_prompt_tokens=source.fact_tokens.prompt_tokens + tokens.prompt_tokens,
            total_candidate_tokens=source.fact_tokens.candidate_tokens + tokens.candidate_tokens,
            total_tokens=source.fact_tokens.total_tokens + tokens.total_tokens,
            cited_facts=cited,
            error=f"Answer generation error: {error}" if error else source.error,
        )

    async def reanswer(self, source_results_dir: str) -> int:
        """Re-answer every stored candidate of the run; returns how many were done."""
        dataset = dataset_dir_for(self.config.dataset_name)
        source = CandidateResults(source_results_dir, self.config.run_name, dataset)
        stored = source.load()
        if stored is None:
            raise FileNotFoundError(f"no candidates stored in {source}")
        destination = CandidateResults(self.config.results_dir, self.config.run_name, dataset)

        semaphore = asyncio.Semaphore(self.config.concurrency)
        done = 0

        async def one_question(record: dict) -> None:
            nonlocal done
            qres = CandidateQuestionResult.from_dict(record)
            question = MedQAQuestion(
                question_id=qres.question_id, question=qres.question, options=qres.options,
                answer_idx=qres.ground_truth, answer=qres.ground_truth_answer,
                meta_info=qres.meta_info,
            )
            todo = [c for c in qres.candidates
                    if not c.error
                    and not destination.candidate_path(qres.question_id, c.candidate_index).exists()]

            async def guarded(c: CandidateResult) -> CandidateResult:
                async with semaphore:
                    return await self.reanswer_candidate(question, c)

            new = await asyncio.gather(*(guarded(c) for c in todo))
            if new:
                out = dataclasses.replace(qres, candidates=list(new))
                out.update_aggregates()
                self.save_results([out])
                done += len(new)

        await asyncio.gather(*(one_question(r) for r in stored["results"]))

        written = destination.load()
        if written:
            results = [CandidateQuestionResult.from_dict(r) for r in written["results"]]
            for r in results:
                r.update_aggregates()
            source_summary = stored.get("summary") or {}
            self.store.write_summary({
                **CandidateWorkflowSummary.from_results(
                    results, model=self.config.model_name, dataset=self.config.dataset_name,
                    config=self.config.dataset_config, split=self.config.dataset_split,
                    n_candidates=source_summary.get("n_candidates") or 0,
                    total_time_seconds=0.0,
                ).to_dict(),
                "reanswered_from": str(source),
            })
        return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-results-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--answer-prompt", choices=sorted(ANSWER_PROMPTS), required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=128)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = CandidateInferenceConfig(
        model_name=args.model, dataset_name=resolve_dataset_name(args.dataset),
        results_dir=args.results_dir, run_name=args.run_name,
        answer_prompt=args.answer_prompt, max_tokens=args.max_tokens,
        temperature=args.temperature, concurrency=args.concurrency,
    )
    config.validate()
    done = asyncio.run(ReanswerWorkflow(config).reanswer(args.source_results_dir))
    logger.info("re-answered %d candidates into %s", done, args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
