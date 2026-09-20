"""Analysis script for MedQA inference results JSON files.

Computes and reports:
- Number of processed questions
- % correct answers (question-level and attempt-level)
- % of correct answers with all 3 attempts correct (simple questions)
- Average input (prompt) and output (candidate) tokens per question
- Total cost ($ USD) consumed based on Gemma 4 / Vertex AI pricing
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from results_store import OneShotResults, split_run_path


def load_difficult_question_ids(csv_path: str | Path) -> set[str]:
    """Read difficult question IDs from CSV file as a set of strings."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Difficult questions CSV file not found: {path}")

    with path.open(mode="r", encoding="utf-8-sig") as f:
        rows = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    if rows and rows[0].lower() in ("question_id", "id", "qid"):
        rows = rows[1:]

    return set(rows)


# Default pricing per 1,000,000 tokens (USD)
# Source: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing?e=48754805
DEFAULT_PRICING = {
    "gemma": {"input": 0.15, "output": 0.60, "cached": 0.015},
    "gpt-oss-120b": {"input": 0.09, "output": 0.36, "cached": 0.009},
    "gpt-oss-20b": {"input": 0.075, "output": 0.30, "cached": 0.0075},
    "gpt-oss": {"input": 0.09, "output": 0.36, "cached": 0.009},
    "gemini-3.8-flash": {"input": 1.50, "output": 7.50, "cached": 0.15},
    "gemini-flash-3.8": {"input": 1.50, "output": 7.50, "cached": 0.15},
    "gemini-3.7-flash": {"input": 1.50, "output": 7.50, "cached": 0.15},
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50, "cached": 0.15},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached": 0.15},
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00, "cached": 0.20},
}


def get_pricing_for_model(
    model_name: str | None = None,
    custom_input: float | None = None,
    custom_output: float | None = None,
    custom_cached: float | None = None,
) -> tuple[float, float, float]:
    """Determine pricing rates (input, output, cached per 1M tokens) for a given model."""
    input_price, output_price, cached_price = 0.15, 0.60, 0.015

    if model_name:
        model_lower = model_name.lower()
        for key, rates in DEFAULT_PRICING.items():
            if key in model_lower:
                input_price = rates["input"]
                output_price = rates["output"]
                cached_price = rates["cached"]
                break

    return (
        custom_input if custom_input is not None else input_price,
        custom_output if custom_output is not None else output_price,
        custom_cached if custom_cached is not None else cached_price,
    )


def _extract_item_metrics(item: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics from a single question result entry."""
    raw_attempts = item.get("attempts") or []
    has_error = bool(item.get("error")) or any(
        bool(a.get("error")) for a in raw_attempts if isinstance(a, dict)
    )

    if raw_attempts:
        total_attempts = len(raw_attempts)
        correct_attempts = sum(
            1 for a in raw_attempts if isinstance(a, dict) and bool(a.get("is_correct"))
        )
        prompt_tokens = sum(
            int(a.get("prompt_tokens") or 0)
            for a in raw_attempts
            if isinstance(a, dict)
        )
        candidate_tokens = sum(
            int(a.get("candidate_tokens") or 0)
            for a in raw_attempts
            if isinstance(a, dict)
        )
        cached_tokens = sum(
            int(a.get("cached_tokens") or 0)
            for a in raw_attempts
            if isinstance(a, dict)
        )
        latency_seconds = sum(
            float(a.get("latency_seconds") or 0.0)
            for a in raw_attempts
            if isinstance(a, dict)
        )
        dict_attempts = [a for a in raw_attempts if isinstance(a, dict)]
        first_attempt = None
        if dict_attempts:
            for a in dict_attempts:
                if a.get("attempt_index") == 0:
                    first_attempt = a
                    break
            if first_attempt is None:
                if any("attempt_index" in a for a in dict_attempts):
                    first_attempt = min(
                        dict_attempts,
                        key=lambda a: a.get("attempt_index", 0),
                    )
                else:
                    first_attempt = dict_attempts[0]
        is_first_correct = (
            bool(first_attempt.get("is_correct"))
            if first_attempt is not None
            else False
        )
    else:
        total_attempts = int(item.get("total_attempts") or 1)
        correct_attempts = (
            int(item["correct_attempts"])
            if item.get("correct_attempts") is not None
            else (1 if item.get("is_correct") else 0)
        )
        prompt_tokens = int(item.get("prompt_tokens") or 0)
        candidate_tokens = int(item.get("candidate_tokens") or 0)
        cached_tokens = int(item.get("cached_tokens") or 0)
        latency_seconds = float(item.get("latency_seconds") or 0.0)
        if item.get("is_first_correct") is not None:
            is_first_correct = bool(item["is_first_correct"])
        elif item.get("first_attempt_correct") is not None:
            is_first_correct = bool(item["first_attempt_correct"])
        elif item.get("is_correct") is not None:
            is_first_correct = bool(item["is_correct"])
        else:
            is_first_correct = False

    # Fallback to top-level fields if attempts lacked them
    prompt_tokens = prompt_tokens or int(item.get("prompt_tokens") or 0)
    candidate_tokens = candidate_tokens or int(item.get("candidate_tokens") or 0)
    cached_tokens = cached_tokens or int(item.get("cached_tokens") or 0)
    latency_seconds = latency_seconds or float(item.get("latency_seconds") or 0.0)

    is_all_correct = (
        bool(item["is_all_correct"])
        if item.get("is_all_correct") is not None
        else (correct_attempts == total_attempts and total_attempts > 0)
    )

    question_id = str(
        item.get("question_id", item.get("id", item.get("qid", "")))
    ).strip()

    return {
        "question_id": question_id,
        "has_error": has_error,
        "total_attempts": total_attempts,
        "correct_attempts": correct_attempts,
        "is_all_correct": is_all_correct,
        "is_any_correct": correct_attempts > 0,
        "is_first_correct": is_first_correct,
        "is_majority_correct": correct_attempts > (total_attempts / 2.0)
        if total_attempts > 0
        else False,
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "cached_tokens": cached_tokens,
        "latency_seconds": latency_seconds,
    }


def analyze_data(
    data: dict[str, Any] | list[dict[str, Any]],
    file_path: str = "",
    input_price: float | None = None,
    output_price: float | None = None,
    cached_price: float | None = None,
    difficult_questions_csv: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute aggregate evaluation metrics and pricing from loaded JSON data.

    If difficult_questions_csv (or difficult_ids) is provided, computes additional
    metrics broken down by 'difficult' and 'other' subsets.
    """
    model_name: str | None = None
    if isinstance(data, dict):
        summary_info = data.get("summary")
        if isinstance(summary_info, dict):
            model_name = summary_info.get("model")
        results_list = data.get("results", [])
    elif isinstance(data, list):
        results_list = data
    else:
        raise ValueError(
            f"Expected dict or list at root of JSON, got {type(data).__name__}"
        )

    items = [_extract_item_metrics(item) for item in results_list]
    total_q = len(items)

    completed_q = sum(1 for q in items if not q["has_error"])
    failed_q = sum(1 for q in items if q["has_error"])

    total_attempts = sum(q["total_attempts"] for q in items)
    total_correct = sum(q["correct_attempts"] for q in items)

    any_correct_q = sum(1 for q in items if q["is_any_correct"])
    all_3_correct_q = sum(1 for q in items if q["is_all_correct"])
    majority_correct_q = sum(1 for q in items if q["is_majority_correct"])
    first_correct_q = sum(1 for q in items if q["is_first_correct"])

    total_prompt = sum(q["prompt_tokens"] for q in items)
    total_candidate = sum(q["candidate_tokens"] for q in items)
    total_cached = sum(q["cached_tokens"] for q in items)
    total_tokens = total_prompt + total_candidate
    total_latency = sum(q["latency_seconds"] for q in items)

    rate_in, rate_out, rate_cached = get_pricing_for_model(
        model_name, input_price, output_price, cached_price
    )

    non_cached_prompt = max(0, total_prompt - total_cached)
    input_cost = (non_cached_prompt / 1_000_000.0) * rate_in
    cached_cost = (total_cached / 1_000_000.0) * rate_cached
    output_cost = (total_candidate / 1_000_000.0) * rate_out
    total_cost = input_cost + cached_cost + output_cost
    standard_cost = (total_prompt / 1_000_000.0) * rate_in + (
        total_candidate / 1_000_000.0
    ) * rate_out

    dist_counts = Counter(q["correct_attempts"] for q in items)
    distribution = dict(sorted(dist_counts.items(), reverse=True))

    res: dict[str, Any] = {
        "file_path": file_path,
        "model_name": model_name,
        "total_questions": total_q,
        "completed_questions": completed_q,
        "failed_questions": failed_q,
        "total_attempts": total_attempts,
        "total_correct_attempts": total_correct,
        "attempt_accuracy_pct": (total_correct / total_attempts * 100.0)
        if total_attempts > 0
        else 0.0,
        "questions_any_correct": any_correct_q,
        "questions_any_correct_pct": (any_correct_q / total_q * 100.0)
        if total_q > 0
        else 0.0,
        "questions_first_correct": first_correct_q,
        "questions_first_correct_pct": (first_correct_q / total_q * 100.0)
        if total_q > 0
        else 0.0,
        "questions_first_attempt_correct": first_correct_q,
        "questions_first_attempt_correct_pct": (first_correct_q / total_q * 100.0)
        if total_q > 0
        else 0.0,
        "questions_all_3_correct": all_3_correct_q,
        "questions_all_3_correct_pct": (all_3_correct_q / total_q * 100.0)
        if total_q > 0
        else 0.0,
        "pct_all_3_of_correct": (all_3_correct_q / any_correct_q * 100.0)
        if any_correct_q > 0
        else 0.0,
        "questions_majority_correct": majority_correct_q,
        "questions_majority_correct_pct": (majority_correct_q / total_q * 100.0)
        if total_q > 0
        else 0.0,
        "simple_questions_count": all_3_correct_q,
        "difficult_questions_count": total_q - all_3_correct_q,
        "total_prompt_tokens": total_prompt,
        "total_candidate_tokens": total_candidate,
        "total_tokens": total_tokens,
        "total_cached_tokens": total_cached,
        "avg_prompt_tokens_per_question": (total_prompt / total_q)
        if total_q > 0
        else 0.0,
        "avg_candidate_tokens_per_question": (total_candidate / total_q)
        if total_q > 0
        else 0.0,
        "avg_total_tokens_per_question": (total_tokens / total_q)
        if total_q > 0
        else 0.0,
        "avg_prompt_tokens_per_attempt": (total_prompt / total_attempts)
        if total_attempts > 0
        else 0.0,
        "avg_candidate_tokens_per_attempt": (total_candidate / total_attempts)
        if total_attempts > 0
        else 0.0,
        "input_price_per_1m": rate_in,
        "output_price_per_1m": rate_out,
        "cached_price_per_1m": rate_cached,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "cached_cost_usd": cached_cost,
        "total_cost_usd": total_cost,
        "standard_total_cost_usd": standard_cost,
        "cache_savings_usd": standard_cost - total_cost,
        "avg_cost_per_question_usd": (total_cost / total_q) if total_q > 0 else 0.0,
        "avg_cost_per_attempt_usd": (total_cost / total_attempts)
        if total_attempts > 0
        else 0.0,
        "total_latency_seconds": total_latency,
        "avg_latency_per_question": (total_latency / total_q) if total_q > 0 else 0.0,
        "attempts_distribution": distribution,
    }

    diff_csv = (
        difficult_questions_csv
        or kwargs.get("difficult_csv")
        or kwargs.get("difficult_questions")
        or kwargs.get("difficult_questions_path")
    )
    difficult_ids: set[str] | None = None
    if "difficult_ids" in kwargs and kwargs["difficult_ids"] is not None:
        difficult_ids = {str(qid).strip() for qid in kwargs["difficult_ids"]}
    elif diff_csv is not None:
        if isinstance(diff_csv, (set, list, tuple)):
            difficult_ids = {str(qid).strip() for qid in diff_csv}
        else:
            difficult_ids = load_difficult_question_ids(diff_csv)

    if difficult_ids is not None:
        diff_items = [q for q in items if q["question_id"] in difficult_ids]
        other_items = [q for q in items if q["question_id"] not in difficult_ids]

        diff_total = len(diff_items)
        diff_all_3 = sum(1 for q in diff_items if q["is_all_correct"])
        diff_any = sum(1 for q in diff_items if q["is_any_correct"])
        diff_first = sum(1 for q in diff_items if q["is_first_correct"])
        diff_all_3_pct = (diff_all_3 / diff_total * 100.0) if diff_total > 0 else 0.0
        diff_any_pct = (diff_any / diff_total * 100.0) if diff_total > 0 else 0.0
        diff_first_pct = (diff_first / diff_total * 100.0) if diff_total > 0 else 0.0

        other_total = len(other_items)
        other_all_3 = sum(1 for q in other_items if q["is_all_correct"])
        other_any = sum(1 for q in other_items if q["is_any_correct"])
        other_first = sum(1 for q in other_items if q["is_first_correct"])
        other_all_3_pct = (
            (other_all_3 / other_total * 100.0) if other_total > 0 else 0.0
        )
        other_any_pct = (other_any / other_total * 100.0) if other_total > 0 else 0.0
        other_first_pct = (
            (other_first / other_total * 100.0) if other_total > 0 else 0.0
        )

        diff_subset_dict = {
            "total_questions": diff_total,
            "questions_all_3_correct": diff_all_3,
            "questions_all_3_correct_pct": diff_all_3_pct,
            "questions_any_correct": diff_any,
            "questions_any_correct_pct": diff_any_pct,
            "questions_first_correct": diff_first,
            "questions_first_correct_pct": diff_first_pct,
            "questions_first_attempt_correct": diff_first,
            "questions_first_attempt_correct_pct": diff_first_pct,
        }
        other_subset_dict = {
            "total_questions": other_total,
            "questions_all_3_correct": other_all_3,
            "questions_all_3_correct_pct": other_all_3_pct,
            "questions_any_correct": other_any,
            "questions_any_correct_pct": other_any_pct,
            "questions_first_correct": other_first,
            "questions_first_correct_pct": other_first_pct,
            "questions_first_attempt_correct": other_first,
            "questions_first_attempt_correct_pct": other_first_pct,
        }

        res["difficult_questions_csv"] = (
            str(diff_csv)
            if diff_csv is not None and not isinstance(diff_csv, (set, list, tuple))
            else None
        )
        res["difficult_subset"] = diff_subset_dict
        res["other_subset"] = other_subset_dict
        res["subsets"] = {
            "difficult": diff_subset_dict,
            "other": other_subset_dict,
            "other_questions": other_subset_dict,
        }
        res["difficult_subset_questions"] = diff_total
        res["difficult_subset_all_3_correct"] = diff_all_3
        res["difficult_subset_all_3_correct_pct"] = diff_all_3_pct
        res["difficult_subset_any_correct"] = diff_any
        res["difficult_subset_any_correct_pct"] = diff_any_pct
        res["difficult_subset_first_correct"] = diff_first
        res["difficult_subset_first_correct_pct"] = diff_first_pct
        res["difficult_subset_first_attempt_correct"] = diff_first
        res["difficult_subset_first_attempt_correct_pct"] = diff_first_pct
        res["other_subset_questions"] = other_total
        res["other_subset_all_3_correct"] = other_all_3
        res["other_subset_all_3_correct_pct"] = other_all_3_pct
        res["other_subset_any_correct"] = other_any
        res["other_subset_any_correct_pct"] = other_any_pct
        res["other_subset_first_correct"] = other_first
        res["other_subset_first_correct_pct"] = other_first_pct
        res["other_subset_first_attempt_correct"] = other_first
        res["other_subset_first_attempt_correct_pct"] = other_first_pct

    return res


def analyze_file(
    file_path: str | Path,
    input_price: float | None = None,
    output_price: float | None = None,
    cached_price: float | None = None,
    difficult_questions_csv: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load and analyze one run: a stored run directory, or a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.is_dir():
        data = OneShotResults(*split_run_path(path)).load()
        if data is None:
            raise FileNotFoundError(f"No results stored in: {path}")
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    return analyze_data(
        data,
        file_path=str(path),
        input_price=input_price,
        output_price=output_price,
        cached_price=cached_price,
        difficult_questions_csv=difficult_questions_csv,
        **kwargs,
    )


def format_report(analysis: dict[str, Any]) -> str:
    """Format analysis metrics into a clear, structured terminal report."""
    lines = [
        "=" * 70,
        " MEDQA INFERENCE RESULTS & PRICING ANALYSIS",
        "=" * 70,
        f"File: {analysis.get('file_path') or 'N/A'}",
    ]
    if analysis.get("model_name"):
        lines.append(f"Model: {analysis['model_name']}")

    lines.extend(
        [
            "-" * 70,
            "1. QUESTION & ATTEMPT COUNTS",
            f"   • Number of processed questions: {analysis['total_questions']}",
            f"   • Successfully completed:        {analysis['completed_questions']}",
            f"   • Failed with error:             {analysis['failed_questions']}",
            f"   • Total inference attempts:      {analysis['total_attempts']}",
            "",
            "2. ACCURACY & CORRECTNESS",
            f"   • Overall attempt accuracy:      {analysis['attempt_accuracy_pct']:.2f}% ({analysis['total_correct_attempts']}/{analysis['total_attempts']} attempts correct)",
            f"   • % Correct (at least 1 correct): {analysis['questions_any_correct_pct']:.2f}% ({analysis['questions_any_correct']}/{analysis['total_questions']} questions)",
            f"   • % Majority correct (>=2 of 3):  {analysis['questions_majority_correct_pct']:.2f}% ({analysis['questions_majority_correct']}/{analysis['total_questions']} questions)",
            "",
            "3. ALL 3 ATTEMPTS CORRECT (SIMPLE QUESTIONS)",
            f"   • % of ALL questions (3/3 correct):        {analysis['questions_all_3_correct_pct']:.2f}% ({analysis['questions_all_3_correct']}/{analysis['total_questions']} questions)",
            f"   • % of CORRECT questions (all 3 correct):  {analysis['pct_all_3_of_correct']:.2f}% ({analysis['questions_all_3_correct']}/{analysis['questions_any_correct']} correct questions)",
            f"   • Simple questions count (all 3 correct):   {analysis['simple_questions_count']}",
            f"   • Difficult questions count (< 3 correct):  {analysis['difficult_questions_count']}",
            "",
            "   Correct attempts distribution per question:",
        ]
    )

    for correct_k, q_count in analysis.get("attempts_distribution", {}).items():
        pct = (
            (q_count / analysis["total_questions"] * 100.0)
            if analysis["total_questions"] > 0
            else 0.0
        )
        lines.append(f"     - {correct_k} correct: {q_count} questions ({pct:.1f}%)")

    lines.extend(
        [
            "",
            "4. TOKEN USAGE & LATENCY",
            f"   • Avg input (prompt) tokens / question:     {analysis['avg_prompt_tokens_per_question']:.2f}",
            f"   • Avg output (candidate) tokens / question: {analysis['avg_candidate_tokens_per_question']:.2f}",
            f"   • Avg total tokens / question:              {analysis['avg_total_tokens_per_question']:.2f}",
            f"   • Avg input tokens / attempt:               {analysis['avg_prompt_tokens_per_attempt']:.2f}",
            f"   • Avg output tokens / attempt:              {analysis['avg_candidate_tokens_per_attempt']:.2f}",
            f"   • Total prompt tokens:                      {analysis['total_prompt_tokens']:,}",
            f"   • Total candidate tokens:                   {analysis['total_candidate_tokens']:,}",
            f"   • Total cached tokens (cache hits):         {analysis['total_cached_tokens']:,}",
            f"   • Total tokens consumed:                    {analysis['total_tokens']:,}",
            f"   • Avg latency per question:                 {analysis['avg_latency_per_question']:.2f}s",
            f"   • Total runtime latency:                    {analysis['total_latency_seconds']:.2f}s",
            "",
            "5. ESTIMATED PRICING & TOTAL COST (USD)",
            f"   (Pricing rates: ${analysis['input_price_per_1m']:.2f}/1M input, ${analysis['output_price_per_1m']:.2f}/1M output, ${analysis['cached_price_per_1m']:.3f}/1M cached)",
            f"   • Input tokens cost:                        ${analysis['input_cost_usd']:.6f}",
            f"   • Output tokens cost:                       ${analysis['output_cost_usd']:.6f}",
            f"   • Cached tokens cost:                       ${analysis['cached_cost_usd']:.6f}",
            f"   • TOTAL $ CONSUMED:                         ${analysis['total_cost_usd']:.6f} USD",
            f"     ↳ (Standard rate without cache savings:   ${analysis['standard_total_cost_usd']:.6f} USD)",
            f"     ↳ (Cache hit discount savings:            ${analysis['cache_savings_usd']:.6f} USD)",
            f"   • Avg cost per question:                    ${analysis['avg_cost_per_question_usd']:.6f} USD",
            f"   • Avg cost per attempt:                     ${analysis['avg_cost_per_attempt_usd']:.6f} USD",
        ]
    )

    if "subsets" in analysis:
        diff = analysis["subsets"]["difficult"]
        other = analysis["subsets"]["other"]
        csv_info = (
            f" ({analysis['difficult_questions_csv']})"
            if analysis.get("difficult_questions_csv")
            else ""
        )
        lines.extend(
            [
                "",
                f"6. SUBSETS ANALYSIS (DIFFICULT VS OTHER QUESTIONS){csv_info}",
                f"   • 'Difficult' questions subset ({diff['total_questions']} questions):",
                f"     - % All 3 attempts correct:        {diff['questions_all_3_correct_pct']:.2f}% ({diff['questions_all_3_correct']}/{diff['total_questions']} questions)",
                f"     - % At least 1 attempt correct:    {diff['questions_any_correct_pct']:.2f}% ({diff['questions_any_correct']}/{diff['total_questions']} questions)",
                f"     - % First attempt correct:         {diff['questions_first_correct_pct']:.2f}% ({diff['questions_first_correct']}/{diff['total_questions']} questions)",
                f"   • 'Other' questions subset ({other['total_questions']} questions):",
                f"     - % All 3 attempts correct:        {other['questions_all_3_correct_pct']:.2f}% ({other['questions_all_3_correct']}/{other['total_questions']} questions)",
                f"     - % At least 1 attempt correct:    {other['questions_any_correct_pct']:.2f}% ({other['questions_any_correct']}/{other['total_questions']} questions)",
                f"     - % First attempt correct:         {other['questions_first_correct_pct']:.2f}% ({other['questions_first_correct']}/{other['total_questions']} questions)",
            ]
        )

    lines.append("=" * 70)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MedQA inference results JSON file and print key scaling metrics with total cost ($ USD)."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="results/single-step/gemma-4-26b-a4b-it-maas",
        help="Directory of a stored run, or a results JSON file",
    )
    parser.add_argument(
        "--difficult-questions-csv",
        "--difficult-csv",
        "--difficult-questions",
        "--difficult-questions-path",
        dest="difficult_questions_csv",
        default=None,
        help="Optional path to difficult questions CSV file to compute subset metrics for difficult and other questions",
    )
    parser.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="Custom price per 1M input tokens in USD (default from pricing table: 0.15 for Gemma 4)",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="Custom price per 1M output tokens in USD (default from pricing table: 0.60 for Gemma 4)",
    )
    parser.add_argument(
        "--cached-price",
        type=float,
        default=None,
        help="Custom price per 1M cached input tokens in USD (default: 0.015 for Gemma 4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format instead of text report",
    )

    args = parser.parse_args(argv)

    try:
        analysis = analyze_file(
            args.file,
            input_price=args.input_price,
            output_price=args.output_price,
            cached_price=args.cached_price,
            difficult_questions_csv=args.difficult_questions_csv,
        )
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return 1
    except Exception as err:
        print(f"Error processing {args.file}: {err}")
        return 1

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print(format_report(analysis))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
