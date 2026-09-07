#!/usr/bin/env python3
"""Script to analyze MedQA verifier outputs and plot accuracy vs candidate generation.

Plots % of questions answered correctly after 1, 2, ..., k candidate generations.
A question is considered answered correctly when at least one candidate generated
up to attempt k has passed all verification checks AND has the correct answer.

Also supports plotting the first-valid candidate selection policy (rejection sampling)
and the unverified candidate baseline for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Set headless matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from analyze_results import get_pricing_for_model  # noqa: E402
from verifier._schemas import CandidateVerificationResult  # noqa: E402
from verifier._schemas import QuestionVerificationResult

logger = logging.getLogger(__name__)


@dataclass
class VerifierCurveData:
    """Evaluation curve metrics across candidate counts k = 1..K."""

    k_values: list[int]
    total_questions: int
    max_candidates: int

    # Primary Metric: At least one candidate in 1..k passed all checks AND answer is correct
    verified_correct_pct: list[float]
    verified_correct_counts: list[int]

    # First Valid Policy: First candidate in 1..k with all facts correct has answer correct
    first_valid_correct_pct: list[float]
    first_valid_correct_counts: list[int]

    # Unverified Baseline: At least one candidate in 1..k has correct answer (ignoring verification)
    unverified_correct_pct: list[float]
    unverified_correct_counts: list[int]

    # Valid Coverage: At least one candidate in 1..k has all facts correct
    has_valid_candidate_pct: list[float]
    has_valid_candidate_counts: list[int]

    # Earliest attempt positions per question
    first_pos_verified_and_correct: dict[str, int | None] = field(default_factory=dict)
    first_pos_valid: dict[str, int | None] = field(default_factory=dict)
    first_pos_unverified_correct: dict[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert curve data to dictionary."""
        return asdict(self)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Convert curve points to tabular rows for CSV export."""
        rows = []
        for i, k in enumerate(self.k_values):
            rows.append(
                {
                    "k": k,
                    "total_examples_evaluated": self.total_questions,
                    "total_questions": self.total_questions,
                    "verified_correct_count": self.verified_correct_counts[i],
                    "verified_correct_pct": self.verified_correct_pct[i],
                    "first_valid_correct_count": self.first_valid_correct_counts[i],
                    "first_valid_correct_pct": self.first_valid_correct_pct[i],
                    "unverified_correct_count": self.unverified_correct_counts[i],
                    "unverified_correct_pct": self.unverified_correct_pct[i],
                    "has_valid_candidate_count": self.has_valid_candidate_counts[i],
                    "has_valid_candidate_pct": self.has_valid_candidate_pct[i],
                }
            )
        return rows


@dataclass
class Step1ComparisonResult:
    """Comparison between Step 3 (Verification Rejection Sampling) and Step 1 (Single-Shot Baseline)."""

    step1_file: str
    total_evaluated_questions: int
    total_compared_questions: int
    step1_correct_count: int
    step1_accuracy_pct: float
    verifier_correct_count: int
    verifier_accuracy_pct: float
    difference_pct: float
    verdict: str  # "BETTER", "LOWER", "TIED"
    is_better: bool
    is_tied: bool
    improved_question_ids: list[str] = field(default_factory=list)
    degraded_question_ids: list[str] = field(default_factory=list)
    both_correct_question_ids: list[str] = field(default_factory=list)
    both_incorrect_question_ids: list[str] = field(default_factory=list)
    missing_in_step1_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison result to a dictionary."""
        return asdict(self)


def load_verifier_results(
    file_path: str | Path,
) -> tuple[dict[str, Any] | None, list[QuestionVerificationResult]]:
    """Load verifier output JSON file.

    Supports both `{ "summary": ..., "results": [ ... ] }` structure and a bare list `[ ... ]`.
    Returns a tuple of (summary_dict_or_None, list_of_question_verification_results).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Verifier results file not found: {path}")

    with path.open(mode="r", encoding="utf-8") as f:
        data = json.load(f)

    summary: dict[str, Any] | None = None
    raw_results: list[dict[str, Any]] = []

    if isinstance(data, dict):
        summary = data.get("summary")
        raw_results = data.get("results") or data.get("data") or []
    elif isinstance(data, list):
        raw_results = data
    else:
        raise ValueError(
            f"Expected JSON object or list in {path}, got {type(data).__name__}"
        )

    results: list[QuestionVerificationResult] = []
    for item in raw_results:
        if isinstance(item, dict):
            try:
                results.append(QuestionVerificationResult.from_dict(item))
            except Exception as e:
                logger.warning(
                    "Failed to parse question result item: %s. Using fallback parser.",
                    e,
                )
                # Fallback minimal construction
                q_id = str(item.get("question_id", ""))
                candidates_evaluated = int(item.get("candidates_evaluated", 0))
                cvs = [
                    CandidateVerificationResult.from_dict(cv)
                    for cv in item.get("candidate_verifications") or []
                    if isinstance(cv, dict)
                ]
                results.append(
                    QuestionVerificationResult(
                        question_id=q_id,
                        meta_info=item.get("meta_info"),
                        question=str(item.get("question", "")),
                        options=dict(item.get("options") or {}),
                        ground_truth=str(item.get("ground_truth", "")),
                        ground_truth_answer=item.get("ground_truth_answer"),
                        candidates_evaluated=candidates_evaluated or len(cvs),
                        found_valid_candidate=bool(
                            item.get("found_valid_candidate", False)
                        ),
                        selected_candidate_index=item.get("selected_candidate_index"),
                        attempt_number=item.get("attempt_number"),
                        predicted_option=item.get("predicted_option"),
                        is_correct=item.get("is_correct"),
                        candidate_verifications=cvs,
                        total_latency_seconds=float(
                            item.get("total_latency_seconds", 0.0)
                        ),
                        total_tokens=int(item.get("total_tokens", 0)),
                        total_prompt_tokens=int(item.get("total_prompt_tokens", 0)),
                        total_candidate_tokens=int(
                            item.get("total_candidate_tokens", 0)
                        ),
                        num_correct_candidates=int(
                            item.get("num_correct_candidates", 0)
                        ),
                        correct_candidates_right_answers=int(
                            item.get("correct_candidates_right_answers", 0)
                        ),
                        correct_candidates_wrong_answers=int(
                            item.get("correct_candidates_wrong_answers", 0)
                        ),
                        first_correct_candidate_pos_assumptions_only=item.get(
                            "first_correct_candidate_pos_assumptions_only"
                        ),
                        first_correct_candidate_pos_all_and_answer=item.get(
                            "first_correct_candidate_pos_all_and_answer"
                        ),
                        error=item.get("error"),
                    )
                )

    return summary, results


def _get_candidate_attempt_number(
    candidate: CandidateVerificationResult, index: int
) -> int:
    """Return 1-indexed attempt number for candidate."""
    if candidate.attempt_number and candidate.attempt_number > 0:
        return candidate.attempt_number
    return index + 1


def compute_verifier_curve(
    results: list[QuestionVerificationResult],
    max_candidates: int | None = None,
    min_candidates: int = 1,
    step: int = 1,
) -> VerifierCurveData:
    """Compute verification accuracy metrics across candidate counts k = 1..K.

    For each question and each candidate limit k:
    1. Verified & Correct (Primary Metric):
       True if there exists at least one candidate among the first k candidates where:
       all_facts_correct is True AND is_correct is True.
    2. First Valid Correct (Rejection Sampling):
       The first candidate among 1..k with all_facts_correct is True has is_correct == True.
    3. Unverified Baseline:
       True if at least one candidate among 1..k has is_correct == True.
    4. Valid Coverage:
       True if at least one candidate among 1..k has all_facts_correct == True.
    """
    total_questions = len(results)
    if total_questions == 0:
        return VerifierCurveData(
            k_values=[],
            total_questions=0,
            max_candidates=0,
            verified_correct_pct=[],
            verified_correct_counts=[],
            first_valid_correct_pct=[],
            first_valid_correct_counts=[],
            unverified_correct_pct=[],
            unverified_correct_counts=[],
            has_valid_candidate_pct=[],
            has_valid_candidate_counts=[],
        )

    # Determine candidate positions per question
    first_pos_verified_and_correct: dict[str, int | None] = {}
    first_pos_valid: dict[str, int | None] = {}
    first_pos_unverified_correct: dict[str, int | None] = {}
    question_candidates_count: list[int] = []

    # Map question_id -> list of (attempt_number, all_facts_correct, is_correct)
    question_candidates_info: dict[str, list[tuple[int, bool, bool]]] = {}

    for q in results:
        q_id = q.question_id
        cands = q.candidate_verifications or []
        question_candidates_count.append(len(cands))

        cand_info: list[tuple[int, bool, bool]] = []
        for idx, cv in enumerate(cands):
            att = _get_candidate_attempt_number(cv, idx)
            all_facts_ok = bool(cv.all_facts_correct and not cv.error)
            ans_ok = bool(cv.is_correct)
            cand_info.append((att, all_facts_ok, ans_ok))

        # Sort by attempt number
        cand_info.sort(key=lambda x: x[0])
        question_candidates_info[q_id] = cand_info

        # 1. Earliest attempt with all facts correct AND answer correct
        earliest_both = next(
            (att for att, all_ok, ans_ok in cand_info if all_ok and ans_ok),
            None,
        )
        # Fallback to schema's cached field if cand_info was empty
        if earliest_both is None and not cand_info:
            earliest_both = q.first_correct_candidate_pos_all_and_answer
        first_pos_verified_and_correct[q_id] = earliest_both

        # 2. Earliest attempt with all facts correct
        earliest_valid = next((att for att, all_ok, _ in cand_info if all_ok), None)
        if earliest_valid is None and not cand_info:
            earliest_valid = q.first_correct_candidate_pos_assumptions_only
        first_pos_valid[q_id] = earliest_valid

        # 3. Earliest attempt with correct answer (unverified)
        earliest_correct = next((att for att, _, ans_ok in cand_info if ans_ok), None)
        first_pos_unverified_correct[q_id] = earliest_correct

    # Determine maximum k
    actual_max = max(question_candidates_count) if question_candidates_count else 0
    if actual_max == 0:
        actual_max = max((q.candidates_evaluated for q in results), default=1)

    max_k = max_candidates if max_candidates is not None else actual_max
    max_k = max(1, max_k)
    min_k = max(1, min(min_candidates, max_k))
    step = max(1, step)

    k_values = list(range(min_k, max_k + 1, step))
    if k_values[-1] != max_k:
        k_values.append(max_k)

    verified_correct_pct: list[float] = []
    verified_correct_counts: list[int] = []
    first_valid_correct_pct: list[float] = []
    first_valid_correct_counts: list[int] = []
    unverified_correct_pct: list[float] = []
    unverified_correct_counts: list[int] = []
    has_valid_candidate_pct: list[float] = []
    has_valid_candidate_counts: list[int] = []

    for k in k_values:
        v_corr_count = 0
        fv_corr_count = 0
        unv_corr_count = 0
        valid_cand_count = 0

        for q in results:
            q_id = q.question_id
            cands = question_candidates_info.get(q_id, [])

            # Filter candidates up to attempt k
            cands_k = [c for c in cands if c[0] <= k]

            # Primary: Any candidate <= k that passed all checks and is correct
            has_v_corr = any(all_ok and ans_ok for _, all_ok, ans_ok in cands_k)
            if not cands_k and first_pos_verified_and_correct.get(q_id) is not None:
                has_v_corr = first_pos_verified_and_correct[q_id] <= k
            if has_v_corr:
                v_corr_count += 1

            # First-Valid: Earliest candidate <= k with all facts correct
            first_v = next((c for c in cands_k if c[1]), None)
            if first_v is not None:
                valid_cand_count += 1
                if first_v[2]:  # answer correct
                    fv_corr_count += 1
            elif not cands_k:
                pos_v = first_pos_valid.get(q_id)
                if pos_v is not None and pos_v <= k:
                    valid_cand_count += 1
                    if (
                        first_pos_verified_and_correct.get(q_id) == pos_v
                        or q.is_correct is True
                    ):
                        fv_corr_count += 1

            # Unverified Baseline: Any candidate <= k with answer correct
            has_unv_corr = any(ans_ok for _, _, ans_ok in cands_k)
            if not cands_k and first_pos_unverified_correct.get(q_id) is not None:
                has_unv_corr = first_pos_unverified_correct[q_id] <= k
            if has_unv_corr:
                unv_corr_count += 1

        verified_correct_counts.append(v_corr_count)
        verified_correct_pct.append(round((v_corr_count / total_questions) * 100.0, 2))

        first_valid_correct_counts.append(fv_corr_count)
        first_valid_correct_pct.append(
            round((fv_corr_count / total_questions) * 100.0, 2)
        )

        unverified_correct_counts.append(unv_corr_count)
        unverified_correct_pct.append(
            round((unv_corr_count / total_questions) * 100.0, 2)
        )

        has_valid_candidate_counts.append(valid_cand_count)
        has_valid_candidate_pct.append(
            round((valid_cand_count / total_questions) * 100.0, 2)
        )

    return VerifierCurveData(
        k_values=k_values,
        total_questions=total_questions,
        max_candidates=max_k,
        verified_correct_pct=verified_correct_pct,
        verified_correct_counts=verified_correct_counts,
        first_valid_correct_pct=first_valid_correct_pct,
        first_valid_correct_counts=first_valid_correct_counts,
        unverified_correct_pct=unverified_correct_pct,
        unverified_correct_counts=unverified_correct_counts,
        has_valid_candidate_pct=has_valid_candidate_pct,
        has_valid_candidate_counts=has_valid_candidate_counts,
        first_pos_verified_and_correct=first_pos_verified_and_correct,
        first_pos_valid=first_pos_valid,
        first_pos_unverified_correct=first_pos_unverified_correct,
    )


def plot_verifier_curve(
    curve_data: VerifierCurveData,
    output_path: str | Path,
    strategy: str = "first_valid",
    title: str | None = None,
    dpi: int = 300,
    log_scale_x: bool = False,
    show_grid: bool = True,
) -> Path:
    """Generate and save the accuracy vs candidate generation plot.

    Args:
        curve_data: VerifierCurveData containing metrics across k=1..K.
        output_path: Path where the plot image (PNG/PDF/SVG) will be saved.
        strategy: Plot mode:
            - 'first_valid': Plots only first-valid candidate selection curve (default).
            - 'all': Plots first-valid and unverified generator baseline.
            (Legacy options 'both' and 'verified' are mapped to first_valid).
        title: Optional custom plot title.
        dpi: Image resolution in dots per inch.
        log_scale_x: Whether to use logarithmic scale on the x-axis.
        show_grid: Whether to show gridlines.

    Returns:
        Path to the saved figure file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not curve_data.k_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(
            0.5,
            0.5,
            "No candidate verification data to plot",
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_file

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    k_vals = curve_data.k_values
    num_pts = len(k_vals)
    use_markers = num_pts <= 35
    marker_size = 6 if num_pts <= 20 else 4

    strategy_clean = strategy.lower().strip()

    # 1. First-Valid Selection (Rejection Sampling Policy)
    # Note: The oracle "ANY candidate" (verified & correct) curve has been removed.
    ax.plot(
        k_vals,
        curve_data.first_valid_correct_pct,
        label="First Valid Candidate (Rejection Sampling Policy)",
        color="#1f77b4",
        linewidth=2.4,
        marker="o" if use_markers else None,
        markersize=marker_size,
        alpha=0.95,
        zorder=4,
    )

    # 2. Unverified Baseline (Raw Generator Pass@k) - optional if requested
    if strategy_clean in ("all", "unverified"):
        ax.plot(
            k_vals,
            curve_data.unverified_correct_pct,
            label="Unverified Baseline (Generator Pass@k)",
            color="#6c757d",
            linewidth=1.8,
            linestyle=":",
            marker="^" if use_markers else None,
            markersize=marker_size,
            alpha=0.8,
            zorder=2,
        )

    # Axes styling
    ax.set_xlabel(
        "Number of Candidate Generations ($k$)", fontsize=12, fontweight="medium"
    )
    ax.set_ylabel("% of Questions Answered Correctly", fontsize=12, fontweight="medium")

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Determine y limits
    all_y = list(curve_data.first_valid_correct_pct)
    if strategy_clean in ("all", "unverified"):
        all_y.extend(curve_data.unverified_correct_pct)

    max_y = max(all_y) if all_y else 100.0
    y_upper = min(105.0, max(20.0, math.ceil(max_y / 10.0) * 10 + 5))
    ax.set_ylim(-2.0, y_upper)

    # X-axis configuration
    if log_scale_x and k_vals[-1] > 10:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=max(0.5, k_vals[0] - 0.5), right=k_vals[-1] + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
        ax.set_axisbelow(True)

    # Title & Subtitle
    plot_title = title or "MedQA Fact Verification: Accuracy vs. Candidate Generation"
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=12)

    # Add text annotation box with run details
    final_acc = (
        curve_data.first_valid_correct_pct[-1]
        if curve_data.first_valid_correct_pct
        else 0.0
    )
    initial_acc = (
        curve_data.first_valid_correct_pct[0]
        if curve_data.first_valid_correct_pct
        else 0.0
    )
    first_k = curve_data.k_values[0] if curve_data.k_values else 1
    last_k = (
        curve_data.k_values[-1] if curve_data.k_values else curve_data.max_candidates
    )
    info_text = (
        f"Examples Evaluated: {curve_data.total_questions}\n"
        f"Max Candidates: {curve_data.max_candidates}\n"
        f"Accuracy @ k={first_k}: {initial_acc:.1f}%\n"
        f"Accuracy @ k={last_k}: {final_acc:.1f}%"
    )
    ax.text(
        0.02,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.85,
        ),
    )

    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved verifier accuracy plot to %s", out_file)
    return out_file


def format_summary_table(
    curve_data: VerifierCurveData,
    checkpoints: list[int] | None = None,
) -> str:
    """Format an ASCII table summarizing accuracy at milestone checkpoints."""
    if not curve_data.k_values:
        return "No verification data available."

    k_to_idx = {k: i for i, k in enumerate(curve_data.k_values)}
    total_q = curve_data.total_questions

    if checkpoints is None:
        default_cps = [1, 2, 3, 5, 10, 20, 25, 50, 100, 250, 500, 1000]
        selected_k = [k for k in default_cps if k in k_to_idx]
        if curve_data.k_values[-1] not in selected_k:
            selected_k.append(curve_data.k_values[-1])
        selected_k = sorted(list(set(selected_k)))
    else:
        selected_k = [k for k in checkpoints if k in k_to_idx]

    headers = [
        "Candidates (k)",
        "Verified & Correct (%)",
        "Verified (Count)",
        "First-Valid (%)",
        "First-Valid (Count)",
        "Valid Coverage (%)",
        "Unverified Pass@k (%)",
    ]
    col_widths = [len(h) for h in headers]

    rows = []
    for k in selected_k:
        idx = k_to_idx[k]
        v_pct = f"{curve_data.verified_correct_pct[idx]:.1f}%"
        v_cnt = f"{curve_data.verified_correct_counts[idx]}/{total_q}"
        fv_pct = f"{curve_data.first_valid_correct_pct[idx]:.1f}%"
        fv_cnt = f"{curve_data.first_valid_correct_counts[idx]}/{total_q}"
        cov_pct = f"{curve_data.has_valid_candidate_pct[idx]:.1f}%"
        unv_pct = f"{curve_data.unverified_correct_pct[idx]:.1f}%"

        row = [str(k), v_pct, v_cnt, fv_pct, fv_cnt, cov_pct, unv_pct]
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
        rows.append(row)

    # Build ASCII table
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_str = (
        "|"
        + "|".join(f" {headers[i]:^{col_widths[i]}} " for i in range(len(headers)))
        + "|"
    )

    table_lines = [sep, header_str, sep]
    for row in rows:
        line = (
            "|"
            + "|".join(f" {row[i]:>{col_widths[i]}} " for i in range(len(row)))
            + "|"
        )
        table_lines.append(line)
    table_lines.append(sep)

    return "\n".join(table_lines)


def compute_evaluation_summary(
    results: list[QuestionVerificationResult],
    summary_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute overall evaluation counts across questions/examples."""
    total_examples = len(results)
    completed_examples = sum(1 for r in results if not r.error)
    failed_examples = sum(1 for r in results if r.error)

    total_candidates = sum(
        r.candidates_evaluated or len(r.candidate_verifications) for r in results
    )

    total_facts = sum(
        len(cv.fact_verifications) for r in results for cv in r.candidate_verifications
    )
    facts_correct = sum(
        1
        for r in results
        for cv in r.candidate_verifications
        for fv in cv.fact_verifications
        if fv.verdict == 1 or fv.is_correct
    )
    facts_incorrect = sum(
        1
        for r in results
        for cv in r.candidate_verifications
        for fv in cv.fact_verifications
        if (fv.verdict == 0 or not fv.is_correct) and not fv.error
    )
    questions_with_valid = sum(1 for r in results if r.found_valid_candidate)
    correct_answers = sum(
        1 for r in results if r.found_valid_candidate and r.is_correct is True
    )

    if total_examples == 0 and summary_meta:
        total_examples = summary_meta.get("total_questions", 0)
        completed_examples = summary_meta.get("completed_questions", 0)
        failed_examples = summary_meta.get("failed_questions", 0)
        total_candidates = summary_meta.get("total_candidates_evaluated", 0)
        total_facts = summary_meta.get("total_facts_verified", 0)
        questions_with_valid = summary_meta.get("questions_with_valid_candidate", 0)
        correct_answers = summary_meta.get("correct_answers", 0)

    completion_rate_pct = (
        round(completed_examples / total_examples * 100.0, 2)
        if total_examples > 0
        else 0.0
    )
    avg_candidates = (
        round(total_candidates / total_examples, 1) if total_examples > 0 else 0.0
    )

    return {
        "total_examples_evaluated": total_examples,
        "total_questions": total_examples,
        "completed_examples": completed_examples,
        "failed_examples": failed_examples,
        "completion_rate_pct": completion_rate_pct,
        "total_candidates_evaluated": total_candidates,
        "avg_candidates_per_example": avg_candidates,
        "total_facts_verified": total_facts,
        "facts_correct": facts_correct,
        "facts_incorrect": facts_incorrect,
        "questions_with_valid_candidate": questions_with_valid,
        "correct_answers": correct_answers,
    }


def format_evaluation_summary(
    eval_summary: dict[str, Any], input_path: str | Path | None = None
) -> str:
    """Format evaluation statistics into a clean text banner."""
    lines = [
        "=" * 80,
        "VERIFIER EVALUATION SUMMARY",
        "=" * 80,
    ]
    if input_path:
        lines.append(f"Input File                    : {input_path}")
    lines.extend(
        [
            f"Total Examples Evaluated      : {eval_summary['total_examples_evaluated']:,}",
            f"  • Successfully Completed    : {eval_summary['completed_examples']:,} ({eval_summary['completion_rate_pct']}%)",
            f"  • Failed with Error         : {eval_summary['failed_examples']:,}",
            f"Total Candidates Evaluated    : {eval_summary['total_candidates_evaluated']:,} (avg {eval_summary['avg_candidates_per_example']} / example)",
            f"Total Medical Facts Verified  : {eval_summary['total_facts_verified']:,} ({eval_summary['facts_correct']:,} correct, {eval_summary['facts_incorrect']:,} incorrect)",
            f"Examples with Valid Candidate : {eval_summary['questions_with_valid_candidate']:,}",
            "=" * 80,
        ]
    )
    return "\n".join(lines)


def compute_token_and_cost_summary(
    results: list[QuestionVerificationResult],
    model_name: str = "gemini-3.8-flash",
) -> dict[str, Any]:
    """Compute token usage, latencies, and estimated verifier cost in USD."""
    total_tokens = sum(r.total_tokens for r in results)
    prompt_tokens = sum(r.total_prompt_tokens for r in results)
    candidate_tokens = sum(r.total_candidate_tokens for r in results)
    total_latency = sum(r.total_latency_seconds for r in results)
    total_candidates = sum(
        r.candidates_evaluated or len(r.candidate_verifications) for r in results
    )

    inp_rate, out_rate, _ = get_pricing_for_model(model_name)
    cost = (prompt_tokens / 1_000_000 * inp_rate) + (
        candidate_tokens / 1_000_000 * out_rate
    )

    num_q = len(results)
    completed_q = sum(1 for r in results if not r.error)
    failed_q = sum(1 for r in results if r.error)

    return {
        "model": model_name,
        "total_examples_evaluated": num_q,
        "total_questions": num_q,
        "completed_examples": completed_q,
        "failed_examples": failed_q,
        "total_candidates_evaluated": total_candidates,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "avg_tokens_per_example": round(total_tokens / num_q, 1) if num_q else 0,
        "avg_tokens_per_question": round(total_tokens / num_q, 1) if num_q else 0,
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_seconds_per_example": round(total_latency / num_q, 2)
        if num_q
        else 0,
        "avg_latency_seconds_per_question": round(total_latency / num_q, 2)
        if num_q
        else 0,
        "estimated_verifier_cost_usd": round(cost, 6),
    }


def export_curve_csv(
    curve_data: VerifierCurveData,
    csv_path: str | Path,
) -> None:
    """Export the curve data points to a CSV file."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = curve_data.to_csv_rows()
    if not rows:
        return

    with path.open(mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Exported curve data to %s", path)


def export_curve_json(
    curve_data: VerifierCurveData,
    json_path: str | Path,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Export the full curve metrics and metadata to a JSON file."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metrics": curve_data.to_dict(),
        "metadata": extra_meta or {},
    }

    with path.open(mode="w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Exported curve JSON to %s", path)


def resolve_step1_file(
    step1_arg: str | Path | None = None,
    input_path: Path | None = None,
) -> Path | None:
    """Resolve the path to the Step 1 single-shot results file.

    If an explicit path is provided, it is returned (or raises FileNotFoundError if absent).
    If no path is provided, searches standard filenames in the current directory and
    input file directory.
    """
    if step1_arg:
        p = Path(step1_arg)
        if not p.is_file():
            raise FileNotFoundError(f"Step 1 results file not found: {p}")
        return p

    search_dirs = []
    if input_path and input_path.parent not in search_dirs:
        search_dirs.append(input_path.parent)
    if Path.cwd() not in search_dirs:
        search_dirs.append(Path.cwd())

    candidate_filenames = [
        "results_gemma4.json",
        "results_one_shot_gemma4.json",
        "results_gemini.json",
        "results_gpt-oss-20b.json",
    ]

    for d in search_dirs:
        for fname in candidate_filenames:
            candidate = d / fname
            if candidate.is_file():
                return candidate

    return None


def load_step1_single_shot_results(
    step1_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load Step 1 results and extract the first candidate (attempt 0) of the single-shot prompt.

    Returns a mapping of question_id -> {
        "question_id": str,
        "predicted_option": str | None,
        "is_correct": bool,
        "ground_truth": str | None,
    }
    """
    path = Path(step1_path)
    if not path.is_file():
        raise FileNotFoundError(f"Step 1 results file not found: {path}")

    with path.open(mode="r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("results") if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = []

    step1_map: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("question_id", "")).strip()
        if not qid:
            continue

        raw_attempts = item.get("attempts") or []
        first_attempt = (
            raw_attempts[0]
            if raw_attempts and isinstance(raw_attempts[0], dict)
            else {}
        )

        is_corr = first_attempt.get("is_correct")
        if is_corr is None:
            is_corr = item.get("is_correct")

        pred_opt = first_attempt.get("predicted_option")
        if pred_opt is None:
            pred_opt = item.get("predicted_option")

        step1_map[qid] = {
            "question_id": qid,
            "predicted_option": pred_opt,
            "is_correct": bool(is_corr),
            "ground_truth": item.get("ground_truth") or item.get("ground_truth_answer"),
        }

    return step1_map


def compare_verifier_to_step1(
    verifier_results: list[QuestionVerificationResult],
    step1_path: str | Path,
    curve_data: VerifierCurveData | None = None,
    max_k: int | None = None,
) -> Step1ComparisonResult:
    """Compare Step 3 verification performance (first valid candidate) with Step 1 single-shot first candidate.

    Determines whether the verification schema yields better, lower, or equal performance on the evaluated questions.
    """
    step1_map = load_step1_single_shot_results(step1_path)

    improved: list[str] = []
    degraded: list[str] = []
    both_correct: list[str] = []
    both_incorrect: list[str] = []
    missing_in_step1: list[str] = []

    s1_correct_count = 0
    verifier_correct_count = 0

    evaluated_k = max_k
    if evaluated_k is None and curve_data is not None and curve_data.k_values:
        evaluated_k = curve_data.k_values[-1]

    compared_count = 0

    for r in verifier_results:
        qid = str(r.question_id).strip()
        if qid not in step1_map:
            missing_in_step1.append(qid)
            continue

        compared_count += 1
        s1_info = step1_map[qid]
        s1_is_corr = bool(s1_info.get("is_correct"))

        if curve_data is not None:
            pos_valid = curve_data.first_pos_valid.get(qid)
            pos_both = curve_data.first_pos_verified_and_correct.get(qid)
            if pos_valid is not None and (
                evaluated_k is None or pos_valid <= evaluated_k
            ):
                v_is_corr = pos_both == pos_valid
            else:
                v_is_corr = False
        else:
            v_is_corr = bool(r.found_valid_candidate and r.is_correct is True)

        if s1_is_corr:
            s1_correct_count += 1
        if v_is_corr:
            verifier_correct_count += 1

        if not s1_is_corr and v_is_corr:
            improved.append(qid)
        elif s1_is_corr and not v_is_corr:
            degraded.append(qid)
        elif s1_is_corr and v_is_corr:
            both_correct.append(qid)
        else:
            both_incorrect.append(qid)

    s1_acc = (s1_correct_count / compared_count * 100.0) if compared_count > 0 else 0.0
    v_acc = (
        (verifier_correct_count / compared_count * 100.0) if compared_count > 0 else 0.0
    )
    diff = round(v_acc - s1_acc, 2)

    if diff > 0:
        verdict = "BETTER"
        is_better = True
        is_tied = False
    elif diff < 0:
        verdict = "LOWER"
        is_better = False
        is_tied = False
    else:
        verdict = "TIED"
        is_better = False
        is_tied = True

    return Step1ComparisonResult(
        step1_file=str(Path(step1_path).name),
        total_evaluated_questions=len(verifier_results),
        total_compared_questions=compared_count,
        step1_correct_count=s1_correct_count,
        step1_accuracy_pct=round(s1_acc, 2),
        verifier_correct_count=verifier_correct_count,
        verifier_accuracy_pct=round(v_acc, 2),
        difference_pct=diff,
        verdict=verdict,
        is_better=is_better,
        is_tied=is_tied,
        improved_question_ids=improved,
        degraded_question_ids=degraded,
        both_correct_question_ids=both_correct,
        both_incorrect_question_ids=both_incorrect,
        missing_in_step1_ids=missing_in_step1,
    )


def format_step1_comparison_summary(comparison: Step1ComparisonResult) -> str:
    """Format an ASCII summary banner comparing Step 3 verifier against Step 1 single-shot."""
    lines = [
        "=" * 80,
        "VERIFICATION vs. STEP 1 (SINGLE-SHOT FIRST CANDIDATE) COMPARISON",
        "=" * 80,
        f"Step 1 Source File           : {comparison.step1_file}",
        f"Evaluated Questions Compared : {comparison.total_compared_questions} / {comparison.total_evaluated_questions}",
        "",
        "Accuracy Comparison:",
        f"  • Step 1 Baseline (Single-Shot Attempt 0) : {comparison.step1_accuracy_pct:.1f}% ({comparison.step1_correct_count}/{comparison.total_compared_questions})",
        f"  • Step 3 Verifier (First Valid Candidate) : {comparison.verifier_accuracy_pct:.1f}% ({comparison.verifier_correct_count}/{comparison.total_compared_questions})",
        f"  • Net Accuracy Difference                  : {comparison.difference_pct:+.1f}%",
        "",
    ]

    if comparison.verdict == "BETTER":
        verdict_str = f"BETTER (+{comparison.difference_pct:.1f}% gain over Step 1)"
        note = "==> RESULT: Schema with verification yields BETTER performance than Step 1!"
    elif comparison.verdict == "LOWER":
        verdict_str = (
            f"LOWER ({comparison.difference_pct:.1f}% decrease compared to Step 1)"
        )
        note = (
            "==> RESULT: Schema with verification yields LOWER performance than Step 1."
        )
    else:
        verdict_str = f"TIED (Same accuracy: {comparison.verifier_accuracy_pct:.1f}%)"
        note = (
            "==> RESULT: Schema with verification yields EQUAL performance to Step 1."
        )

    lines.append(f"Performance Check Verdict    : {verdict_str}")
    lines.append(note)
    lines.append("")
    lines.append("Question Transitions:")

    imp_q = (
        f" (IDs: {', '.join(comparison.improved_question_ids[:5])}{'...' if len(comparison.improved_question_ids) > 5 else ''})"
        if comparison.improved_question_ids
        else ""
    )
    deg_q = (
        f" (IDs: {', '.join(comparison.degraded_question_ids[:5])}{'...' if len(comparison.degraded_question_ids) > 5 else ''})"
        if comparison.degraded_question_ids
        else ""
    )
    bc_q = (
        f" (IDs: {', '.join(comparison.both_correct_question_ids[:5])}{'...' if len(comparison.both_correct_question_ids) > 5 else ''})"
        if comparison.both_correct_question_ids
        else ""
    )
    bi_q = (
        f" (IDs: {', '.join(comparison.both_incorrect_question_ids[:5])}{'...' if len(comparison.both_incorrect_question_ids) > 5 else ''})"
        if comparison.both_incorrect_question_ids
        else ""
    )

    lines.append(
        f"  • Improved (Step 1 Wrong -> Verifier Correct) : {len(comparison.improved_question_ids)}{imp_q}"
    )
    lines.append(
        f"  • Degraded (Step 1 Correct -> Verifier Wrong) : {len(comparison.degraded_question_ids)}{deg_q}"
    )
    lines.append(
        f"  • Both Correct                                : {len(comparison.both_correct_question_ids)}{bc_q}"
    )
    lines.append(
        f"  • Both Incorrect                              : {len(comparison.both_incorrect_question_ids)}{bi_q}"
    )

    if comparison.missing_in_step1_ids:
        lines.append(
            f"  • Not Found in Step 1 File                   : {len(comparison.missing_in_step1_ids)}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MedQA verifier outputs and plot % of questions answered "
            "correctly after 1, 2, ..., k candidate generations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input file: either positional or --input / -i
    parser.add_argument(
        "positional_input",
        nargs="?",
        default=None,
        help="Path to verifier JSON results file (e.g. results_step3_verified.json).",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_file",
        default="results_step3_verified.json",
        help="Path to verifier JSON results file.",
    )

    # Plot output
    parser.add_argument(
        "--output",
        "-o",
        dest="output_file",
        default="verifier_accuracy_curve.png",
        help="Path to save the output accuracy curve plot image (PNG/PDF/SVG).",
    )

    # Strategy / metric selection
    parser.add_argument(
        "--strategy",
        "-s",
        choices=["first_valid", "all", "both", "verified"],
        default="first_valid",
        help=(
            "Plot mode: 'first_valid' (rejection sampling policy, default), "
            "or 'all' (first_valid + unverified generator baseline)."
        ),
    )

    # Candidate range configuration
    parser.add_argument(
        "--max-candidates",
        "-k",
        type=int,
        default=None,
        help="Maximum candidate generation count k to analyze up to (default: max available in data).",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=1,
        help="Minimum candidate generation count k to start from.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size between candidate checkpoints.",
    )

    # Plot styling
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the generated plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for the generated figure.",
    )
    parser.add_argument(
        "--log-scale-x",
        action="store_true",
        default=False,
        help="Use logarithmic scale for candidate generations on x-axis.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip generating plot image and only output summary table and exports.",
    )

    # Export options
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional path to export curve table to CSV.",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional path to export curve metrics to JSON.",
    )

    # Step 1 Baseline comparison
    parser.add_argument(
        "--step1-file",
        "--step1-results",
        dest="step1_file",
        type=str,
        default=None,
        help=(
            "Optional path to Step 1 inference results JSON (e.g. results_gemma4.json) "
            "to check whether verification yields better performance than Step 1 single-shot first candidate. "
            "If omitted, automatically checks for matching Step 1 results in current directory."
        ),
    )
    parser.add_argument(
        "--no-step1-comparison",
        action="store_true",
        default=False,
        help="Disable Step 1 comparison check.",
    )

    # Pricing & model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-flash",
        help="Model identifier used for calculating token cost.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve input file (positional argument takes precedence if specified)
    input_path = args.positional_input or args.input_file
    logger.info("Loading verifier results from %s", input_path)

    try:
        summary_meta, results = load_verifier_results(input_path)
    except Exception as e:
        logger.error("Failed to load %s: %s", input_path, e)
        return 1

    total_q = len(results)
    eval_summary = compute_evaluation_summary(results, summary_meta)

    # Prominently print evaluation summary banner
    print()
    print(format_evaluation_summary(eval_summary, input_path=input_path))

    if total_q == 0:
        print("No verification results found in input file.")
        return 0

    # Compute curve data
    curve_data = compute_verifier_curve(
        results=results,
        max_candidates=args.max_candidates,
        min_candidates=args.min_candidates,
        step=args.step,
    )

    # Print summary milestone table
    print("\n" + "=" * 80)
    print(
        f"VERIFIER ACCURACY vs. CANDIDATE GENERATION (k) — {total_q} Examples Evaluated"
    )
    print("=" * 80)
    print(format_summary_table(curve_data))

    # Token & Cost Metrics
    token_summary = compute_token_and_cost_summary(results, model_name=args.model)
    print("\n" + "-" * 50)
    print("VERIFICATION RESOURCE USAGE")
    print("-" * 50)
    print(f"Total Examples Evaluated   : {eval_summary['total_examples_evaluated']:,}")
    print(
        f"Total Candidates Evaluated : {token_summary['total_candidates_evaluated']:,}"
    )
    print(f"Total Tokens               : {token_summary['total_tokens']:,}")
    print(f"  Prompt Tokens            : {token_summary['prompt_tokens']:,}")
    print(f"  Candidate Tokens         : {token_summary['candidate_tokens']:,}")
    print(f"Avg Tokens / Example       : {token_summary['avg_tokens_per_example']:,}")
    print(
        f"Total Latency              : {token_summary['total_latency_seconds']:,.2f}s"
    )
    print(
        f"Estimated Verifier Cost    : ${token_summary['estimated_verifier_cost_usd']:.4f} USD ({args.model})"
    )
    print("-" * 50)

    # Step 1 Baseline Comparison Check
    step1_comparison = None
    if not args.no_step1_comparison:
        try:
            step1_path = resolve_step1_file(
                args.step1_file, input_path=Path(input_path)
            )
            if step1_path:
                step1_comparison = compare_verifier_to_step1(
                    verifier_results=results,
                    step1_path=step1_path,
                    curve_data=curve_data,
                    max_k=args.max_candidates,
                )
                print("\n" + format_step1_comparison_summary(step1_comparison))
            elif args.step1_file:
                logger.warning("Step 1 file not found: %s", args.step1_file)
        except Exception as e:
            logger.warning("Could not execute Step 1 baseline comparison: %s", e)

    # Plotting
    if not args.no_plot:
        out_fig = plot_verifier_curve(
            curve_data=curve_data,
            output_path=args.output_file,
            strategy=args.strategy,
            title=args.title,
            dpi=args.dpi,
            log_scale_x=args.log_scale_x,
        )
        print(f"\nPlot saved successfully to: {out_fig.resolve()}")

    # Exports
    if args.export_csv:
        export_curve_csv(curve_data, args.export_csv)
        print(f"CSV data exported to: {Path(args.export_csv).resolve()}")

    if args.export_json:
        export_curve_json(
            curve_data,
            args.export_json,
            extra_meta={
                "eval_summary": eval_summary,
                "token_summary": token_summary,
                "step1_comparison": step1_comparison.to_dict()
                if step1_comparison
                else None,
                "input_file": str(input_path),
            },
        )
        print(f"JSON metrics exported to: {Path(args.export_json).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
