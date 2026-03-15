import argparse
import asyncio
import json
from collections import Counter

from anthropic import AnthropicVertex
from google import genai
from google.genai.types import HttpOptions

from _eval import eval_results
from _prompts import PROMPT_V1, PROMPT_V2, PROMPT_V3, PROMPT_V4
from _utils import (
    CLUSTERS_CSV_PATH,
    GOLDEN_DATA_PATH,
    _run_with_retry,
    build_classification_components,
    format_cluster_descriptions,
    get_anthropic_decider,
    get_gemini_decider,
    load_cluster_descriptions,
    load_eval_results,
    load_golden_dataset,
)

GEMINI_MODEL = "gemini-3-pro-preview"
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _enum_to_title(enum_value: str, cluster_descriptions: list[dict]) -> str:
    """Convert an uppercase-underscore enum value back to its human-readable cluster title."""
    for c in cluster_descriptions:
        if c["title"].replace(" ", "_").upper() == enum_value:
            return c["title"]
    return enum_value


async def classify_with_consensus(
    gemini_decide,
    anthropic_decide,
    submission: dict,
    formatted: str,
    cluster_descriptions: list[dict],
):
    title, text = submission["title"], submission["text"]

    # ── Phase 1: Independent classification WITH reasoning ────────────
    gemini_result, anthropic_result = await asyncio.gather(
        _run_with_retry(gemini_decide, title, text, formatted),
        _run_with_retry(anthropic_decide, title, text, formatted),
    )

    if gemini_result.informational_need == anthropic_result.informational_need:
        return gemini_result.informational_need

    g_name = _enum_to_title(
        gemini_result.informational_need.value, cluster_descriptions
    )
    a_name = _enum_to_title(
        anthropic_result.informational_need.value, cluster_descriptions
    )
    print(f"  Disagreement: Gemini='{g_name}', Anthropic='{a_name}'")

    # ── Phase 2: Judge — both models evaluate BOTH reasonings ────────
    # Key differences from cross-pollination:
    #  - Each model sees BOTH reasonings (full picture, not just the
    #    opposing view), so it can compare reasoning quality directly.
    #  - "Evaluate" framing avoids anchoring — the model judges two
    #    experts instead of defending its own prior answer.
    #  - Order is swapped per model to counter position bias AND make
    #    each model see the OTHER's reasoning in the privileged first slot.
    #  - Full category set remains available (not constrained to two).
    gemini_judge_ctx = (
        f"\n\nTwo experts analyzed this post and reached different conclusions:\n\n"
        f"Expert A chose '{a_name}':\n\"{anthropic_result.reasoning}\"\n\n"
        f"Expert B chose '{g_name}':\n\"{gemini_result.reasoning}\"\n\n"
        f"Evaluate both analyses and classify the post yourself. "
        f"You are not limited to these two options."
    )
    anthropic_judge_ctx = (
        f"\n\nTwo experts analyzed this post and reached different conclusions:\n\n"
        f"Expert A chose '{g_name}':\n\"{gemini_result.reasoning}\"\n\n"
        f"Expert B chose '{a_name}':\n\"{anthropic_result.reasoning}\"\n\n"
        f"Evaluate both analyses and classify the post yourself. "
        f"You are not limited to these two options."
    )

    gemini_judge, anthropic_judge = await asyncio.gather(
        _run_with_retry(gemini_decide, title, text, formatted, gemini_judge_ctx),
        _run_with_retry(anthropic_decide, title, text, formatted, anthropic_judge_ctx),
    )

    # ── Resolution ────────────────────────────────────────────────────
    # If both judges agree → strong consensus (they saw the full picture)
    if gemini_judge.informational_need == anthropic_judge.informational_need:
        winner_name = _enum_to_title(
            gemini_judge.informational_need.value, cluster_descriptions
        )
        print(f"  Resolved (judge consensus) → {winner_name}")
        return gemini_judge.informational_need

    # Majority vote across all 4 results
    all_results = [gemini_result, anthropic_result, gemini_judge, anthropic_judge]
    all_votes = [r.informational_need.value for r in all_results]
    counts = Counter(all_votes)
    winner_val, winner_count = counts.most_common(1)[0]

    if winner_count >= 3:
        winner_name = _enum_to_title(winner_val, cluster_descriptions)
        print(f"  Resolved ({winner_count}/4 majority) → {winner_name}")
        for r in all_results:
            if r.informational_need.value == winner_val:
                return r.informational_need

    # Still tied — run a single tiebreaker: Gemini judges once more
    # with the two JUDGE reasonings (meta-judge). This 5th vote
    # guarantees a majority among the original two candidates.
    print("  Tie, running tiebreaker...")
    gj_name = _enum_to_title(
        gemini_judge.informational_need.value, cluster_descriptions
    )
    aj_name = _enum_to_title(
        anthropic_judge.informational_need.value, cluster_descriptions
    )
    tiebreaker_ctx = (
        f"\n\nTwo experts analyzed this post and reached different conclusions:\n\n"
        f"Expert A chose '{aj_name}':\n\"{anthropic_judge.reasoning}\"\n\n"
        f"Expert B chose '{gj_name}':\n\"{gemini_judge.reasoning}\"\n\n"
        f"Evaluate both analyses and classify the post yourself. "
        f"You are not limited to these two options."
    )
    tiebreaker = await _run_with_retry(
        gemini_decide, title, text, formatted, tiebreaker_ctx
    )
    all_votes.append(tiebreaker.informational_need.value)
    counts = Counter(all_votes)
    winner_val = counts.most_common(1)[0][0]
    winner_name = _enum_to_title(winner_val, cluster_descriptions)
    print(f"  Resolved (tiebreaker) → {winner_name}")
    for r in [*all_results, tiebreaker]:
        if r.informational_need.value == winner_val:
            return r.informational_need


async def run(
    out_filename: str,
    temperature: float,
    prompt_template: str,
    gemini_client,
    anthropic_client,
):
    submissions = load_golden_dataset(GOLDEN_DATA_PATH)
    cluster_descriptions = load_cluster_descriptions(CLUSTERS_CSV_PATH)
    formatted = format_cluster_descriptions(cluster_descriptions)

    # Reasoning enabled — gives each model chain-of-thought for better
    # initial classification and enables cross-pollination on disagreements.
    InformationalNeedEnum, InformationalNeed, allowed_values = (
        build_classification_components(cluster_descriptions, include_reasoning=True)
    )

    gemini_decide = get_gemini_decider(
        GEMINI_MODEL, InformationalNeed, prompt_template, temperature, gemini_client
    )
    anthropic_decide = get_anthropic_decider(
        model_name=ANTHROPIC_MODEL,
        expected_output=InformationalNeed,
        prompt_template=prompt_template,
        temperature=temperature,
        client=anthropic_client,
    )

    predicted_labels = []
    true_labels = []
    results_log = []
    start = 0
    for i, submission in enumerate(submissions[start:]):
        try:
            predicted_enum = await classify_with_consensus(
                gemini_decide,
                anthropic_decide,
                submission,
                formatted,
                cluster_descriptions,
            )
        except ValueError:
            print(
                f"Skipping submission {submission['submission_id']}: too many retries"
            )
            continue

        predicted_name = [
            c["title"]
            for c in cluster_descriptions
            if c["title"].replace(" ", "_").upper() == predicted_enum.value
        ][0]

        true_label = submission["cluster"]
        predicted_labels.append(predicted_name)
        true_labels.append(true_label)

        results_log.append(
            {
                "submission_id": submission["submission_id"],
                "true_label": true_label,
                "predicted_label": predicted_name,
            }
        )

        if i % 10 == 0:
            print(
                f"Processed {i + start + 1}/{len(submissions)}: '{true_label}' -> '{predicted_name}'"
            )

    with open(out_filename, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nResults saved to {out_filename}")

    return (predicted_labels, true_labels)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Read existing results file and compute metrics without running clustering.",
    )
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--filename",
        default=None,
        help="Results file name (relative to script directory).",
    )
    parser.add_argument(
        "--region",
        default="us-east5",
        help="Vertex AI region for Anthropic models (default: us-east5).",
    )
    args = parser.parse_args()

    gemini_client = genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        project="kuligin-sandbox1",
        location="global",
    )
    anthropic_client = AnthropicVertex(
        project_id="kuligin-sandbox1", region=args.region
    )

    if args.eval_only:
        predicted_labels, true_labels = load_eval_results(args.filename)
    else:
        prompt_templates = {
            "v1": PROMPT_V1,
            "v2": PROMPT_V2,
            "v3": PROMPT_V3,
            "v4": PROMPT_V4,
        }
        prompt_template = prompt_templates.get(args.prompt, PROMPT_V4)

        predicted_labels, true_labels = await run(
            out_filename=args.filename,
            temperature=args.temperature,
            prompt_template=prompt_template,
            gemini_client=gemini_client,
            anthropic_client=anthropic_client,
        )

    eval_results(predicted_labels, true_labels)


if __name__ == "__main__":
    asyncio.run(main())
