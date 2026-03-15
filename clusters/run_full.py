import argparse
import asyncio
import json

from google import genai
from google.genai.types import HttpOptions

from _prompts import PROMPT_V1, PROMPT_V2, PROMPT_V3, PROMPT_V4
from _utils import (
    CLUSTERS_CSV_PATH,
    DEFAULT_DISEASE,
    build_classification_components,
    classify_submission,
    format_cluster_descriptions,
    get_anthropic_decider,
    get_gemini_decider,
    load_cluster_descriptions,
    load_submissions,
)
from run_consensus import ANTHROPIC_MODEL, GEMINI_MODEL


async def run(
    out_filename: str,
    temperature: float,
    prompt_template: str,
    client,
    submissions_path: str,
    disease: str,
    clusters_csv_path: str,
    use_anthropic: bool = False,
):
    submissions = load_submissions(submissions_path)
    cluster_descriptions = load_cluster_descriptions(clusters_csv_path)
    formatted = format_cluster_descriptions(cluster_descriptions)

    InformationalNeedEnum, InformationalNeed, allowed_values = (
        build_classification_components(cluster_descriptions)
    )

    if use_anthropic:
        func_decide = get_anthropic_decider(
            ANTHROPIC_MODEL, InformationalNeed, prompt_template, temperature, client
        )
    else:
        func_decide = get_gemini_decider(
            GEMINI_MODEL, InformationalNeed, prompt_template, temperature, client
        )

    results_log = []
    start = 0
    for i, submission in enumerate(submissions[start:]):
        submission_id = submission["permalink"]  # .split("/")[4]
        try:
            predicted_enums = await classify_submission(
                func_decide,
                submission,
                formatted,
                parallel_samples=1,
                disease=disease,
            )
            predicted_enum = predicted_enums[0]
        except ValueError:
            print(f"Skipping submission {submission_id}: too many retries")
            continue

        predicted_name = [
            c["title"]
            for c in cluster_descriptions
            if c["title"].replace(" ", "_").upper() == predicted_enum.value
        ][0]

        results_log.append(
            {
                "submission_id": submission_id,
                "title": submission["title"],
                "predicted_label": predicted_name,
            }
        )

        if i % 10 == 0:
            print(f"Processed {i + start + 1}/{len(submissions)}: '{predicted_name}'")

        with open(out_filename, "w") as f:
            json.dump(results_log, f, indent=2)

    with open(out_filename, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nResults saved to {out_filename}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--filename", default=None, help="Output file path.")
    parser.add_argument(
        "--submissions_path",
        default="./submissions.json",
        help="Path to submissions JSON file.",
    )
    parser.add_argument(
        "--disease",
        default=DEFAULT_DISEASE,
        help="Disease name for classification context.",
    )
    parser.add_argument(
        "--clusters_csv_path",
        default=CLUSTERS_CSV_PATH,
        help="Path to clusters CSV file.",
    )
    parser.add_argument(
        "--use-anthropic",
        action="store_true",
        help="Use an Anthropic model via Vertex AI instead of Gemini.",
    )
    parser.add_argument(
        "--region",
        default="us-east5",
        help="Vertex AI region for Anthropic models (default: us-east5).",
    )
    args = parser.parse_args()

    if args.use_anthropic:
        from anthropic import AnthropicVertex

        client = AnthropicVertex(project_id="kuligin-sandbox1", region=args.region)
    else:
        client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=True,
            project="kuligin-sandbox1",
            location="global",
        )

    prompt_templates = {
        "v1": PROMPT_V1,
        "v2": PROMPT_V2,
        "v3": PROMPT_V3,
        "v4": PROMPT_V4,
    }
    prompt_template = prompt_templates.get(args.prompt, PROMPT_V4)

    await run(
        out_filename=args.filename,
        temperature=args.temperature,
        prompt_template=prompt_template,
        client=client,
        submissions_path=args.submissions_path,
        disease=args.disease,
        clusters_csv_path=args.clusters_csv_path,
        use_anthropic=args.use_anthropic,
    )


if __name__ == "__main__":
    asyncio.run(main())
