import argparse
import asyncio
import json
import os

from google import genai
from google.genai import types

from _ehr import create_ehr_hint
from _prompts import _PROMPT_STEP_1


async def generate_single_profile(
    client: genai.Client,
    prompt_text: str,
    model: str = "gemini-3.7-flash",
    max_revisions: int = 2,
) -> str:
    """Generates a single psychological profile with revision steps using native Google GenAI SDK."""
    config = types.GenerateContentConfig(
        tools=[{"google_search": {}}],
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )
    draft = response.text or ""

    current_profile = draft
    for _ in range(max_revisions):
        refine_prompt = (
            f"Reflect on and improve the following psychological profile generated for a patient:\n\n"
            f"{current_profile}\n\n"
            f"Provide an updated, comprehensive, and accurate psychological profile."
        )
        refine_response = await client.aio.models.generate_content(
            model=model,
            contents=refine_prompt,
            config=config,
        )
        if refine_response.text:
            current_profile = refine_response.text

    return current_profile


def _save_profiles(output_path: str, profiles: list[str]) -> None:
    """Saves generated profiles to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"profiles": profiles}, f, ensure_ascii=False, indent=2)


async def main(
    ehr_path: str, num_profiles: int = 20, project: str = "kuligin-sandbox-502813"
):
    """Generates psychological profiles and saves them to a file."""
    ehr_hint = create_ehr_hint(ehr_path)
    prompt_text = _PROMPT_STEP_1.format(ehr_hint=ehr_hint)

    client = genai.Client(
        http_options=types.HttpOptions(api_version="v1"),
        vertexai=True,
        project=project,
        location="global",
    )

    profiles = []
    suffix = os.path.splitext(os.path.basename(ehr_path))[0]
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(num_profiles):
        profile = await generate_single_profile(client, prompt_text)
        print(f"Generated profile {i + 1}/{num_profiles}")
        profiles.append(profile)

        output_path = os.path.join(results_dir, f"profiles_{suffix}.json")
        await asyncio.to_thread(_save_profiles, output_path, profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate psychological profiles from an EHR JSON file."
    )
    parser.add_argument("--ehr_path", help="Path to the EHR JSON file")
    parser.add_argument(
        "--num_profiles", type=int, default=20, help="Number of profiles to generate"
    )
    parser.add_argument(
        "--project", default="kuligin-sandbox-502813", help="GCP project ID"
    )
    args = parser.parse_args()
    asyncio.run(
        main(args.ehr_path, num_profiles=args.num_profiles, project=args.project)
    )
