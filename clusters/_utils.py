import asyncio
import csv
import json
import os
import random
import traceback
from enum import Enum

from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

DEFAULT_DISEASE = "breast cancer"
MAX_RETRIES = 10
MAX_CONCURRENT = 4
MAX_EMPTY_RESULTS = 6

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

CLUSTERS_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DATA_PATH = os.path.join(CLUSTERS_DIR, "golden_data_enriched.csv")
CLUSTERS_CSV_PATH = os.path.join(CLUSTERS_DIR, "clusters.csv")
_MAP_LABELS = {
    "Clinical Decision-Making (including Risk Reduction, Surgery, and Surveillance) AND Peer experience seeking": "Clinical Decision-Making (including Risk Reduction, Surgery, and Surveillance)"
}


def load_cluster_descriptions(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        return [{"title": row[0], "description": row[1]} for row in reader if row]


def load_submissions(path: str) -> list:
    with open(path, "r") as rf:
        return json.load(rf)["submissions"]


def load_golden_dataset(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "title": row["title"],
                "text": row["text"],
                "cluster": row["cluster"],
                "submission_id": row["submission_id"],
            }
            for row in reader
        ]


def save_results(path: str, clusters: list) -> None:
    with open(path, "w") as wf:
        wf.write(json.dumps({"clusters": clusters}))


def format_cluster_descriptions(cluster_descriptions: list[dict]) -> str:
    return "\n".join(
        f"InformationalNeed_{i + 1}:\nTITLE:{c['title']}\nDESCRIPTION:{c['description']}\n"
        for i, c in enumerate(cluster_descriptions)
    )


def build_classification_components(cluster_descriptions, include_reasoning=True):
    allowed_values = [
        c["title"].replace(" ", "_").upper() for c in cluster_descriptions
    ]
    InformationalNeedEnum = Enum(
        "InformationalNeedEnum", {v: v for v in allowed_values}
    )

    _REASONING_DESCRIPTION = "Step-by-step reasoning identifying the author's dominant informational need before selecting a category"

    def _from_anthropic_tool_block(cls, tool_block):
        kwargs = {
            "informational_need": InformationalNeedEnum[
                tool_block.input["informational_need"].upper()
            ]
        }
        if include_reasoning:
            kwargs["reasoning"] = tool_block.input.get("reasoning", "")
        return cls(**kwargs)

    def _to_anthropic_tool():
        properties = {}
        required = []
        if include_reasoning:
            properties["reasoning"] = {
                "type": "string",
                "description": _REASONING_DESCRIPTION,
            }
            required.append("reasoning")
        properties["informational_need"] = {
            "type": "string",
            "enum": allowed_values,
            "description": "Patient's informational need",
        }
        required.append("informational_need")
        return {
            "name": "classify",
            "description": "Classify the patient's informational need",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

    if include_reasoning:

        class InformationalNeed(BaseModel):
            reasoning: str = Field(description=_REASONING_DESCRIPTION)
            informational_need: InformationalNeedEnum = Field(
                description="Patient's informational need"
            )
            from_anthropic_tool_block = classmethod(_from_anthropic_tool_block)
            to_anthropic_tool = staticmethod(_to_anthropic_tool)

    else:

        class InformationalNeed(BaseModel):
            informational_need: InformationalNeedEnum = Field(
                description="Patient's informational need"
            )
            from_anthropic_tool_block = classmethod(_from_anthropic_tool_block)
            to_anthropic_tool = staticmethod(_to_anthropic_tool)

    return InformationalNeedEnum, InformationalNeed, allowed_values


async def _run_with_retry(
    decide_func,
    title: str,
    text: str,
    formatted: str,
    collaboration_context: str = None,
    disease: str = DEFAULT_DISEASE,
):
    for retry in range(1, MAX_RETRIES + 1):
        try:
            result = await decide_func(
                title=title,
                text=text,
                disease=disease,
                formatted=formatted,
                collaboration_context=collaboration_context,
            )
            if result is not None:
                return result
        except Exception:
            traceback.print_exc()
        if retry >= MAX_RETRIES:
            raise ValueError()
        delay = min(2**retry + random.uniform(0, 1), 60)
        print(
            f"result is None, retrying in {delay:.1f}s (attempt {retry}/{MAX_RETRIES})"
        )
        await asyncio.sleep(delay)


def load_eval_results(filename: str) -> tuple[list, list]:
    print(f"Loading results from {filename}")
    with open(filename) as f:
        results_log = json.load(f)
    submissions = load_golden_dataset(GOLDEN_DATA_PATH)
    predicted_labels = [r["predicted_label"] for r in results_log]
    true_labels = [r["true_label"] for r in results_log]
    true_labels = [_MAP_LABELS.get(el, el) for el in true_labels]
    incorrect = [
        r
        for r in results_log
        if r["predicted_label"] != _MAP_LABELS.get(r["true_label"], r["true_label"])
    ]
    print(f"\nIncorrectly classified: {len(incorrect)}/{len(results_log)}")
    for r in incorrect:
        print(
            f"  [{r['submission_id']}] true='{r['true_label']}' predicted='{r['predicted_label']}'"
        )
        submission = next(
            s for s in submissions if s["submission_id"] == r["submission_id"]
        )
        print(submission)
    return predicted_labels, true_labels


def get_gemini_decider(
    model_name,
    expected_output,
    prompt_template: str,
    temperature: float = 0.0,
    client=None,
):
    async def _decide(
        title: str,
        text: str,
        disease: str,
        formatted: str,
        collaboration_context: str = None,
    ):
        async with semaphore:
            prompt = prompt_template.format(
                title=title, text=text, disease=disease, formatted=formatted
            )
            if collaboration_context:
                prompt += collaboration_context
            result = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=expected_output,
                    temperature=temperature,
                ),
            )
            return result.parsed

    return _decide


def get_anthropic_decider(
    model_name,
    expected_output,
    prompt_template: str,
    temperature: float = 0.0,
    client=None,
):
    async def _decide(
        title: str,
        text: str,
        disease: str,
        formatted: str,
        collaboration_context: str = None,
    ):
        async with semaphore:
            prompt = prompt_template.format(
                title=title, text=text, disease=disease, formatted=formatted
            )
            if collaboration_context:
                prompt += collaboration_context
            response = await asyncio.to_thread(
                lambda: client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    temperature=temperature,
                    tools=[expected_output.to_anthropic_tool()],
                    tool_choice={"type": "tool", "name": "classify"},
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            tool_use_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if tool_use_block is None:
                return None
            return expected_output.from_anthropic_tool_block(tool_use_block)

    return _decide


async def classify_submission(
    decide_func,
    submission: dict,
    formatted: str,
    parallel_samples: int,
    disease: str = DEFAULT_DISEASE,
) -> list:
    tasks = [
        _run_with_retry(
            decide_func,
            submission["title"],
            submission["text"],
            formatted,
            disease=disease,
        )
        for _ in range(parallel_samples)
    ]
    results = await asyncio.gather(*tasks)

    empty_results = len([r for r in results if r is None])
    if empty_results > MAX_EMPTY_RESULTS:
        raise ValueError()

    return [result.informational_need for result in results if result is not None]
