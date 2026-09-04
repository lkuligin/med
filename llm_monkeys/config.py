"""Configuration module for MedQA inference experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "vertex_ai/google/gemma-4-26b-a4b-it-maas"

SUPPORTED_MODELS: dict[str, str] = {
    "gemma-4-26b": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
    "gpt-oss-20b": "vertex_ai/openai/gpt-oss-20b-maas",
    "gemini-3.8-flash": "vertex_ai/gemini-3.8-flash",
}


def register_litellm_model_pricing() -> None:
    """Register custom pricing and metadata for models in LiteLLM's model_cost dict."""
    try:
        import litellm

        gpt_20b_info = {
            "input_cost_per_token": 0.075 / 1_000_000,
            "output_cost_per_token": 0.30 / 1_000_000,
            "cache_read_input_token_cost": 0.0075 / 1_000_000,
            "litellm_provider": "vertex_ai",
            "max_input_tokens": 131072,
            "max_output_tokens": 32768,
            "max_tokens": 32768,
            "mode": "chat",
            "supports_function_calling": False,
            "supports_reasoning": True,
            "source": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing?e=48754805#openais-models",
        }
        pricing_entries = {
            "vertex_ai/openai/gpt-oss-20b-maas": gpt_20b_info,
        }
        litellm.model_cost.update(pricing_entries)
    except Exception:
        pass


register_litellm_model_pricing()


def resolve_model_name(model_name: str | None) -> str:
    """Resolve a model name or alias to its canonical Vertex AI LiteLLM identifier."""
    if not model_name or not model_name.strip():
        return DEFAULT_MODEL
    cleaned = model_name.strip()
    lower = cleaned.lower()
    if lower in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[lower]
    if lower.startswith("openai/gpt-oss"):
        return f"vertex_ai/{cleaned}"
    if lower.startswith("google/gemma"):
        return f"vertex_ai/{cleaned}"
    return cleaned


@dataclass
class InferenceConfig:
    """Configuration for running one-shot LLM inference on MedQA."""

    model_name: str = DEFAULT_MODEL
    temperature: float = 0.8
    max_tokens: int = 1024
    system_instruction: str = (
        "You are an expert physician taking a medical licensing board examination. "
        "Answer all questions accurately with careful clinical reasoning."
    )

    project_id: str | None = field(
        default_factory=lambda: (
            os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT")
        )
    )
    location: str = field(
        default_factory=lambda: os.getenv("VERTEXAI_LOCATION", "global")
    )

    dataset_name: str = "bigbio/med_qa"
    dataset_config: str = "med_qa_en_source"
    dataset_split: str = "test"
    limit: int | None = None
    offset: int = 0

    n_attempts: int = 3
    concurrency: int = 2
    max_retries: int = 5
    rate_limit_max_retries: int = 10
    max_parse_retries: int = 3
    base_retry_delay: float = 2.0
    max_retry_delay: float = 60.0
    litellm_num_retries: int = 3

    output_filepath: str = "results_one_shot_gemma4.json"
    save_every_n: int = 10
