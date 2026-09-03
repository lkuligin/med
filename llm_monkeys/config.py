"""Configuration module for MedQA inference experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    """Configuration for running one-shot LLM inference on MedQA."""

    model_name: str = "vertex_ai/google/gemma-4-26b-a4b-it-maas"
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
    max_parse_retries: int = 3
    base_retry_delay: float = 2.0
    max_retry_delay: float = 30.0

    output_filepath: str = "results_one_shot_gemma4.json"
    save_every_n: int = 10
