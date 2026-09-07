"""Configuration module for MedQA inference experiments and workflows."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

from inference._prompts import (
    DEFAULT_ANSWER_SYSTEM_INSTRUCTION,
    DEFAULT_FACT_SYSTEM_INSTRUCTION,
)
from verifier._prompts import DEFAULT_VERIFIER_SYSTEM_INSTRUCTION

DEFAULT_MODEL = "vertex_ai/google/gemma-4-26b-a4b-it-maas"
DEFAULT_VERIFIER_MODEL = "gemini-3.8-flash"

SUPPORTED_MODELS: dict[str, str] = {
    "gemma-4-26b": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
    "gpt-oss-20b": "vertex_ai/openai/gpt-oss-20b-maas",
    "gemini-3.8-flash": "vertex_ai/gemini-3.8-flash",
    "gemini-3-flash-preview": "vertex_ai/gemini-3-flash-preview",
}

DEFAULT_ONE_SHOT_INSTRUCTION = (
    "You are an expert physician taking a medical licensing board examination. "
    "Answer all questions accurately with careful clinical reasoning."
)


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


class WorkloadType(str, Enum):
    """Supported inference workload types."""

    ONE_SHOT = "one_shot"
    CANDIDATE = "candidate"
    VERIFIER = "verifier"


@dataclass
class BaseInferenceConfig:
    """Base configuration for MedQA inference experiments and workflows."""

    model_name: str = DEFAULT_MODEL
    temperature: float = 0.8
    max_tokens: int = 1024

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

    concurrency: int = 2
    max_retries: int = 5
    rate_limit_max_retries: int = 10
    max_parse_retries: int = 3
    base_retry_delay: float = 2.0
    max_retry_delay: float = 60.0
    litellm_num_retries: int = 3

    output_filepath: str = "results_one_shot_gemma4.json"

    @property
    def resolved_model_name(self) -> str:
        """Resolve this configuration's model name to its canonical identifier."""
        return resolve_model_name(self.model_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls: type[Self], data: dict[str, Any], ignore_unknown: bool = True
    ) -> Self:
        """Instantiate configuration from a dictionary.

        Args:
            data: Key-value mapping of configuration fields.
            ignore_unknown: If True, ignore keys that are not recognized fields.

        Returns:
            Instance of the configuration class.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        if ignore_unknown:
            filtered = {k: v for k, v in data.items() if k in valid_fields}
        else:
            unknown = set(data) - valid_fields
            if unknown:
                raise ValueError(
                    f"Unknown configuration keys for {cls.__name__}: {sorted(unknown)}"
                )
            filtered = data
        return cls(**filtered)

    def copy_with(self: Self, **overrides: Any) -> Self:
        """Create a new instance with the specified field overrides."""
        return dataclasses.replace(self, **overrides)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.rate_limit_max_retries < 0:
            raise ValueError(
                f"rate_limit_max_retries must be >= 0, got {self.rate_limit_max_retries}"
            )


@dataclass
class OneShotInferenceConfig(BaseInferenceConfig):
    """Configuration for running one-shot LLM inference on MedQA (Step 1)."""

    system_instruction: str = DEFAULT_ONE_SHOT_INSTRUCTION
    n_attempts: int = 3
    concurrency: int = 2
    output_filepath: str = "results_one_shot_gemma4.json"
    save_every_n: int = 10

    def validate(self) -> None:
        """Validate one-shot specific parameters."""
        super().validate()
        if self.n_attempts < 1:
            raise ValueError(f"n_attempts must be >= 1, got {self.n_attempts}")
        if self.save_every_n < 1:
            raise ValueError(f"save_every_n must be >= 1, got {self.save_every_n}")


# Backward-compatibility alias for Step 1
InferenceConfig = OneShotInferenceConfig


@dataclass
class CandidateInferenceConfig(BaseInferenceConfig):
    """Configuration for running candidate-based multi-step MedQA inference (Step 2)."""

    concurrency: int = 4
    output_filepath: str = "results_step2_gemma4_candidates.json"

    fact_system_instruction: str = DEFAULT_FACT_SYSTEM_INSTRUCTION
    answer_system_instruction: str = DEFAULT_ANSWER_SYSTEM_INSTRUCTION
    difficult_questions_path: str = "difficult_questions.csv"

    n_candidates: int = 1000
    save_every_n_candidates: int = 25
    save_every_n_questions: int = 1

    def validate(self) -> None:
        """Validate candidate-specific parameters."""
        super().validate()
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")
        if self.save_every_n_candidates < 1:
            raise ValueError(
                f"save_every_n_candidates must be >= 1, got {self.save_every_n_candidates}"
            )
        if self.save_every_n_questions < 1:
            raise ValueError(
                f"save_every_n_questions must be >= 1, got {self.save_every_n_questions}"
            )


@dataclass
class VerifierConfig(BaseInferenceConfig):
    """Configuration for running fact verification on step 2 candidates (Step 3)."""

    model_name: str = DEFAULT_VERIFIER_MODEL
    temperature: float = 0.0
    max_tokens: int = 512
    concurrency: int = 4
    input_filepath: str = "results_step2_gemma4_candidates.json"
    output_filepath: str = "results_step3_verified.json"
    system_instruction: str = DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
    save_every_n_questions: int = 1
    max_candidates_per_question: int | None = None
    early_stop_facts: bool = False
    early_stop_candidates: bool = True

    def validate(self) -> None:
        """Validate verifier-specific parameters."""
        super().validate()
        if self.save_every_n_questions < 1:
            raise ValueError(
                f"save_every_n_questions must be >= 1, got {self.save_every_n_questions}"
            )
        if (
            self.max_candidates_per_question is not None
            and self.max_candidates_per_question < 1
        ):
            raise ValueError(
                f"max_candidates_per_question must be >= 1, got {self.max_candidates_per_question}"
            )


def create_config(
    workload: str | WorkloadType = WorkloadType.ONE_SHOT,
    **kwargs: Any,
) -> BaseInferenceConfig:
    """Create an inference configuration for the specified workload type.

    Args:
        workload: Workload type ('one_shot', 'candidate', 'verifier', 'step1', 'step2', 'step3').
        **kwargs: Configuration overrides passed to constructor.

    Returns:
        OneShotInferenceConfig, CandidateInferenceConfig, or VerifierConfig instance.
    """
    key = (
        str(workload.value if isinstance(workload, WorkloadType) else workload)
        .lower()
        .strip()
    )
    if key in ("one_shot", "oneshot", "step1"):
        return OneShotInferenceConfig(**kwargs)
    elif key in ("candidate", "candidates", "candidate_inference", "step2"):
        return CandidateInferenceConfig(**kwargs)
    elif key in ("verifier", "verify", "step3"):
        return VerifierConfig(**kwargs)
    else:
        raise ValueError(
            f"Unknown workload type: {workload!r}. Supported types: 'one_shot', 'candidate', 'verifier'"
        )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VERIFIER_MODEL",
    "SUPPORTED_MODELS",
    "DEFAULT_ONE_SHOT_INSTRUCTION",
    "DEFAULT_FACT_SYSTEM_INSTRUCTION",
    "DEFAULT_ANSWER_SYSTEM_INSTRUCTION",
    "DEFAULT_VERIFIER_SYSTEM_INSTRUCTION",
    "WorkloadType",
    "BaseInferenceConfig",
    "OneShotInferenceConfig",
    "InferenceConfig",
    "CandidateInferenceConfig",
    "VerifierConfig",
    "create_config",
    "register_litellm_model_pricing",
    "resolve_model_name",
]
