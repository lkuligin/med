"""Configuration module for MedQA inference experiments and workflows."""

from __future__ import annotations

import dataclasses
import importlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

from results_store import (
    DEFAULT_CANDIDATES_FILE,
    DEFAULT_ONE_SHOT_FILE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_VERIFIED_FILE,
    run_name_for,
)

from inference._prompts import (
    ANSWER_PROMPTS,
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_ANSWER_SYSTEM_INSTRUCTION,
    DEFAULT_FACT_PROMPT,
    DEFAULT_FACT_SYSTEM_INSTRUCTION,
    FACT_PROMPTS,
)
from verifier._prompts import (
    DEFAULT_JUDGE_PROMPT,
    DEFAULT_VERIFIER_SYSTEM_INSTRUCTION,
    JUDGE_PROMPTS,
)

DEFAULT_MODEL = "vertex_ai/google/gemma-4-26b-a4b-it-maas"
DEFAULT_VERIFIER_MODEL = "gemini-3.8-flash"

SUPPORTED_MODELS: dict[str, str] = {
    "gemma-4-26b": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
    "google/gemma-4-26b-a4b-it-maas": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
    "gpt-oss-20b": "vertex_ai/openai/gpt-oss-20b-maas",
    "gpt-oss-20b-maas": "vertex_ai/openai/gpt-oss-20b-maas",
    "openai/gpt-oss-20b-maas": "vertex_ai/openai/gpt-oss-20b-maas",
    "gemini-3.8-flash": "vertex_ai/gemini-3.8-flash",
    "gemini-flash-3.8": "vertex_ai/gemini-3.8-flash",
}

DEFAULT_DATASET = "bigbio/med_qa"
DEFAULT_DATASET_CONFIG = "med_qa_en_source"
DEFAULT_DATASET_SPLIT = "test"

MEDBULLETS_DATASET = "mkieffer/Medbullets"
MEDBULLETS_SPLIT = "op5_test"

SUPPORTED_DATASETS: dict[str, str] = {
    "med_qa": "bigbio/med_qa",
    "medqa": "bigbio/med_qa",
    "bigbio/med_qa": "bigbio/med_qa",
    "medbullets": "mkieffer/Medbullets",
    "medbullets_op5": "mkieffer/Medbullets",
    "mkieffer/medbullets": "mkieffer/Medbullets",
}

DEFAULT_ONE_SHOT_INSTRUCTION = (
    "You are an expert physician taking a medical licensing board examination. "
    "Answer all questions accurately with careful clinical reasoning."
)


def register_litellm_model_pricing() -> None:
    """Register custom pricing and metadata for models in LiteLLM's model_cost dict."""
    try:
        import litellm

        litellm.suppress_debug_info = True

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
            "openai/gpt-oss-20b-maas": gpt_20b_info,
            "gpt-oss-20b-maas": gpt_20b_info,
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

    if lower.startswith("vertex_ai/"):
        unprefixed = lower[len("vertex_ai/") :]
        if unprefixed in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[unprefixed]
        if unprefixed.startswith("gemini-flash-"):
            candidate = f"gemini-{unprefixed[len('gemini-flash-') :]}-flash"
            if candidate in SUPPORTED_MODELS:
                return SUPPORTED_MODELS[candidate]
            return f"vertex_ai/{candidate}"
        return cleaned

    if lower.startswith("gemini-flash-"):
        candidate = f"gemini-{lower[len('gemini-flash-') :]}-flash"
        if candidate in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[candidate]
        return f"vertex_ai/{candidate}"

    if lower.startswith("openai/gpt-oss"):
        return f"vertex_ai/{cleaned}"
    if lower.startswith("google/gemma"):
        return f"vertex_ai/{cleaned}"
    if lower.startswith("gpt-oss"):
        model_part = cleaned if cleaned.endswith("-maas") else f"{cleaned}-maas"
        return f"vertex_ai/openai/{model_part}"
    if lower.startswith("gemma"):
        return f"vertex_ai/google/{cleaned}"
    if lower.startswith("gemini"):
        return f"vertex_ai/{cleaned}"
    return cleaned


MODEL_FACTORY_ENV = "MEDQA_MODEL_FACTORY"
RESULTS_STORE_ENV = "MEDQA_RESULTS_STORE"


def _resolve_factory(env_var: str) -> Any | None:
    """Resolve an optional ``"module.path:callable"`` override from the environment.

    Returns None when the variable is unset, which is what keeps the default
    behaviour exactly as it was for anyone who has not opted in.

    Raises:
        RuntimeError: If the spec is malformed or cannot be imported. Failing
            loudly is deliberate: an override that silently does nothing sends
            work somewhere other than where it was asked to go, and that is not
            visible until the results are read.
    """
    spec = os.getenv(env_var)
    if not spec or not spec.strip():
        return None

    cleaned = spec.strip()
    module_name, sep, attr = cleaned.partition(":")
    if not sep or not module_name.strip() or not attr.strip():
        raise RuntimeError(
            f"{env_var} must look like 'module.path:callable', got {cleaned!r}"
        )

    try:
        module = importlib.import_module(module_name.strip())
    except Exception as exc:
        raise RuntimeError(
            f"{env_var}: cannot import module {module_name.strip()!r}: {exc}"
        ) from exc

    try:
        factory = getattr(module, attr.strip())
    except AttributeError as exc:
        raise RuntimeError(
            f"{env_var}: module {module_name.strip()!r} has no attribute {attr.strip()!r}"
        ) from exc

    if not callable(factory):
        raise RuntimeError(f"{env_var}: {cleaned!r} is not callable")
    return factory


def resolve_model_factory() -> Any | None:
    """The model-factory override declared via MEDQA_MODEL_FACTORY, or None.

    The callable receives the active configuration and must return an ADK
    ``BaseLlm``. Unset, callers fall back to the stock LiteLlm adapter.
    """
    return _resolve_factory(MODEL_FACTORY_ENV)


def resolve_results_store() -> Any | None:
    """The results-store override declared via MEDQA_RESULTS_STORE, or None.

    The callable receives the active configuration and must return an object
    with ``save(payload)``, ``load()`` and a readable ``str()``. Unset, results
    are written as one JSON file per run, which is what every entry point here
    has always done.
    """
    return _resolve_factory(RESULTS_STORE_ENV)

def resolve_dataset_name(dataset_name: str | None) -> str:
    """Resolve dataset name or alias to canonical Hugging Face dataset identifier."""
    if not dataset_name or not dataset_name.strip():
        return DEFAULT_DATASET
    cleaned = dataset_name.strip()
    lower = cleaned.lower()
    if lower in SUPPORTED_DATASETS:
        return SUPPORTED_DATASETS[lower]
    return cleaned


def is_medbullets_dataset(dataset_name: str | None) -> bool:
    """Check if a dataset name or alias corresponds to the MedBullets dataset."""
    if not dataset_name:
        return False
    lower = dataset_name.strip().lower()
    return (
        "medbullets" in lower
        or lower == "mkieffer/medbullets"
        or lower == MEDBULLETS_DATASET.lower()
    )


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

    dataset_name: str = DEFAULT_DATASET
    dataset_config: str | None = DEFAULT_DATASET_CONFIG
    dataset_split: str = DEFAULT_DATASET_SPLIT
    limit: int | None = None
    offset: int = 0

    concurrency: int = 2
    max_retries: int = 5
    rate_limit_max_retries: int = 10
    max_parse_retries: int = 3
    base_retry_delay: float = 2.0
    max_retry_delay: float = 60.0
    litellm_num_retries: int = 3

    # Where a run's results go. output_filepath is what the default store
    # writes; results_dir and run_name are read by a store that keeps a
    # directory per run, and mean nothing to the default one.
    output_filepath: str = DEFAULT_ONE_SHOT_FILE
    results_dir: str = DEFAULT_RESULTS_DIR
    run_name: str | None = None

    @property
    def resolved_run_name(self) -> str:
        """Directory name for this run, derived from the model unless set.

        Naming it explicitly is what keeps two runs of one model under
        different settings from landing in the same directory and blending
        into a single dataset.
        """
        return self.run_name or run_name_for(self.resolved_model_name)

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
    output_filepath: str = DEFAULT_ONE_SHOT_FILE
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

    # Ask the fact step for a JSON schema, which is how every stored run was
    # generated. Turn it off for a model that answers the schema by satisfying
    # it emptily: gpt-oss-20b returned {"facts": []} for a third of its
    # candidates - complete, valid, and useless - spending 98% of its output
    # on reasoning first. Unconstrained it writes the facts as prose, which
    # parse_medical_facts already reads through its bulleted fallback.
    structured_facts: bool = True

    concurrency: int = 4
    output_filepath: str = DEFAULT_CANDIDATES_FILE

    # Which entry of FACT_PROMPTS asks for the facts. "reference" is the
    # authors' wording; anything else is an experiment and wants its own
    # run name, since the store does not tell prompts apart on resume.
    fact_prompt: str = DEFAULT_FACT_PROMPT
    # None takes the system instruction of fact_prompt; a string overrides it.
    fact_system_instruction: str | None = None
    # Which entry of ANSWER_PROMPTS asks for the answer; same rules as fact_prompt.
    answer_prompt: str = DEFAULT_ANSWER_PROMPT
    # None takes the system instruction of answer_prompt; a string overrides it.
    answer_system_instruction: str | None = None
    difficult_questions_path: str = "difficult_questions.csv"

    n_candidates: int = 1000
    save_every_n_candidates: int = 25
    save_every_n_questions: int = 1

    @property
    def resolved_fact_system_instruction(self) -> str:
        """The fact generator's system instruction, after any override."""
        return (self.fact_system_instruction
                or FACT_PROMPTS[self.fact_prompt].system_instruction)

    @property
    def resolved_answer_system_instruction(self) -> str:
        """The answer step's system instruction, after any override."""
        return (self.answer_system_instruction
                or ANSWER_PROMPTS[self.answer_prompt].system_instruction)

    def validate(self) -> None:
        """Validate candidate-specific parameters."""
        super().validate()
        if self.answer_prompt not in ANSWER_PROMPTS:
            raise ValueError(
                f"answer_prompt must be one of {sorted(ANSWER_PROMPTS)}, got {self.answer_prompt!r}"
            )
        if self.fact_prompt not in FACT_PROMPTS:
            raise ValueError(
                f"fact_prompt must be one of {sorted(FACT_PROMPTS)}, got {self.fact_prompt!r}"
            )
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
    temperature: float = 1.0
    max_tokens: int = 1024
    concurrency: int = 4
    input_filepath: str = DEFAULT_CANDIDATES_FILE
    output_filepath: str = DEFAULT_VERIFIED_FILE
    judge_name: str | None = None
    # Which entry of JUDGE_PROMPTS asks for the verdicts. "reference" is the
    # authors' prompt; anything else is an experiment and wants its own
    # judge name, since the store does not tell prompts apart on resume.
    judge_prompt: str = DEFAULT_JUDGE_PROMPT
    # None takes the system instruction of judge_prompt; a string overrides it.
    system_instruction: str | None = None
    save_every_n_questions: int = 1
    max_candidates_per_question: int | None = None
    early_stop_facts: bool = False
    early_stop_candidates: bool = True
    # Fill the slots a run's last questions leave idle by judging further
    # candidates of them side by side. Off by default, and deliberately not
    # inferred from anything: the extra candidates are extra requests, which
    # on a judge that charges per call is money spent to finish a few minutes
    # sooner. Turn it on for a judge that runs on hardware already paid for.
    speculate_tail: bool = False
    # How long one fact check may take before it is cancelled and retried.
    # 120 suits a judge that answers without thinking; measured on GLM-5.3,
    # p99 is 174s and the slowest check took 838s, so a tenth of a percent of
    # the work was being cancelled and retried five times over - which adds
    # load, which cancels more. Raise it for a judge that reasons.
    request_timeout: float = 120.0
    # How many candidates of one question may be judged side by side once
    # questions run out. Bounded because a question's candidates share their
    # prompt prefix, so cache-aware routing sends them all to one replica: the
    # width is that replica's queue depth, and a deep queue of long
    # generations is what request_timeout starts cancelling. Four suited a
    # single-replica judge; a faster one tolerates more.
    speculate_width: int = 4
    difficult_questions: str | None = None

    @property
    def resolved_judge_name(self) -> str:
        """Directory name for this judge's verdicts, derived from its model.

        run_name names the run being verified, since the verdicts are stored
        beside the candidates they judge; this names the judge within it.
        """
        return self.judge_name or run_name_for(self.resolved_model_name)

    @property
    def resolved_system_instruction(self) -> str:
        """The judge's system instruction, after any override."""
        return self.system_instruction or JUDGE_PROMPTS[self.judge_prompt].system_instruction

    def validate(self) -> None:
        """Validate verifier-specific parameters."""
        super().validate()
        if self.judge_prompt not in JUDGE_PROMPTS:
            raise ValueError(
                f"judge_prompt must be one of {sorted(JUDGE_PROMPTS)}, got {self.judge_prompt!r}"
            )
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
    "DEFAULT_DATASET",
    "DEFAULT_DATASET_CONFIG",
    "DEFAULT_DATASET_SPLIT",
    "MEDBULLETS_DATASET",
    "MEDBULLETS_SPLIT",
    "SUPPORTED_DATASETS",
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
    "resolve_model_factory",
    "MODEL_FACTORY_ENV",
    "RESULTS_STORE_ENV",
    "resolve_results_store",

    "resolve_dataset_name",
    "is_medbullets_dataset",
]
