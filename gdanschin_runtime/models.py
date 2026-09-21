"""The models a run may use, each under a unique name.

A name identifies a configuration, not just a model: the same weights served
with thinking disabled, or judged at a different temperature, are different
entries, because they produce different results and their outputs must not
land in the same directory.

    from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS, describe

    describe()                      # what is available
    BASE_MODELS["gemma-4-26b"]      # one configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaseModel:
    """A model that generates answers and candidate reasoning paths."""

    name: str
    gateway_model: str
    max_tokens: int
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # Set when the model is served from our own GPUs instead of the gateway.
    # The value is an OpenAI-compatible base URL, and gateway_model is then the
    # name that endpoint answers to. Nothing here imports gpu_serving: the only
    # thing the two share is the URL, which is what makes the serving package
    # separable.
    base_url: str = ""


@dataclass(frozen=True)
class JudgeModel:
    """A model that verifies facts."""

    name: str
    gateway_model: str
    max_tokens: int
    temperature: float
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# max_tokens is per model and measured, not guessed. Too low a budget does not
# raise: the reply comes back truncated or empty, the fact parser finds nothing,
# and the candidate is quietly worthless. The values below leave roughly a
# factor of two over the largest response seen on difficult questions.
BASE_MODELS: dict[str, BaseModel] = {
    "gemma-4-26b": BaseModel(
        name="gemma-4-26b",
        gateway_model="gemma-4-26b-internal",
        max_tokens=1024,
        note="the model the original experiment used; peaks around 1100 tokens",
    ),
    "gpt-oss-120b": BaseModel(
        name="gpt-oss-120b",
        gateway_model="gpt-oss-120b",
        max_tokens=4096,
        note="second SLM named in the README; fewer, larger facts; peaks near 2900",
    ),
    "deepseek-v4-flash": BaseModel(
        name="deepseek-v4-flash",
        gateway_model="deepseek-v4-flash",
        max_tokens=4096,
    ),
    "qwen3.6-27b-nr": BaseModel(
        name="qwen3.6-27b-nr",
        gateway_model="qwen3.6-27b-noreasoning",
        max_tokens=4096,
        note="thinking disabled; with it on the whole budget goes to reasoning",
    ),
    "gemma-4-26b-local": BaseModel(
        name="gemma-4-26b-local",
        gateway_model="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
        max_tokens=1024,
        note="the same weights as gemma-4-26b, served on our own cards. A "
             "separate entry, not a flag, because the run name is the results "
             "directory: mixing the two backends into one directory would "
             "destroy the comparison they exist for",
    ),
    "qwen3.8-27b-nr": BaseModel(
        name="qwen3.8-27b-nr",
        gateway_model="qwen3.8-27b-noreasoning",
        max_tokens=4096,
        note="thinking disabled",
    ),
}

JUDGE_MODELS: dict[str, JudgeModel] = {
    "gemini-3.8-flash": JudgeModel(
        name="gemini-3.8-flash",
        gateway_model="gemini-3.8-flash",
        max_tokens=512,
        temperature=1.0,
        note="the reference default; note temperature 1.0 makes verdicts non-deterministic",
    ),
}


def describe() -> None:
    """Print the registry."""
    print("base models (generate candidates):")
    for m in BASE_MODELS.values():
        where = f" @ {m.base_url}" if m.base_url else ""
        print(f"  {m.name:20} {m.gateway_model:26} max_tokens={m.max_tokens}{where}")
        if m.note:
            print(f"    {m.note}")
    print("\njudges (verify facts):")
    for j in JUDGE_MODELS.values():
        print(f"  {j.name:20} {j.gateway_model:26} max_tokens={j.max_tokens} t={j.temperature}")
        if j.note:
            print(f"    {j.note}")


if __name__ == "__main__":
    # python3 -m gdanschin_runtime.models - the registry from a shell, for when
    # a --base or --judge name has to be typed and only the notebook knew them.
    describe()


__all__ = ["BaseModel", "JudgeModel", "BASE_MODELS", "JUDGE_MODELS", "describe"]
