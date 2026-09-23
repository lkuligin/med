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

import os
from dataclasses import dataclass, field
from typing import Any


# Where a model we serve ourselves answers. Named once: a second server on
# another port, or a tunnel from a laptop, is then one variable rather than an
# edit to every entry. gpu_serving is not imported for this - the URL is the
# entire contract between the two, which is what keeps that package separable.
LOCAL_BASE_URL = os.environ.get("MEDQA_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")


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
def served_locally(name: str, served_name: str, max_tokens: int,
                   note: str = "", **extra: Any) -> BaseModel:
    """Declare a model served from our own GPUs rather than the gateway.

    The name is the results directory, so a locally served model gets its own
    entry rather than a flag on an existing one: one switch would mix both
    backends into the same directory and destroy the comparison the local run
    exists to make. The "-local" suffix is the convention that makes the pair
    obvious in a listing.

    `served_name` is what the endpoint answers to - read it from /v1/models
    rather than guessing, since it is set by --served-model-name on the server
    and need not match any name used here.
    """
    if not name.endswith("-local"):
        raise ValueError(f"{name!r}: locally served entries end in -local, so "
                         f"that a results directory says where it came from")
    return BaseModel(name=name, gateway_model=served_name, max_tokens=max_tokens,
                     note=note, base_url=LOCAL_BASE_URL, extra=extra)


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
    "deepseek-v4-flash-think-high": BaseModel(
        name="deepseek-v4-flash-think-high",
        gateway_model="deepseek-v4-flash-think-high",
        max_tokens=8192,
        note="thinking on at reasoning_effort=high; the gateway serves it "
             "without thinking by default, which is deepseek-v4-flash. "
             "Peaks at 2373 output tokens over a 15-answer smoke run.",
    ),
    "glm5.3-flash": BaseModel(
        name="glm5.3-flash",
        gateway_model="glm5.3-flash",
        max_tokens=4096,
        note="320B total, 18B active; the largest open-weight model the "
             "internal gateway serves, and from none of our candidates' families",
    ),
    "qwen3.6-27b-nr": BaseModel(
        name="qwen3.6-27b-nr",
        gateway_model="qwen3.6-27b-noreasoning",
        max_tokens=4096,
        note="thinking disabled; with it on the whole budget goes to reasoning",
    ),
    "gemma-4-26b-local": served_locally(
        "gemma-4-26b-local", "google/gemma-4-26B-A4B-it", max_tokens=4096,
        note="the same weights as gemma-4-26b, on our own cards. 4096 rather "
             "than the gateway entry's 1024: at 1024, 36 of 1273 MedQA and 7 "
             "of 308 MedBullets questions hit the ceiling on some attempt, "
             "which returns a truncated answer with no error. Note this makes "
             "the two entries no longer directly comparable on exactly those "
             "questions - the gateway baselines were produced at 1024.",
    ),
    "gpt-oss-120b-local": served_locally(
        "gpt-oss-120b-local", "openai/gpt-oss-120b", max_tokens=4096,
        note="the same weights the gateway serves as gpt-oss-120b. MXFP4 with "
             "4 of 128 experts live per token, so 61 GB and one card despite "
             "the name. Budget matches the gateway entry's, which records a "
             "peak near 2900 tokens.",
    ),
    "qwen3.5-4b-nr-local": served_locally(
        "qwen3.5-4b-nr-local", "Qwen/Qwen3.5-4B", max_tokens=4096,
        note="same family, same thinking switch as the rest of Qwen3.5.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3.5-9b-nr-local": served_locally(
        "qwen3.5-9b-nr-local", "Qwen/Qwen3.5-9B", max_tokens=4096,
        note="the largest of the small Qwen, and the last rung of that ladder.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3.6-27b-nr-local": served_locally(
        "qwen3.6-27b-nr-local", "Qwen/Qwen3.6-27B-FP8", max_tokens=4096,
        note="the size twin of qwen3.8-27b-nr-local: same family, same FP8, "
             "same dense 27B, one generation earlier. The gateway serves it "
             "too but has no run here, so the comparison that matters is "
             "against 3.8 rather than against the gateway.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3.5-122b-a10b-local": served_locally(
        "qwen3.5-122b-a10b-local", "Qwen/Qwen3.5-122B-A10B-FP8", max_tokens=4096,
        note="the most knowledge that fits on our four cards: 122B total, 10B "
             "active. Served tp2 over two cards with two replicas, so it "
             "answers at roughly half the parallelism of the tp1 models - "
             "fine for step 1, which is 308 or 1273 questions, not 48000 "
             "candidates.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "minimax-m2.5-local": served_locally(
        "minimax-m2.5-local", "MiniMaxAI/MiniMax-M2.5", max_tokens=4096,
        note="the only large non-Qwen candidate that runs on this SGLang, "
             "which is the whole reason to spend cards on it: every model "
             "above it in the agreement table is a Qwen, and a Qwen judging "
             "Qwen candidates cannot tell us whether that agreement is "
             "shared knowledge or shared family.",
    ),
    "qwen3.6-35b-a3b-nr-local": served_locally(
        "qwen3.6-35b-a3b-nr-local", "Qwen/Qwen3.6-35B-A3B-FP8", max_tokens=4096,
        note="35 GB on disk but roughly 3B active per token - the lightest "
             "of the large models by compute, lighter than gpt-oss-20b. "
             "Judge it as a cheap MoE, not as a 35B model.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3.5-2b-nr-local": served_locally(
        "qwen3.5-2b-nr-local", "Qwen/Qwen3.5-2B", max_tokens=4096,
        note="same family and the same thinking switch as 0.8B, for the same "
             "measured reason: served as-is the answer lands in "
             "reasoning_content and content comes back empty.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3.5-0.8b-nr-local": served_locally(
        "qwen3.5-0.8b-nr-local", "Qwen/Qwen3.5-0.8B", max_tokens=4096,
        note="the smallest model we can serve, and the first where the smoke "
             "run found something: served as-is it answers entirely inside a "
             "thinking block, so content comes back empty and every reply "
             "looks blank. Measured - 1075 characters in reasoning_content "
             "and nothing in content, the same 1075 in content once thinking "
             "is off. Hence -nr, as for Qwen3.8.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "gpt-oss-20b-local": served_locally(
        "gpt-oss-20b-local", "openai/gpt-oss-20b", max_tokens=4096,
        note="the smaller GPT-OSS; there is no third size. No gateway "
             "counterpart, so it is compared against the other local models "
             "and gemini. Same reasoning parser as the 120B.",
    ),
    "gemma-4-e2b-local": served_locally(
        "gemma-4-e2b-local", "google/gemma-4-E2B-it", max_tokens=4096,
        note="the smallest Gemma 4. No gateway counterpart, so the comparison "
             "is against the other local models and gemini. Budget starts "
             "generous because nothing has measured this model's peak yet - "
             "the smoke run is what settles it.",
    ),
    "qwen3.8-27b-nr-local": served_locally(
        "qwen3.8-27b-nr-local", "Qwen/Qwen3.8-27B-FP8", max_tokens=4096,
        note="the FP8 checkpoint the gateway serves, on our own cards. The "
             "thinking switch is not optional here: the gateway entry is the "
             "-noreasoning one, and with thinking on the model spends the "
             "whole budget reasoning and returns an empty string. Serving it "
             "thinking would compare two different configurations.",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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


def difficult_run(base: str) -> str:
    """Where step 1 over only the difficult list is stored, for a model.

    Step 1 proper covers the whole split and is stored under the model's own
    name. A run over just the difficult questions is the same model at another
    coverage, and one name for both would leave a directory nothing describes.
    """
    return f"{base}-difficult"


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


__all__ = ["BaseModel", "JudgeModel", "BASE_MODELS", "JUDGE_MODELS",
           "difficult_run", "describe"]
