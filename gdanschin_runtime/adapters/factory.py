"""Route llm_monkeys' model construction through the LLM gateway.

Enabled by pointing the hook at this module:

    export MEDQA_MODEL_FACTORY="gdanschin_runtime.adapters.factory:build"

llm_monkeys builds every agent's model in one place (model_factory.build_model),
which consults that variable and calls this in place of the stock LiteLlm.
Unset, the reference behaves exactly as its authors left it, against Vertex AI.

Nothing here subclasses BaseLlm: ADK's LiteLlm already forwards unknown keyword
arguments to litellm, and the gateway speaks litellm's dialects. That keeps
usage metadata, streaming and retries as ADK implements them, rather than as we
would have reimplemented them - and token accounting silently returning zeros is
exactly the failure a hand-written adapter invites.
"""

from __future__ import annotations

from typing import Any

from gdanschin_runtime.gateway import completion_kwargs

# llm_monkeys names models for Vertex AI. Map those onto gateway names so the
# reference's own defaults and --model aliases keep working unchanged.
# Candidate generation goes to the INTERNAL gateway. The rate limit is per
# user, not per model or per backend, so sending step 2 through the external
# gateway spends the same quota the judge needs - and the external gateway is
# not the place for sustained parallel load in the first place. The same Gemma
# is served internally; at temperature 0 both return identical answers and
# token counts.
MODEL_MAP: dict[str, str] = {
    "vertex_ai/google/gemma-4-26b-a4b-it-maas": "gemma-4-26b-internal",
    "gemma-4-26b": "gemma-4-26b-internal",
    "vertex_ai/openai/gpt-oss-20b-maas": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-120b",
    "vertex_ai/gemini-3.8-flash": "gemini-3.8-flash",
    "gemini-3.8-flash": "gemini-3.8-flash",
    "vertex_ai/gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
}


def gateway_name(model_name: str) -> str:
    """Translate an llm_monkeys model name into a gateway one."""
    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]
    # Strip a vertex_ai/ prefix and try again, so a name we have not listed
    # still has a chance rather than being sent verbatim to the wrong place.
    stripped = model_name.removeprefix("vertex_ai/")
    return MODEL_MAP.get(stripped, stripped)


def build(config: Any):
    """Build an ADK model for `config`, served by the gateway."""
    from google.adk.models.lite_llm import LiteLlm

    kwargs = completion_kwargs(gateway_name(config.resolved_model_name))
    return LiteLlm(**kwargs)


__all__ = ["build", "gateway_name", "MODEL_MAP"]
