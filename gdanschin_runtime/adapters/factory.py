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

from gdanschin_runtime import _bootstrap
from gdanschin_runtime.gateway import completion_kwargs
from gdanschin_runtime.models import BASE_MODELS

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


# Our gateway names are registered as canonical when the runtime is set up;
# doing it again here covers llm_monkeys loading this factory on its own.
_bootstrap.register_gateway_names()


def gateway_name(model_name: str) -> str:
    """Translate an llm_monkeys model name into a gateway one."""
    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]
    # Strip a vertex_ai/ prefix and try again, so a name we have not listed
    # still has a chance rather than being sent verbatim to the wrong place.
    stripped = model_name.removeprefix("vertex_ai/")
    return MODEL_MAP.get(stripped, stripped)


# Models we serve ourselves, keyed by the name their endpoint answers to.
# Checked before the gateway, because those names contain a slash and would
# otherwise be read as "provider/model" and routed somewhere wrong.
LOCAL_MODELS = {m.gateway_model: m for m in BASE_MODELS.values() if m.base_url}


def build(config: Any):
    """Build an ADK model for `config`, served by the gateway or by our GPUs."""
    from google.adk.models.lite_llm import LiteLlm

    # The run name first, the served name second. Two entries can share one
    # served name - the same weights with thinking on and off are two entries
    # by design - and keying only on the served name lets whichever was
    # written last silently win. That is not a crash: the run completes,
    # under the name that was asked for, measuring the other configuration.
    name = config.resolved_model_name
    run = getattr(config, "run_name", None)
    local = (BASE_MODELS.get(run) if run else None)
    if local is not None and not local.base_url:
        local = None
    if local is None:
        local = LOCAL_MODELS.get(name) or LOCAL_MODELS.get(
            name.removeprefix("vertex_ai/"))
    if local is not None:
        # SGLang speaks the OpenAI dialect, so litellm needs the openai prefix
        # and some key; the server does not check it. No token here on purpose:
        # this endpoint is on loopback and there is nothing to authenticate to.
        return LiteLlm(model=f"openai/{local.gateway_model}",
                       api_base=local.base_url, api_key="local", **local.extra)

    kwargs = completion_kwargs(gateway_name(name))
    return LiteLlm(**kwargs)


__all__ = ["build", "gateway_name", "MODEL_MAP", "LOCAL_MODELS"]
