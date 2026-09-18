"""Single construction point for the LLM backing every ADK agent.

By default this returns the stock LiteLlm adapter, so behaviour is identical to
the original implementation. Setting the MEDQA_MODEL_FACTORY environment
variable to an import spec ``"module.path:callable"`` redirects construction to
that callable, which receives the active configuration and returns an ADK
``BaseLlm``. This lets an external runtime supply its own model client without
provider-specific code leaking into this package.

Note that the factory is keyed on the configuration object, so a factory can
distinguish workloads via ``isinstance`` (e.g. keep VerifierConfig on Vertex AI
while routing candidate inference elsewhere). It cannot distinguish the fact
agent from the answer agent, since both share one CandidateInferenceConfig;
for per-agent control, inject prebuilt agents into the workflow instead.
"""

from __future__ import annotations

from typing import Any

from google.adk.models.lite_llm import LiteLlm

from config import BaseInferenceConfig, resolve_model_factory


def build_model(config: BaseInferenceConfig) -> Any:
    """Build the model for an agent, honouring the MEDQA_MODEL_FACTORY override."""
    factory = resolve_model_factory()
    if factory is not None:
        return factory(config)
    return LiteLlm(model=config.resolved_model_name)


__all__ = ["build_model"]
