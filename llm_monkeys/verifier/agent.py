"""Agent and runner factory module for fact verification using Google ADK."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import VerifierConfig, register_litellm_model_pricing
from verifier._prompts import DEFAULT_VERIFIER_SYSTEM_INSTRUCTION
from verifier._schemas import FactVerification

# Suppress noisy LiteLLM warnings when running Gemini models via LiteLLM adapter
os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")


def _setup_litellm(retries: int) -> None:
    """Register custom model pricing and configure LiteLLM retries."""
    register_litellm_model_pricing()
    try:
        import litellm

        litellm.num_retries = retries
    except Exception:
        pass


def create_fact_verifier_agent(
    config: VerifierConfig | None = None,
    output_schema: type[Any] | None = FactVerification,
) -> Agent:
    """Create an ADK Agent configured as an LLM-as-a-judge for binary fact verification.

    Args:
        config: Verifier configuration. Defaults to VerifierConfig().
        output_schema: Schema for structured output. Defaults to FactVerification.

    Returns:
        Configured ADK Agent.
    """
    cfg = config or VerifierConfig()
    _setup_litellm(cfg.litellm_num_retries)

    return Agent(
        name="medqa_fact_verifier",
        model=LiteLlm(model=cfg.resolved_model_name),
        instruction=cfg.system_instruction or DEFAULT_VERIFIER_SYSTEM_INSTRUCTION,
        output_schema=output_schema,
        generate_content_config=types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            response_mime_type="application/json" if output_schema else None,
        ),
    )


def create_runner(
    agent: Agent,
    session_service: InMemorySessionService | None = None,
    app_name: str = "medqa_verifier_app",
) -> Runner:
    """Create an ADK Runner for executing fact verification invocations.

    Args:
        agent: ADK Agent to wrap.
        session_service: Session service for multi-turn and session isolation.
        app_name: ADK application identifier.

    Returns:
        ADK Runner instance.
    """
    return Runner(
        agent=agent,
        session_service=session_service or InMemorySessionService(),
        app_name=app_name,
    )


__all__ = [
    "create_fact_verifier_agent",
    "create_runner",
]
