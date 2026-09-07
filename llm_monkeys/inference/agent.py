"""Agent and runner factory module for candidate-based MedQA inference using ADK."""

from __future__ import annotations

from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import CandidateInferenceConfig, register_litellm_model_pricing
from inference._schemas import MedicalFacts


def _build_agent(
    name: str,
    instruction: str,
    config: CandidateInferenceConfig | None = None,
    output_schema: type[Any] | None = None,
) -> Agent:
    """Helper to initialize an ADK Agent with LiteLLM model adapter."""
    register_litellm_model_pricing()
    cfg = config or CandidateInferenceConfig()
    try:
        import litellm

        litellm.num_retries = cfg.litellm_num_retries
    except Exception:
        pass

    return Agent(
        name=name,
        model=LiteLlm(model=cfg.resolved_model_name),
        instruction=instruction,
        output_schema=output_schema,
        generate_content_config=types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            response_mime_type="application/json" if output_schema else None,
        ),
    )


def create_fact_generation_agent(
    config: CandidateInferenceConfig | None = None,
) -> Agent:
    """Create an ADK Agent configured with structured output schema for atomic medical facts."""
    cfg = config or CandidateInferenceConfig()
    return _build_agent(
        name="medqa_fact_generator",
        instruction=cfg.fact_system_instruction,
        config=cfg,
        output_schema=MedicalFacts,
    )


def create_answer_generation_agent(
    config: CandidateInferenceConfig | None = None,
) -> Agent:
    """Create an ADK Agent configured for clinical reasoning and answer generation from facts."""
    cfg = config or CandidateInferenceConfig()
    return _build_agent(
        name="medqa_answer_generator",
        instruction=cfg.answer_system_instruction,
        config=cfg,
    )


def create_runner(
    agent: Agent,
    session_service: InMemorySessionService | None = None,
    app_name: str = "medqa_candidate_app",
) -> Runner:
    """Create an ADK Runner for executing agent invocations."""
    return Runner(
        agent=agent,
        session_service=session_service or InMemorySessionService(),
        app_name=app_name,
    )


__all__ = [
    "create_answer_generation_agent",
    "create_fact_generation_agent",
    "create_runner",
]
