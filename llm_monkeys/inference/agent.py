"""Agent and runner factory module for candidate-based MedQA inference using ADK."""

from __future__ import annotations

from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import CandidateInferenceConfig, register_litellm_model_pricing
from inference._schemas import MedicalFacts
from model_factory import build_model


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
        model=build_model(cfg),
        instruction=instruction,
        output_schema=output_schema,
        generate_content_config=types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            response_mime_type="application/json" if output_schema else None,
        ),
    )


# Said in words only when the schema is not there to say it. Left free, a model
# invents its own shape - gpt-oss-20b answered one candidate in three with
# `{"id": 1, "statement": ...}` objects and another with entity-attribute rows,
# neither of which is a sentence a judge can verify.
FREE_FORM_FACT_SHAPE = (
    "\n\nReturn only a JSON object of the form "
    '{"facts": ["...", "..."]}, where each element is one complete '
    "self-contained sentence. Do not wrap the sentences in objects and do not "
    "add any other field."
)


def create_fact_generation_agent(
    config: CandidateInferenceConfig | None = None,
) -> Agent:
    """Create an ADK Agent configured with structured output schema for atomic medical facts."""
    cfg = config or CandidateInferenceConfig()
    structured = getattr(cfg, "structured_facts", True)
    instruction = cfg.resolved_fact_system_instruction
    if not structured:
        instruction += FREE_FORM_FACT_SHAPE
    return _build_agent(
        name="medqa_fact_generator",
        instruction=instruction,
        config=cfg,
        output_schema=MedicalFacts if structured else None,
    )


def create_answer_generation_agent(
    config: CandidateInferenceConfig | None = None,
) -> Agent:
    """Create an ADK Agent configured for clinical reasoning and answer generation from facts."""
    cfg = config or CandidateInferenceConfig()
    return _build_agent(
        name="medqa_answer_generator",
        instruction=cfg.resolved_answer_system_instruction,
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
