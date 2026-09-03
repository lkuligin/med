"""Agent initialization module using Google ADK and LiteLLM adapter."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from config import InferenceConfig


def create_medqa_agent(config: InferenceConfig | None = None) -> Agent:
    """Create a Google ADK Agent configured with LiteLLM for Vertex AI MAAS."""
    cfg = config or InferenceConfig()
    return Agent(
        name="medqa_evaluator",
        model=LiteLlm(model=cfg.model_name),
        instruction=cfg.system_instruction,
    )


def create_runner(
    agent: Agent,
    session_service: InMemorySessionService | None = None,
    app_name: str = "medqa_inference_app",
) -> Runner:
    """Create an ADK Runner for executing agent invocations."""
    return Runner(
        agent=agent,
        session_service=session_service or InMemorySessionService(),
        app_name=app_name,
    )
