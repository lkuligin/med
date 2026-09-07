"""Unit tests for verifier.agent module."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from config import VerifierConfig
from verifier._schemas import FactVerification
from verifier.agent import create_fact_verifier_agent, create_runner


def test_create_fact_verifier_agent():
    config = VerifierConfig(model_name="gemini-3-flash-preview")
    agent = create_fact_verifier_agent(config)

    assert isinstance(agent, Agent)
    assert agent.name == "medqa_fact_verifier"
    assert agent.output_schema == FactVerification
    assert agent.model.model == "vertex_ai/gemini-3-flash-preview"
    assert agent.generate_content_config is not None
    assert agent.generate_content_config.response_mime_type == "application/json"
    assert agent.generate_content_config.temperature == config.temperature
    assert agent.generate_content_config.max_output_tokens == config.max_tokens


def test_create_fact_verifier_agent_default():
    agent = create_fact_verifier_agent()

    assert isinstance(agent, Agent)
    assert agent.name == "medqa_fact_verifier"
    assert agent.output_schema == FactVerification
    assert agent.model.model == VerifierConfig().resolved_model_name
    assert agent.generate_content_config is not None
    assert agent.generate_content_config.response_mime_type == "application/json"


def test_create_fact_verifier_agent_no_output_schema():
    agent = create_fact_verifier_agent(output_schema=None)

    assert isinstance(agent, Agent)
    assert agent.output_schema is None
    assert agent.generate_content_config is not None
    assert agent.generate_content_config.response_mime_type is None


def test_create_runner():
    config = VerifierConfig()
    agent = create_fact_verifier_agent(config)
    runner = create_runner(agent)

    assert isinstance(runner, Runner)
    assert runner.agent == agent
    assert runner.app_name == "medqa_verifier_app"


def test_create_runner_custom_session_service():
    session_service = InMemorySessionService()
    agent = create_fact_verifier_agent()
    runner = create_runner(
        agent, session_service=session_service, app_name="custom_verifier_app"
    )

    assert runner.agent == agent
    assert runner.session_service == session_service
    assert runner.app_name == "custom_verifier_app"
