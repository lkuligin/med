"""Unit tests for inference.agent module."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.runners import Runner

from config import CandidateInferenceConfig
from inference._schemas import MedicalFacts
from inference.agent import (
    create_answer_generation_agent,
    create_fact_generation_agent,
    create_runner,
)


def test_create_fact_generation_agent():
    config = CandidateInferenceConfig(
        model_name="vertex_ai/google/gemma-4-26b-a4b-it-maas",
        temperature=0.7,
        max_tokens=512,
    )
    agent = create_fact_generation_agent(config)
    assert isinstance(agent, Agent)
    assert agent.name == "medqa_fact_generator"
    assert agent.output_schema == MedicalFacts
    assert agent.generate_content_config is not None
    assert agent.generate_content_config.temperature == 0.7
    assert agent.generate_content_config.max_output_tokens == 512
    assert agent.generate_content_config.response_mime_type == "application/json"


def test_create_answer_generation_agent():
    config = CandidateInferenceConfig(
        model_name="vertex_ai/google/gemma-4-26b-a4b-it-maas",
        temperature=0.5,
        max_tokens=2048,
    )
    agent = create_answer_generation_agent(config)
    assert isinstance(agent, Agent)
    assert agent.name == "medqa_answer_generator"
    assert agent.output_schema is None
    assert agent.generate_content_config is not None
    assert agent.generate_content_config.temperature == 0.5
    assert agent.generate_content_config.max_output_tokens == 2048
    assert agent.generate_content_config.response_mime_type is None


def test_create_runner():
    config = CandidateInferenceConfig()
    agent = create_fact_generation_agent(config)
    runner = create_runner(agent)
    assert isinstance(runner, Runner)
    assert runner.agent == agent


def test_fact_agent_uses_reference_instruction_by_default():
    from inference._prompts import DEFAULT_FACT_SYSTEM_INSTRUCTION

    agent = create_fact_generation_agent(CandidateInferenceConfig())
    assert agent.instruction == DEFAULT_FACT_SYSTEM_INSTRUCTION


def test_fact_agent_uses_the_chosen_prompt():
    from inference._prompts import FACT_PROMPTS

    agent = create_fact_generation_agent(CandidateInferenceConfig(fact_prompt="background"))
    assert agent.instruction == FACT_PROMPTS["background"].system_instruction


def test_explicit_fact_instruction_overrides_the_prompt():
    config = CandidateInferenceConfig(fact_prompt="background", fact_system_instruction="custom")
    assert create_fact_generation_agent(config).instruction == "custom"
