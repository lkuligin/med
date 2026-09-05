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
        model_name="vertex_ai/google/gemma-4-26b-a4b-it-maas"
    )
    agent = create_fact_generation_agent(config)
    assert isinstance(agent, Agent)
    assert agent.name == "medqa_fact_generator"
    assert agent.output_schema == MedicalFacts


def test_create_answer_generation_agent():
    config = CandidateInferenceConfig(
        model_name="vertex_ai/google/gemma-4-26b-a4b-it-maas"
    )
    agent = create_answer_generation_agent(config)
    assert isinstance(agent, Agent)
    assert agent.name == "medqa_answer_generator"
    assert agent.output_schema is None


def test_create_runner():
    config = CandidateInferenceConfig()
    agent = create_fact_generation_agent(config)
    runner = create_runner(agent)
    assert isinstance(runner, Runner)
    assert runner.agent == agent
