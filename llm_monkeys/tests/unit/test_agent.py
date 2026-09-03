"""Unit tests for agent and runner creation."""

from google.adk.agents import Agent
from google.adk.runners import Runner
from one_shot.agent import create_medqa_agent, create_runner
from config import InferenceConfig


def test_create_medqa_agent():
    config = InferenceConfig(model_name="vertex_ai/google/gemma-4-26b-a4b-it-maas")
    agent = create_medqa_agent(config)
    assert isinstance(agent, Agent)
    assert agent.name == "medqa_evaluator"
    assert "medical licensing board examination" in agent.instruction


def test_create_medqa_agent_default():
    agent = create_medqa_agent()
    assert isinstance(agent, Agent)
    assert agent.name == "medqa_evaluator"


def test_create_runner():
    config = InferenceConfig()
    agent = create_medqa_agent(config)
    runner = create_runner(agent)
    assert isinstance(runner, Runner)
    assert runner.agent == agent
