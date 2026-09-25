"""That every step the plan starts is told which model, judge and dataset."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime import run_plan  # noqa: E402
from gdanschin_runtime.models import BASE_MODELS, JUDGE_MODELS  # noqa: E402


def flags(command: list[str]) -> dict[str, str]:
    """The --name value pairs of a command, for asking what it carries."""
    return {name: value for name, value in zip(command, command[1:])
            if name.startswith("--")}


@pytest.mark.parametrize("base", ["gemma-4-26b", "gpt-oss-120b", "qwen3.8-27b-nr"])
def test_generation_is_told_the_model_not_only_the_run_name(base):
    command = run_plan.Step("stage", "candidates", base, "candidates", 20).command()
    carried = flags(command)

    assert carried["--model"] == BASE_MODELS[base].gateway_model
    assert carried["--run-name"] == base
    assert carried["--max-tokens"] == str(BASE_MODELS[base].max_tokens)
    assert carried["--n-candidates"] == "20"


@pytest.mark.parametrize("base", ["gemma-4-26b", "qwen3.8-27b-nr"])
def test_step_1_is_told_the_model(base):
    carried = flags(run_plan.Step("stage", "single-step", base, "one-shot").command())

    assert carried["--base"] == base
    assert carried["--max-tokens"] == str(BASE_MODELS[base].max_tokens)


def test_the_judge_is_told_its_settings_not_just_its_name():
    """--judge-name names a directory. Without the rest, a judge named for one
    setting runs at another, and the verdicts say nothing about which."""
    carried = flags(run_plan.Step("stage", "verdicts", "gemma-4-26b", "verdicts").command())
    judge = JUDGE_MODELS[run_plan.JUDGE]

    assert carried["--judge-name"] == run_plan.JUDGE
    assert carried["--model"] == judge.gateway_model
    assert carried["--temperature"] == str(judge.temperature)
    assert carried["--max-tokens"] == str(judge.max_tokens)


def test_every_step_names_the_dataset():
    for kind in ("one-shot", "candidates", "verdicts"):
        command = run_plan.Step("stage", kind, "gemma-4-26b", kind, 20).command()
        assert flags(command)["--dataset"] == run_plan.DATASET_NAME


def test_the_temperature_override_reaches_the_command(monkeypatch):
    monkeypatch.setattr(run_plan, "JUDGE_TEMPERATURE", "0.0")
    carried = flags(run_plan.Step("stage", "verdicts", "gemma-4-26b", "verdicts").command())
    assert carried["--temperature"] == "0.0"


def test_step_1_over_the_split_takes_the_model_name_and_the_list_run_does_not():
    """The split is step 1 proper; the difficult-list run is the aside."""
    full = run_plan.Step("stage", "single-step full", "gemma-4-26b", "one-shot-full")
    listed = run_plan.Step("stage", "single-step", "gemma-4-26b", "one-shot")

    assert full.run_name == "gemma-4-26b"
    assert listed.run_name == "gemma-4-26b-difficult"
    assert flags(full.command())["--run-name"] == "gemma-4-26b"
