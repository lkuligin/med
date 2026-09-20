"""Where a run's results go, and who decides.

The layout is chosen in one place, so these tests cover that place: the
default, the hook that replaces it, and the promise that a workflow does not
care which one it got.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import CandidateInferenceConfig, InferenceConfig, VerifierConfig
from results_store import (
    CandidateResults,
    OneShotResults,
    SingleFileResults,
    SingleFileVerification,
    VerificationResults,
    build_store,
)

PER_RECORD = "results_store:per_record"


@pytest.fixture
def unhooked(monkeypatch):
    """No override, which is what anyone who has opted into nothing gets."""
    monkeypatch.delenv("MEDQA_RESULTS_STORE", raising=False)


def make_mock_event(text: str) -> MagicMock:
    part = MagicMock(text=text, thought=False)
    return MagicMock(content=MagicMock(parts=[part]), usage_metadata=None)


def test_default_is_one_file_per_run(unhooked):
    one_shot = build_store(InferenceConfig(output_filepath="a.json"))
    candidates = build_store(CandidateInferenceConfig(output_filepath="b.json"))
    verdicts = build_store(
        VerifierConfig(output_filepath="c.json", input_filepath="b.json")
    )

    assert isinstance(one_shot, SingleFileResults)
    assert isinstance(candidates, SingleFileResults)
    assert isinstance(verdicts, SingleFileVerification)
    assert str(one_shot) == "a.json"
    # Step 3 reads its candidates from the file step 2 wrote.
    assert str(verdicts.candidates) == "b.json"


def test_the_hook_chooses_a_directory_per_run(monkeypatch):
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)

    one_shot = build_store(InferenceConfig(run_name="r"))
    candidates = build_store(CandidateInferenceConfig(run_name="r"))
    verdicts = build_store(VerifierConfig(run_name="r", judge_name="j"))

    assert isinstance(one_shot, OneShotResults)
    assert isinstance(candidates, CandidateResults)
    assert isinstance(verdicts, VerificationResults)
    assert isinstance(verdicts.candidates, CandidateResults)


@pytest.mark.parametrize(
    "spec",
    [
        "results_store",                      # no callable named
        "no_such_module:per_record",          # module that cannot be imported
        "results_store:no_such_callable",     # name that is not there
        "results_store:DEFAULT_RESULTS_DIR",  # name that is not callable
    ],
)
def test_a_broken_hook_is_refused_rather_than_ignored(monkeypatch, spec):
    """An override that quietly did nothing would write a run somewhere other
    than where it was asked to, and nothing downstream could tell."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", spec)
    with pytest.raises(RuntimeError):
        build_store(InferenceConfig())


def test_one_file_round_trip(tmp_path):
    store = SingleFileResults(tmp_path / "run.json")
    payload = {"summary": {"accuracy": 1.0}, "results": [{"question_id": "7"}]}

    store.save(payload)

    assert store.load() == payload
    assert json.loads((tmp_path / "run.json").read_text()) == payload


def test_an_empty_path_writes_nothing_and_reads_nothing(tmp_path):
    """Passing no path is how a caller says it only wants the summary."""
    store = SingleFileResults("")
    store.save({"results": [{"question_id": "7"}]})
    assert store.load() is None
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_does_not_parse_reads_as_nothing(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{ truncated")
    assert SingleFileResults(path).load() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["one file", "directory per run"])
async def test_a_workflow_reads_back_what_it_wrote_either_way(
    layout, tmp_path, monkeypatch, sample_questions, workflow_factory
):
    """The workflow neither knows nor cares which layout it was given."""
    if layout == "one file":
        monkeypatch.delenv("MEDQA_RESULTS_STORE", raising=False)
        kwargs: dict[str, Any] = {"output_filepath": str(tmp_path / "run.json")}
    else:
        monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
        kwargs = {"results_dir": str(tmp_path / "results"), "run_name": "test-run"}

    async def answer(user_id, session_id, new_message):
        yield make_mock_event("Final Answer: Option A")

    workflow = workflow_factory(
        n_attempts=1, runner=MagicMock(run_async=answer), **kwargs
    )
    summary, _ = await workflow.run(questions=sample_questions)

    stored = workflow.store.load()
    assert summary.completed == 2
    assert {item["question_id"] for item in stored["results"]} == {"0", "1"}
    assert {item["predicted_option"] for item in stored["results"]} == {"A"}

    wrote_a_file = (tmp_path / "run.json").is_file()
    wrote_a_directory = (tmp_path / "results" / "single-step" / "test-run").is_dir()
    assert (wrote_a_file, wrote_a_directory) == (layout == "one file",
                                                 layout != "one file")
