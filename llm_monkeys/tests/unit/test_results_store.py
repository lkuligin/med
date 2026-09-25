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

import results_store
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


def test_the_dataset_is_the_first_level(monkeypatch, tmp_path):
    """Two datasets number their questions from zero, so they cannot share."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    medqa = build_store(CandidateInferenceConfig(run_name="r", results_dir=str(tmp_path)))
    bullets = build_store(
        CandidateInferenceConfig(
            run_name="r", results_dir=str(tmp_path), dataset_name="mkieffer/Medbullets"
        )
    )
    assert medqa.directory == tmp_path / "med_qa" / "facts-pipeline" / "r"
    assert bullets.directory == tmp_path / "medbullets" / "facts-pipeline" / "r"


def test_step_3_follows_the_run_it_verifies(monkeypatch, tmp_path):
    """The verifier loads no dataset, so its own dataset_name says nothing
    about the run it was pointed at; the run's own place does."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    (tmp_path / "medbullets" / "facts-pipeline" / "r").mkdir(parents=True)

    store = build_store(
        VerifierConfig(run_name="r", judge_name="j", results_dir=str(tmp_path))
    )

    assert store.directory == tmp_path / "medbullets" / "facts-pipeline" / "r"
    assert store.candidates.directory == store.directory


def test_a_named_dataset_wins_when_several_hold_the_run(monkeypatch, tmp_path):
    """Saying which dataset to verify is enough, even when the name is reused."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    for dataset in ("med_qa", "medbullets"):
        (tmp_path / dataset / "facts-pipeline" / "r").mkdir(parents=True)

    store = build_store(
        VerifierConfig(run_name="r", judge_name="j", results_dir=str(tmp_path),
                       dataset_name="mkieffer/Medbullets")
    )

    assert store.directory == tmp_path / "medbullets" / "facts-pipeline" / "r"


def test_a_run_under_datasets_that_were_not_named_is_refused(monkeypatch, tmp_path):
    """Two datasets hold it, neither is the one asked for: guessing which the
    verdicts belong to is exactly what must not happen quietly."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    for dataset in ("medbullets", "pubmedqa"):
        (tmp_path / dataset / "facts-pipeline" / "r").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="none of those is"):
        build_store(VerifierConfig(run_name="r", judge_name="j",
                                   results_dir=str(tmp_path)))


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


def test_two_datasets_keep_their_own_records(monkeypatch, tmp_path):
    """Question 0 of MedQA and question 0 of MedBullets are different
    questions. One model, one run name, two datasets: neither may land on the
    other, and each has to read back exactly what it wrote."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    answers = {"bigbio/med_qa": "A", "mkieffer/Medbullets": "B"}

    stores = {}
    for dataset, answer in answers.items():
        stores[dataset] = build_store(
            InferenceConfig(run_name="gemma", results_dir=str(tmp_path),
                            dataset_name=dataset)
        )
        stores[dataset].save({
            "summary": {"dataset": dataset},
            "results": [{"question_id": "0", "predicted_option": answer},
                        {"question_id": "1", "predicted_option": answer}],
        })

    for dataset, answer in answers.items():
        records = stores[dataset].read()
        assert sorted(records) == ["0", "1"]
        assert {r["predicted_option"] for r in records.values()} == {answer}
        # Not equality: the summary also carries the sampling settings now.
        # What this test is about is that each dataset kept its own.
        assert stores[dataset].read_summary()["dataset"] == dataset


def test_two_models_keep_their_own_records(monkeypatch, tmp_path):
    """One dataset, two models: the run name is what keeps them apart, and a
    run stored under the wrong name is a run answered by the wrong model."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    answers = {"gemma-4-26b": "A", "gpt-oss-120b": "B"}

    for run, answer in answers.items():
        build_store(
            InferenceConfig(run_name=run, results_dir=str(tmp_path))
        ).save({
            "summary": {"model": run},
            "results": [{"question_id": "0", "predicted_option": answer}],
        })

    for run, answer in answers.items():
        store = build_store(InferenceConfig(run_name=run, results_dir=str(tmp_path)))
        assert store.directory == tmp_path / "med_qa" / "single-step" / run
        assert store.read()["0"]["predicted_option"] == answer
        assert store.read_summary()["model"] == run


def test_two_datasets_keep_their_own_candidates_and_verdicts(monkeypatch, tmp_path):
    """The same, for the two steps that keep a directory per question."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    for dataset, fact in (("bigbio/med_qa", "medqa fact"),
                          ("mkieffer/Medbullets", "medbullets fact")):
        candidates = build_store(
            CandidateInferenceConfig(run_name="gemma", results_dir=str(tmp_path),
                                     dataset_name=dataset)
        )
        candidates.save({
            "summary": {"n_candidates": 1},
            "results": [{"question_id": "0", "question": dataset,
                         "candidates": [{"candidate_index": 0, "facts": [fact]}]}],
        })
        verdicts = build_store(
            VerifierConfig(run_name="gemma", judge_name="judge",
                           results_dir=str(tmp_path), dataset_name=dataset)
        )
        verdicts.save({
            "summary": {},
            "results": [{"question_id": "0", "found_valid_candidate": True,
                         "candidate_verifications": [
                             {"candidate_index": 0, "fact": fact}]}],
        })

    for dataset, fact in (("bigbio/med_qa", "medqa fact"),
                          ("mkieffer/Medbullets", "medbullets fact")):
        candidates = build_store(
            CandidateInferenceConfig(run_name="gemma", results_dir=str(tmp_path),
                                     dataset_name=dataset)
        )
        stored = candidates.load()["results"]
        assert len(stored) == 1
        assert stored[0]["question"] == dataset
        assert stored[0]["candidates"][0]["facts"] == [fact]

        verdicts = build_store(
            VerifierConfig(run_name="gemma", judge_name="judge",
                           results_dir=str(tmp_path), dataset_name=dataset)
        )
        judged = verdicts.load()["results"]
        assert len(judged) == 1
        assert judged[0]["candidate_verifications"][0]["fact"] == fact


@pytest.mark.asyncio
async def test_a_run_does_not_resume_from_another_dataset(
    tmp_path, monkeypatch, sample_questions, workflow_factory
):
    """Resuming reads what is stored for this run - and a run of the same name
    on another dataset answers other questions, so it must not count."""
    monkeypatch.setenv("MEDQA_RESULTS_STORE", PER_RECORD)
    already = build_store(
        InferenceConfig(run_name="test-run", results_dir=str(tmp_path / "results"),
                        dataset_name="bigbio/med_qa")
    )
    already.save({
        "summary": None,
        "results": [{"question_id": q.question_id, "predicted_option": "A",
                     "is_correct": True, "correct_attempts": 1,
                     "total_attempts": 1, "attempts": [{"attempt_index": 0,
                     "predicted_option": "A", "is_correct": True}]}
                    for q in sample_questions],
    })

    asked = 0

    async def answer(user_id, session_id, new_message):
        nonlocal asked
        asked += 1
        yield make_mock_event("Final Answer: Option C")

    workflow = workflow_factory(
        n_attempts=1,
        results_dir=str(tmp_path / "results"),
        run_name="test-run",
        dataset_name="mkieffer/Medbullets",
        runner=MagicMock(run_async=answer),
    )
    await workflow.run(questions=sample_questions)

    assert asked == len(sample_questions), "the MedQA answers were taken for these"
    assert {r["predicted_option"] for r in workflow.store.read().values()} == {"C"}
    # and the questions it did not answer are where they were
    assert {r["predicted_option"] for r in already.read().values()} == {"A"}


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
    # The dataset comes first: a question id means nothing without it.
    wrote_a_directory = (
        tmp_path / "results" / "med_qa" / "single-step" / "test-run"
    ).is_dir()
    assert (wrote_a_file, wrote_a_directory) == (layout == "one file",
                                                 layout != "one file")


# --- what the run sampled with -------------------------------------------

def test_the_summary_records_what_the_run_sampled_with(tmp_path):
    """Two runs of one model differ by temperature and token budget more than
    by anything else. Unrecorded, a directory can only be compared with
    another by trusting its name."""
    store = results_store.OneShotResults(
        tmp_path, "a-run", "med_qa",
        sampling={"temperature": 0.8, "max_tokens": 4096, "n_attempts": 3},
    )
    store.save({"results": [], "summary": {"accuracy": 0.7}})

    written = json.loads(store.summary_path.read_text())
    assert written["accuracy"] == 0.7
    assert written["sampling"]["temperature"] == 0.8
    assert written["sampling"]["max_tokens"] == 4096


def test_a_store_without_sampling_writes_the_summary_unchanged(tmp_path):
    store = results_store.OneShotResults(tmp_path, "a-run", "med_qa")
    store.save({"results": [], "summary": {"accuracy": 0.7}})
    assert json.loads(store.summary_path.read_text()) == {"accuracy": 0.7}


def test_a_mid_run_save_does_not_erase_the_summary(tmp_path):
    # Unchanged behaviour, guarded because the merge above rewrote this path.
    store = results_store.OneShotResults(
        tmp_path, "a-run", "med_qa", sampling={"temperature": 0.8})
    store.save({"results": [], "summary": {"accuracy": 0.7}})
    store.save({"results": [], "summary": None})
    assert json.loads(store.summary_path.read_text())["accuracy"] == 0.7


def test_sampling_leaves_out_what_a_config_does_not_have(tmp_path):
    """A judge has a temperature and no n_candidates; a generator the other
    way round. Absent fields must not appear as nulls that read like a
    setting of None."""
    class Judgeish:
        resolved_model_name = "gemini-3.8-flash"
        temperature = 1.0
        max_tokens = 512

    found = results_store.sampling_of(Judgeish())
    assert found == {"model": "gemini-3.8-flash", "temperature": 1.0,
                     "max_tokens": 512}


def test_a_bare_list_is_still_readable():
    """Upstream's loaders documented accepting a bare list as well as the
    wrapped payload. Reading only the wrapped one makes step 3 a TypeError and
    every resume a silent full re-run, so the leniency has to survive."""
    from results_store import stored_results

    wrapped = {"summary": {}, "results": [{"question_id": "1"}]}
    bare = [{"question_id": "1"}]

    assert stored_results(wrapped) == [{"question_id": "1"}]
    assert stored_results(bare) == [{"question_id": "1"}]
    assert stored_results(None) == []
    assert stored_results({"summary": {}}) == []
    assert stored_results([{"question_id": "1"}, "junk"]) == [{"question_id": "1"}]


def test_one_shot_records_keep_the_id_the_dataset_spelled(tmp_path):
    """A padded id survives being read back.

    The key used to be parsed out of the file name with int(), so medbullets
    question 001 came back as "1" while the record inside, the candidate store,
    the difficult list and the dataset all said "001". Every join between them
    then missed, and silently: the questions below 100 simply were not counted.
    """
    store = OneShotResults(tmp_path, "run", "medbullets")
    store.save({
        "summary": None,
        "results": [
            {"question_id": "001", "is_correct": True},
            {"question_id": "010", "is_correct": False},
            {"question_id": "100", "is_correct": True},
        ],
    })

    assert store.question_path("001").is_file(), "the file is named as the id is"
    assert sorted(store.read()) == ["001", "010", "100"]
    assert store.read("001")["is_correct"] is True
    assert [r["question_id"] for r in store.load()["results"]] == ["001", "010", "100"]


def test_one_shot_keeps_medqa_ids_unpadded(tmp_path):
    """The other spelling is left alone: MedQA says 7, not 007."""
    store = OneShotResults(tmp_path, "run", "med_qa")
    store.save({"summary": None,
                "results": [{"question_id": "7"}, {"question_id": "0"}]})

    assert sorted(store.read()) == ["0", "7"]
