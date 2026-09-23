"""That an analysis covers the questions the run works on, and no others.

Two ways of getting this wrong were live at once: reading a run over every
question it holds rather than over the list it was given, and comparing a
padded id with a bare one as strings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime import inspect_run  # noqa: E402


class FakeOneShot:
    """A one-shot store, keyed the way the dataset hands its ids over."""

    def __init__(self, records: dict):
        self.records = records

    def __call__(self, *args, **kwargs):
        return self

    def read(self) -> dict:
        return self.records


def answered(correct: int, attempts: int = 1,
             opening: bool | None = None) -> dict:
    """A one-shot record: the tally, and how the first attempt went."""
    first = correct > 0 if opening is None else opening
    return {"correct_attempts": correct, "total_attempts": attempts,
            "attempts": [{"attempt_index": 0, "is_correct": first}]}


def test_a_padded_id_and_a_bare_one_are_the_same_question(monkeypatch):
    # medbullets writes 001 in its difficult-questions list and in the
    # per-candidate store, while the one-shot store keys that question 1.
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        FakeOneShot({str(i): answered(1) for i in range(1, 10)}))

    measured = inspect_run._one_shot(
        "gemma-4-26b", [f"{i:03d}" for i in range(1, 10)], "medbullets")

    assert measured.covered == 9, "every question of the list has an answer"
    assert measured.attempt0 == 100.0


def test_a_question_with_no_one_shot_answer_is_left_out(monkeypatch):
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        FakeOneShot({"1": answered(1), "2": answered(0)}))

    measured = inspect_run._one_shot(
        "gemma-4-26b", ["001", "002", "003"], "medbullets")

    assert measured.covered == 2, "003 was never answered, so it counts neither way"
    assert measured.averaged == 50.0


def test_the_first_attempt_is_not_the_average_of_the_attempts(monkeypatch):
    """The author's baseline is attempt 0 alone, so it is measured that way
    rather than labelled that way and computed as something else."""
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        FakeOneShot({"1": answered(2, attempts=3, opening=False)}))

    measured = inspect_run._one_shot("gemma-4-26b", ["001"], "medbullets")

    assert measured.attempt0 == 0.0, "the first attempt was wrong"
    assert round(measured.averaged, 1) == 66.7, "two of its three were right"


def test_a_run_is_read_over_the_list_not_over_what_it_holds(monkeypatch):
    """A list can be shortened between runs, and nothing stored is thrown away.

    The questions it drops are the easy ones, so leaving them in flatters every
    number the pipeline reports.
    """
    monkeypatch.setattr(inspect_run, "_difficult_ids",
                        lambda dataset=None: ["001", "003"])
    stored = [{"question_id": q} for q in ("001", "002", "003", "004")]

    kept = inspect_run._on_list(stored, "medbullets")

    assert [q["question_id"] for q in kept] == ["001", "003"]


def test_the_list_is_matched_whatever_way_the_ids_are_written(monkeypatch):
    monkeypatch.setattr(inspect_run, "_difficult_ids",
                        lambda dataset=None: ["1", "3"])
    stored = [{"question_id": q} for q in ("001", "002", "003")]

    kept = inspect_run._on_list(stored, "medbullets")

    assert [q["question_id"] for q in kept] == ["001", "003"]


def test_a_dataset_with_no_list_yet_is_read_whole(monkeypatch):
    monkeypatch.setattr(inspect_run, "_difficult_ids", lambda dataset=None: None)
    stored = [{"question_id": "1"}, {"question_id": "2"}]

    assert inspect_run._on_list(stored, "medbullets") == stored


class FakeCandidates:
    """A per-candidate store, which knows only which questions it holds."""

    def __init__(self, ids: list[str]):
        self.ids = ids

    def questions(self) -> list[str]:
        return self.ids


def test_a_chart_reads_the_questions_the_curve_on_it_was_drawn_from(monkeypatch):
    """The baseline line and the curve went through different paths once, and
    the chart ended up comparing 165 questions against 308."""
    monkeypatch.setattr(inspect_run, "_difficult_ids",
                        lambda dataset=None: ["1", "3"])

    kept = inspect_run._questions_on_list(
        FakeCandidates(["001", "002", "003"]), "medbullets")

    assert kept == ["001", "003"]


def test_with_no_list_a_chart_reads_the_whole_run(monkeypatch):
    monkeypatch.setattr(inspect_run, "_difficult_ids", lambda dataset=None: None)

    store = FakeCandidates(["1", "2"])

    assert inspect_run._questions_on_list(store, "med_qa") == ["1", "2"]


def test_step_1_is_found_under_the_difficult_run_beside_it(monkeypatch):
    """A step 2 run is named for the model, and so is the whole-split step 1
    run; a step 1 run over only the difficult list is named
    <model>-difficult, because one name for two coverages leaves a directory
    nothing can describe. Every reader of a step 2 run needs a step 1 next
    door, and finding it is not the reader's job."""
    stored = {"qwen-local-difficult": {"1": answered(1)}}
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        lambda root, run, dataset: FakeOneShot(
                            stored.get(run, {})))

    assert inspect_run._step1_run("qwen-local", "medbullets") == "qwen-local-difficult"


def test_a_baseline_without_a_difficult_run_falls_back_to_the_split(monkeypatch):
    """The split answers the difficult questions too, so it is a baseline."""
    stored = {"gemma": {"1": answered(1)}}
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        lambda root, run, dataset: FakeOneShot(
                            stored.get(run, {})))

    assert inspect_run._step1_run("gemma", "medbullets") == "gemma"


def test_a_model_with_no_step_1_anywhere_resolves_to_nothing(monkeypatch):
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        lambda root, run, dataset: FakeOneShot({}))

    assert inspect_run._step1_run("gemma", "medbullets") is None


def test_the_whole_split_block_asks_for_the_split(monkeypatch):
    """The same model can have both runs, and the two callers want different
    ones: a pipeline baseline wants the difficult-list run beside it, while
    the block headed WHOLE SPLIT wants the split. Reading the first there
    printed "WHOLE SPLIT, 483 questions" for a split of 1273."""
    stored = {"gemma-difficult": {"1": answered(1)}, "gemma": {"2": answered(1)}}
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        lambda root, run, dataset: FakeOneShot(
                            stored.get(run, {})))

    assert inspect_run._step1_run("gemma", "medbullets") == "gemma-difficult"
    assert inspect_run._step1_run("gemma", "medbullets",
                                  prefer_full=True) == "gemma"


def test_asking_for_the_split_falls_back_when_there_is_none(monkeypatch):
    stored = {"gemma-difficult": {"1": answered(1)}}
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        lambda root, run, dataset: FakeOneShot(
                            stored.get(run, {})))

    assert inspect_run._step1_run("gemma", "medbullets",
                                  prefer_full=True) == "gemma-difficult"


def test_the_x_axis_stays_readable_however_far_k_goes():
    """Every other k is fine to twenty and a smear at a hundred, where fifty
    labels overlap. Whatever the range, the axis should carry about a dozen
    ticks on round numbers."""
    for count in (10, 20, 50, 100, 200):
        stride = inspect_run._tick_stride(count)
        assert count / stride <= 14, count
        assert stride in (1, 2, 5, 10, 20, 25, 50, 100), stride


def test_a_medbullets_run_gets_its_split_and_not_the_default():
    """Medbullets has no config and its split is op5_test. Naming only the
    dataset leaves the default "test", and the run stops on Unknown split
    before it reaches a model - which is how this was found."""
    from gdanschin_runtime.run_step1 import _dataset_kwargs

    medbullets = _dataset_kwargs("medbullets")
    assert medbullets["dataset_split"] == "op5_test"
    assert medbullets["dataset_config"] is None
    # MedQA keeps the defaults it has always had.
    assert "dataset_split" not in _dataset_kwargs("med_qa")
    assert _dataset_kwargs(None) == {}
