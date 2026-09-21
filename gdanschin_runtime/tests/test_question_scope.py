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


def answered(correct: int, attempts: int = 1) -> dict:
    return {"correct_attempts": correct, "total_attempts": attempts}


def test_a_padded_id_and_a_bare_one_are_the_same_question(monkeypatch):
    # medbullets writes 001 in its difficult-questions list and in the
    # per-candidate store, while the one-shot store keys that question 1.
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        FakeOneShot({str(i): answered(1) for i in range(1, 10)}))

    rate, covered = inspect_run._one_shot(
        "gemma-4-26b", [f"{i:03d}" for i in range(1, 10)], "medbullets")

    assert covered == 9, "every question of the list has a one-shot answer"
    assert rate == 100.0


def test_a_question_with_no_one_shot_answer_is_left_out(monkeypatch):
    monkeypatch.setattr(inspect_run, "OneShotResults",
                        FakeOneShot({"1": answered(1), "2": answered(0)}))

    rate, covered = inspect_run._one_shot(
        "gemma-4-26b", ["001", "002", "003"], "medbullets")

    assert covered == 2, "003 was never answered, so it counts neither way"
    assert rate == 50.0


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
