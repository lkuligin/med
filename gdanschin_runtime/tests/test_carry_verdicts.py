"""Verdicts carried between two copies of a run that hold the same facts.

The facts and the judge's word on them come from the source; the answers, and
so which verdicts count as right, from the target's own candidates.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime.carry_verdicts import carry  # noqa: E402

RUN, JUDGE = "model", "judge"


def question_dir(store: Path) -> Path:
    return store / "med_qa" / "facts-pipeline" / RUN / "question_0"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def candidates(store: Path, answers: str, facts: dict[int, list[str]] | None = None) -> None:
    for i, option in enumerate(answers):
        write(question_dir(store) / f"iteration_{i}.json",
              {"question_id": "0", "ground_truth": "A",
               "candidate": {"candidate_index": i, "facts": (facts or {}).get(i, [f"fact {i}"]),
                             "predicted_option": option, "is_correct": option == "A"}})


def verdicts(store: Path, valid: list[bool]) -> None:
    judge = question_dir(store) / JUDGE
    for i, ok in enumerate(valid):
        write(judge / f"iteration_{i}.json",
              {"question_id": "0", "ground_truth": "A", "judge": JUDGE,
               "verification": {"candidate_index": i, "attempt_number": i + 1,
                                "facts": [f"fact {i}"], "fact_verifications": [],
                                "all_facts_correct": ok, "predicted_option": "B",
                                "is_correct": False, "total_tokens": 10,
                                "total_latency_seconds": 1.0}})
    write(judge / "result.json", {"question_id": "0", "question": "q", "ground_truth": "A",
                                  "candidates_evaluated": len(valid),
                                  "total_latency_seconds": 9.0})


@pytest.fixture
def stores(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    candidates(source, "BBB")
    verdicts(source, [False, True, True])
    candidates(target, "BAC")
    return source, target


def test_the_answers_are_the_targets_own(stores):
    source, target = stores
    assert carry(source, target, RUN, JUDGE) == 1
    judge = question_dir(target) / JUDGE
    assert sorted(p.name for p in judge.iterdir()) == [
        "iteration_0.json", "iteration_1.json", "iteration_2.json", "result.json"]
    assert read(judge / "iteration_1.json")["verification"]["predicted_option"] == "A"
    assert read(judge / "iteration_1.json")["verification"]["is_correct"] is True
    result = read(judge / "result.json")
    assert (result["selected_candidate_index"], result["predicted_option"], result["is_correct"]) == (1, "A", True)
    assert (result["correct_candidates_right_answers"], result["correct_candidates_wrong_answers"]) == (1, 1)
    assert result["first_correct_candidate_pos_all_and_answer"] == 2
    assert result["total_latency_seconds"] == 9.0


def test_first_valid_stops_where_the_authors_rule_stops(stores):
    source, target = stores
    carry(source, target, RUN, JUDGE, first_valid=True)
    judge = question_dir(target) / JUDGE
    assert sorted(p.name for p in judge.glob("iteration_*.json")) == ["iteration_0.json", "iteration_1.json"]
    result = read(judge / "result.json")
    assert (result["candidates_evaluated"], result["num_correct_candidates"]) == (2, 1)
    assert (result["total_tokens"], result["total_latency_seconds"]) == (20, 2.0)
    assert result["is_correct"] is True


def test_with_no_valid_candidate_every_verdict_is_kept(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    candidates(source, "AA")
    verdicts(source, [False, False])
    candidates(target, "AA")
    carry(source, target, RUN, JUDGE, first_valid=True)
    result = read(question_dir(target) / JUDGE / "result.json")
    assert (result["candidates_evaluated"], result["found_valid_candidate"], result["is_correct"]) == (2, False, None)


def test_other_facts_stop_the_carry_before_anything_is_written(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    candidates(source, "BBB")
    verdicts(source, [False, True, True])
    candidates(target, "BAC", facts={1: ["another fact"]})
    with pytest.raises(ValueError):
        carry(source, target, RUN, JUDGE)
    assert not (question_dir(target) / JUDGE).exists()
    assert not (question_dir(target) / f"{JUDGE}.incoming").exists()


def test_a_question_the_target_has_verdicts_for_is_left_alone(stores):
    source, target = stores
    write(question_dir(target) / JUDGE / "result.json", {"question_id": "0", "is_correct": False})
    assert carry(source, target, RUN, JUDGE) == 0
    assert read(question_dir(target) / JUDGE / "result.json") == {"question_id": "0", "is_correct": False}


def test_the_summary_says_where_the_verdicts_came_from(stores):
    source, target = stores
    write(source / "med_qa" / "facts-pipeline" / RUN / f"summary.{JUDGE}.json",
          {"model": "m", "temperature": 1.0, "sampling": {"judge_prompt": "reference"}})
    carry(source, target, RUN, JUDGE, first_valid=True)
    summary = read(target / "med_qa" / "facts-pipeline" / RUN / f"summary.{JUDGE}.json")
    assert summary["carried_from"] == str(source / "med_qa" / "facts-pipeline" / RUN)
    assert summary["carried_verdicts"] == "up to the first valid candidate"
    assert summary["sampling"] == {"judge_prompt": "reference"}
    assert (summary["correct_answers"], summary["total_candidates_evaluated"]) == (1, 2)
