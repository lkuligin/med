"""Unit tests for verifier._dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier._dataset import Step2CandidateData, Step2QuestionData, load_step2_results


def test_step2_candidate_data_from_dict_and_to_dict():
    data = {
        "candidate_index": 2,
        "facts": ["Fact 1", "Fact 2"],
        "predicted_option": "B",
        "is_correct": False,
        "answer_raw_response": "Raw text",
    }
    cand = Step2CandidateData.from_dict(data)
    assert cand.candidate_index == 2
    assert cand.facts == ["Fact 1", "Fact 2"]
    assert cand.predicted_option == "B"
    assert cand.is_correct is False
    assert cand.answer_raw_response == "Raw text"

    as_dict = cand.to_dict()
    assert as_dict["candidate_index"] == 2
    assert as_dict["facts"] == ["Fact 1", "Fact 2"]
    assert as_dict["predicted_option"] == "B"
    assert as_dict["is_correct"] is False
    assert as_dict["answer_raw_response"] == "Raw text"


def test_step2_question_data_from_dict_and_to_dict():
    data = {
        "question_id": "101",
        "question": "What is the diagnosis?",
        "options": {"A": "Asthma", "B": "COPD"},
        "ground_truth": "A",
        "ground_truth_answer": "Asthma",
        "candidates": [
            {
                "candidate_index": 0,
                "facts": ["Reversible airway obstruction indicates asthma."],
                "predicted_option": "A",
                "is_correct": True,
            }
        ],
    }
    q = Step2QuestionData.from_dict(data)
    assert q.question_id == "101"
    assert q.question == "What is the diagnosis?"
    assert q.ground_truth == "A"
    assert len(q.candidates) == 1
    assert q.candidates[0].candidate_index == 0
    assert q.candidates[0].is_correct is True

    as_dict = q.to_dict()
    assert as_dict["question_id"] == "101"
    assert as_dict["ground_truth"] == "A"
    assert len(as_dict["candidates"]) == 1
    assert as_dict["candidates"][0]["candidate_index"] == 0


def test_load_step2_results_wrapped_and_bare(tmp_path: Path):
    sample_data = {
        "summary": {"total_questions": 2},
        "results": [
            {
                "question_id": "0",
                "question": "Q0",
                "options": {"A": "OptA"},
                "ground_truth": "A",
                "candidates": [],
            },
            {
                "question_id": "1",
                "question": "Q1",
                "options": {"B": "OptB"},
                "ground_truth": "B",
                "candidates": [],
            },
        ],
    }
    p_wrapped = tmp_path / "wrapped.json"
    with open(p_wrapped, "w") as f:
        json.dump(sample_data, f)

    loaded = load_step2_results(p_wrapped)
    assert len(loaded) == 2
    assert loaded[0].question_id == "0"
    assert loaded[1].question_id == "1"

    # Test limit and offset
    loaded_slice = load_step2_results(p_wrapped, limit=1, offset=1)
    assert len(loaded_slice) == 1
    assert loaded_slice[0].question_id == "1"

    # Test bare list format
    p_bare = tmp_path / "bare.json"
    with open(p_bare, "w") as f:
        json.dump(sample_data["results"], f)

    loaded_bare = load_step2_results(p_bare)
    assert len(loaded_bare) == 2

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        load_step2_results(tmp_path / "nonexistent.json")

    # Test invalid JSON format: dict without 'results' key
    p_invalid_dict = tmp_path / "invalid_dict.json"
    with open(p_invalid_dict, "w") as f:
        json.dump({"foo": "bar"}, f)
    with pytest.raises(ValueError, match="Expected 'results' key in JSON dict"):
        load_step2_results(p_invalid_dict)

    # Test invalid JSON format: primitive / unexpected type
    p_invalid_type = tmp_path / "invalid_type.json"
    with open(p_invalid_type, "w") as f:
        json.dump(12345, f)
    with pytest.raises(ValueError, match="Unexpected JSON format"):
        load_step2_results(p_invalid_type)
