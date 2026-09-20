"""Unit tests for verifier._dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from results_store import CandidateResults
from verifier._dataset import (
    Step2CandidateData,
    Step2QuestionData,
    load_step2_results,
)


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


def test_load_step2_results_from_store(tmp_path: Path):
    store = CandidateResults(tmp_path, "test-run")
    store.save(
        {
            "summary": {"total_questions": 2},
            "results": [
                {
                    "question_id": "0",
                    "question": "Q0",
                    "options": {"A": "OptA"},
                    "ground_truth": "A",
                    "candidates": [{"candidate_index": 0, "facts": ["Fact 0"]}],
                },
                {
                    "question_id": "1",
                    "question": "Q1",
                    "options": {"B": "OptB"},
                    "ground_truth": "B",
                    "candidates": [{"candidate_index": 0, "facts": ["Fact 1"]}],
                },
            ],
        }
    )

    loaded = load_step2_results(store)
    assert len(loaded) == 2
    assert loaded[0].question_id == "0"
    assert loaded[1].question_id == "1"

    # Limit and offset
    loaded_slice = load_step2_results(store, limit=1, offset=1)
    assert len(loaded_slice) == 1
    assert loaded_slice[0].question_id == "1"

    # A run with nothing stored
    with pytest.raises(FileNotFoundError):
        load_step2_results(CandidateResults(tmp_path, "no-such-run"))


