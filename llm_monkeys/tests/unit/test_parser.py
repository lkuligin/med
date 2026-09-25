"""Unit tests for answer extraction and evaluation logic."""

import pytest

from one_shot.parser import (
    evaluate_prediction,
    extract_predicted_option,
    rescore_results,
)


@pytest.mark.parametrize(
    "response_text,valid_options,expected",
    [
        ("A", ["A", "B", "C", "D"], "A"),
        ("A.", ["A", "B", "C", "D"], "A"),
        ("(B)", ["A", "B", "C", "D"], "B"),
        ("Answer: C", ["A", "B", "C", "D"], "C"),
        ("Answer: (C)", ["A", "B", "C", "D"], "C"),
        ("**Answer:** D. Nitrofurantoin is safe", ["A", "B", "C", "D", "E"], "D"),
        ("The correct answer is E.", ["A", "B", "C", "D", "E"], "E"),
        ("The best option is (B).", ["A", "B", "C", "D"], "B"),
        ("Final Answer: A", ["A", "B", "C", "D"], "A"),
        ("Option C is the most appropriate next step.", ["A", "B", "C", "D"], "C"),
        (
            "C\n\n**Explanation:**\nThe patient presents with several classic findings...",
            ["A", "B", "C", "D", "E"],
            "C",
        ),
        (
            "Answer: B\n\nExplanation: This scenario involves a medical error...",
            ["A", "B", "C", "D", "E"],
            "B",
        ),
        (
            "Cholesterol embolization is the correct answer because...",
            {
                "A": "Renal papillary necrosis",
                "B": "Allergic interstitial nephritis",
                "C": "Cholesterol embolization",
            },
            "C",
        ),
        (
            "<think>\nLet us analyze Option A and Option B.\nOption A is incorrect.\nOption B is correct.\n</think>\nFinal Answer: Option B",
            ["A", "B", "C", "D"],
            "B",
        ),
        (
            "<think>\nConsider A vs C.\n</think>\n### Answer\nOption C",
            ["A", "B", "C", "D"],
            "C",
        ),
        (
            "### Final Answer: D",
            ["A", "B", "C", "D"],
            "D",
        ),
        (
            "The answer is Option A",
            ["A", "B", "C", "D"],
            "A",
        ),
        ("Reasoning.\n\nFINAL ANSWER: [Option B]", ["A", "B", "C", "D"], "B"),
        ("**Option D** is wrong.\n\nFINAL ANSWER: [Option B]", ["A", "B", "C", "D"], "B"),
        ("FINAL ANSWER: **[Option C]**", ["A", "B", "C", "D"], "C"),
        ("FINAL ANSWER: [C]", ["A", "B", "C", "D"], "C"),
        ("FINAL ANSWER: **C**", ["A", "B", "C", "D"], "C"),
        ("FINAL ANSWER: Based on the above, the answer is D", ["A", "B", "C", "D"], "D"),
        ("", ["A", "B", "C"], None),
        ("I cannot answer this question.", ["A", "B", "C"], None),
    ],
)
def test_extract_predicted_option(response_text, valid_options, expected):
    result = extract_predicted_option(response_text, valid_options)
    assert result == expected


def test_evaluate_prediction():
    assert evaluate_prediction("A", "A") is True
    assert evaluate_prediction("a", "A") is True
    assert evaluate_prediction("B", "A") is False
    assert evaluate_prediction(None, "A") is False
    assert evaluate_prediction("A", "") is False


def test_rescore_results_reparses_raw_responses():
    items = [
        {
            "question_id": "1",
            "ground_truth": "B",
            "attempts": [
                {"raw_response": "FINAL ANSWER: [Option B]",
                 "predicted_option": "D", "is_correct": False},
                {"raw_response": "", "predicted_option": "B", "is_correct": True},
            ],
        }
    ]
    rescored = rescore_results(items)
    assert rescored[0]["attempts"][0]["predicted_option"] == "B"
    assert rescored[0]["attempts"][0]["is_correct"] is True
    # No raw response: the stored values stand.
    assert rescored[0]["attempts"][1]["is_correct"] is True
    # The input is left as it was.
    assert items[0]["attempts"][0]["predicted_option"] == "D"


def test_rescore_results_recomputes_question_level_fields():
    items = [
        {
            "question_id": "1",
            "ground_truth": "B",
            "predicted_option": "D",
            "is_correct": False,
            "is_all_correct": False,
            "correct_attempts": 1,
            "attempts": [
                {"raw_response": "FINAL ANSWER: [Option B]", "is_correct": False},
                {"raw_response": "FINAL ANSWER: B", "is_correct": True},
            ],
        }
    ]
    item = rescore_results(items)[0]
    assert item["predicted_option"] == "B"
    assert item["correct_attempts"] == 2
    assert item["is_all_correct"] is True
    assert item["is_correct"] is True
