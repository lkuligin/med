"""Unit tests for answer extraction and evaluation logic."""

import pytest

from one_shot.parser import evaluate_prediction, extract_predicted_option


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
