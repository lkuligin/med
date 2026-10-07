"""Cohen's kappa and observed agreement, on cases whose answers are known."""

import pytest

from paper.irr import agreement, cohen_kappa


def test_identical_raters_agree_fully():
    result = agreement([1, 0, 1, 1, 0], [1, 0, 1, 1, 0])
    assert result.observed == 1.0
    assert result.kappa == pytest.approx(1.0)


def test_the_textbook_two_by_two():
    # 50 items: both yes 20, A yes B no 5, A no B yes 10, both no 15.
    # Observed 35/50 = 0.7; chance 0.5*0.6 + 0.5*0.4 = 0.5; kappa 0.4.
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    result = agreement(a, b)
    assert result.observed == pytest.approx(0.7)
    assert result.expected == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.4)


def test_opposite_raters_on_balanced_labels_score_minus_one():
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_labels_need_not_be_binary():
    a = ["A", "B", "C", "A", "B", "C"]
    b = ["A", "B", "C", "A", "C", "B"]
    result = agreement(a, b)
    assert result.observed == pytest.approx(4 / 6)
    # Each label 1/3 for both raters: chance 3 * (1/3)^2 = 1/3.
    assert result.kappa == pytest.approx((4 / 6 - 1 / 3) / (1 - 1 / 3))


def test_rare_disagreement_under_a_dominant_label_gives_a_low_kappa():
    # 98% observed agreement, yet kappa well below it: the paradox to report
    # observed agreement beside kappa for.
    a = [1] * 98 + [0, 1]
    b = [1] * 98 + [1, 0]
    result = agreement(a, b)
    assert result.observed == pytest.approx(0.98)
    assert result.kappa < 0.0


def test_kappa_is_undefined_when_both_raters_never_vary():
    result = agreement([1, 1, 1], [1, 1, 1])
    assert result.observed == 1.0
    assert result.kappa is None


def test_raters_must_label_the_same_items():
    with pytest.raises(ValueError):
        agreement([1, 0], [1])
    with pytest.raises(ValueError):
        agreement([], [])
