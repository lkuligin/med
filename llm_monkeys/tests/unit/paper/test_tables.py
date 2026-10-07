"""The rules behind the paper's numbers, on cases whose answers are known."""

import pytest

from paper.tables import (first_valid_position, mcnemar, paired_fact_verdicts, pass_at_k,
                          resolve, vote)


def test_a_tied_vote_goes_to_the_option_sampled_first():
    assert vote(["B", "A", "A", "B"], "B") == 1
    assert vote(["B", "A", "A", "B"], "A") == 0
    assert vote(["A", "C", "C", None], "C") == 1


def test_a_vote_with_no_answers_is_wrong():
    assert vote([None, None], "A") == 0


def test_pass_at_k_matches_its_closed_form():
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    # 1 - C(7,2)/C(10,2) = 1 - 21/45
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - 21 / 45)
    # fewer wrong samples than k: some sample in any k is right
    assert pass_at_k(10, 8, 3) == 1.0


def test_mcnemar_follows_the_papers_formula():
    a = [0] * 10 + [1] * 2 + [1] * 5
    b = [1] * 10 + [0] * 2 + [1] * 5
    result = mcnemar(a, b)
    assert (result["improved"], result["degraded"]) == (10, 2)
    # Z = 8 / sqrt(12); p = erfc(Z / sqrt 2) / 2
    assert result["p"] == pytest.approx(0.01046, abs=1e-4)
    assert result["direction"] == "better"


def test_mcnemar_hides_the_direction_in_p_and_reports_it_apart():
    result = mcnemar([1] * 10 + [0] * 2, [0] * 10 + [1] * 2)
    assert result["p"] == pytest.approx(0.01046, abs=1e-4)
    assert result["direction"] == "worse"


def test_a_difference_of_one_counts_as_none():
    assert mcnemar([0, 1, 1], [1, 0, 0])["p"] == pytest.approx(0.5)
    assert mcnemar([1, 1], [1, 1])["direction"] == "same"


def test_the_first_valid_candidate_is_counted_from_one():
    verdicts = {0: (False, (0, 1)), 1: (False, (1, 0)), 2: (True, (1, 1)), 3: (True, (1,))}
    assert first_valid_position(verdicts, 50) == 3


def test_no_valid_candidate_costs_all_k():
    assert first_valid_position({0: (False, (0,)), 1: (False, (0,))}, 100) == 100


def test_facts_pair_up_only_where_both_verifiers_read_them():
    a = {0: (True, (1, 1, None)), 1: (False, (1, 0)), 2: (True, (1,))}
    b = {0: (False, (1, 0, 1)), 1: (False, (1,)), 3: (True, (1,))}
    # candidate 0: the third fact is unreadable for A; candidate 1: lengths differ
    assert paired_fact_verdicts(a, b) == ([1, 1], [1, 0])


def test_ids_resolve_as_the_reference_does():
    assert resolve(["7", "007", "12"], {"7", "12"}) == ["7", "7", "12"]
    assert resolve(["1", "2"], {"001", "002"}) == ["001", "002"]
    assert resolve(["0"], {"0"}) == ["0"]
    assert resolve(["999"], {"1"}) == []
