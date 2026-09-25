"""That a question id means the same question wherever it is written down.

MedQA spells its ids 7, medbullets spells the same position 007, and the two
have to meet: the stores and the dataset use the dataset's spelling, while a
difficult-questions list is written by hand and may use either. Comparing them
as plain strings is what dropped every medbullets question below 100 - two
thirds of the list - from a join, leaving a plausible number behind.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime.question_ids import resolve, spellings  # noqa: E402


def test_the_spellings_are_the_ones_the_reference_tries():
    assert spellings("7") == ("7", "7", "007")
    assert spellings("007") == ("007", "7", "007")
    assert spellings(7) == ("7", "7", "007")


def test_zero_is_not_swallowed_by_stripping():
    # "0".lstrip("0") is "", which would match nothing and drop question 0 of
    # MedQA - the first question of the split.
    assert "0" in spellings("0")
    assert resolve(["0"], {"0"}) == {"0"}
    assert resolve(["000"], {"0"}) == {"0"}


def test_a_padded_list_finds_bare_ids_and_the_other_way_round():
    assert resolve(["001", "002"], {"1", "2", "3"}) == {"1", "2"}
    assert resolve(["1", "2"], {"001", "002", "003"}) == {"001", "002"}


def test_the_answer_is_in_the_spelling_the_store_uses():
    # Whatever the list says, what comes back is what the store can be indexed
    # with; handing back the list's spelling would only move the mismatch.
    assert resolve(["7"], {"007"}) == {"007"}
    assert resolve(["007"], {"7"}) == {"7"}


def test_an_id_that_names_nothing_is_dropped():
    # A list may name a question a run does not hold, exactly as the
    # reference's own matching drops it.
    assert resolve(["1", "999"], {"1", "2"}) == {"1"}
    assert resolve(["999"], {"1"}) == set()


def test_ids_that_are_not_numbers_are_left_alone():
    assert resolve(["abc"], {"abc"}) == {"abc"}
    assert resolve(["abc"], {"0abc"}) == set()
