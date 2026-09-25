"""Match question ids from a list against the ids a dataset actually uses.

Ids are carried in the spelling the dataset gives them: MedQA says 7, medbullets
says 007, and both are stored, exported and reported that way. A hand-written
or hand-edited list of question ids is the one place the two can disagree, so
that is the one place that forgives it.

The rule is the reference's own, from `load_difficult_questions` in
`inference/_dataset.py`: try the id as written, then without leading zeros,
then padded to three digits. Doing it anywhere else - keying a store by a
"normalised" id, say - is what silently dropped every medbullets question below
100 from a join, two thirds of the list, leaving a plausible-looking number
behind.
"""

from __future__ import annotations

from typing import Any, Iterable


def spellings(question_id: Any) -> tuple[str, ...]:
    """The forms a listed id may take, in the order the reference tries them."""
    text = str(question_id).strip()
    return (text, text.lstrip("0") or "0", text.zfill(3))


def resolve(listed: Iterable[Any], known: Iterable[Any]) -> set[str]:
    """The known ids that `listed` names, in the spelling `known` uses.

    Ids in the list that match nothing are dropped, exactly as the reference
    drops them: a list can name a question a run does not hold.
    """
    known = {str(k) for k in known}
    out: set[str] = set()
    for question_id in listed:
        for candidate in spellings(question_id):
            if candidate in known:
                out.add(candidate)
                break
    return out


__all__ = ["resolve", "spellings"]
