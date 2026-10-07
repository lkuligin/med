"""Inter-rater agreement between two raters labelling the same items.

    from paper.irr import agreement, cohen_kappa

    agreement([1, 1, 0, 1], [1, 0, 0, 1])   # Agreement(n=4, observed=0.75, kappa=0.5, ...)
    cohen_kappa(verdicts_a, verdicts_b)

Pure functions over two equally long sequences of labels, with no knowledge of
where the labels come from; any hashable labels work, not only 0/1. Used in the
paper for two verifiers judging the same facts.

Cohen's kappa corrects the share of matching labels for the agreement two
raters would reach by chance with their own label frequencies:
kappa = (observed - expected) / (1 - expected). When one label dominates (a
verifier that accepts 99% of facts), expected agreement is close to 1 and kappa
is low even when the raters almost never differ, so report the observed
agreement beside it.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class Agreement:
    n: int                          # items both raters labelled
    observed: float                 # share of items with the same label
    expected: float                 # share expected by chance
    kappa: float | None             # Cohen's kappa; None when chance agreement is certain
    rates_a: dict[Hashable, float]  # each label's share among rater A's labels
    rates_b: dict[Hashable, float]


def agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> Agreement:
    """Observed agreement and Cohen's kappa of two raters over the same items.

    Kappa is None when both raters give one and the same label to every item:
    agreement by chance is then certain and kappa is undefined.
    """
    if len(a) != len(b):
        raise ValueError(f"raters labelled different numbers of items: {len(a)} and {len(b)}")
    if not a:
        raise ValueError("no items to compare")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = collections.Counter(a), collections.Counter(b)
    rates_a = {label: c / n for label, c in count_a.items()}
    rates_b = {label: c / n for label, c in count_b.items()}
    expected = sum(rates_a[label] * rates_b.get(label, 0.0) for label in rates_a)
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return Agreement(n=n, observed=observed, expected=expected, kappa=kappa,
                     rates_a=rates_a, rates_b=rates_b)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa alone; see agreement()."""
    return agreement(a, b).kappa


__all__ = ["Agreement", "agreement", "cohen_kappa"]
