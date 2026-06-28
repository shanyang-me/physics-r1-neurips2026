"""Baseline bin-packing heuristics in priority form — these define "the bar".

Each is a `priority_fn(item, remaining, capacity) -> list[float]`. The verifier places
the item in the feasible bin with the highest score (ties -> lowest index).

  first_fit  : prefer the earliest-opened bin            -> score = -index
  best_fit   : prefer the tightest bin that still fits   -> score = -remaining
  worst_fit  : prefer the emptiest bin                    -> score =  remaining

first_fit_decreasing is offline (it reorders the stream), so it is provided as a
standalone evaluator rather than a priority_fn.
"""
from __future__ import annotations

from typing import List

from .instances import Instance
from .verifier import PackResult, simulate


def first_fit(item: float, remaining: List[float], capacity: float) -> List[float]:
    return [-i for i in range(len(remaining))]


def best_fit(item: float, remaining: List[float], capacity: float) -> List[float]:
    return [-r for r in remaining]


def worst_fit(item: float, remaining: List[float], capacity: float) -> List[float]:
    return [r for r in remaining]


BASELINES = {
    "first_fit": first_fit,
    "best_fit": best_fit,
    "worst_fit": worst_fit,
}


def first_fit_decreasing(instance: Instance) -> PackResult:
    """Offline FFD: sort items descending, then first-fit. Returned as a PackResult."""
    ordered = Instance(
        items=tuple(sorted(instance.items, reverse=True)),
        capacity=instance.capacity,
        dist=instance.dist,
        seed=instance.seed,
    )
    return simulate(first_fit, ordered)
