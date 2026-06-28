"""Airtight online-bin-packing verifier.

Design invariant (anti-reward-hacking):
    The candidate program supplies ONLY a per-bin priority score. The verifier itself
    performs placement and enforces capacity feasibility. Therefore a candidate can
    influence *which* feasible bin is chosen, but can NEVER produce an infeasible
    packing or under-count bins. A crashing / malformed candidate degrades to a safe
    first-fit fallback rather than being rewarded.

A `priority_fn` has signature:
    priority_fn(item: float, remaining: list[float], capacity: float) -> list[float]
returning one score per currently-open bin (same length as `remaining`). The item is
placed in the feasible bin (remaining >= item) with the highest score; ties broken by
lowest index; if no bin fits, a new bin is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .instances import Instance

_EPS = 1e-9

PriorityFn = Callable[[float, List[float], float], List[float]]


@dataclass
class PackResult:
    bins_used: int
    lower_bound: int
    feasible: bool          # always True if simulate() returns; placement is enforced
    fallbacks: int          # how many steps fell back due to bad candidate output

    @property
    def excess(self) -> int:
        return self.bins_used - self.lower_bound

    @property
    def frac_excess(self) -> float:
        return self.excess / self.lower_bound if self.lower_bound else 0.0


def simulate(priority_fn: PriorityFn, instance: Instance) -> PackResult:
    """Deterministically pack `instance.items` (online) using `priority_fn` for bin
    selection. Capacity feasibility is enforced by the verifier, not the candidate."""
    cap = instance.capacity
    remaining: List[float] = []
    fallbacks = 0

    for item in instance.items:
        feasible = [i for i, r in enumerate(remaining) if r + _EPS >= item]
        if not feasible:
            remaining.append(cap - item)          # open a new bin (always feasible)
            continue

        scores = None
        try:
            scores = priority_fn(item, list(remaining), cap)
        except Exception:
            scores = None

        valid = (
            isinstance(scores, (list, tuple))
            and len(scores) == len(remaining)
            and all(isinstance(s, (int, float)) and s == s for s in scores)  # s==s rejects NaN
        )
        if not valid:
            fallbacks += 1
            choice = feasible[0]                   # safe first-fit fallback
        else:
            # max score among feasible bins, ties -> lowest index
            choice = max(feasible, key=lambda i: (scores[i], -i))

        # Hard feasibility guard (defensive; choice already came from `feasible`).
        if remaining[choice] + _EPS < item:
            fallbacks += 1
            choice = feasible[0]
        remaining[choice] -= item

    return PackResult(
        bins_used=len(remaining),
        lower_bound=instance.lower_bound(),
        feasible=True,
        fallbacks=fallbacks,
    )


def evaluate(priority_fn: PriorityFn, instances: List[Instance]) -> dict:
    """Mean fractional excess over a list of instances (lower is better)."""
    if not instances:
        return {"mean_frac_excess": 0.0, "mean_bins": 0.0, "n": 0, "total_fallbacks": 0}
    results = [simulate(priority_fn, inst) for inst in instances]
    n = len(results)
    return {
        "mean_frac_excess": sum(r.frac_excess for r in results) / n,
        "mean_excess": sum(r.excess for r in results) / n,
        "mean_bins": sum(r.bins_used for r in results) / n,
        "n": n,
        "total_fallbacks": sum(r.fallbacks for r in results),
    }
