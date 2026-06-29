"""Bin-packing as a Domain (the EvoTune-comparable reproduction family).

Score = -mean_fractional_excess (higher is better). The candidate's `priority` is fed
straight to the airtight verifier, which enforces feasibility — so the score cannot be
hacked by overfilling. Selection happens on `train`; generalization is judged on the
held-out `test` and `ood` splits.
"""
from __future__ import annotations

from typing import Callable, List

from .base import Domain, EvalResult
from ..binpacking.instances import make_splits
from ..binpacking import baselines as B
from ..binpacking.verifier import evaluate as bp_evaluate


_SEED = '''\
def priority(item, remaining, capacity):
    # linear scoring over each open bin's remaining capacity.
    # seed coefficients are 0 -> ties -> first-fit; search tunes them.
    out = []
    for r in remaining:
        score = 0.0 * r + 0.0 * (r - item) + 0.0 * (r * r)
        out.append(score)
    return out
'''


class BinPackingDomain(Domain):
    name = "binpacking"
    fn_name = "priority"

    def __init__(self, dist: str = "uniform"):
        self.dist = dist
        self._splits = make_splits(dist=dist).as_dict()

    def spec(self) -> str:
        return (
            "Online bin packing (capacity normalized to 1.0). Items arrive one at a "
            "time and must be placed immediately into an open bin that has room, or a "
            "new bin is opened. Write `priority(item, remaining, capacity)` returning "
            "one score per open bin (a list the same length as `remaining`, the list "
            "of bins' remaining capacities). The item is placed in the feasible bin "
            "with the HIGHEST score. Goal: minimize the number of bins used. Beat "
            "best-fit (priority = -remaining)."
        )

    def seed_sources(self) -> List[str]:
        return [_SEED]

    def train_split(self) -> str:
        return "train"

    def report_splits(self) -> List[str]:
        return ["test", "ood"]

    def evaluate(self, fn: Callable, split: str) -> EvalResult:
        insts = self._splits[split]
        stats = bp_evaluate(fn, insts)
        return EvalResult(
            score=-stats["mean_frac_excess"],
            correct=True,  # feasibility is enforced by the verifier
            detail=stats,
        )

    def baseline_score(self, split: str) -> float:
        insts = self._splits[split]
        return -bp_evaluate(B.best_fit, insts)["mean_frac_excess"]
