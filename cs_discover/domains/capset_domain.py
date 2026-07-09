"""Cap-set as a Domain. Single deterministic construction (no instance splits), so it
serves as the *second problem family* for the cross-problem-transfer axis."""
from __future__ import annotations

from typing import Callable, List

from .base import Domain, EvalResult
from .capset import greedy_cap, is_cap, lexicographic_priority


_SEED = '''\
def priority(vector, n):
    # weighted sum over components; seed weights are 0 -> lexicographic-ish order.
    s = 0.0
    for i, v in enumerate(vector):
        s += 0.0 * v * (i + 1) + 0.0 * (v * v) + 0.0 * ((v + i) % 3)
    return s
'''


class CapSetDomain(Domain):
    name = "capset"
    fn_name = "priority"

    def __init__(self, n: int = 4):
        self.n = n
        self._max_size = 3 ** n  # loose normalizer

    def spec(self) -> str:
        return (
            f"Cap set in F_3^{self.n}: find the largest subset of length-{self.n} "
            "vectors over {0,1,2} with NO three distinct vectors summing to the zero "
            "vector (mod 3). Write `priority(vector, n)` returning a float; vectors are "
            "added greedily in descending priority, skipping any that would complete a "
            "line. Goal: maximize the cap size. Beat the lexicographic baseline."
        )

    def seed_sources(self) -> List[str]:
        return [_SEED]

    def reference_sources(self) -> List[str]:
        lexicographic = (
            "def priority(vector, n):\n"
            "    return -sum(val * (3 ** i) for i, val in enumerate(vector))\n"
        )
        return [_SEED, lexicographic]

    def evaluate(self, fn: Callable, split: str) -> EvalResult:
        cap = greedy_cap(fn, self.n)
        ok = is_cap(cap)  # airtight re-verification
        size = len(cap) if ok else 0
        return EvalResult(score=float(size), correct=ok, detail={"size": size, "n": self.n})

    def baseline_score(self, split: str) -> float:
        cap = greedy_cap(lexicographic_priority, self.n)
        return float(len(cap))
