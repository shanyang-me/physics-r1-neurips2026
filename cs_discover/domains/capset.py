"""Cap-set construction in F_3^n — the 2nd discovery family (FunSearch's headline math
problem), needed so cross-problem transfer (L2) is testable.

A *cap set* is a subset of F_3^n with no three distinct points x,y,z summing to 0
(mod 3) — i.e. no three-term line. The discovery artifact is a `priority(vector, n)`
that orders the 3^n vectors; a greedy constructor adds them in priority order, skipping
any that would complete a line. The score is the resulting set size (larger = better;
the construction is a valid cap by construction, and verified airtight).
"""
from __future__ import annotations

from itertools import product
from typing import Callable, List, Tuple

Vector = Tuple[int, ...]


def all_vectors(n: int) -> List[Vector]:
    return list(product(range(3), repeat=n))


def is_cap(points) -> bool:
    """Airtight verifier: True iff no three distinct points form a line (sum ≡ 0)."""
    pts = list(points)
    pset = set(pts)
    if len(pset) != len(pts):
        return False  # duplicates
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            z = tuple((-(x + y)) % 3 for x, y in zip(a, b))
            if z in pset and z != a and z != b:
                return False
    return True


def greedy_cap(priority: Callable[[Vector, int], float], n: int) -> List[Vector]:
    """Greedily build a cap by adding vectors in descending priority order, skipping
    any vector that would complete a line with two already-chosen points."""
    vecs = all_vectors(n)
    ordered = sorted(vecs, key=lambda v: priority(v, n), reverse=True)
    chosen: List[Vector] = []
    chosen_set = set()
    for v in ordered:
        ok = True
        for u in chosen:
            z = tuple((-(a + b)) % 3 for a, b in zip(u, v))
            if z in chosen_set:  # z != u, z != v provable for distinct u,v over F_3
                ok = False
                break
        if ok:
            chosen.append(v)
            chosen_set.add(v)
    return chosen


def lexicographic_priority(v: Vector, n: int) -> float:
    """Baseline ordering: lexicographic over F_3^n."""
    return -sum(val * (3 ** i) for i, val in enumerate(v))
