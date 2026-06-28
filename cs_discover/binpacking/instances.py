"""Seeded online-bin-packing instance generators + train/val/test/OOD splits.

Instances are *generated* (not downloaded), so the suite is contamination-free by
construction — a winning heuristic cannot have been memorized from a public dataset.
Each split uses a disjoint base seed so the streams never overlap.

Capacity is normalized to 1.0; item sizes live in (0, 1].
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Instance:
    items: tuple              # the arrival stream (online order matters)
    capacity: float = 1.0
    dist: str = "uniform"
    seed: int = 0

    def lower_bound(self) -> int:
        """Trivial volume lower bound: ceil(sum(items) / capacity)."""
        total = sum(self.items)
        # ceil without floating drift
        import math
        return max(1, math.ceil(total / self.capacity - 1e-9))


def _sample_item(rng: random.Random, dist: str, capacity: float) -> float:
    if dist == "uniform":
        x = rng.uniform(0.10, 0.70)
    elif dist == "weibull":
        # FunSearch-style Weibull item sizes, clipped into (0, capacity].
        x = rng.weibullvariate(0.45, 3.0)
    else:
        raise ValueError(f"unknown dist: {dist!r}")
    return min(max(x, 1e-3), capacity)


def gen_instance(n: int, seed: int, dist: str = "uniform", capacity: float = 1.0) -> Instance:
    rng = random.Random(seed)
    items = tuple(_sample_item(rng, dist, capacity) for _ in range(n))
    return Instance(items=items, capacity=capacity, dist=dist, seed=seed)


# Disjoint seed bands per split so train/val/test/ood streams never collide.
_SEED_BANDS = {
    "train": 0,
    "val": 1_000_000,
    "test": 2_000_000,
    "ood": 3_000_000,
}


@dataclass
class Splits:
    train: List[Instance] = field(default_factory=list)
    val: List[Instance] = field(default_factory=list)
    test: List[Instance] = field(default_factory=list)
    ood: List[Instance] = field(default_factory=list)

    def as_dict(self):
        return {"train": self.train, "val": self.val, "test": self.test, "ood": self.ood}


def make_splits(
    dist: str = "uniform",
    n_train: int = 40,
    n_val: int = 20,
    n_test: int = 40,
    n_ood: int = 20,
    size: int = 50,
    ood_size: int = 500,
    base_seed: int = 42,
) -> Splits:
    """Build disjoint instance splits.

    L1 generalization is measured on `test` (held-out instances, same size as train).
    OOD generalization is measured on `ood` (much larger streams, size `ood_size`).
    """
    def band(name, count, n):
        b = _SEED_BANDS[name] + base_seed
        return [gen_instance(n, seed=b + i, dist=dist) for i in range(count)]

    return Splits(
        train=band("train", n_train, size),
        val=band("val", n_val, size),
        test=band("test", n_test, size),
        ood=band("ood", n_ood, ood_size),
    )
