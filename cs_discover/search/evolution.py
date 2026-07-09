"""The propose -> test -> refine loop (FunSearch / EvoTune-style evolutionary search).

This is the harness the RL trainer plugs into: swap the frozen `Proposer` for a learned
policy and add a weight-update step on (context, program, reward). Selection signal is
the TRAIN split; reported numbers come from held-out splits — so improvement reflects
L1 generalization, not memorization of the search instances.

`budget` counts **proposer calls** (candidate generations) — the matched-compute unit
for the RL-vs-frozen-search comparison (CS_DISCOVERY_EVAL.md §2). A `cache` keyed on
program source avoids re-running the verifier on duplicate proposals (common for both
mutation and LLM proposers, and the substrate for reusing trajectories in RL). Set
`workers > 1` to issue proposer calls concurrently — the key scaling lever when the
proposer is a slow LLM (subprocess/IO-bound, so threads give real parallelism).
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..domains.base import Domain, EvalResult
from .proposer import Program, Proposer


@dataclass
class SearchResult:
    best: Program
    seed_score: float
    baseline_score: float
    history: List[float] = field(default_factory=list)   # best train score over time
    n_evaluated: int = 0        # proposer calls made (== budget consumed)
    n_verified: int = 0         # actual verifier calls (cache misses)
    n_cache_hits: int = 0
    n_failed: int = 0           # proposals that failed to propose/compile/verify
    report: Dict[str, float] = field(default_factory=dict)  # held-out split -> score


def _safe_propose(proposer, domain, population, rng):
    try:
        return proposer.propose(domain, population, rng)
    except Exception:  # noqa: BLE001 — a flaky LLM call is a failed step, not a crash
        return None


def evolve(
    domain: Domain,
    proposer: Proposer,
    budget: int = 200,
    pop_cap: int = 40,
    seed: int = 0,
    workers: int = 1,
    cache: Optional[Dict[str, Optional[EvalResult]]] = None,
) -> SearchResult:
    """Run evolutionary discovery search for `budget` proposer calls."""
    rng = random.Random(seed)
    train = domain.train_split()
    cache = cache if cache is not None else {}

    stats = {"verify": 0, "hits": 0}

    def eval_cached(src: str):
        if src in cache:
            stats["hits"] += 1
            return cache[src]
        res = domain.safe_evaluate(src, train)
        cache[src] = res
        stats["verify"] += 1
        return res

    # seed population
    population: List[Program] = []
    for src in domain.seed_sources():
        res = eval_cached(src)
        score = res.score if (res and res.correct) else float("-inf")
        population.append(Program(src, score))
    seed_score = max(p.score for p in population)
    best = max(population, key=lambda p: p.score)

    history: List[float] = [best.score]
    n_eval = 0
    n_fail = 0

    def ingest(src):
        nonlocal best, n_fail
        if src is None:
            n_fail += 1
            history.append(best.score)
            return
        res = eval_cached(src)
        if res is None or not res.correct:
            n_fail += 1
            history.append(best.score)
            return
        child = Program(src, res.score)
        population.append(child)
        if child.score > best.score:
            best = child
        if len(population) > pop_cap:
            population.sort(key=lambda p: p.score, reverse=True)
            del population[pop_cap:]
        history.append(best.score)

    if workers <= 1:
        while n_eval < budget:
            ingest(_safe_propose(proposer, domain, population, rng))
            n_eval += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while n_eval < budget:
                b = min(workers, budget - n_eval)
                snapshot = list(population)  # concurrent proposers read a fixed snapshot
                futures = [
                    pool.submit(
                        _safe_propose, proposer, domain, snapshot,
                        random.Random(seed * 1_000_003 + n_eval + i),
                    )
                    for i in range(b)
                ]
                n_eval += b
                for f in futures:
                    ingest(f.result())

    report = {}
    for split in domain.report_splits():
        r = domain.safe_evaluate(best.source, split)
        report[split] = r.score if r else float("-inf")

    return SearchResult(
        best=best,
        seed_score=seed_score,
        baseline_score=domain.baseline_score(domain.report_splits()[0]),
        history=history,
        n_evaluated=n_eval,
        n_verified=stats["verify"],
        n_cache_hits=stats["hits"],
        n_failed=n_fail,
        report=report,
    )
