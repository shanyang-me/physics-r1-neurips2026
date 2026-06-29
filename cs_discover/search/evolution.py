"""The propose -> test -> refine loop (FunSearch / EvoTune-style evolutionary search).

This is the harness the RL trainer plugs into: swap the frozen `Proposer` for a learned
policy and add a weight-update step on (context, program, reward). Selection signal is
the TRAIN split; reported numbers come from held-out splits — so improvement reflects
L1 generalization, not memorization of the search instances.

Budget is measured in **candidate evaluations** (verifier calls) — the unit used for
the matched-compute RL-vs-frozen-search comparison (see CS_DISCOVERY_EVAL.md §2).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..domains.base import Domain
from .proposer import Program, Proposer


@dataclass
class SearchResult:
    best: Program
    seed_score: float
    baseline_score: float
    history: List[float] = field(default_factory=list)   # best train score over time
    n_evaluated: int = 0
    n_failed: int = 0
    report: Dict[str, float] = field(default_factory=dict)  # held-out split -> score


def evolve(
    domain: Domain,
    proposer: Proposer,
    budget: int = 200,
    pop_cap: int = 40,
    seed: int = 0,
) -> SearchResult:
    """Run evolutionary discovery search for `budget` candidate evaluations."""
    rng = random.Random(seed)
    train = domain.train_split()

    # seed population
    population: List[Program] = []
    for src in domain.seed_sources():
        res = domain.safe_evaluate(src, train)
        score = res.score if (res and res.correct) else float("-inf")
        population.append(Program(src, score))
    seed_score = max(p.score for p in population)
    best = max(population, key=lambda p: p.score)

    history: List[float] = [best.score]
    n_eval = 0
    n_fail = 0

    while n_eval < budget:
        src = proposer.propose(domain, population, rng)
        res = domain.safe_evaluate(src, train)
        n_eval += 1
        if res is None or not res.correct:
            n_fail += 1
            history.append(best.score)
            continue
        child = Program(src, res.score)
        population.append(child)
        if child.score > best.score:
            best = child
        # bound population to the top pop_cap by score
        if len(population) > pop_cap:
            population.sort(key=lambda p: p.score, reverse=True)
            del population[pop_cap:]
        history.append(best.score)

    # held-out report for the best program
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
        n_failed=n_fail,
        report=report,
    )
