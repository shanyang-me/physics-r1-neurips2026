"""Collect preference data for RL-training the proposer (EvoTune-style DPO signal).

For each *context* (a set of parent programs, exactly what an LLMProposer conditions
on), we sample several candidate completions, score them with the verifier, and pair a
higher-reward completion (chosen) against a lower-reward one (rejected) — sharing the
same prompt. Fine-tuning the proposer to prefer `chosen` over `rejected` is the
"proposer learns" step that distinguishes CS-Discover from a frozen FunSearch loop.

Runnable on CPU with the model-free MutationProposer (for tests / plumbing); pass the
Claude LLMProposer to collect real preference data (each context = several model calls).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

from ..domains.base import Domain
from ..search.proposer import LLMProposer, Program, Proposer


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float

    @property
    def margin(self) -> float:
        return self.chosen_score - self.rejected_score


def sample_contexts(domain: Domain, n_contexts: int, seed: int, k_parents: int = 2) -> List[List[Program]]:
    """Build varied parent-sets by mutating the seed program a few times each."""
    rng = random.Random(seed)
    train = domain.train_split()
    contexts = []
    for _ in range(n_contexts):
        parents, base = [], domain.seed_sources()[0]
        for _ in range(k_parents):
            src = domain.mutate(base, rng)
            res = domain.safe_evaluate(src, train)
            score = res.score if (res and res.correct) else float("-inf")
            parents.append(Program(src, score))
            base = src
        contexts.append(parents)
    return contexts


def collect_preferences(
    domain: Domain,
    proposer: Proposer,
    n_contexts: int = 8,
    samples_per_context: int = 4,
    margin: float = 1e-9,
    max_pairs_per_context: int = 3,
    seed: int = 0,
) -> List[PreferencePair]:
    """Sample completions per context and emit (chosen > rejected) preference pairs."""
    rng = random.Random(seed)
    train = domain.train_split()
    prompt_builder = LLMProposer()  # prompt format the trained proposer will learn
    pairs: List[PreferencePair] = []

    for ctx in sample_contexts(domain, n_contexts, seed):
        prompt = prompt_builder.build_prompt(domain, ctx)
        cands = []
        for _ in range(samples_per_context):
            try:
                src = proposer.propose(domain, ctx, rng)
            except Exception:  # noqa: BLE001
                continue
            res = domain.safe_evaluate(src, train)
            if res is None or not res.correct:
                continue
            cands.append((src, res.score))
        # form preference pairs from distinct-score candidates
        ctx_pairs = []
        for (sa, ca), (sb, cb) in combinations(cands, 2):
            if abs(ca - cb) <= margin or sa == sb:
                continue
            if ca >= cb:
                ctx_pairs.append(PreferencePair(prompt, sa, sb, ca, cb))
            else:
                ctx_pairs.append(PreferencePair(prompt, sb, sa, cb, ca))
        ctx_pairs.sort(key=lambda p: p.margin, reverse=True)
        pairs.extend(ctx_pairs[:max_pairs_per_context])

    return pairs
