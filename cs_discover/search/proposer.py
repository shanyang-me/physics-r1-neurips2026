"""Proposers — the swappable component of the discovery loop.

  MutationProposer : no-LLM operator (perturbs float coefficients). Lets the whole
                     pipeline run + be tested WITHOUT model access, and is the
                     frozen-proposer control for the headline ablation.
  LLMProposer      : drop-in that asks an LLM to rewrite the best programs. The RL
                     trainer's policy is itself an LLMProposer whose weights are
                     updated from discovery reward (vs FunSearch's frozen LLM).

A Proposer maps (domain, parents) -> candidate source string.
"""
from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..domains.base import Domain


class Program:
    __slots__ = ("source", "score")

    def __init__(self, source: str, score: float = float("-inf")):
        self.source = source
        self.score = score


class Proposer(ABC):
    @abstractmethod
    def propose(self, domain: Domain, parents: List[Program], rng: random.Random) -> str:
        ...


class MutationProposer(Proposer):
    """Frozen, model-free baseline: mutate a sampled parent's float coefficients."""

    def propose(self, domain: Domain, parents: List[Program], rng: random.Random) -> str:
        parent = parents[rng.randrange(len(parents))] if parents else None
        base = parent.source if parent else domain.seed_sources()[0]
        return domain.mutate(base, rng)


_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


class LLMProposer(Proposer):
    """Frozen-LLM proposer (FunSearch/EvoTune-style). `call_fn(prompt) -> text` is any
    chat-completion callable; inject a trained policy here for the RL variant."""

    def __init__(self, call_fn: Optional[Callable[[str], str]] = None, k_parents: int = 2):
        self.call_fn = call_fn
        self.k_parents = k_parents

    def build_prompt(self, domain: Domain, parents: List[Program]) -> str:
        shown = sorted(parents, key=lambda p: p.score, reverse=True)[: self.k_parents]
        examples = "\n\n".join(
            f"# score = {p.score:.5f}\n{p.source}" for p in shown
        )
        return (
            f"{domain.spec()}\n\n"
            f"Here are the best programs found so far (higher score is better):\n\n"
            f"{examples}\n\n"
            f"Write an improved version of `{domain.fn_name}`. Return ONLY a Python "
            f"code block."
        )

    def propose(self, domain: Domain, parents: List[Program], rng: random.Random) -> str:
        if self.call_fn is None:
            raise NotImplementedError(
                "LLMProposer needs a call_fn (model backend) injected."
            )
        text = self.call_fn(self.build_prompt(domain, parents))
        m = _CODE_BLOCK.search(text or "")
        return m.group(1).strip() if m else (text or "").strip()
