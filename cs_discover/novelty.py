"""Novelty gate — "did it discover this, or recall/trivially-copy it?"

Mirrors stage-1 of this repo's contamination audit (`audit/audit_two_stage.py`): a
5-gram Jaccard overlap, here applied to a discovered *program's source* against a
reference corpus of known/baseline programs. A discovered heuristic that is essentially
the baseline scores low novelty; a structurally new one scores high.

This is the cheap, dependency-free first stage. The embedding-cosine second stage (the
audit's mxbai step) can be layered on later for semantic near-duplicate detection. The
design docs put a novelty gate *inside the reward loop* — this module is that gate's
core scorer; the RL trainer will subtract a penalty when novelty is low.
"""
from __future__ import annotations

import re
from typing import Iterable, List


def _tokens(text: str) -> List[str]:
    # collapse comments + whitespace so novelty reflects code structure, not prose.
    text = re.sub(r"#.*", " ", text)
    return re.findall(r"[A-Za-z_]\w*|[^\sA-Za-z_]", text)


def _ngrams(tokens: List[str], n: int = 5) -> set:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: str, b: str, n: int = 5) -> float:
    ga, gb = _ngrams(_tokens(a), n), _ngrams(_tokens(b), n)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def max_similarity(source: str, references: Iterable[str], n: int = 5) -> float:
    sims = [jaccard(source, ref, n) for ref in references]
    return max(sims) if sims else 0.0


def novelty_score(source: str, references: Iterable[str], n: int = 5) -> float:
    """1.0 = maximally novel vs the reference corpus; 0.0 = identical to a reference."""
    return 1.0 - max_similarity(source, references, n)


def is_novel(source: str, references: Iterable[str], threshold: float = 0.5, n: int = 5) -> bool:
    return novelty_score(source, references, n) >= threshold
