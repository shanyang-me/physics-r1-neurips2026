"""Serialize preference data to the JSONL formats the trainers expect.

  DPO : {"prompt", "chosen", "rejected"}            (trl DPOTrainer / verl)
  SFT : {"prompt", "completion"}                     (warm-start on top programs)

Wrapping in ```python fences matches how LLMProposer parses completions, so the trained
proposer emits code the search loop can extract unchanged.
"""
from __future__ import annotations

import json
from typing import Iterable, List

from .collect import PreferencePair


def _fence(src: str) -> str:
    return f"```python\n{src.strip()}\n```"


def write_dpo_jsonl(pairs: Iterable[PreferencePair], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps({
                "prompt": p.prompt,
                "chosen": _fence(p.chosen),
                "rejected": _fence(p.rejected),
                "chosen_score": p.chosen_score,
                "rejected_score": p.rejected_score,
            }) + "\n")
            n += 1
    return n


def write_sft_jsonl(pairs: Iterable[PreferencePair], path: str) -> int:
    """One SFT record per unique (prompt, chosen) — warm-start on winning programs."""
    seen = set()
    n = 0
    with open(path, "w") as f:
        for p in pairs:
            key = (p.prompt, p.chosen)
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({"prompt": p.prompt, "completion": _fence(p.chosen)}) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
