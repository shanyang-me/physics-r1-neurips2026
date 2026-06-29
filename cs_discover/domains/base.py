"""Domain abstraction — the pluggable contract that makes cross-problem transfer (L2)
testable.

Every discovery problem family implements `Domain`. The discovery artifact is always a
Python *program* defining a `priority(...)` function (the FunSearch representation,
shared by bin-packing and cap-set). The domain compiles it in a restricted namespace
and turns it into a scalar score where **higher is better**.

NOTE ON SANDBOXING: `compile_program` restricts builtins and blocks imports, which is
enough for a local mutation/LLM proposer in a research prototype. It is NOT a security
sandbox (no hard CPU/memory/wall-clock isolation). Treat candidate code as untrusted
before any networked deployment.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# --- restricted execution -------------------------------------------------------

_SAFE_BUILTIN_NAMES = [
    "abs", "min", "max", "sum", "len", "range", "sorted", "float", "int", "bool",
    "enumerate", "zip", "map", "filter", "round", "pow", "list", "tuple", "dict",
    "set", "all", "any", "reversed", "divmod", "True", "False", "None",
]
import builtins as _b
_SAFE_BUILTINS = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_b, n)}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})


class CompileError(Exception):
    pass


def compile_program(source: str, fn_name: str) -> Callable:
    """Compile `source` and return its `fn_name` function, with imports/builtins
    restricted. Raises CompileError on any failure."""
    if "import" in source:
        raise CompileError("imports are not permitted in candidate programs")
    g = {"__builtins__": _SAFE_BUILTINS, "math": math}
    try:
        code = compile(source, "<candidate>", "exec")
        exec(code, g)  # globals == locals so helper fns see each other + math
    except Exception as e:  # noqa: BLE001
        raise CompileError(f"compile/exec failed: {e!r}")
    fn = g.get(fn_name)
    if not callable(fn):
        raise CompileError(f"program does not define callable {fn_name!r}")
    return fn


# --- float mutation (the no-LLM proposer's operator) ----------------------------

_FLOAT_RE = re.compile(r"-?\d+\.\d+")


def perturb_floats(source: str, rng, scale: float = 0.6, p: float = 0.6) -> str:
    """Stochastically perturb numeric float literals in `source`. Guarantees at least
    one change so search always makes progress."""
    changed = [0]

    def repl(m):
        val = float(m.group(0))
        if rng.random() < p:
            changed[0] += 1
            return f"{round(val + rng.uniform(-1, 1) * scale, 4)}"
        return m.group(0)

    out = _FLOAT_RE.sub(repl, source)
    if changed[0] == 0:  # force one change
        floats = list(_FLOAT_RE.finditer(source))
        if floats:
            m = floats[rng.randrange(len(floats))]
            val = float(m.group(0)) + rng.uniform(-1, 1) * scale
            out = source[: m.start()] + f"{round(val, 4)}" + source[m.end():]
    return out


# --- domain contract ------------------------------------------------------------

@dataclass
class EvalResult:
    score: float            # higher is better (verifier output, normalized)
    correct: bool           # passed the hard correctness gate
    detail: dict = field(default_factory=dict)


class Domain(ABC):
    name: str = "domain"
    fn_name: str = "priority"

    @abstractmethod
    def spec(self) -> str:
        """Natural-language problem statement for an LLM proposer."""

    @abstractmethod
    def seed_sources(self) -> List[str]:
        """Initial (often trivial) program(s) to seed search."""

    @abstractmethod
    def evaluate(self, fn: Callable, split: str) -> EvalResult:
        """Score a compiled candidate on a named split. Higher score = better."""

    @abstractmethod
    def baseline_score(self, split: str) -> float:
        """Reference score of the strongest standard baseline on a split."""

    # split plumbing (overridden by instance-based domains)
    def train_split(self) -> str:
        return "all"

    def report_splits(self) -> List[str]:
        return ["all"]

    # default mutation operator: perturb float coefficients
    def mutate(self, source: str, rng) -> str:
        return perturb_floats(source, rng)

    # convenience: compile + evaluate, returning a safe failing result on error
    def safe_evaluate(self, source: str, split: str) -> Optional[EvalResult]:
        try:
            fn = compile_program(source, self.fn_name)
        except CompileError:
            return None
        try:
            return self.evaluate(fn, split)
        except Exception:  # noqa: BLE001
            return None
