"""Discovery domains (pluggable problem families) for CS-Discover.

Each domain implements the `Domain` contract in base.py. Multiple families behind one
interface are what make the cross-problem-transfer (L2) claim testable.
"""
from .base import Domain, EvalResult, compile_program, CompileError
from .binpacking_domain import BinPackingDomain
from .capset_domain import CapSetDomain

REGISTRY = {
    "binpacking": BinPackingDomain,
    "capset": CapSetDomain,
}


def make_domain(name: str, **kwargs) -> Domain:
    if name not in REGISTRY:
        raise KeyError(f"unknown domain {name!r}; available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
