"""Scaling-infra tests: candidate cache, concurrency, and the novelty gate.
Run: python cs_discover/tests/test_scale.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cs_discover.domains import make_domain
from cs_discover.search import MutationProposer, evolve
from cs_discover import novelty


def test_cache_reduces_verifier_calls():
    dom = make_domain("capset", n=3)
    cache = {}
    r1 = evolve(dom, MutationProposer(), budget=80, seed=3, cache=cache)
    # a warm cache should make a second identical run mostly cache hits.
    r2 = evolve(dom, MutationProposer(), budget=80, seed=3, cache=cache)
    assert r2.n_cache_hits > r1.n_cache_hits
    assert r2.n_verified < r1.n_verified          # fewer real verifier calls the 2nd time
    assert r1.n_verified + r1.n_cache_hits >= r1.n_evaluated - r1.n_failed


def test_workers_run_and_stay_valid():
    dom = make_domain("binpacking", dist="uniform")
    res = evolve(dom, MutationProposer(), budget=40, seed=1, workers=4)
    assert res.n_evaluated == 40
    # best is still a real, correct, compilable improvement over the seed
    assert res.best.score >= res.seed_score
    assert dom.safe_evaluate(res.best.source, "test") is not None


def test_novelty_flags_duplicate_and_rewards_difference():
    dom = make_domain("capset", n=3)
    refs = dom.reference_sources()
    # an exact copy of a reference is minimally novel...
    assert novelty.novelty_score(refs[0], refs) == 0.0
    assert not novelty.is_novel(refs[0], refs, threshold=0.5)
    # ...a structurally different program is highly novel.
    other = (
        "def priority(vector, n):\n"
        "    return sum((a*b) % 3 for a in vector for b in vector) * 1.7 - vector[0]\n"
    )
    assert novelty.novelty_score(other, refs) > 0.5
    assert novelty.is_novel(other, refs, threshold=0.5)


def test_discovered_capset20_is_novel():
    # the Claude-discovered size-20 program must read as genuinely novel vs baselines.
    from cs_discover.discovered import capset_n4_claude_size20 as disc
    import inspect
    dom = make_domain("capset", n=4)
    src = inspect.getsource(disc.priority)
    assert novelty.novelty_score(src, dom.reference_sources()) > 0.8


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} scale/novelty tests passed.")


if __name__ == "__main__":
    _run_all()
