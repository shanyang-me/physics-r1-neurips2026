"""End-to-end discovery-loop tests across both domains. The loop must:
  - run without model access (MutationProposer),
  - improve the held-out score over the seed,
  - only ever return correct (verified) programs.
Run: python cs_discover/tests/test_search.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cs_discover.domains import make_domain, compile_program
from cs_discover.search import MutationProposer, evolve


def test_binpacking_search_improves_over_seed():
    dom = make_domain("binpacking", dist="uniform")
    res = evolve(dom, MutationProposer(), budget=200, seed=1)
    # best beats the trivial seed on the selection split...
    assert res.best.score > res.seed_score, (res.best.score, res.seed_score)
    # ...and generalizes to held-out test (>= seed on test).
    seed_test = dom.safe_evaluate(dom.seed_sources()[0], "test").score
    assert res.report["test"] >= seed_test - 1e-9


def test_capset_mock_proposer_saturates_at_2n():
    # HONEST FINDING (see README "Findings"): greedy construction with any SEPARABLE
    # priority saturates at exactly 2^n. The MutationProposer only tunes coefficients
    # of a separable priority, so it cannot beat 2^n — this is precisely the gap a
    # non-separable LLM/RL proposer must close (the EvoTune/FunSearch motivation).
    dom = make_domain("capset", n=4)
    res = evolve(dom, MutationProposer(), budget=200, seed=1)
    assert dom.safe_evaluate(res.best.source, "all").correct
    assert res.best.score == 16.0 == res.seed_score, (res.best.score, res.seed_score)
    assert res.report["all"] <= 20.0  # never exceeds the known max for n=4


def test_search_only_returns_correct_programs():
    dom = make_domain("capset", n=3)
    res = evolve(dom, MutationProposer(), budget=120, seed=2)
    r = dom.safe_evaluate(res.best.source, "all")
    assert r is not None and r.correct


def test_two_domains_share_one_interface():
    # the L2-transfer precondition: identical loop runs on both families unchanged.
    for name, kw in (("binpacking", {}), ("capset", {"n": 3})):
        dom = make_domain(name, **kw)
        res = evolve(dom, MutationProposer(), budget=60, seed=0)
        assert res.n_evaluated == 60
        compile_program(res.best.source, dom.fn_name)  # best is always compilable


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} search tests passed.")


if __name__ == "__main__":
    _run_all()
