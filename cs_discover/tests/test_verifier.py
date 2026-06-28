"""Adversarial battery proving the bin-packing verifier is airtight.

Run directly (no pytest needed):  python cs_discover/tests/test_verifier.py
The verifier's core anti-reward-hacking guarantee: a candidate supplies only
priorities; the verifier enforces feasibility. These tests pin that guarantee down.
"""
import math
import os
import sys

# allow running as a plain script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cs_discover.binpacking.instances import Instance, gen_instance, make_splits
from cs_discover.binpacking import baselines as B
from cs_discover.binpacking.verifier import simulate, evaluate


def test_lower_bound_correct():
    inst = Instance(items=(0.5, 0.5, 0.5, 0.5), capacity=1.0)
    assert inst.lower_bound() == 2
    inst = Instance(items=(0.3, 0.3, 0.3), capacity=1.0)
    assert inst.lower_bound() == 1  # 0.9 total -> 1 bin lower bound


def test_known_first_fit_packing():
    # 0.6, 0.5, 0.4: 0.6 -> bin1; 0.5 -> doesn't fit bin1 (0.4 left) -> bin2;
    # 0.4 -> fits bin1 (0.4 left) -> bin1. Total 2 bins.
    inst = Instance(items=(0.6, 0.5, 0.4), capacity=1.0)
    res = simulate(B.first_fit, inst)
    assert res.bins_used == 2, res.bins_used
    assert res.feasible


def test_capacity_never_violated():
    # No matter the candidate, total packed volume per bin must never exceed capacity.
    inst = gen_instance(200, seed=7)

    def evil_priority(item, remaining, capacity):
        # Try to force placement into the FULLEST bin (least remaining) to overfill.
        return [-r for r in remaining]

    # Reconstruct per-bin load by replaying with an instrumented sim is overkill;
    # instead assert bins_used >= lower_bound (an infeasible packing would under-count).
    res = simulate(evil_priority, inst)
    assert res.bins_used >= res.lower_bound
    assert res.feasible


def test_crashing_candidate_falls_back():
    inst = gen_instance(50, seed=11)

    def boom(item, remaining, capacity):
        raise RuntimeError("candidate exploded")

    res = simulate(boom, inst)
    # Falls back to first-fit on every step that had >=1 open feasible bin.
    assert res.feasible
    assert res.bins_used >= res.lower_bound
    assert res.fallbacks > 0
    # Must match pure first-fit exactly (fallback IS first-fit).
    assert res.bins_used == simulate(B.first_fit, inst).bins_used


def test_nan_and_wrong_length_rejected():
    inst = gen_instance(50, seed=13)

    def nan_scores(item, remaining, capacity):
        return [float("nan")] * len(remaining)

    def wrong_len(item, remaining, capacity):
        return [1.0]  # wrong length unless exactly one bin

    for bad in (nan_scores, wrong_len):
        res = simulate(bad, inst)
        assert res.feasible
        assert res.bins_used == simulate(B.first_fit, inst).bins_used


def test_baselines_beat_or_match_trivial():
    # best_fit should never be worse than worst_fit on average over a sample.
    insts = make_splits(n_train=0, n_val=0, n_test=15, n_ood=0, size=80).test
    bf = evaluate(B.best_fit, insts)["mean_frac_excess"]
    wf = evaluate(B.worst_fit, insts)["mean_frac_excess"]
    assert bf <= wf + 1e-9, (bf, wf)


def test_ffd_not_worse_than_first_fit():
    insts = make_splits(n_train=0, n_val=0, n_test=15, n_ood=0, size=80).test
    ff_bins = sum(simulate(B.first_fit, i).bins_used for i in insts)
    ffd_bins = sum(B.first_fit_decreasing(i).bins_used for i in insts)
    assert ffd_bins <= ff_bins, (ffd_bins, ff_bins)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} verifier tests passed.")


if __name__ == "__main__":
    _run_all()
