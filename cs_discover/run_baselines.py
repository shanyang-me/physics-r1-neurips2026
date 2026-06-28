"""Print the baseline bar for online bin-packing across all splits.

This establishes the reference numbers that any discovered heuristic — and the
RL-vs-FunSearch comparison — must beat. Metric: mean fractional excess over the
volume lower bound (lower is better).

Usage:  python -m cs_discover.run_baselines [--dist uniform|weibull]
"""
from __future__ import annotations

import argparse

from cs_discover.binpacking.instances import make_splits
from cs_discover.binpacking import baselines as B
from cs_discover.binpacking.verifier import evaluate, simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="uniform", choices=["uniform", "weibull"])
    args = ap.parse_args()

    splits = make_splits(dist=args.dist).as_dict()

    print(f"Online bin-packing baselines  (dist={args.dist})")
    print("metric = mean fractional excess over volume lower bound (lower is better)\n")
    header = f"{'split':<8}{'n':>4}  " + "".join(f"{name:>12}" for name in B.BASELINES) + f"{'ffd':>12}"
    print(header)
    print("-" * len(header))

    for split_name, insts in splits.items():
        row = f"{split_name:<8}{len(insts):>4}  "
        for name, fn in B.BASELINES.items():
            row += f"{evaluate(fn, insts)['mean_frac_excess']:>12.4f}"
        # FFD (offline) mean fractional excess
        if insts:
            ffd_vals = [B.first_fit_decreasing(i) for i in insts]
            ffd_fe = sum(r.frac_excess for r in ffd_vals) / len(ffd_vals)
        else:
            ffd_fe = 0.0
        row += f"{ffd_fe:>12.4f}"
        print(row)

    print("\nLower is better. best_fit and ffd are the strongest baselines to beat.")


if __name__ == "__main__":
    main()
