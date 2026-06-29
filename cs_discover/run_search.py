"""Run propose->test->refine discovery search on a domain (frozen MutationProposer
baseline) and report best-found vs seed vs the standard baseline on held-out splits.

Usage:
    python -m cs_discover.run_search --domain binpacking --budget 300
    python -m cs_discover.run_search --domain capset --n 4 --budget 300
"""
from __future__ import annotations

import argparse

from cs_discover.domains import make_domain
from cs_discover.search import MutationProposer, evolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="binpacking", choices=["binpacking", "capset"])
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4, help="cap-set dimension")
    ap.add_argument("--dist", default="uniform", choices=["uniform", "weibull"])
    args = ap.parse_args()

    kwargs = {"n": args.n} if args.domain == "capset" else {"dist": args.dist}
    domain = make_domain(args.domain, **kwargs)

    res = evolve(domain, MutationProposer(), budget=args.budget, seed=args.seed)

    print(f"domain         : {domain.name}")
    print(f"evaluations    : {res.n_evaluated}  (failed: {res.n_failed})")
    print(f"seed score     : {res.seed_score:+.5f}")
    print(f"baseline score : {res.baseline_score:+.5f}")
    print(f"best (train)   : {res.best.score:+.5f}")
    print("held-out report (higher is better):")
    for split, sc in res.report.items():
        tag = "  >= baseline" if sc >= res.baseline_score else "  < baseline"
        print(f"  {split:<6}: {sc:+.5f}{tag}")
    print("\nbest program:\n")
    print(res.best.source)


if __name__ == "__main__":
    main()
