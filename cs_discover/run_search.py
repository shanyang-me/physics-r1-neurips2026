"""Run propose->test->refine discovery search on a domain (frozen MutationProposer
baseline) and report best-found vs seed vs the standard baseline on held-out splits.

Usage:
    python -m cs_discover.run_search --domain binpacking --budget 300
    python -m cs_discover.run_search --domain capset --n 4 --budget 300
"""
from __future__ import annotations

import argparse

from cs_discover.domains import make_domain
from cs_discover.search import MutationProposer, LLMProposer, evolve, make_claude_call_fn
from cs_discover.novelty import novelty_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="binpacking", choices=["binpacking", "capset"])
    ap.add_argument("--proposer", default="mutation", choices=["mutation", "claude"])
    ap.add_argument("--model", default="claude-sonnet-4-5", help="model for --proposer claude")
    ap.add_argument("--timeout", type=int, default=120, help="per-call timeout (claude)")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1, help="concurrent proposer calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4, help="cap-set dimension")
    ap.add_argument("--dist", default="uniform", choices=["uniform", "weibull"])
    ap.add_argument("--novelty", action="store_true", help="report novelty of best vs baselines")
    args = ap.parse_args()

    kwargs = {"n": args.n} if args.domain == "capset" else {"dist": args.dist}
    domain = make_domain(args.domain, **kwargs)

    if args.proposer == "claude":
        proposer = LLMProposer(make_claude_call_fn(model=args.model, timeout=args.timeout))
    else:
        proposer = MutationProposer()

    res = evolve(domain, proposer, budget=args.budget, seed=args.seed, workers=args.workers)

    print(f"domain         : {domain.name}")
    print(f"proposer calls : {res.n_evaluated}  (verified: {res.n_verified}, "
          f"cache hits: {res.n_cache_hits}, failed: {res.n_failed})")
    print(f"seed score     : {res.seed_score:+.5f}")
    print(f"baseline score : {res.baseline_score:+.5f}")
    print(f"best (train)   : {res.best.score:+.5f}")
    print("held-out report (higher is better):")
    for split, sc in res.report.items():
        tag = "  >= baseline" if sc >= res.baseline_score else "  < baseline"
        print(f"  {split:<6}: {sc:+.5f}{tag}")
    if args.novelty:
        nov = novelty_score(res.best.source, domain.reference_sources())
        print(f"novelty vs baselines : {nov:.3f}  (1.0 = novel, 0.0 = duplicate)")
    print("\nbest program:\n")
    print(res.best.source)


if __name__ == "__main__":
    main()
