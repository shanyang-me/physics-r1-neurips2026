"""Build a DPO/SFT preference dataset from proposer samples on a domain.

    # cheap plumbing check (no model):
    python -m cs_discover.rl.build_dataset --domain capset --n 4 --proposer mutation \\
        --contexts 8 --samples 5 --out-dir data_rl

    # real preference data from the Claude proposer (each context = several calls):
    python -m cs_discover.rl.build_dataset --domain capset --n 5 --proposer claude \\
        --contexts 20 --samples 6 --workers 4 --out-dir data_rl
"""
from __future__ import annotations

import argparse
import os

from ..domains import make_domain
from ..search import MutationProposer, LLMProposer, make_claude_call_fn
from .collect import collect_preferences
from .dataset import write_dpo_jsonl, write_sft_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="capset", choices=["binpacking", "capset"])
    ap.add_argument("--proposer", default="mutation", choices=["mutation", "claude"])
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dist", default="uniform", choices=["uniform", "weibull"])
    ap.add_argument("--contexts", type=int, default=8)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data_rl")
    args = ap.parse_args()

    kwargs = {"n": args.n} if args.domain == "capset" else {"dist": args.dist}
    domain = make_domain(args.domain, **kwargs)
    proposer = (
        LLMProposer(make_claude_call_fn(model=args.model))
        if args.proposer == "claude" else MutationProposer()
    )

    pairs = collect_preferences(
        domain, proposer, n_contexts=args.contexts,
        samples_per_context=args.samples, seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.domain}_{args.proposer}"
    n_dpo = write_dpo_jsonl(pairs, os.path.join(args.out_dir, f"{tag}_dpo.jsonl"))
    n_sft = write_sft_jsonl(pairs, os.path.join(args.out_dir, f"{tag}_sft.jsonl"))
    print(f"collected {len(pairs)} preference pairs -> {args.out_dir}/{tag}_dpo.jsonl "
          f"({n_dpo} DPO, {n_sft} SFT)")


if __name__ == "__main__":
    main()
