# cs_discover

Prototype for **CS-Discover** — an RL-trained agentic discovery engine for verifiable
computer-science problems. See the design docs in the repo root:

- `CS_DISCOVERY.md` — the thesis ("FunSearch, but the proposer learns")
- `CS_DISCOVERY_EVAL.md` — data / eval / success criteria (pre-registered)
- `LUMI_INTEGRATION.md` — how this plugs into the Lumi Research Manager

## Build order (the verifier IS the dataset)

This package is built bottom-up so the **scoring substrate is trustworthy before any
RL**:

1. **`binpacking/instances.py`** — seeded instance generators + train/val/test/OOD
   splits (contamination-free by construction).
2. **`binpacking/verifier.py`** — deterministic, *airtight* online-bin-packing
   simulator. The candidate only supplies a per-bin **priority**; the verifier itself
   enforces capacity feasibility, so a candidate **cannot** cheat by overfilling.
3. **`binpacking/baselines.py`** — first-fit, best-fit, worst-fit (priority form) +
   first-fit-decreasing (offline). These define "the bar."
4. **`tests/test_verifier.py`** — adversarial battery proving the verifier is
   airtight (rejects infeasible placement, survives crashing/NaN candidates, lower
   bound correct).

Not yet built (next phases): the multi-turn propose→test→refine agent loop, the
frozen-LLM evolutionary-search baseline (FunSearch), and the RL trainer (verl/GRPO).

## Quickstart

```bash
python -m cs_discover.run_baselines          # print the baseline bar on all splits
python cs_discover/tests/test_verifier.py    # run the airtightness battery
```

## Status

Tier 0 (plumbing/sanity) substrate only — see `CS_DISCOVERY_EVAL.md` §3 for the
tiered success criteria this is the foundation for.
