# cs_discover

Prototype for **CS-Discover** — an RL-trained agentic discovery engine for verifiable
computer-science problems. See the design docs in the repo root:

- `CS_DISCOVERY.md` — the thesis + positioning (vs FunSearch / AlphaEvolve / EvoTune)
- `CS_DISCOVERY_EVAL.md` — data / eval / success criteria (pre-registered)
- `LANDSCAPE.md` — cited landscape (late-2025 → mid-2026)
- `LUMI_INTEGRATION.md` — how this plugs into the Lumi Research Manager

## Layout

```
binpacking/        airtight online bin-packing verifier + instances + baselines
domains/           the pluggable Domain contract (enables cross-problem transfer / L2)
  base.py            Domain ABC, restricted code execution, float-mutation operator
  binpacking_domain  bin-packing as a Domain (EvoTune-comparable family)
  capset.py          cap-set verifier + greedy constructor (F_3^n)
  capset_domain      cap-set as a Domain (2nd family, for the transfer axis)
search/            propose -> test -> refine loop (the RL trainer plugs in here)
  proposer.py        Proposer ABC; MutationProposer (no-LLM), LLMProposer (drop-in)
  evolution.py       evolutionary search; budget = candidate evaluations
run_baselines.py   print the bin-packing baseline bar
run_search.py      run discovery search on a domain, report vs baseline on held-out
tests/             adversarial verifier battery + end-to-end search tests
```

## Build order (the verifier IS the dataset)

Built bottom-up so the scoring substrate is trustworthy before any RL:
1. **Verifiers first** (`binpacking/verifier.py`, `domains/capset.py`) — airtight;
   the candidate supplies only a `priority(...)`, the verifier enforces all
   constraints, so scores cannot be hacked.
2. **Domains** wrap each family behind one `Domain` interface — the precondition for
   the cross-problem-transfer (L2) claim.
3. **Search loop** runs propose→test→refine with a swappable proposer. The current
   `MutationProposer` needs no model (it perturbs float coefficients) and is the
   **frozen-proposer control** for the headline RL-vs-frozen ablation.

## Quickstart

```bash
python -m cs_discover.run_baselines               # bin-packing baseline bar
python -m cs_discover.run_search --domain binpacking --budget 400
python -m cs_discover.run_search --domain capset --n 4 --budget 300
python cs_discover/tests/test_verifier.py         # 7/7 airtight bin-packing
python cs_discover/tests/test_capset.py           # 6/6 cap-set verifier
python cs_discover/tests/test_search.py           # 4/4 end-to-end loop
```

## Findings (honest baselines — these motivate the learned proposer)

The model-free `MutationProposer` reaches each domain's standard baseline **but cannot
exceed it**:

- **Bin-packing:** from a trivial first-fit seed, search *rediscovers best-fit* (it
  converges to best-fit-equivalent coefficients) and matches the best-fit baseline on
  held-out test/ood — it does not beat it. A separable linear priority's optimum on
  this distribution is ≈ best-fit.
- **Cap-set:** greedy construction with **any separable priority saturates at exactly
  2^n** (8 / 16 / 32 for n=3/4/5), well below the known maxima (9 / 20 / 45). Tuning
  coefficients — even per-(position,value) weights — never beats 2^n. Beating it
  *requires a non-separable priority function*, which coefficient mutation cannot
  express.

**Why this matters:** the headroom above these baselines (toward FFD/optimal for
bin-packing, toward the true cap maxima) is precisely what a *learned, non-separable*
proposer must claim. This is the FunSearch/EvoTune gap, demonstrated empirically — and
the target the RL trainer is measured against.

## Not yet built (next phases)

- `LLMProposer` backend — wire a model `call_fn` (the frozen-LLM FunSearch baseline).
- RL trainer — fine-tune the proposer on (context, program, reward); compare to the
  frozen proposer at matched evaluation budget (the headline ablation).
- Novelty/contamination audit hook (reuse repo `audit/`) as a reward-loop gate.
- A 3rd family held out for the cross-problem-transfer (L2) measurement.

## Status

Tier 0 (substrate) complete + the search harness runs end-to-end on two families with
no model dependency. See `CS_DISCOVERY_EVAL.md` §3 for the tiered success criteria.
