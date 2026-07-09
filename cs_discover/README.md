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
search/            propose -> test -> refine loop (workers + candidate cache)
  proposer.py        Proposer ABC; MutationProposer (no-LLM), LLMProposer (drop-in)
  claude_backend.py  LLMProposer backend on the Claude subscription (frozen baseline)
  evolution.py       evolutionary search; budget = proposer calls
novelty.py         "discovered or recalled?" gate (5-gram Jaccard, mirrors audit/)
rl/                make the proposer LEARN (EvoTune-style DPO) — see rl/README.md
  collect.py         completions -> (chosen > rejected) preference pairs
  dataset.py         DPO / SFT JSONL; build_dataset.py CLI
  train_dpo.py       GPU DPO fine-tune of an open-weights proposer (trl)
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

### LLMProposer (Claude) closes that gap — live result

Swapping the frozen model-free proposer for the Claude-backed `LLMProposer` (the
frozen-LLM FunSearch baseline) **breaks the 2^n ceiling on the first try**:

| proposer | cap-set n=4 | cap-set n=5 |
| -------- | :---------: | :---------: |
| separable ceiling (2^n) / model-free control | 16 | 32 |
| **LLMProposer (Claude, frozen LLM)** | **20** (optimal) | **40** (89% of opt) |
| known optimum | 20 | 45 |

- **n=4:** Claude reached the **proven maximum (20)** in 10 successful evaluations
  (budget 12), using min digit count, variance, pairwise modular interactions →
  `discovered/capset_n4_claude_size20.py`.
- **n=5:** Claude reached **40** (+8 over the 2^n=32 ceiling, ~89% of the optimum 45)
  in 14 successful evals with **4 concurrent workers**, novelty 0.978. The program is
  far richer (F_3 quadratic/cubic forms, higher moments, F_3* structure) →
  `discovered/capset_n5_claude_size40.py`.

Both verified as valid caps. Reproduce:

```bash
python -m cs_discover.run_search --domain capset --n 4 --proposer claude --budget 12
python -m cs_discover.run_search --domain capset --n 5 --proposer claude --budget 16 --workers 4 --novelty
```

This is the proposer-quality axis demonstrated on our own substrate, and it **scales
with n** — the harder the problem, the wider the gap over the separable ceiling. Honest
caveat: these optima are *known*, so this proves the *mechanism*, not a novel
discovery; a real discovery claim needs open-n (n≥7) plus the novelty audit (now
wired). Concurrency (`--workers`) addresses the earlier ~22-min sequential wall-clock.

## Not yet built (next phases)

- Execute the RL loop end-to-end on a GPU: collect Claude preference data →
  `train_dpo.py` an open proposer → serve → run the matched-budget ablation. The
  pipeline + trainer are in `rl/` (GPU-gated); only execution remains.
- Iterate collect→train→search rounds (EvoTune's alternation).
- A 3rd family held out for the cross-problem-transfer (L2) measurement.
- Larger cap-set n (open optima, n≥7) for a genuine discovery claim.

## Done

- `LLMProposer` on the Claude subscription — the frozen-LLM FunSearch baseline. Reaches
  optimal (20) at n=4 and 40 (89% of optimum) at n=5, beating the model-free control.
- Scaling: `--workers` concurrency + source-keyed candidate cache.
- Novelty gate (`novelty.py`) — reward-loop-ready "discovered or recalled?" scorer.
- RL preference-data pipeline (`rl/`, CPU-tested) + GPU-ready DPO trainer + the
  pre-registered ablation protocol (`rl/README.md`).

## Status

Tier 0 (substrate) complete + the search harness runs end-to-end on two families with
no model dependency. See `CS_DISCOVERY_EVAL.md` §3 for the tiered success criteria.
