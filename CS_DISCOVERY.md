# CS-Discover — An RL-Trained Agent that Discovers New Things in Computer Science

> Recommended concretization of the Science-Gym vision (`AGENTIC_SCIENCE.md`),
> specialized to the one domain where the verifier is free: **computer science**.
> Build an agentic policy that proposes → tests → refines candidate solutions to
> verifiable CS problems, **RL-trained on the discovery signal itself**. Reuses the
> GSPO/DAPO/verl RLVR stack and the contamination audit from this repo.

---

## 0. Thesis

> **"FunSearch, but the proposer learns" is no longer the contribution — it is
> validated prior art.** As of mid-2026 (see `LANDSCAPE.md`), **EvoTune** (EPFL, Apr
> 2025) already RL-fine-tunes the proposer's weights from discovery signal and beats
> FunSearch on bin-packing/TSP; **ThetaEvolve** (Nov 2025) does test-time RL on the
> proposer. So the bet is *de-risked, not novel*. CS-Discover's contribution is the
> **conjunction nobody has shipped**: a discovery agent that is (1) general-purpose,
> (2) beyond toy combinatorial tasks, (3) RL-trained-proposer, (4) verifiable-reward,
> (5) **novelty/contamination-audited inside the reward loop**, and (6)
> **cross-problem generalizing**. Existing systems have at most a few of these.

**Lead with the two least-solved pieces, which this repo is uniquely set up for:**
- **(6) cross-problem transfer** — does the RL-trained proposer get better at
  *discovering on held-out problems* (the L2 split in `CS_DISCOVERY_EVAL.md`)? No
  system has shown this; EvoTune/ThetaEvolve are per-problem.
- **(5) novelty audit + hardened verifier in the loop** — reuse this repo's `audit/`
  as a novelty gate. The 2026 reward-hacking literature ([LLMs Gaming Verifiers],
  [Fuzzing RLVR Verifiers]) makes verifier integrity + contamination control the
  credibility moat the AI-scientist field has failed to clear.

CS is the right target because the **artifact and its verifier share one medium**:
code executes, proofs check, benchmarks self-score. Verifiable reward is native and
abundant — no LLM judge required. This is exactly why every real machine-discovery
result lives in CS-adjacent domains. Keep verifiers cheap and hard-to-game
(CUDA-L1 / KernelBench / SWE-Gym style) to hold integrity + compute tractable.

## 1. What "discovery" means here

Discovery = **beat the best-known solution on a problem with a cheap verifier and a
huge search space.** Taxonomy of targets:

1. **Algorithmic** — faster/smaller algorithms (cf. AlphaDev sort, AlphaTensor
   matrix-mult). Reward = correct × fast.
2. **Combinatorial constructions / open math-CS** — better bounds (cap sets, Ramsey,
   packing; cf. FunSearch). Reward = valid construction × beats SOTA.
3. **Heuristics for NP-hard** — SAT, bin-packing, TSP, scheduling. Reward = solution
   quality on held-out instances.
4. **Program / kernel optimization** — faster correct GPU kernels (KernelBench),
   compiler passes, circuits. Reward = correct × benchmarked speedup.
5. **Empirical ML / systems** — better architectures/optimizers/data structures.
   Reward = held-out metric (this is the Lumi-RE coder agent, generalized).

Unifying structure: **search over programs/constructions against a verifiable
objective.**

## 2. Positioning (the gap this fills)

| System | General | Proposer learns | Beyond toy tasks | Novelty-audited | Cross-problem transfer |
| ------ | :-: | :-: | :-: | :-: | :-: |
| AlphaTensor / AlphaDev | ✗ | ✓ (RL) | ✓ | ✗ | ✗ |
| FunSearch / AlphaEvolve | ✓ | ✗ (frozen) | ✓ | ✗ | ✗ |
| **EvoTune / ThetaEvolve** | ~ | **✓ (RL)** | ✗ (toy/combinatorial) | ✗ | ✗ (per-problem) |
| AI-Scientist / co-scientist | ✓ | ✗ | ✓ | ✗ (prime critique target) | ✗ |
| **CS-Discover** | ✓ | ✓ | → goal | **✓ (`audit/` in loop)** | **✓ (the L2 claim)** |

EvoTune already owns the "proposer learns" cell — so CS-Discover's open cells are the
last two columns: **novelty-audited-in-the-loop** and **cross-problem transfer**.
Those are the contribution; everything left of them is reproduction of validated work.

## 3. The two crux problems

### 3a. Discovery vs memorization *(we already own the tooling)*

The killer question for any AI-discovery claim: *did it find this or recall it?*
Requires held-out open problems **and a contamination/novelty audit**. This repo's
`audit/` (5-gram Jaccard + mxbai embedding cosine) is repurposed as the **novelty
verifier** — a candidate solution must be both correct *and* audited as
non-memorized to count. Non-obvious, high-credibility reuse.

### 3b. Reward hacking / verifier integrity *(red-team track)*

In a discovery system the verifier *is* the reward, so the agent will try to game it:
hardcoding instances, exploiting evaluator bugs, overfitting the test set, returning
solutions that pass the checker but don't generalize. An airtight verifier + an
exploit probe suite + incidence-over-training curves are essential and
interview-valuable.

## 4. Reward design

```
r = correctness_gate(candidate)                       # hard gate: verified execution / proof
    * improvement_over_best_known(candidate)          # dense discovery signal
    + novelty_bonus(candidate)                        # diversity vs known + audit pass
    - hacking_penalty(candidate)                      # verifier-exploit detectors
```

Sparse-reward mitigations: partial-correctness shaping, progress toward a better
objective value, diversity bonus to avoid collapse onto a memorized answer.

## 5. Agent loop

Multi-turn, tool-using: read problem → propose candidate (program/construction) →
run verifier (execute / check / benchmark) → observe score + profile → refine →
submit best. Tools: code-exec sandbox, profiler, the verifier, the novelty audit,
literature search. Train with multi-turn GRPO/GSPO, observation-token masking,
trajectory advantage — reuse the Physics-R1 verl recipe.

## 6. The headline ablation (this is the paper figure)

**RL-trained agent vs frozen-LLM evolutionary search (FunSearch baseline), matched
compute.** If training the proposer beats searching with a frozen one, you've shown
RL **amortizes scientific search**. That comparison is the contribution.

## 7. MVP (cheap, CPU-runnable, finishable)

1. Pick one verifiable domain with a known SOTA and an airtight verifier. First
   target: **FunSearch-style heuristic discovery (online bin-packing or a small
   combinatorial construction)** — CPU-only, clean "beat best-known heuristic"
   signal, no GPU cluster.
2. Build the env: problem spec + candidate executor + held-out instance scorer +
   novelty audit (reuse `audit/`).
3. Train a small code model with multi-turn GRPO to propose → test → refine.
4. Results: beats-baseline curve, the **RL-vs-frozen-search ablation**, a
   contamination/novelty report, and a reward-hacking incidence curve.

**Stretch:** KernelBench (GPU kernel discovery); competitive-programming with hidden
tests; AlphaProof-style Lean constructions; the hierarchical Lumi orchestration
(Commander routes to specialist discoverers) on top.

## 8. Honest framing

A *genuinely novel* discovery is rare and not guaranteable on a prototype timeline.
What is demonstrable — and what makes the artifact hireable — is the **mechanism**:
the RL agent finds verified, audited-as-non-memorized solutions that beat strong
baselines *and* frozen-LLM search, with verifier integrity defended by a red-team. An
actual new construction is upside, not the deliverable.

## 9. Hiring narrative

> *"I built an autonomous multi-agent research system (Lumi), shipped single-turn
> RLVR with GSPO+DAPO (Physics-R1), and built a contamination audit for genuine
> novelty. CS-Discover fuses them: a general agentic policy **RL-trained to discover**
> verifiable CS results — algorithms, constructions, heuristics, kernels — that
> amortizes search into its weights (vs FunSearch's frozen LLM), with my audit
> pipeline as the novelty verifier and a reward-hacking red-team on the scoring
> loop."*

Hits every frontier thread: agentic RL + long-horizon credit assignment, RLVR +
reward hacking + verifier integrity, AI-for-discovery, contamination/eval rigor, and
real ML-systems (verl/vLLM/async rollouts) — and it *is* the Codex-RE / RL-for-code
story.
