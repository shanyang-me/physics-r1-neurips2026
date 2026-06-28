# CS-Discover — Data, Eval & Success Criteria (Experiment Design)

> The foundation doc for `CS_DISCOVERY.md`. For an RLVR-discovery system, **data +
> eval + success criteria *are* the experiment.** Defined and pre-registered here
> *before* any training, so the build conforms to the measurement, not vice versa.

---

## 0. Two framing insights

1. **The verifier IS the dataset.** There are no labels — reward quality = verifier
   quality. A buggy or exploitable verifier silently invalidates every result.
   Building + adversarially testing the verifier suite is the core "data" work.
2. **Two levels of generalization, two splits:**
   - **L1 — program → held-out instances.** The *discovered program* must generalize
     to instances it never saw during search.
   - **L2 — agent → held-out problems.** The *RL-trained agent* must do well on
     *problems* not in its training set. L2 is the entire "learns to discover /
     amortizes search" claim and needs its own train-problems / test-problems split.

## 1. Data — a verifiable task suite (not a labeled corpus)

- **Problem families** with airtight programmatic verifiers + a **known baseline /
  SOTA** that defines "improvement."
  - **MVP: online bin-packing** (FunSearch flagship; CPU-only). Discovery = a
    priority/scoring program for bin choice. Baselines: first-fit, best-fit,
    first-fit-decreasing; reference: the FunSearch heuristic.
  - Others: cap-set / admissible-set construction (pure verification), sorting
    networks (correctness + comparator count), SAT/TSP heuristics (standard
    benchmarks + generators), KernelBench (stretch, GPU).
- **Instance generators + train/val/test splits** (by seed). The program is the
  artifact; instances are the eval substrate. Programs are scored on **held-out
  instances** never seen during search.
- **OOD instance shifts** — train n=50, test n=500 (and shifted size distributions):
  does the heuristic generalize across scale/distribution?
- **Train-problems / test-problems split** — for the L2 transfer claim.
- **Known-answer reproduction set** — validates the verifier, calibrates the agent,
  and measures the **memorization floor** (instant known-SOTA emission ⇒ suspect
  leakage).
- **Contamination control** — novel / perturbed instances + the repo's `audit/`
  pipeline as a novelty gate, so "discovery" ≠ retrieval.
- **Baselines as data** — best-known human heuristic; frozen-LLM single-shot;
  **frozen-LLM + evolutionary search (= FunSearch)**; plus RL ablations.

## 2. Eval

### Metrics

| Axis | Metrics |
| ---- | ------- |
| **Discovery** | improvement Δ over best-known on held-out; strict beat-rate; **verified-correctness rate** (hard gate before any Δ counts); **novelty-pass rate** (audit) |
| **Efficiency** *(the thesis)* | improvement **vs verifier-call / token budget** — RL should reach a quality bar with *fewer* candidate evaluations than FunSearch |
| **Generalization** | L1 held-out-instance + OOD; **L2 held-out-problem transfer** |
| **Integrity** | reward-hack incidence (passes train scorer, fails stricter independent / held-out verifier) tracked over training; verifier-exploit audit; contamination verdicts on winners |

### Baselines (apples-to-apples, matched budget)

1. Best-known human heuristic / SOTA — the bar for "discovery."
2. Frozen-LLM single-shot (no search).
3. **Frozen-LLM + evolutionary search = FunSearch** — the key comparison.
4. RL-trained agent (ours).
5. Ablations: outcome-only vs dense reward; − novelty bonus; − process shaping.

### Hygiene (non-negotiable)

- Frozen verifier across all methods; held-out sets never touched during
  training/search.
- **Matched compute budget**, unit defined explicitly (verifier calls).
- **Multiple seeds, mean ± CI** — RL is high-variance; single-seed numbers are not
  credible.

### The matched-compute honesty trap

Preempt the obvious pushback ("but you paid for RL training"). Report **both**:
- (a) **inference-time** efficiency at matched verifier-call budget — RL should win
  *per new problem*;
- (b) **total cost including training**, amortized over the problem suite — RL wins
  only once the suite is large enough.

Stating this distinction is itself a signal of systems judgment.

## 3. Success criteria (tiered, pre-registered, falsifiable)

- **Tier 0 — Plumbing / sanity.** Env stable; verifier passes an adversarial
  unit-test battery; on a known-answer problem the agent recovers known SOTA
  (validates verifier *and* agent).
- **Tier 1 — RL learns (minimum real result).** Held-out beat-rate / improvement
  strictly rises vs initialization across seeds (significant); trained agent beats
  frozen single-shot; reward-hack incidence stays bounded / falls.
- **Tier 2 — Thesis holds (hireable result).** RL agent **≥ FunSearch at matched
  compute** on held-out problems; agent **transfers to held-out problems (L2)**;
  winners pass the novelty/contamination audit.
- **Tier 3 — Discovery (upside, not required).** ≥1 solution **strictly beats
  best-known** on held-out, verified and audited-as-novel.

### Pre-registration & falsifier

- Fix splits, budget unit, seed count, and statistical test **before** running.
- **Falsifier:** if RL never beats frozen search at matched budget, the amortization
  thesis fails — reporting that cleanly is still a credible (negative) result.
- **Go/no-go:** a Tier-1 failure means the env/reward/verifier is broken, not the
  idea — fix the harness before scaling.

## 4. MVP instantiation (bin-packing, concrete)

- **Instances:** item-size streams (uniform + Weibull), capacities fixed; splits by
  seed; sizes {50, 100} train / {500} OOD test.
- **Discovery artifact:** a `priority(item, bins)` program selecting the bin.
- **Verifier:** deterministic exact packing simulation → bin count vs lower bound
  (⌈Σsizes / capacity⌉); airtight = exhaustive sim, no sampling.
- **Primary metric:** mean fractional excess over lower bound on held-out instances.
- **Headline figure:** improvement vs verifier-call budget, RL agent vs FunSearch,
  ≥5 seeds, mean ± CI.
- **Integrity panel:** hack-incidence curve + novelty-audit table for winners.
