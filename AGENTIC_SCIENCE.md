# Science-Gym — A Trainable Agentic Scientific Research System

> Vision doc. Evolve **Lumi** (a prompted multi-agent research system) into a
> **domain-pluggable, RL-trainable** agentic scientist: one shared agentic backbone
> that can be trained for *any* science area by plugging in a domain pack (corpus +
> tools + verifier). Reuses the GSPO/DAPO/verl RLVR stack from **Physics-R1**.
>
> Companion: `AGENTIC_RL.md` (the Lumi-RE coding wedge) becomes **domain pack (B)**
> of this larger system.

---

## 0. The thesis in one sentence

FunSearch / AlphaEvolve / AlphaProof showed that **verifiable-reward RL beats
prompting** inside narrow scientific domains. Science-Gym generalizes that pattern
**across sciences** via a single pluggable verifier interface — so the same agentic
policy becomes trainable for physics, chemistry, ML, math, … by swapping the pack.

This is the trainable counterpart to *prompted* systems like Google's AI co-scientist
or Sakana's AI Scientist: same multi-agent shape, but the policy is **learned against
verifiable rewards**, not hand-prompted.

## 1. The one hard problem: verifiable reward for open-ended science

Open-ended research isn't directly gradeable. But research **decomposes** into
sub-problems that are, and there is a clean taxonomy of reward sources:

1. **Simulation-as-oracle** *(most scalable)* — predict → run a simulator → score.
   Physics engines, MD/DFT, CFD, circuit/climate/epidemiology sims. Cheap, abundant,
   domain-swappable.
2. **Held-out empirical / database truth** — Materials Project, PDB, PubChem, reaction
   yields, gene expression. Predict a measured quantity.
3. **Formal verification** — Lean/Coq for math; symbolic-consistency checks. The
   Physics-R1 dimensional / conservation checkers generalize here.
4. **Reproduction reward** *(the killer source)* — re-derive a *known published*
   headline number from the method description. The literature becomes a giant
   self-supervised label set: abundant **and** verifiable.
5. **Outcome→process bootstrapping** — learn a process reward model from which
   intermediate steps led to verifiable success (long-horizon credit assignment).
6. **Judge / rubric** — only for the irreducibly soft parts (writing, novelty), used
   sparingly with reward-hacking guards.

## 2. Architecture — the domain-pluggable "Science Gym"

Generality = **shared backbone + swappable pack**. Every science area implements one
contract:

```
ScienceDomain:
  corpus()        # literature + dataset retrieval index
  task_sampler()  # curriculum of research tasks by difficulty
  tools()         # simulators, CAS, RDKit, Lean, code-exec, lab/DB APIs
  verifier()      # the oracle -> scalar reward (taxonomy in section 1)
  rubric_judge()  # soft fallback for un-verifiable outputs
```

One agentic policy learns the **transferable meta-skills** — decompose, retrieve,
hypothesize → design → run → verify, self-check, know-when-to-stop. The
domain-specific part is *only* the pack. "Train it for a new science" literally means:
**ship a pack + run domain RLVR on the shared backbone.**

## 3. Structure — hierarchical agentic RL (Lumi's cast becomes the policy hierarchy)

- **Commander** — high-level manager policy: pick pipeline stage, allocate specialist,
  stop/continue, spend budget. Reward = sparse final outcome + dense verified subgoals.
- **Specialists** — low-level skill policies, each with a *local* verifiable reward:
  - **Scout** (retrieval) — reward = citation grounding / recall against held-out refs
  - **Architect** (experiment design) — reward = does the design actually test the
    hypothesis (checked in simulation)
  - **Coder** (research engineering) — reward = tests pass / reproduction (the
    `AGENTIC_RL.md` Lumi-RE wedge)
  - **Datasmith** (data) — reward = data-validity / leakage checks
- Train workers on local rewards first, then the manager on top (HiPER-style options /
  manager-worker), or jointly.

## 4. How "trainable for any science area" is *proven*, not asserted

- Pretrain the backbone on **cheap-verifier domains** (code, math, sim-physics) that
  teach the research loop.
- Then show it **adapts to a held-out science faster than from-scratch**.
- That **transfer curve is the whole pitch in one figure**: shared agentic skills
  transfer; only the pack changes.

## 5. Self-improving curriculum

Mine the literature to auto-generate **reproduction tasks with known answers** → the
system grades itself, climbs difficulty, then attempts genuine extensions. A
verifiable *floor* under open-ended self-play — the scalable-oversight story.

## 6. Reward hacking at science scale *(red-team track — high interview value)*

Failure modes a serious system must probe and harden against: p-hacking / metric
gaming, cherry-picking favorable sims, hallucinated citations, fitting-the-verifier
rather than the science, reward-model exploitation. Ship an exploit probe suite +
hardened verifiers + incidence-over-training curves.

## 7. Honest framing

A fully general "AI scientist that learns any field" is an **open frontier problem**,
not a 3-month build. What is *hireable and finishable* is the credible wedge that
proves the architecture:

> the domain-pluggable RL environment + the verifiable-reward taxonomy + **2 domains
> sharing one trained backbone** + a **transfer/generalization result** + a
> **reward-hacking red-team**.

## 8. MVP

1. Define the `ScienceDomain` interface + multi-turn env loop (reuse verl, GSPO/DAPO,
   observation-token masking from Physics-R1).
2. Instantiate **pack (A) Physics** — reuse `reward/reward_physics.py` verifiers + a
   sympy/physics simulator + reproduction tasks from the olympiad corpus.
3. Instantiate **pack (B) Code/ML-RE** — the Lumi-RE-Gym harness (`AGENTIC_RL.md`).
4. Train **one shared backbone** with multi-turn GRPO via reproduction + test rewards.
5. Demo **transfer to pack (C) chemistry** (RDKit property prediction): does the shared
   backbone adapt faster than scratch? That N=2→3 result *is* "trainable for any
   science area," demonstrated.

**Stretch:** hierarchical Commander; learned process-reward model; literature-mined
auto-curriculum; more packs (biology, materials, math/Lean).

## 9. Hiring narrative

> *"I built an autonomous multi-agent research system (Lumi, prompted), and shipped
> single-turn RLVR with GSPO+DAPO (Physics-R1). Science-Gym makes the system
> **learned**: a domain-pluggable agentic-RL environment where one backbone trains
> against verifiable rewards (simulation, reproduction, formal checks) and **transfers
> across sciences** — with a reward-hacking red-team. It generalizes the
> FunSearch/AlphaProof verifiable-reward recipe from one domain to many."*

Hits every frontier thread at once: agentic RL (hierarchical, long-horizon), RLVR +
reward hacking + scalable oversight, AI-for-science, and real ML-systems
(verl/vLLM/async rollouts).
