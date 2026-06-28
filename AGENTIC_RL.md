# PhysGym — Turning Physics-R1 into an Agentic RL Prototype

> Design / proposal doc for evolving this repo from **single-turn RLVR** into an
> **agentic, tool-use RL** prototype. Intended as a frontier-lab hiring artifact:
> small enough to actually finish and run, deep enough to show research taste.

---

## 1. What this repo is today

`physics-r1-neurips2026` is a **single-turn RLVR pipeline** for visual olympiad physics:

- **`reward/reward_physics.py`** — a 5-component dense verifiable reward computed
  *passively* from the final transcript:
  `r = r_ans + r_fmt + r_dim + r_sym + r_cons`, clipped to `[-1, 1]`, with an
  `AUDIT_LAMBDA` contamination down-weighting hook on `r_ans`.
- **`audit/`** — two-stage contamination audit (5-gram Jaccard → mxbai embedding
  cosine) of the train pool against held-out evals.
- **`data/make_splits.py`** — taxonomy-tagged (concept × difficulty × modality)
  train/eval split builder.
- **`eval/` + `judge/`** — vLLM batched eval harnesses + Sonnet LLM-judge
  alignment (strict + liberal).
- Training is **verl GRPO/GSPO**: one rollout → one scalar reward → policy update.

It is a strong RLVR + eval-integrity story. What it is **not** yet: agentic. The
policy never *acts* — it emits one chain of thought and the verifiers grade the
transcript after the fact.

## 2. The core reframe

**The reward components are already verifiers. Agentic RL turns them into tools the
policy calls mid-rollout.**

| Today (passive shaping)                          | Agentic (active tool)                                  |
| ------------------------------------------------ | ------------------------------------------------------ |
| `reward_dimensional()` grades units in final text | `units(expr)` — dimensional check the agent calls      |
| `reward_symbolic()` checks a `\frac` parses        | `cas(expr)` — sympy solve/simplify the agent invokes   |
| `reward_conservation()` penalizes violations       | `check_conservation(...)` queried before submitting    |
| `_base_score()` final grade                        | terminal `submit(answer)` → outcome reward             |

One-sentence pitch: *"We already built the verifiers; the prototype makes them an
interactive environment and learns the tool-use policy with multi-turn RL."* That
narrative demonstrates RLVR fluency, environment design, and credit assignment on a
domain where we already own audited data and trusted evals.

## 3. PhysGym — the environment

A gym-like, multi-turn environment. State = problem (text + optional diagram) +
running scratchpad + tool-call history. The policy interleaves reasoning with tool
calls until it emits a terminal `submit`.

**Tools**

- `python(code)` — sandboxed numpy/scipy/sympy interpreter (subprocess, no net, time + mem cap)
- `cas(expr)` — symbolic solve / simplify / dimensional analysis (sympy)
- `units(expr)` — dimensional consistency (reuse `reward_dimensional` internals)
- `check_conservation(...)` — energy/momentum balance (reuse `reward_conservation`)
- `read_diagram(bbox)` — crop/zoom the problem image (visual physics → visual tool
  use, a strong differentiator)
- `submit(answer)` — terminal action; triggers the verifiable outcome reward

**Reward (trajectory)**

```
R = r_outcome (_base_score on submitted answer)        # terminal, verifiable
  + Σ small process rewards (valid + successful tool calls)
  - penalties (malformed calls, hacking patterns, step budget overrun)
```

**RL changes vs today**

- Multi-turn rollout loop (verl already supports tool-calling agents).
- **Mask tool-output / observation tokens from the loss** (don't train on what the
  env said, only on what the policy generated).
- Trajectory-level advantage (GRPO group over full trajectories); optional
  turn-level credit as a stretch.

## 4. Three frontier-signal research angles (go deep on 1–2)

### 4a. Reward-hacking red-team → hardening  *(highest hiring value)*

The current dense reward is gameable **right now**:

- `reward_symbolic()` (`reward/reward_physics.py:366`) fires on a **single**
  parseable `\frac{a}{b}` *anywhere* in the text → +0.20 for a junk fraction.
- `reward_format()` rewards **any** non-empty `\boxed{}` → +0.10 for an empty-ish box.

A policy can farm **+0.30** with a throwaway fraction and a boxed token, no real
reasoning. Plan: (1) build a probe that exploits this and measure incidence over
training; (2) harden — require the `\frac` to be causally on the solution path, or
replace heuristic shaping with a process verifier; (3) re-measure. Reward-hacking
analysis is exactly what RL/alignment teams live in.

### 4b. Process supervision vs outcome-only

Promote the existing Sonnet judge to a **step-level** verifier and run the clean
ablation: outcome-only vs dense-shaped vs process-supervised. A real learning-curve
comparison is a portfolio-grade result.

### 4c. Self-verification / auto-grading

Use `cas` + a small simulator to *generate* verifiable rewards for problems lacking
gold answers (bootstrap reward from tools). Directly relevant to scalable oversight.

## 5. Scoping — the killer is over-scoping

**MVP (runs on 1 GPU):**
- Text-only subset (defer VL), Qwen2.5-3B / Qwen3-4B.
- 2 tools: `python`, `cas`. Terminal reward + reward-hack probe.
- ~200 problems from the audited pool.
- Deliverable: training curve + reward-hacking case study.

**Stretch:** add VL + `read_diagram`; process rewards; the 3-way ablation; the
self-verification loop.

## 6. Portfolio deliverables

- PhysGym as a clean, standalone, `pip install -e .` package with a gym-style API.
- 1–2 small trained checkpoints (reproducible on modest hardware).
- A 3–5 page tech report with curves: pass@1, tool-success rate, **reward-hack
  incidence over training**.
- The reward-hacking case study as the centerpiece.

## 7. Mapping to hiring threads

- **Agents / tool-use RL** — §3 env design + multi-turn credit assignment.
- **RLVR / reward hacking** — §4a/§4b, verifiable rewards + scalable oversight.
- **Multimodal reasoning** — VL + `read_diagram` visual tool use.

## 8. Suggested first steps

1. Carve out `physgym/` package skeleton: `env.py`, `tools/`, `reward.py`
   (re-exporting `reward_physics.compute_score`).
2. Stand up the `python` + `cas` tools with a hard sandbox.
3. Write the reward-hack probe against current `r_sym`/`r_fmt` and record baseline
   incidence — this is the cheapest high-signal result.
4. Wire the multi-turn loop into verl; train the MVP; plot curves.
