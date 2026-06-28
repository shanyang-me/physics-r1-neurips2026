# Lumi-RE — Turning the Lumi Research System into an Agentic RL Prototype

> Design / proposal doc. Goal: evolve **Lumi** (a prompted multi-agent research
> orchestration system) into an **agentic RL** prototype where one agent's policy is
> *learned* via multi-turn RLVR — reusing the GSPO/DAPO/verl stack already shipped in
> **Physics-R1**. Intended as a frontier-lab hiring artifact aligned to the RL Canon
> plan (M7: agentic RL) and the Codex RE / RL-for-code endgame.

---

## 1. The two assets this builds on

**Lumi Research Manager** — a Next.js multi-agent research platform with a
`lumi-research` MCP + Notion-backed projects DB. Role-prompted agents — Scout
(literature), Theorist (hypotheses), Architect (experiment design), Coder
(implementation), Datasmith (data), Commander (coordinator), Writer/Planner — run
turn-based "meetings" through a research pipeline stage machine
(DEFINE→SURVEY→PLAN→HYPOTHESIZE→DESIGN→IMPLEMENT→EXECUTE→WRITE), writing inventory
back to the DB. Human-in-the-loop, permission-gated.

> **Lumi today is a *prompted* harness. Nothing is trained.** The agents are LLMs
> with roles and tools.

**Physics-R1** (`physics-r1-neurips2026`) — a shipped single-turn RLVR pipeline:
5-component verifiable reward (`reward/reward_physics.py`), contamination audit,
verl GRPO/GSPO + DAPO. This is the **reusable RL stack**, not the thing being
converted.

## 2. The core insight

Lumi already *is* the three pillars of agentic RL — environment, reward signal,
policy — in hand-written form. Turning it into an agentic RL prototype =
**make one agent's policy learned via multi-turn RLVR.** That single move:

- ships **M7** of the RL Canon plan ("extend a GRPO loop to a multi-turn tool-use
  environment"),
- sets up the **Codex RE / RL-for-code** specialization directly,
- fuses the two real strengths — agentic systems + LLM-RL — into one artifact,
- stays *finishable* by scoping to a **verifiable** slice.

## 3. Headline prototype — **Lumi-RE-Gym**

Don't RL-train the whole multi-agent system (open-ended research isn't gradeable).
Carve out the **Coder / research-engineer** agent: RE tasks are *verifiable* (tests
pass, metric improves, experiment reproduces) — exactly the Codex RE setting.

- **Task distribution** — real RE tasks mined from Lumi's own project history +
  synthesized: "make this test pass", "implement this eval harness", "reproduce
  metric X", "fix this bug". Self-bootstrapping: Lumi generates its own curriculum.
- **Environment** — sandboxed repo. Observation = repo state + task spec + tool
  outputs. Tools = shell / file-edit / run-tests / search / `lumi-research` MCP.
  Episode = multi-turn until `submit` or step/cost budget.
- **Reward** — verifiable outcome (tests pass / metric delta / experiment completes)
  + light process shaping (valid tool calls, partial test pass) − penalties (reward
  hacking, budget overrun).
- **RL** — multi-turn GRPO/GSPO, **observation-token masking** (train only on policy
  tokens), trajectory-level advantage, async rollouts. Reuse the Physics-R1 verl
  recipe. Start with a 3–7B code model for reproducibility.

## 4. Three frontier-signal angles (go deep on 1–2)

1. **Long-horizon credit assignment** — outcome-only vs process vs turn-level
   rewards. *The* M7 question and a standard interview probe.
2. **Reward hacking in RE agents** *(highest hiring value)* — the canonical
   SWE-agent failure: deleting/weakening the failing test, hardcoding expected
   output, overfitting the metric. Build an exploit probe + hardened verifier +
   incidence-over-training curve. Gold for an Anthropic/OpenAI interview.
3. **Hierarchical manager-worker RL** — train **Commander** as a meta-policy routing
   to sub-agents (options framework / HiPER). Higher ceiling, higher risk.

## 5. Scoping — the killer is over-scoping

**MVP (finishable):** one agent (Coder), one verifiable task family, sandboxed repo,
~100–300 tasks, 3–7B model, 2–3 tools, terminal reward + reward-hack probe.
Deliverable: training curve (resolve rate) + reward-hack incidence + short writeup.

**Stretch:** process rewards; hierarchical Commander; auto-curriculum (Lumi proposes
harder tasks); a Scout/retrieval-RL variant (reward = citation grounding / answer
faithfulness via judge).

**Honest risks:**
- Lumi's "experiments" are partly DB records, not real artifacts — you must build a
  *real* executable sandbox + task harness for verifiable reward.
- Compute for multi-turn rollouts.
- Resist the urge to train the full multi-agent system.

## 6. Portfolio narrative

The thread is the value, and it's already coherent:

> *"I built an autonomous multi-agent research system (Lumi, prompted). I shipped
> single-turn RLVR with GSPO+DAPO (Physics-R1). Lumi-RE makes the research-engineer
> agent **learned** — multi-turn RLVR on verifiable coding tasks, reusing that stack
> — with a reward-hacking red-team. That is the Physics-R1 → RL-for-code → Codex RE
> arc."*

Maps to hiring threads: agents/tool-use RL (§3), RLVR/reward-hacking/scalable
oversight (§4), and ML-systems (reusing real verl/vLLM infra).

## 7. Suggested first steps

1. Build a minimal sandboxed RE task harness: repo snapshot + task spec + hidden
   tests + a deterministic verifier. Seed ~30 tasks from Lumi history.
2. Stand up the multi-turn env loop with `shell` + `edit` + `run_tests` tools.
3. Write the reward-hack probe (test-deletion / output-hardcoding) and record
   baseline incidence — cheapest high-signal result.
4. Wire multi-turn GRPO into verl with observation masking; train the MVP; plot
   resolve rate + hack incidence.
