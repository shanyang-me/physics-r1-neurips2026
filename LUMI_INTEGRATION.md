# Integrating CS-Discover into the Lumi Research Manager

> How the RL-trained discovery engine (`CS_DISCOVERY.md`) lives inside **Lumi** — the
> existing multi-agent research orchestration system. The fit is two-way, and it
> forces the one upgrade Lumi most needs (real execution).

---

## 0. The two-way fit

- **Lumi drives CS-Discover** — the discovery engine is a trainable *skill* one agent
  invokes during the research pipeline.
- **CS-Discover completes Lumi** — Lumi's surrounding agents are what turn a raw
  heuristic into an actual *research finding*: Scout finds the SOTA to beat, Theorist
  frames a scoreable claim, Architect sets the eval protocol, Datasmith builds the
  instance generators + novelty audit, Writer writes it up, Commander gates.

## 1. The gap CS-Discover forces Lumi to close

In the DreamDojo session, Lumi's own Coder audited the repo and found the project
"lives in the research manager database, **not as local code**." Today Lumi *records*
experiments as inert DB rows; it does not *run* them. CS-Discover requires a **real
execution substrate** — which is exactly the missing piece that turns Lumi from
planning-theater into a system that actually does science. Building it is
non-negotiable and is itself a portfolio-worthy contribution.

## 2. Build phases

- **Phase 0 — Standalone engine.** Build CS-Discover as a standalone service (the
  bin-packing MVP from `CS_DISCOVERY_EVAL.md`), decoupled from the web app. Isolates
  the hard RL work from integration risk. This is the `cs_discover/` scaffold.
- **Phase 1 — Give Lumi a real execution substrate.** A sandboxed **Verifier/Runner
  service** + a **discovery-task registry** in the projects DB (problem families,
  splits, baselines, best-known, a leaderboard of discovered programs). Exposed via
  the `lumi-research` MCP as new tools: `run_candidate`, `score_on_heldout`,
  `audit_novelty`, `submit_discovery`, `get_leaderboard`.
- **Phase 2 — Wire it in as the Discoverer agent.** Add a **Discoverer** specialist
  (or upgrade Coder) whose *policy is the RL-trained CS-Discover checkpoint*. Commander
  routes discovery tasks to it; winners + scores + novelty verdicts are written back as
  inventory. This is the **first trained component** in an otherwise-prompted system —
  the concrete "Lumi becomes trainable" milestone.
- **Phase 3 — Close the loop.** Map each pipeline stage to a real verifiable action:
  - SURVEY (Scout) → pull SOTA/baselines (defines the bar) + contamination corpus
  - HYPOTHESIZE (Theorist) → "heuristic family X beats baseline Y" (scoreable)
  - DESIGN (Architect) → the eval protocol (splits, budget, seeds, success tiers)
  - DATA (Datasmith) → instance generators + train/val/test/OOD + novelty audit
  - EXECUTE (Discoverer) → the RL discovery run → leaderboard
  - WRITE (Writer) → auto-report with the RL-vs-FunSearch figure + integrity panel
  - Commander → go/no-go on the success tiers
- **Phase 4 — The flywheel.** Lumi auto-generates discovery tasks (curriculum), runs
  the Discoverer, and the resulting (problem, trajectory, reward) trajectories become
  **training data for the next Discoverer policy**. Lumi is simultaneously the task
  generator and the RL data factory → self-improving loop. *(Stretch: RL-train
  Commander's routing — hierarchical agentic RL.)*

## 3. Integration mechanics

- **Engine** — Python service (verl/vLLM, GPU). Heavy RL training runs offline,
  producing a served checkpoint the Discoverer agent calls.
- **Bridge** — new `lumi-research` MCP tools (above); the prompted agents call cheap
  tools during meetings, training happens out-of-band.
- **DB schema additions** — `DiscoveryTask`, `InstanceSplit`, `Candidate`, `Score`,
  `NoveltyVerdict`, `Leaderboard`.
- **Safety** — keep human-in-the-loop gates on compute-spending actions (consistent
  with Lumi's current permission-gated design).

## 4. Honest scoping

- Only the **Discoverer** is trained at first; the rest of the cast stays prompted
  orchestration. Hierarchical training of Commander is a later stretch.
- The execution substrate (Phase 1) is a real infra dependency Lumi lacks — sequence
  it before any "Lumi is trainable" claims.
- Do not couple the standalone engine (Phase 0) to the web app until Tier-2 success
  criteria hold for the engine alone.

## 5. Why this is a strong hiring artifact

It demonstrates the full stack a frontier lab probes for: an **autonomous research
system** (Lumi), a **trainable verifiable-discovery engine** (CS-Discover, the RLVR +
agentic-RL core), the **systems work** to make experiments real (execution substrate +
MCP + DB), and the **rigor** to defend novelty (contamination audit) and reward
integrity (red-team). The closed loop — *find a verified, novel CS result and document
it end-to-end* — is the AI-for-science thesis made concrete and learnable.
