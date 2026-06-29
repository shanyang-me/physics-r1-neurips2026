# Landscape — AI-Driven Discovery & Agentic RL for Discovery (late 2025 → mid-2026)

> Cited landscape review backing `CS_DISCOVERY.md`. Sourcing caveat: arxiv.org,
> openai.com, metr.org, sakana.ai and several primary hosts were proxy-blocked
> (HTTP 403) during research, so preprint-level claims rest on multi-source secondary
> corroboration (Nature, Quanta, DeepMind/Google/FutureHouse blogs, Tao's mathstodon)
> and are flagged where unverified. Re-check primaries before any external writeup.

## The crux up front

The field is dominated by **frozen LLM + evolutionary/search outer loop** (FunSearch,
AlphaEvolve, OpenEvolve, ShinkaEvolve): the proposer's weights are never updated;
"learning" lives in the prompt context + program database. **The pattern CS-Discover
targets — RL-training the proposer on discovery outcomes — now exists, but only at
small scale** (EvoTune, ThetaEvolve), and they beat their frozen-proposer baselines.
No one, as of mid-2026, has fielded a *general-purpose, frontier-scale, RL-trained,
novelty-audited, cross-problem-generalizing* discovery proposer. **That is the white
space — narrowed, not closed.**

## 1. Verifiable program / algorithm discovery

**Frozen-LLM evolutionary (dominant paradigm):**
- [FunSearch](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) (DeepMind, Dec 2023, *Nature*) — LLM + evaluator + evolution over a scoring *function*; new cap-set + bin-packing heuristics. Frozen LLM; evolves one function.
- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind, May 2025; arXiv 2506.13131) — Gemini evolutionary agent editing whole codebases; 48-mult 4×4 complex matmul (first over Strassen's 49 since 1969), ~0.7% Borg compute + ~1% Gemini train-time production wins. **Frozen proposer** (Gemini ensemble, not fine-tuned); closed; compute-heavy.
- [Math discovery at scale](https://arxiv.org/abs/2511.02864) (DeepMind/UCLA/Brown w/ **Terence Tao**, Nov 2025; [Quanta Apr 2026](https://www.quantamagazine.org/the-ai-revolution-in-math-has-arrived-20260413/)) — AlphaEvolve on 67 math problems: improved 23, matched ~36. Still frozen-proposer search; needs expert framing.
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve), [ShinkaEvolve](https://sakana.ai/shinka-evolve/) (Sakana, Sep 2025; arXiv 2509.19349), [DeepEvolve](https://arxiv.org/html/2510.06056v1), [CodeEvolve](https://arxiv.org/abs/2510.14150) — open AlphaEvolve-style reimplementations; sample-efficiency, retrieval, cross-file edits. **All frozen proposers** (ShinkaEvolve's bandit only *selects among* models).

**RL-trained weights, AlphaZero-style (not an LLM proposer):**
- [AlphaTensor](https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/) (2022), [AlphaDev](https://deepmind.google/blog/alphadev-discovers-faster-sorting-algorithms/) (2023, in LLVM libc++), [AlphaProof + AlphaGeometry 2](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/) (IMO-2024 silver; *Nature* Nov 2025). RL on verifiable outcomes, but bespoke games / Lean prover — not a general program-discovery proposer loop; per-problem test-time RL is very compute-heavy.

**★ RL-trained proposer LLMs (direct prior art for our thesis):**
- [EvoTune](https://arxiv.org/abs/2504.05108) (EPFL CLAIRE, Apr 2025) — **the system CS-Discover described.** Alternates evolutionary search with **RL fine-tuning of the proposer's weights (DPO)** using discoveries as reward; explicitly motivated by "FunSearch treats the LLM as a static generator." Beats FunSearch on **bin-packing, TSP, flatpack** with small models (Llama-3.2-1B, Phi-3.5-Mini, Granite-3.1-2B). Limitation: small tasks + small models; per-problem; not frontier-scale.
- [ThetaEvolve](https://arxiv.org/abs/2511.23473) (Nov 2025) — single LLM scaling **in-context + RL at test time** (MAP-Elites); new best-known on circle packing + an autocorrelation inequality, faster than ShinkaEvolve. Preprint; narrow; single-LLM.
- [AlphaResearch](https://arxiv.org/abs/2511.08522) (Nov 2025) — dual-environment propose/verify/optimize. **Unverified** whether proposer is trained vs frozen.

## 2. Autonomous "AI scientist" / end-to-end research agents

- [Sakana AI Scientist v1](https://arxiv.org/abs/2408.06292) (Aug 2024) / [v2](https://arxiv.org/abs/2504.08066) (Apr 2025) — end-to-end ideate→experiment→write→review; v2 passed an **ICLR 2025 workshop** review (paper later withdrawn by agreement, still had errors).
- [Google AI co-scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) (Feb 2025) — multi-agent Gemini hypothesis generator (generate–debate–rank); drug-repurposing + AMR hypotheses. A generator, not end-to-end; lab- not clinically-validated. Closest in spirit to Lumi.
- [FutureHouse Platform](https://www.futurehouse.org/research-announcements/launching-futurehouse-platform-ai-agents) (May 2025), [Robin](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system) (ripasudil dAMD repurposing *hypothesis*; *Nature* status **unverified**), [Edison/Kosmos](https://www.futurehouse.org/research-announcements/announcing-edison-scientific) (Nov 2025, vendor-reported), [Intology Zochi](https://www.intology.ai/blog/zochi-acl) (ACL 2025 main; human manuscript prep — covered skeptically).

**Novelty/quality critiques (the credibility moat):**
- ["Bold Claims, Mixed Results"](https://arxiv.org/abs/2502.14297) (Feb 2025) — result hallucination, low novelty (trivial recombination).
- ["The More You Automate, the Less You See"](https://arxiv.org/abs/2509.08713) (Sep 2025) — benchmark misselection, **data leakage**, metric misuse, post-hoc selection; web-scale contamination makes "discoveries" recall.
- ["Why LLMs Aren't Scientists Yet"](https://arxiv.org/pdf/2601.03315) (Jan 2026) — fail on genuine inference; "polished but superficial" flood.

## 3. Agentic RL for code / research-engineering + RLVR

**Benchmarks/environments:** [RE-Bench](https://arxiv.org/abs/2411.15114) (METR), [MLE-bench](https://arxiv.org/abs/2410.07095) + [PaperBench](https://arxiv.org/abs/2504.01848) (OpenAI), [SWE-bench](https://arxiv.org/abs/2310.06770) + Verified ([swebench.com](https://www.swebench.com)), [SWE-Gym](https://arxiv.org/abs/2412.21139) (first *training* env), [PrimeIntellect Environments Hub](https://www.primeintellect.ai/blog/environments) + [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers), [Reasoning Gym](https://arxiv.org/abs/2505.24760) (procedural verifiers, reward-hack-resistant; NeurIPS 2025 spotlight).

**RLVR methods:** [GRPO](https://arxiv.org/abs/2402.03300) (DeepSeekMath), [DAPO](https://arxiv.org/abs/2503.14476) (ByteDance), [GSPO](https://arxiv.org/abs/2507.18071) (Qwen, sequence-level; Qwen3), [VAPO](https://arxiv.org/abs/2504.05118), [ProRL](https://arxiv.org/abs/2505.24864) (NVIDIA), [ARPO](https://arxiv.org/abs/2507.19849) (multi-turn tool-use). Survey: [Landscape of Agentic RL for LLMs](https://arxiv.org/abs/2509.02547) — central problem = credit assignment over sparse delayed rewards.

**RL that DISCOVERS (improves a real metric):** [CUDA-L1](https://arxiv.org/abs/2507.14111) (Jul 2025) — contrastive RL for CUDA kernels, ~3.12× avg over 250 KernelBench kernels (correctness-gated). Otherwise the real discovery cases are AlphaEvolve (evolutionary) + EvoTune/ThetaEvolve (RL proposer, small).

## 4. Open problems / debates

- **(a) Novelty vs memorization/contamination** — the strongest critique; leakage + post-hoc selection are documented. **A novelty/contamination audit is the credibility moat.**
- **(b) Search vs learning** — open. EvoTune/ThetaEvolve: RL-on-proposer beats frozen *at small scale*; AlphaEvolve: frozen + massive search still wins at frontier. Counter-camp: [GEPA](https://arxiv.org/pdf/2507.19457) (reflective prompt evolution > RL), [ES at Scale](https://arxiv.org/abs/2509.24372). Likely scale/signal-density dependent.
- **(c) Reward hacking / verifier integrity** — [LLMs Gaming Verifiers](https://arxiv.org/abs/2604.15149) (2026) + [Fuzzing RLVR Verifiers](https://arxiv.org/html/2606.01066v1) (2026): RLVR models enumerate instance labels that pass extensional checks. First-class design problem.
- **(d) Cross-problem generalization** — weak everywhere; EvoTune/ThetaEvolve are per-problem; no proposer yet learns *transferable* discovery skill. **The clearest unsolved frontier.**
- **(e) Compute cost** — AlphaEvolve expensive/closed; RL-training the proposer adds training compute atop search compute (the central efficiency tension).

## What changed since early-2025

- **The proposer started learning** — EvoTune (Apr) + ThetaEvolve (Nov) update proposer weights and beat frozen baselines: our thesis, de-risked at small scale.
- Open-source caught the *recipe* (OpenEvolve/ShinkaEvolve/CodeEvolve), not the scale.
- Real audited math results (Tao-coauthored, Nov 2025) gave the paradigm credibility.
- Critique literature matured to failure-mode taxonomies + **verifier reward-hacking** results → novelty audits + verifier hardening are now table stakes.

## Implications for CS-Discover (repositioning)

- **Do not claim to be first to RL-train the proposer.** Cite **EvoTune** as direct
  prior art and position against it. The bet is *validated*, not novel.
- **The defensible contribution is the conjunction nobody has shipped:** (1)
  general-purpose, (2) beyond toy combinatorial tasks, (3) RL-trained proposer, (4)
  verifiable reward, (5) **novelty/contamination audit in the reward loop**, (6)
  **cross-problem transfer**. Existing systems have at most a few of these.
- **Lead with the two hardest, least-solved pieces — which we are already set up for:**
  **(d) cross-problem generalization** (our L2 split) and **(a)+(c) novelty audit +
  hardened verifier in the loop** (reuse `audit/`; the 2026 reward-hacking literature
  makes this the moat).
- **Keep verifiers cheap and hard-to-game** (CUDA-L1 / KernelBench / SWE-Gym style) to
  hold integrity + compute tractable. The bin-packing MVP is the EvoTune-comparable
  starting point; the *novel* result is transfer to held-out problems + the audited
  novelty guarantee.
