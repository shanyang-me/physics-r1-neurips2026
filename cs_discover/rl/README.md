# cs_discover/rl — making the proposer learn

The step that separates CS-Discover from a frozen FunSearch/AlphaEvolve loop. The
frozen Claude `LLMProposer` is the **baseline**; a DPO-fine-tuned **open-weights**
proposer is the **learning** variant. (Claude via CLI is closed/frozen — it cannot be
weight-updated — so the trainable proposer must be an open model on a GPU.)

## Pipeline

```
search runs / sampling  ──collect.py──▶  preference pairs (chosen ≻ rejected)
                        ──dataset.py──▶  DPO + SFT JSONL
   open model  ─────────train_dpo.py──▶  trained proposer checkpoint  (GPU)
   serve (vLLM) ───▶ LLMProposer(call_fn=local)  ─── matched-budget ablation ───▶ result
```

## Steps

1. **Collect data** (CPU; use Claude for real signal):
   ```bash
   python -m cs_discover.rl.build_dataset --domain capset --n 5 --proposer claude \
       --contexts 20 --samples 6 --out-dir data_rl
   ```
   Emits `capset_claude_dpo.jsonl` ({prompt, chosen, rejected}) + an SFT file.

2. **Train** (GPU; needs trl/transformers/torch/peft):
   ```bash
   python -m cs_discover.rl.train_dpo --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
       --data data_rl/capset_claude_dpo.jsonl --out checkpoints/proposer-dpo --lora
   ```

3. **Serve + ablate.** Serve the checkpoint (vLLM), wrap it as
   `LLMProposer(call_fn=local_call_fn)`, and run the headline comparison.

## The headline ablation (pre-registered in CS_DISCOVERY_EVAL.md)

At **matched proposer-call budget**, compare on held-out problems:

| arm | proposer |
| --- | -------- |
| frozen-search baseline | untrained open model (or Claude) |
| **learning** | DPO-trained open model |

Primary question: does the trained proposer reach equal/greater discovery score with
**fewer proposer calls** (amortized search)? Report inference-time efficiency *and*
training-inclusive cost (the honesty trap in the eval doc). Cross-problem transfer
(L2): train on {bin-packing, cap-set n≤5}, test on a held-out family / larger n.

## Status

- `collect.py`, `dataset.py`, `build_dataset.py` — runnable + tested on CPU (mock
  proposer). Real data via the Claude proposer.
- `train_dpo.py` — GPU-ready (trl DPOTrainer, optional LoRA); not run in CI.
- Not yet: a served open checkpoint + the executed ablation (needs GPU); iterate
  collect→train→search rounds (EvoTune's alternation).
