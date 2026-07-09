"""RL layer — make the proposer LEARN (EvoTune-style), the step that separates
CS-Discover from a frozen FunSearch/AlphaEvolve loop.

  collect.py        sample proposer completions -> (chosen > rejected) preference pairs
  dataset.py        serialize to DPO / SFT JSONL
  build_dataset.py  CLI to produce a dataset from a domain + proposer
  train_dpo.py      GPU DPO fine-tune of an open-weights proposer (requires trl/torch)

The frozen Claude proposer is the FunSearch baseline; a DPO-trained open proposer is the
learning variant. Compare them at matched proposer-call budget (see README.md).
"""
from .collect import PreferencePair, collect_preferences, sample_contexts
from .dataset import write_dpo_jsonl, write_sft_jsonl, read_jsonl
