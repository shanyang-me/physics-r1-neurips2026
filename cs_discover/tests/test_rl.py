"""RL preference-data pipeline tests (CPU, model-free). Verifies the dataset builder
produces well-formed DPO/SFT data; the GPU trainer (train_dpo.py) is not exercised here.
Run: python cs_discover/tests/test_rl.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cs_discover.domains import make_domain
from cs_discover.search import MutationProposer
from cs_discover.rl import collect_preferences, write_dpo_jsonl, write_sft_jsonl, read_jsonl


def test_preference_pairs_are_well_ordered():
    dom = make_domain("binpacking", dist="uniform")
    pairs = collect_preferences(dom, MutationProposer(), n_contexts=6, samples_per_context=5, seed=1)
    assert len(pairs) > 0
    for p in pairs:
        assert p.chosen_score > p.rejected_score          # chosen strictly preferred
        assert p.margin > 0
        assert p.chosen != p.rejected
        assert isinstance(p.prompt, str) and len(p.prompt) > 0


def test_dpo_and_sft_jsonl_roundtrip():
    # bin-packing (not cap-set): mutation varies bin-packing scores, so pairs exist.
    dom = make_domain("binpacking", dist="uniform")
    pairs = collect_preferences(dom, MutationProposer(), n_contexts=6, samples_per_context=5, seed=2)
    assert len(pairs) > 0
    with tempfile.TemporaryDirectory() as d:
        dpo_p = os.path.join(d, "dpo.jsonl")
        sft_p = os.path.join(d, "sft.jsonl")
        n_dpo = write_dpo_jsonl(pairs, dpo_p)
        n_sft = write_sft_jsonl(pairs, sft_p)
        assert n_dpo == len(pairs)
        assert 0 < n_sft <= n_dpo
        dpo = read_jsonl(dpo_p)
        for r in dpo:
            assert set(["prompt", "chosen", "rejected"]).issubset(r)
            assert r["chosen"].startswith("```python") and r["rejected"].startswith("```python")
            assert r["chosen_score"] > r["rejected_score"]
        sft = read_jsonl(sft_p)
        for r in sft:
            assert "prompt" in r and r["completion"].startswith("```python")


def test_no_pairs_when_no_score_variation():
    # a proposer that always returns the seed produces no distinct-score pairs.
    dom = make_domain("capset", n=3)

    class ConstProposer:
        def propose(self, domain, parents, rng):
            return domain.seed_sources()[0]

    pairs = collect_preferences(dom, ConstProposer(), n_contexts=4, samples_per_context=4, seed=0)
    assert pairs == []


def test_capset_mutation_yields_no_pairs_by_saturation():
    # Consequence of the 2^n separable ceiling: every valid mutation of the separable
    # seed scores exactly 2^n, so there is no reward variation to learn from -> a
    # frozen model-free proposer generates ZERO training signal on cap-set. Only a
    # non-separable (LLM) proposer produces learnable preference pairs here.
    dom = make_domain("capset", n=4)
    pairs = collect_preferences(dom, MutationProposer(), n_contexts=6, samples_per_context=5, seed=1)
    assert pairs == []


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} rl pipeline tests passed.")


if __name__ == "__main__":
    _run_all()
