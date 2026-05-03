#!/usr/bin/env python3
"""Detect text overlap between corpus subsets.

Uses 5-gram shingle Jaccard. Reports pairs with Jaccard >= threshold.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("$HOME/isometry/physics-o1/data_collection")
MASTER = ROOT / "master_corpus.jsonl"
PHYX = Path("/tmp/phyx_baseline/phyx_1000.jsonl")


def normalize(text):
    text = text.lower()
    text = re.sub(r"\\[a-zA-Z]+\b", " ", text)
    text = re.sub(r"[\$\{\}\[\]()]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingles(text, k=5):
    words = text.split()
    if len(words) < k:
        return frozenset({tuple(words)})
    return frozenset(tuple(words[i:i+k]) for i in range(len(words) - k + 1))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def load_corpus_records():
    by_subset = defaultdict(list)
    for line in open(MASTER):
        r = json.loads(line)
        subset = r["metadata"].get("corpus_subset", "?")
        text = r["messages"][0]["content"]
        by_subset[subset].append({
            "id": r["source"], "text": text, "norm": normalize(text),
        })
    return by_subset


def load_eval_olympiad_v2():
    out = []
    for line in open(ROOT / "splits/eval_olympiad_v2.jsonl"):
        r = json.loads(line)
        text = r["messages"][0]["content"]
        out.append({"id": r["source"], "text": text, "norm": normalize(text)})
    return out


def load_phyx():
    if not PHYX.exists():
        return []
    out = []
    for i, line in enumerate(open(PHYX)):
        r = json.loads(line)
        text = r.get("question") or r.get("messages", [{}])[0].get("content", "")
        out.append({"id": f"phyx_{i}", "text": text, "norm": normalize(text)})
    return out


def find_overlaps(set_a, set_b, label, jac_thresh=0.4, k=5):
    print(f"\n=== {label} ===  ({len(set_a)} × {len(set_b)})")
    # Pre-compute shingles for the smaller set
    if len(set_a) > len(set_b):
        small, big = set_b, set_a
    else:
        small, big = set_a, set_b
    for r in small:
        r["sh"] = shingles(r["norm"], k=k)
    overlaps = []
    for ra in big:
        sh_a = shingles(ra["norm"], k=k)
        for rb in small:
            j = jaccard(sh_a, rb["sh"])
            if j >= jac_thresh:
                overlaps.append({
                    "a": ra["id"], "b": rb["id"], "jaccard": round(j, 3),
                    "a_text": ra["text"][:120], "b_text": rb["text"][:120],
                })
    print(f"  → {len(overlaps)} overlaps with Jaccard ≥ {jac_thresh}")
    if overlaps:
        for o in overlaps[:5]:
            print(f"  HIT j={o['jaccard']}: {o['a'][:40]} <-> {o['b'][:40]}")
    return overlaps


def main():
    print("Loading...")
    by_subset = load_corpus_records()
    train = by_subset.get("train", [])
    eval_mini = by_subset.get("eval_mini", [])
    eval_full = by_subset.get("eval_full", [])
    eval_oly = by_subset.get("eval_olympiad", [])
    novel = by_subset.get("novel_pool", [])
    eval_oly_v2 = load_eval_olympiad_v2()
    phyx = load_phyx()
    print(f"  train={len(train)} eval_mini={len(eval_mini)} eval_full={len(eval_full)} "
          f"eval_oly={len(eval_oly)} novel={len(novel)} oly_v2={len(eval_oly_v2)} "
          f"phyx={len(phyx)}")

    overlaps = {}

    # 1. CRITICAL: train ↔ eval_olympiad_v2
    overlaps["train_vs_eval_olympiad_v2"] = find_overlaps(
        train, eval_oly_v2, "train ↔ eval_olympiad_v2 (CRITICAL)",
        jac_thresh=0.4,
    )
    # 2. train ↔ existing eval splits
    overlaps["train_vs_eval_mini"] = find_overlaps(
        train, eval_mini, "train ↔ eval_mini",
        jac_thresh=0.4,
    )
    overlaps["train_vs_eval_full"] = find_overlaps(
        train, eval_full, "train ↔ eval_full",
        jac_thresh=0.4,
    )
    overlaps["train_vs_eval_oly"] = find_overlaps(
        train, eval_oly, "train ↔ eval_olympiad (existing)",
        jac_thresh=0.4,
    )
    # 3. novel ↔ PhyX
    if phyx:
        overlaps["novel_vs_phyx"] = find_overlaps(
            novel, phyx, "novel_pool ↔ PhyX 1000q",
            jac_thresh=0.4,
        )

    out = ROOT / "dedup_report.json"
    with open(out, "w") as f:
        json.dump({k: v[:50] for k, v in overlaps.items()}, f, indent=2)
    print(f"\nReport: {out}")
    total = sum(len(v) for v in overlaps.values())
    print(f"\nSUMMARY: {total} suspected overlaps")
    for k, v in overlaps.items():
        print(f"  {k}: {len(v)}")
    if total == 0:
        print("\n✓ Clean — no contamination at Jaccard ≥ 0.4")


if __name__ == "__main__":
    main()
