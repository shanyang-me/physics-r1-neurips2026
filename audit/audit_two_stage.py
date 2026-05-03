#!/usr/bin/env python3
"""Remove train_pool records that overlap (J >= 0.4) with any eval split.

Reads master_corpus.jsonl + dedup_report.json, drops the offending train IDs,
writes train_pool_dedup.jsonl. Report counts removed by source family.
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path("$HOME/isometry/physics-o1/data_collection")
SPLITS = ROOT / "splits"
REPORT = ROOT / "dedup_report.json"
OUT_TRAIN = SPLITS / "train_pool_dedup.jsonl"


def main():
    rep = json.load(open(REPORT))
    # The report stores "a" and "b" keys; "a" is the smaller set in find_overlaps,
    # so we need to extract train IDs from whichever side they appeared on.
    drop_ids = set()
    for key in ["train_vs_eval_mini", "train_vs_eval_full", "train_vs_eval_oly"]:
        for o in rep.get(key, []):
            for fld in ("a", "b"):
                src = o[fld]
                # Train IDs are the ones not matching the eval format markers
                # Easier: drop both sides; if they're the same record (same ID
                # in train and eval) the dedup handles itself.
                drop_ids.add(src)

    # Load existing train pool
    kept = []
    dropped = []
    by_family_dropped = Counter()
    for line in open(SPLITS / "train_pool.jsonl"):
        r = json.loads(line)
        # Source might be in r["source"] or r["metadata"]._split_tags.uid
        sid = r.get("source") or r.get("metadata", {}).get("_split_tags", {}).get("uid")
        if sid in drop_ids:
            dropped.append(r)
            tags = r.get("metadata", {}).get("_split_tags", {})
            by_family_dropped[tags.get("source_bucket", "?")] += 1
        else:
            kept.append(r)

    print(f"Train pool original: {len(kept) + len(dropped)}")
    print(f"Dropped (overlap with eval splits): {len(dropped)}")
    print(f"Kept: {len(kept)}")
    print(f"\nDropped by source family:")
    for k, v in by_family_dropped.most_common():
        print(f"  {k:20s}: {v}")

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT_TRAIN}")

    # Also re-run dedup check to confirm
    print("\n=== Re-checking dedup AFTER cleanup ===")
    print("(Run dedup_check.py with train_pool_dedup.jsonl swapped in for verification)")


if __name__ == "__main__":
    main()
