"""Build held-out evaluation splits for Physics-o1 (NeurIPS D&B submission).

Aggregates 5 input JSONL files, tags each problem with (source, concept,
difficulty, modality), and emits 4 disjoint splits:
  1. eval_mini.jsonl      - 500 problems, balanced across concept x difficulty
  2. eval_full.jsonl      - 2000 problems, preserving source distribution
  3. eval_olympiad.jsonl  - 500 problems, hardest subset
  4. train_pool.jsonl     - the rest (~10K)

Plus a markdown stats report.

Usage:
    python make_splits.py \
        --rl-sft  /workspace/rl_sft_matched/rl_data.jsonl \
        --physics-se /path/to/physics_se.jsonl \
        --openstax   /path/to/openstax_physics.jsonl \
        --ugphysics  /workspace/additional_data/ugphysics_train.jsonl \
        --physreason /workspace/additional_data/physreason_train.jsonl \
        --out-dir    ./splits \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# -----------------------------------------------------------------------------
# Concept taxonomy
# -----------------------------------------------------------------------------
# Seven canonical concept buckets for reporting.
CONCEPTS = ["Mechanics", "EM", "QM", "Thermo", "Waves", "Optics", "Modern"]

# Keyword patterns for concept inference. Checked in order; first match wins.
# Patterns target SE tags, OpenStax `topic`, PhysReason `theorem` list, plus
# free-text question fallbacks.
CONCEPT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Optics",   re.compile(r"optic|lens|mirror|refract|diffract|interferenc|polariz|thin.?film|telescope|microscope", re.I)),
    ("Waves",    re.compile(r"wave|sound|acoustic|doppler|standing.?wave|string.?vibr|harmonic.?motion|oscillat|pendulum", re.I)),
    ("QM",       re.compile(r"quantum|wavefunction|schrodinger|schr.?dinger|spin(?!dle)|qubit|hilbert|operator|commutat|uncertainty|eigenstat|bra.?ket|entangle", re.I)),
    ("Modern",   re.compile(r"relativ|lorentz|general.?relat|special.?relat|cosmolog|black.?hole|nuclear|particle.?phys|standard.?model|qft|quantum.?field|field.?theory|gauge|string.?theory", re.I)),
    ("Thermo",   re.compile(r"thermo|entropy|heat|temperature|ideal.?gas|carnot|kinetic.?theory|statistic.?mech|boltzmann|partition.?function|phase.?transition|fluid.?stat|viscos", re.I)),
    ("EM",       re.compile(r"electr|magnet|capacitor|inductor|resistor|current|voltage|maxwell|coulomb|gauss|faraday|ampere|circuit|emf|electrostat|electrodynam|induction", re.I)),
    ("Mechanics",re.compile(r"mechanic|newton|force|momentum|kinetic|potential.?energy|kinematic|dynamic|lagrang|hamilton|rotation|torque|rigid.?body|projectile|gravit|collision|friction|fluid.?dynam", re.I)),
]


def infer_concept(record: dict) -> str:
    """Return a canonical concept bucket for the record."""
    meta = record.get("metadata") or {}
    # Pool all "topical" strings we can find.
    hay_parts: list[str] = []
    if isinstance(meta, dict):
        tags = meta.get("tags")
        if isinstance(tags, list):
            hay_parts.extend(str(t) for t in tags)
        for k in ("topic", "book", "theorem", "subject", "category", "chapter"):
            v = meta.get(k)
            if isinstance(v, str):
                hay_parts.append(v)
            elif isinstance(v, list):
                hay_parts.extend(str(x) for x in v)

    # Question text as last-ditch fallback.
    try:
        question = record["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        question = ""
    hay_meta = " ".join(hay_parts)

    # First try to match on the (stronger) metadata fields.
    for name, pat in CONCEPT_PATTERNS:
        if hay_meta and pat.search(hay_meta):
            return name
    # Fall back to question body (first 400 chars only — keep it cheap).
    q_snip = question[:400]
    for name, pat in CONCEPT_PATTERNS:
        if pat.search(q_snip):
            return name
    return "Mechanics"  # default bucket


# -----------------------------------------------------------------------------
# Difficulty normalization
# -----------------------------------------------------------------------------
# Target 3 buckets: "low", "mid", "high".
DIFFICULTY_MAP = {
    # PhysReason
    "easy": "low",
    "knowledge": "low",
    "medium": "mid",
    "difficult": "high",
    "hard": "high",
    # OpenStax
    "intro": "low",
    "undergrad": "mid",
    "graduate": "high",
}


def infer_difficulty(record: dict, source_bucket: str) -> str:
    meta = record.get("metadata") or {}
    if isinstance(meta, dict):
        raw = meta.get("difficulty")
        if isinstance(raw, str) and raw.lower() in DIFFICULTY_MAP:
            return DIFFICULTY_MAP[raw.lower()]
        # Physics-SE: use vote score as a proxy for difficulty / depth.
        # Higher-scored Qs tend to be harder / more advanced.
        score = meta.get("score")
        if isinstance(score, (int, float)):
            if score >= 25:
                return "high"
            if score >= 10:
                return "mid"
            return "low"

    # Source-level priors.
    if source_bucket == "physreason":
        return "mid"
    if source_bucket == "ugphysics":
        return "high"  # university-level, no per-row difficulty
    if source_bucket == "rl_sft":
        return "mid"
    if source_bucket == "openstax":
        return "low"
    return "mid"


# -----------------------------------------------------------------------------
# Source bucket + modality
# -----------------------------------------------------------------------------
SOURCE_BUCKETS = ["rl_sft", "physics_se", "openstax", "ugphysics", "physreason"]


def has_images(record: dict) -> bool:
    imgs = record.get("images")
    if isinstance(imgs, list) and len(imgs) > 0:
        return True
    # Also detect inline <image> placeholders in the question.
    try:
        q = record["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        q = ""
    return "<image>" in q


# -----------------------------------------------------------------------------
# Loading / tagging
# -----------------------------------------------------------------------------
def load_jsonl(path: str, source_bucket: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{i} JSON decode error: {e}")
                continue
            rec["_source_bucket"] = source_bucket
            rec["_row_idx"] = i
            # Stable global id (helps cross-referencing & dedup).
            src_name = rec.get("source") or f"{source_bucket}_{i}"
            rec["_uid"] = f"{source_bucket}::{src_name}"
            rec["_concept"] = infer_concept(rec)
            rec["_difficulty"] = infer_difficulty(rec, source_bucket)
            rec["_multimodal"] = has_images(rec)
            out.append(rec)
    return out


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------
def stratified_sample(
    records: list[dict],
    n_target: int,
    strata_key,
    rng: random.Random,
) -> list[dict]:
    """Sample n_target records, balancing across strata as much as possible.

    If some strata are too small, we top up from the largest strata.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[strata_key(r)].append(r)

    # Shuffle each bucket.
    for k in buckets:
        rng.shuffle(buckets[k])

    n_buckets = len(buckets)
    if n_buckets == 0:
        return []
    per_bucket = n_target // n_buckets
    remainder = n_target - per_bucket * n_buckets

    picked: list[dict] = []
    leftover_pool: list[dict] = []
    # Iterate in deterministic order (sorted keys).
    sorted_keys = sorted(buckets.keys(), key=lambda x: tuple(str(v) for v in x))
    for k in sorted_keys:
        bucket = buckets[k]
        take = min(per_bucket, len(bucket))
        picked.extend(bucket[:take])
        leftover_pool.extend(bucket[take:])

    # Now spread the remainder + any unmet per-bucket quota across leftovers.
    short = n_target - len(picked)
    rng.shuffle(leftover_pool)
    picked.extend(leftover_pool[:short])
    rng.shuffle(picked)
    return picked


def proportional_sample(
    records: list[dict],
    n_target: int,
    strata_key,
    rng: random.Random,
) -> list[dict]:
    """Sample preserving the source distribution (proportional allocation)."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[strata_key(r)].append(r)
    for k in buckets:
        rng.shuffle(buckets[k])

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return []
    # Floor allocation per bucket, then distribute remainder by largest fractional part.
    raw = {k: n_target * len(v) / total for k, v in buckets.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    frac = sorted(((raw[k] - alloc[k], k) for k in buckets), reverse=True)
    short = n_target - sum(alloc.values())
    for _, k in frac[:short]:
        alloc[k] += 1

    picked: list[dict] = []
    for k, n in alloc.items():
        picked.extend(buckets[k][:n])
    rng.shuffle(picked)
    return picked


def hardness_score(r: dict) -> float:
    """Higher = harder. Used to pick the olympiad split."""
    s = 0.0
    diff = r.get("_difficulty")
    s += {"low": 0.0, "mid": 1.0, "high": 2.0}.get(diff, 1.0)
    bucket = r.get("_source_bucket")
    # Source priors: physreason difficult > ugphysics > openstax_undergrad.
    if bucket == "physreason":
        s += 2.0
        meta = r.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("difficulty") == "difficult":
            s += 2.0
    elif bucket == "ugphysics":
        s += 1.5
    elif bucket == "physics_se":
        meta = r.get("metadata") or {}
        score = meta.get("score", 0) if isinstance(meta, dict) else 0
        s += min(2.0, float(score) / 50.0)
    elif bucket == "openstax":
        meta = r.get("metadata") or {}
        book = (meta.get("book") or "").lower() if isinstance(meta, dict) else ""
        if "vol 3" in book or "vol3" in book or "vol 2" in book:
            s += 0.5  # optics/QM/thermo volumes skew harder
    return s


# -----------------------------------------------------------------------------
# Stats writer
# -----------------------------------------------------------------------------
def fmt_table(
    rows: list[list[str]],
    headers: list[str],
) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    out = [head, sep]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def summarize(records: list[dict]) -> dict:
    s = {
        "n": len(records),
        "by_source": Counter(r["_source_bucket"] for r in records),
        "by_concept": Counter(r["_concept"] for r in records),
        "by_difficulty": Counter(r["_difficulty"] for r in records),
        "multimodal": sum(1 for r in records if r["_multimodal"]),
    }
    return s


def write_stats_md(out_path: Path, splits: dict[str, list[dict]], total: int) -> None:
    lines: list[str] = []
    lines.append("# Physics-o1 Split Statistics\n")
    lines.append(f"Total aggregated problems: **{total}**\n")
    lines.append("Splits are disjoint by `_uid`.\n")

    for name, recs in splits.items():
        s = summarize(recs)
        lines.append(f"\n## {name}  (n={s['n']})\n")

        lines.append("### By source")
        rows = [[src, s["by_source"].get(src, 0)] for src in SOURCE_BUCKETS]
        rows.append(["TOTAL", s["n"]])
        lines.append(fmt_table(rows, ["source", "count"]))
        lines.append("")

        lines.append("### By concept")
        rows = [[c, s["by_concept"].get(c, 0)] for c in CONCEPTS]
        rows.append(["TOTAL", s["n"]])
        lines.append(fmt_table(rows, ["concept", "count"]))
        lines.append("")

        lines.append("### By difficulty")
        rows = [[d, s["by_difficulty"].get(d, 0)] for d in ("low", "mid", "high")]
        rows.append(["TOTAL", s["n"]])
        lines.append(fmt_table(rows, ["difficulty", "count"]))
        lines.append("")

        lines.append("### Modality")
        rows = [
            ["text-only", s["n"] - s["multimodal"]],
            ["multimodal", s["multimodal"]],
        ]
        lines.append(fmt_table(rows, ["modality", "count"]))
        lines.append("")

        # Source x difficulty cross-tab.
        lines.append("### Source x difficulty")
        xtab: dict[tuple[str, str], int] = defaultdict(int)
        for r in recs:
            xtab[(r["_source_bucket"], r["_difficulty"])] += 1
        rows = []
        for src in SOURCE_BUCKETS:
            row = [src]
            for d in ("low", "mid", "high"):
                row.append(xtab[(src, d)])
            rows.append(row)
        lines.append(fmt_table(rows, ["source", "low", "mid", "high"]))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def strip_internal(r: dict) -> dict:
    """Drop internal fields but keep a canonical annotated block in metadata."""
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    meta = dict(out.get("metadata") or {}) if isinstance(out.get("metadata"), dict) else {}
    meta["_split_tags"] = {
        "source_bucket": r["_source_bucket"],
        "concept": r["_concept"],
        "difficulty": r["_difficulty"],
        "multimodal": r["_multimodal"],
        "uid": r["_uid"],
    }
    out["metadata"] = meta
    return out


def dump_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(strip_internal(r), ensure_ascii=False))
            f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rl-sft", required=True)
    p.add_argument("--physics-se", required=True)
    p.add_argument("--openstax", required=True)
    p.add_argument("--ugphysics", required=True)
    p.add_argument("--physreason", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini-n", type=int, default=500)
    p.add_argument("--full-n", type=int, default=2000)
    p.add_argument("--olympiad-n", type=int, default=500)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading sources")
    specs = [
        ("rl_sft",     args.rl_sft),
        ("physics_se", args.physics_se),
        ("openstax",   args.openstax),
        ("ugphysics",  args.ugphysics),
        ("physreason", args.physreason),
    ]
    all_records: list[dict] = []
    for bucket, path in specs:
        recs = load_jsonl(path, bucket)
        print(f"  {bucket:<11s} n={len(recs):>5d}   ({path})")
        all_records.extend(recs)

    # Dedup by _uid (rare but possible across sources; keep first).
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in all_records:
        if r["_uid"] in seen:
            continue
        seen.add(r["_uid"])
        deduped.append(r)
    print(f"  total={len(all_records)}  dedup={len(deduped)}")
    rng.shuffle(deduped)  # deterministic shuffle under seed

    # ------------------------------------------------------------------
    # Build olympiad split first (hardest subset), so it gets first pick
    # of rare difficult items.
    # ------------------------------------------------------------------
    print("[2/5] Sampling olympiad split")
    olympiad_pool = sorted(deduped, key=hardness_score, reverse=True)
    # Take top-K by hardness, but with a per-source cap to ensure diversity
    # across the 3 requested hardness sources.
    per_source_cap = {"physreason": 250, "ugphysics": 200, "openstax": 75,
                      "physics_se": 100, "rl_sft": 50}
    olympiad: list[dict] = []
    picked_uids: set[str] = set()
    caps = defaultdict(int)
    for r in olympiad_pool:
        if len(olympiad) >= args.olympiad_n:
            break
        bucket = r["_source_bucket"]
        if caps[bucket] >= per_source_cap.get(bucket, 0):
            continue
        olympiad.append(r)
        caps[bucket] += 1
        picked_uids.add(r["_uid"])
    # Pad if we ran out of capped items.
    if len(olympiad) < args.olympiad_n:
        for r in olympiad_pool:
            if r["_uid"] in picked_uids:
                continue
            olympiad.append(r)
            picked_uids.add(r["_uid"])
            if len(olympiad) >= args.olympiad_n:
                break

    remaining = [r for r in deduped if r["_uid"] not in picked_uids]
    print(f"  olympiad n={len(olympiad)}  remaining={len(remaining)}")

    # ------------------------------------------------------------------
    # eval_mini: balanced across (concept x difficulty).
    # ------------------------------------------------------------------
    print("[3/5] Sampling eval_mini")
    eval_mini = stratified_sample(
        remaining,
        args.mini_n,
        strata_key=lambda r: (r["_concept"], r["_difficulty"]),
        rng=rng,
    )
    picked_uids.update(r["_uid"] for r in eval_mini)
    remaining = [r for r in remaining if r["_uid"] not in {m["_uid"] for m in eval_mini}]
    print(f"  eval_mini n={len(eval_mini)}  remaining={len(remaining)}")

    # ------------------------------------------------------------------
    # eval_full: proportional across source (representative).
    # ------------------------------------------------------------------
    print("[4/5] Sampling eval_full")
    eval_full = proportional_sample(
        remaining,
        args.full_n,
        strata_key=lambda r: r["_source_bucket"],
        rng=rng,
    )
    picked_uids_full = {r["_uid"] for r in eval_full}
    train_pool = [r for r in remaining if r["_uid"] not in picked_uids_full]
    print(f"  eval_full n={len(eval_full)}  train_pool n={len(train_pool)}")

    # Sanity: disjoint sets.
    uids_mini = {r["_uid"] for r in eval_mini}
    uids_full = {r["_uid"] for r in eval_full}
    uids_oly = {r["_uid"] for r in olympiad}
    uids_trn = {r["_uid"] for r in train_pool}
    assert uids_mini.isdisjoint(uids_full)
    assert uids_mini.isdisjoint(uids_oly)
    assert uids_full.isdisjoint(uids_oly)
    assert uids_trn.isdisjoint(uids_mini | uids_full | uids_oly)
    total_coverage = len(uids_mini) + len(uids_full) + len(uids_oly) + len(uids_trn)
    assert total_coverage == len(deduped), (total_coverage, len(deduped))

    # ------------------------------------------------------------------
    # Write splits + stats.
    # ------------------------------------------------------------------
    print("[5/5] Writing outputs")
    splits = {
        "eval_mini":     eval_mini,
        "eval_full":     eval_full,
        "eval_olympiad": olympiad,
        "train_pool":    train_pool,
    }
    for name, recs in splits.items():
        out_path = out_dir / f"{name}.jsonl"
        dump_jsonl(out_path, recs)
        print(f"  wrote {out_path}  n={len(recs)}")

    write_stats_md(out_dir / "split_stats.md", splits, total=len(deduped))
    print(f"  wrote {out_dir / 'split_stats.md'}")

    # Machine-readable stats too.
    summary = {
        "seed": args.seed,
        "total": len(deduped),
        "splits": {
            name: {
                "n": s["n"],
                "by_source": dict(s["by_source"]),
                "by_concept": dict(s["by_concept"]),
                "by_difficulty": dict(s["by_difficulty"]),
                "multimodal": s["multimodal"],
            }
            for name, recs in splits.items()
            for s in (summarize(recs),)
        },
    }
    (out_dir / "split_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'split_stats.json'}")


if __name__ == "__main__":
    main()
