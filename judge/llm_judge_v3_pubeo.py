"""LLM-judge v3 for PhysUniBench-OE: cached clean gold + tail fallback.

Reads the per-id clean gold cache built from sonnet_physunibench_judged.jsonl,
then for each model's `*_physunibench*.jsonl` (raw or v2-judged) re-runs the
best-match alignment v2 logic but with:

  (a) gold = cached clean per-subpart list (e.g., ["2.68nC", "7853.1W"])
      instead of the verbose paragraph that confused v2.

  (b) F1 fallback: when pred_boxed_list is empty/[''], scan the raw_tail
      for likely numeric/symbolic answers (regex: '=\\s*[0-9\\-+]+...',
      'is\\s+approximately', 'answer:\\s+...') and use those as candidate preds.

Output: <input>_judged_v3.jsonl with judge_problem_correct (liberal) +
        strict_correct (any normalized boxed pred matches any gold).

Usage:
  python llm_judge_v3_pubeo.py --in <model>_physunibench*.jsonl --workers 6
"""
from __future__ import annotations
import argparse, json, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os as _os
CLAUDE_BIN = _os.path.expanduser('~/.local/bin/claude')
TIMEOUT = 90
GOLD_CACHE_PATH = '/tmp/tbd20_results/pub_oe_gold_cache.json'


def call_judge(prompt: str) -> tuple[str, str | None]:
    try:
        r = subprocess.run(
            [CLAUDE_BIN, '--print', '--model', 'claude-sonnet-4-5', prompt],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        if r.returncode != 0:
            return '', f'rc={r.returncode}'
        return r.stdout, None
    except Exception as e:
        return '', f'exc: {str(e)[:120]}'


JUDGE_PROMPT = """You are grading a physics olympiad answer.

GOLD answer: {gold}

CANDIDATE predictions (one or more, separated by ===):
{preds}

Task: decide whether ANY candidate prediction is mathematically/physically equivalent to the gold.
Allow: different but equivalent algebraic forms; trivial unit/format differences ("450 N" == "450 \\text{{ N}}");
       rounding within 2% relative tolerance; trailing prose ("approximately", "to the right");
       different variable names mapping cleanly.
Reject: different magnitude, different sign, different functional form, missing or wrong physical content.

Respond with EXACTLY one word: YES or NO."""


def parse_yes_no(t):
    if not t:
        return None
    m = re.search(r'\b(YES|NO)\b', t.strip().upper())
    return m and (m.group(1) == 'YES')


# F1 fallback: extract candidate answers from response tail when no boxed
# Common patterns in CoT tails ending with answer
TAIL_PATTERNS = [
    r'(?:answer|Answer|ANSWER)\s*(?:is|:|=)\s*\$?([^\n.$]{1,80})',
    r'(?:therefore|Therefore|thus|Thus|so|hence|Hence)\s*[,]?\s*([^\n.]{1,80})',
    r'=\s*([0-9.\-+]+(?:\s*\\?[a-zA-Z][a-zA-Z/^_{}\\\d.\s\-]*)?)',
    r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
]


def fallback_extract(tail: str) -> list[str]:
    if not tail:
        return []
    cands = []
    for pat in TAIL_PATTERNS:
        for m in re.finditer(pat, tail):
            c = m.group(1).strip()
            if c and len(c) < 120 and c not in cands:
                cands.append(c)
    return cands[:8]  # cap


def normalize(s):
    if not s:
        return ''
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\s+', '', s)
    s = s.replace('$', '').replace('\\,', '').replace('\\!', '').replace('\\;', '')
    s = s.replace(',', '')
    return s.lower()


def strict_match_any(preds: list[str], gold: str) -> bool:
    g = normalize(gold)
    if not g:
        return False
    for p in preds:
        np = normalize(p)
        if np == g:
            return True
        # tolerate numeric prefix in pred (e.g., "8.48nC" gold vs "8.48nCdirected" pred)
        if g and (np.startswith(g) or g in np):
            return True
    return False


def judge_record(rec, gold_cache):
    rid = str(rec.get('id'))
    gold_subs = gold_cache.get(rid, [])
    if not gold_subs:
        # No clean gold available → can't grade
        return {**rec, 'judge_problem_correct_v3': False, 'strict_correct_v3': False,
                'gold_v3': [], 'cand_preds_v3': [], 'err_v3': 'no gold in cache'}

    raw_preds = rec.get('pred_boxed_list', []) or []
    preds = [p for p in raw_preds if p and p.strip()]
    used_fallback = False
    if not preds:
        tail = rec.get('raw_tail', '') or ''
        preds = fallback_extract(tail)
        used_fallback = bool(preds)

    if not preds:
        return {**rec, 'judge_problem_correct_v3': False, 'strict_correct_v3': False,
                'gold_v3': gold_subs, 'cand_preds_v3': [], 'err_v3': 'no preds even after fallback'}

    # Strict pass: any pred normalized equals any gold normalized
    strict_ok = all(strict_match_any(preds, g) for g in gold_subs)

    # Liberal pass: per-gold judge call, AND across all subs
    pred_block = '\n===\n'.join(preds[:6])  # cap candidate list
    sub_results = []
    for g in gold_subs:
        if strict_match_any(preds, g):
            sub_results.append({'gold': g, 'judge': True, 'src': 'strict-norm'})
            continue
        out, err = call_judge(JUDGE_PROMPT.format(gold=g, preds=pred_block))
        v = parse_yes_no(out)
        sub_results.append({'gold': g, 'judge': bool(v) if v is not None else False,
                            'raw': out[:60], 'err': err, 'src': 'sonnet-judge'})

    liberal_ok = all(s['judge'] for s in sub_results)

    return {**rec, 'judge_problem_correct_v3': liberal_ok, 'strict_correct_v3': strict_ok,
            'gold_v3': gold_subs, 'cand_preds_v3': preds, 'used_fallback_v3': used_fallback,
            'sub_results_v3': sub_results}


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.in_path)
    if not inp.is_absolute():
        inp = Path('/tmp/tbd20_results') / inp
    out = inp.with_name(inp.stem.replace('_judged_v2', '').replace('_judged', '') + '_judged_v3.jsonl')

    gold_cache = json.load(open(GOLD_CACHE_PATH))

    items = load(inp)
    by_id = {}
    for r in items:
        by_id[str(r.get('id'))] = r  # last wins
    items = list(by_id.values())
    if args.limit:
        items = items[:args.limit]

    done = set()
    if out.exists():
        for r in load(out):
            done.add(str(r.get('id')))
    todo = [r for r in items if str(r['id']) not in done]
    print(f'[{inp.name}] total={len(items)} done={len(done)} todo={len(todo)} workers={args.workers}', flush=True)

    if not todo:
        rs = load(out)
        s = sum(1 for r in rs if r.get('strict_correct_v3'))
        l = sum(1 for r in rs if r.get('judge_problem_correct_v3'))
        print(f'\n[FINAL] {inp.name}: strict={s}/{len(rs)} = {100*s/len(rs):.2f}% | liberal={l}/{len(rs)} = {100*l/len(rs):.2f}%', flush=True)
        return

    lock = threading.Lock()
    n_done = n_strict = n_lib = n_fb = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_record, r, gold_cache): r for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock:
                    with open(out, 'a') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    n_done += 1
                    if rec.get('strict_correct_v3'): n_strict += 1
                    if rec.get('judge_problem_correct_v3'): n_lib += 1
                    if rec.get('used_fallback_v3'): n_fb += 1
                    if n_done % 25 == 0 or n_done == len(todo):
                        el = time.time() - t0
                        rate = n_done / max(el, 1)
                        eta = (len(todo) - n_done) / max(rate, 1e-6)
                        print(f'  [{n_done}/{len(todo)}] strict={n_strict} ({100*n_strict/n_done:.1f}%) '
                              f'liberal={n_lib} ({100*n_lib/n_done:.1f}%) tail-fallback={n_fb} '
                              f'eta={eta:.0f}s', flush=True)
            except Exception as e:
                print(f'  [ERR] {e}', flush=True)

    rs = load(out)
    s = sum(1 for r in rs if r.get('strict_correct_v3'))
    l = sum(1 for r in rs if r.get('judge_problem_correct_v3'))
    fb = sum(1 for r in rs if r.get('used_fallback_v3'))
    print(f'\n[FINAL] {inp.name}: strict={s}/{len(rs)} = {100*s/len(rs):.2f}% | '
          f'liberal={l}/{len(rs)} = {100*l/len(rs):.2f}% | tail-fallback used on {fb} records', flush=True)


if __name__ == '__main__':
    main()
