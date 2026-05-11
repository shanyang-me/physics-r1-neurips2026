"""LLM-judge v2: best-match alignment.

For each gold[i], scan all pred_boxed_list[j] and ask the judge "does any pred equal gold[i]?".
Counts gold[i] as correct if ANY pred matches. Avoids the parser's strict positional misalignment.

Sonnet-as-judge over saved generations. Skips empty preds.

Usage:
  python llm_judge_v2.py --in Physics-R1-step40_physreason.jsonl --workers 6
"""
from __future__ import annotations
import argparse, json, subprocess, threading, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os as _os
CLAUDE_BIN = _os.path.expanduser('~/.local/bin/claude')
TIMEOUT = 90


def call_judge(prompt: str) -> tuple[str, str | None]:
    try:
        r = subprocess.run(
            [CLAUDE_BIN, '--print', '--model', 'claude-sonnet-4-5', prompt],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        if r.returncode != 0:
            return '', f'rc={r.returncode}: {r.stderr[:120]}'
        return r.stdout, None
    except subprocess.TimeoutExpired:
        return '', 'timeout'
    except Exception as e:
        return '', f'exc: {str(e)[:120]}'


JUDGE_PROMPT = """You are grading a physics olympiad answer.

GOLD answer: {gold}

CANDIDATE predictions (one or more, separated by ===):
{preds}

Task: decide whether ANY of the candidate predictions is mathematically/physically equivalent to the gold answer.
Allow: different but equivalent algebraic forms (e.g. \\sqrt{{2gh}}/\\tan(theta) == \\sqrt{{2gh}}\\cot(theta));
       trivial unit/format differences ("450 N" == "450", "9.8 m/s^2" == "9.8");
       rounding within 1% relative tolerance;
       trailing prose like "to the right", "approximately";
       different variable names that map cleanly.
Reject: different magnitude, different sign, different functional form, missing or wrong physical content.

Respond with EXACTLY one word: YES (if any candidate matches) or NO. No other text."""


def parse_yes_no(text: str) -> bool | None:
    if not text:
        return None
    m = re.search(r'\b(YES|NO)\b', text.strip().upper())
    return m and (m.group(1) == 'YES')


def judge_record(rec: dict) -> dict:
    bench = rec.get('benchmark', '')
    raw_preds = rec.get('pred_boxed_list', []) or []
    preds = [p for p in raw_preds if p and p.strip()]
    gold = rec.get('gold')

    if bench == 'physreason':
        gold_list = gold if isinstance(gold, list) else [gold]
    else:
        # physunibench: gold is verbose paragraph; try to extract sub-parts via (a)/(b) markers
        gold_str = gold if isinstance(gold, str) else (gold[0] if gold else '')
        sub_marks = re.split(r'\n\s*\(([a-d])\)\s*[:.]?\s*', gold_str)
        if len(sub_marks) >= 3:
            gold_list = [sub_marks[2 + 2*i] for i in range((len(sub_marks)-1)//2)]
        else:
            gold_list = [gold_str]

    if not preds:
        return {**rec, 'judge_subs': [{'gold': g, 'preds_n': 0, 'judge': False} for g in gold_list],
                'judge_problem_correct': False}

    pred_block = '\n===\n'.join(preds)

    judgments = []
    for g in gold_list:
        out, err = call_judge(JUDGE_PROMPT.format(gold=g, preds=pred_block))
        v = parse_yes_no(out)
        judgments.append({'gold': g, 'preds_n': len(preds), 'judge': bool(v) if v is not None else False,
                          'raw': out[:80], 'err': err})

    sub_correct = [j['judge'] for j in judgments]
    problem_correct = bool(sub_correct) and all(sub_correct)
    return {**rec, 'judge_subs': judgments, 'judge_sub_correct': sub_correct,
            'judge_problem_correct': problem_correct}


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.in_path)
    if not inp.is_absolute():
        inp = Path('/tmp/tbd20_results') / inp
    out = inp.with_name(inp.stem + '_judged_v2.jsonl')

    items = load_jsonl(inp)
    by_id = {}
    for r in items:
        by_id[str(r.get('id'))] = r
    items = list(by_id.values())
    if args.limit > 0:
        items = items[:args.limit]

    done = set()
    if out.exists():
        for r in load_jsonl(out):
            done.add(str(r.get('id')))
    todo = [r for r in items if str(r['id']) not in done]
    print(f'[{inp.name}] total={len(items)} done={len(done)} todo={len(todo)} workers={args.workers}', flush=True)

    if not todo:
        return

    lock = threading.Lock()
    t0 = time.time()
    n_done = 0
    n_correct = 0

    def append(rec):
        with lock:
            with open(out, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_record, r): r for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                append(rec)
                n_done += 1
                if rec.get('judge_problem_correct'):
                    n_correct += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    rate = n_done / max(el, 1)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f'  [{n_done}/{len(todo)}] judged-correct={n_correct}/{n_done} '
                          f'({100*n_correct/max(n_done,1):.1f}%) eta={eta:.0f}s', flush=True)
            except Exception as e:
                print(f'  [ERR] {e}', flush=True)

    rs = load_jsonl(out)
    total = len(rs)
    correct = sum(1 for r in rs if r.get('judge_problem_correct'))
    print(f'\n[FINAL] {inp.name}: {correct}/{total} = {100*correct/max(total,1):.2f}%', flush=True)


if __name__ == '__main__':
    main()
