"""Sonnet-judge step40 olympiad_v2 raws — strict (boxed-match) + liberal (judge).

Input: physics-r1-step40_eval_olympiad_v2.jsonl with keys {response, gold, ...}
Output: physics-r1-step40_eval_olympiad_v2_judged.jsonl with judge_problem_correct (liberal)
        and strict_correct (any boxed pred string-equals gold after normalization).
"""
import argparse, json, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CLAUDE_BIN = '$HOME/.local/bin/claude'
TIMEOUT = 90


def extract_boxed(text):
    """Extract all \\boxed{...} contents (handles 1 level of nesting)."""
    out = []
    i = 0
    while True:
        i = text.find('\\boxed{', i)
        if i < 0:
            break
        i += len('\\boxed{')
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[i:j-1].strip())
            i = j
        else:
            break
    return out


def normalize(s):
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\s+', '', s)
    s = s.replace('$', '').replace('\\,', '').replace('\\!', '').replace('\\;', '')
    return s.lower()


def strict_match(preds, gold_str):
    g = normalize(gold_str)
    for p in preds:
        if normalize(p) == g:
            return True
        # also accept gold as substring of pred
        if g and g in normalize(p):
            return True
    return False


def call_judge(prompt):
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

GOLD (full reference solution; the final numeric/symbolic answer is what matters):
{gold}

CANDIDATE answers (extracted from the model's \\boxed{{}} markers):
{preds}

(If the candidate emitted no \\boxed{{}}, the candidate is the last 600 chars of its full response:)
{tail}

Task: decide whether the candidate's final answer is mathematically/physically equivalent to the gold's final answer.
Allow: different but equivalent algebraic forms; trivial unit/format differences; rounding within 2% relative tolerance; trailing prose.
Reject: different magnitude, different sign, different functional form, missing or wrong physical content, no answer.

Respond with EXACTLY one word: YES or NO."""


def parse_yes_no(t):
    if not t:
        return None
    m = re.search(r'\b(YES|NO)\b', t.strip().upper())
    return m and (m.group(1) == 'YES')


def judge_record(rec):
    resp = rec.get('response', '') or ''
    gold = rec.get('gold', '') or ''
    boxed = extract_boxed(resp)
    strict = strict_match(boxed, gold) if boxed else False
    if not gold:
        return {**rec, 'pred_boxed_list': boxed, 'strict_correct': False,
                'judge_problem_correct': False, 'judge_raw': '', 'judge_err': 'no gold'}
    pred_block = '\n===\n'.join(boxed) if boxed else '[no \\boxed{} extracted]'
    tail = resp[-600:]
    out, err = call_judge(JUDGE_PROMPT.format(gold=gold[:3000], preds=pred_block, tail=tail))
    v = parse_yes_no(out)
    liberal = bool(v) if v is not None else False
    return {**rec, 'pred_boxed_list': boxed, 'strict_correct': strict,
            'judge_problem_correct': liberal, 'judge_raw': out[:80], 'judge_err': err}


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    inp = Path(args.in_path)
    out = inp.with_name(inp.stem + '_judged.jsonl')
    items = load(inp)

    done = set()
    if out.exists():
        for r in load(out):
            done.add(str(r.get('source')) + '||' + str(r.get('topic', '')))
    todo = [r for r in items if (str(r.get('source')) + '||' + str(r.get('topic', ''))) not in done]
    print(f'[{inp.name}] total={len(items)} done={len(done)} todo={len(todo)}', flush=True)

    if not todo:
        return

    lock = threading.Lock()
    n_done = 0
    n_strict = 0
    n_lib = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_record, r): r for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock:
                    with open(out, 'a') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    n_done += 1
                    if rec.get('strict_correct'):
                        n_strict += 1
                    if rec.get('judge_problem_correct'):
                        n_lib += 1
                    if n_done % 25 == 0 or n_done == len(todo):
                        el = time.time() - t0
                        rate = n_done / max(el, 1)
                        eta = (len(todo) - n_done) / max(rate, 1e-6)
                        print(f'  [{n_done}/{len(todo)}] strict={n_strict} ({100*n_strict/n_done:.1f}%) liberal={n_lib} ({100*n_lib/n_done:.1f}%) eta={eta:.0f}s', flush=True)
            except Exception as e:
                print(f'  [ERR] {e}', flush=True)

    rs = load(out)
    s = sum(1 for r in rs if r.get('strict_correct'))
    l = sum(1 for r in rs if r.get('judge_problem_correct'))
    print(f'\n[FINAL] {inp.name}: strict={s}/{len(rs)} = {100*s/len(rs):.2f}% | liberal={l}/{len(rs)} = {100*l/len(rs):.2f}%', flush=True)


if __name__ == '__main__':
    main()
