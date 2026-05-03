"""PhyX-3k MC eval via closed APIs (GPT-4o + Gemini 2.5 Pro).

Loads /tmp/tbd20_results/PhyX_MC.tsv (3000 multimodal rows), sends image + question
to each API, extracts A/B/C/D letter, scores against gold. Resume-safe (skips ids
already in output JSONL).

Usage:
  python3 eval_phyx_3k_api.py --model gpt-4o --workers 30
  python3 eval_phyx_3k_api.py --model gemini-2.5-pro --workers 20
"""
import argparse, asyncio, base64, csv, json, os, re, sys, time
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def extract_letter(r):
    if not r:
        return 'X'
    if '</think>' in r:
        r = r.split('</think>')[-1]
    r = r.strip()
    m = re.search(r'\\boxed\{([A-D])\}', r)
    if m: return m.group(1)
    for p in [
        r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\*\*([A-D])\*\*\s*$",
        r"\b([A-D])\b\s*\.?\s*$",
        r"\(([A-D])\)",
    ]:
        m = re.search(p, r)
        if m: return m.group(1)
    letters = re.findall(r'\b([A-D])\b', r)
    return letters[-1] if letters else 'X'


def load_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                done.add(str(json.loads(line)['id']))
            except Exception:
                pass
    return done


async def call_gpt4o(client, row, model='gpt-4o', max_retries=4):
    prompt = row['question'] + "\n\nThink briefly, then put your final letter (A/B/C/D) in \\boxed{}."
    img_b64 = row['image']
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = await client.chat.completions.create(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                        {'type': 'text', 'text': prompt},
                    ],
                }],
                max_tokens=2048,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ''
            return text, time.time() - t0, None
        except Exception as e:
            wait = 2 ** attempt + 1
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
            else:
                return '', time.time() - t0, f'attempt {attempt+1}: {str(e)[:200]}'


async def call_gemini(row, model='gemini-2.5-pro', max_retries=4):
    """Uses google-generativeai sync API in a thread."""
    import google.generativeai as genai
    prompt = row['question'] + "\n\nThink briefly, then put your final letter (A/B/C/D) in \\boxed{}."
    img_bytes = base64.b64decode(row['image'])

    def _sync_call():
        m = genai.GenerativeModel(model)
        try:
            r = m.generate_content(
                [{'mime_type': 'image/jpeg', 'data': img_bytes}, prompt],
                generation_config={'max_output_tokens': 2048, 'temperature': 0.0},
            )
            try:
                return r.text or ''
            except Exception:
                cs = getattr(r, 'candidates', [])
                if cs:
                    return ''.join(getattr(p, 'text', '') for p in getattr(cs[0].content, 'parts', []))
                return ''
        except Exception as e:
            raise e

    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            text = await asyncio.to_thread(_sync_call)
            if not text:
                last_err = 'empty response (likely safety block)'
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt + 1)
                    continue
                return '', time.time() - t0, f'attempt {attempt+1}: empty'
            return text, time.time() - t0, None
        except Exception as e:
            last_err = str(e)
            wait = 2 ** attempt + 1
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
            else:
                return '', time.time() - t0, f'attempt {attempt+1}: {last_err[:200]}'


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['gpt-4o', 'gemini-2.5-pro'])
    ap.add_argument('--tsv', default='/tmp/tbd20_results/PhyX_MC.tsv')
    ap.add_argument('--out-dir', default='/tmp/tbd20_results')
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    out_path = os.path.join(args.out_dir, f'phyx_3k_{args.model}.jsonl')
    rows = load_tsv(args.tsv)
    if args.limit:
        rows = rows[:args.limit]

    done = load_done(out_path)
    todo = [r for r in rows if str(r['index']) not in done]
    print(f'[{args.model}] tsv={len(rows)} done={len(done)} todo={len(todo)} workers={args.workers}', flush=True)

    if not todo:
        print('Nothing to do.')
        return

    if args.model == 'gpt-4o':
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
    else:
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        client = None

    sem = asyncio.Semaphore(args.workers)
    lock = asyncio.Lock()
    n_done = 0
    n_correct = 0
    t0 = time.time()

    async def task(row):
        nonlocal n_done, n_correct
        async with sem:
            if args.model == 'gpt-4o':
                text, lat, err = await call_gpt4o(client, row)
            else:
                text, lat, err = await call_gemini(row)
            pred = extract_letter(text)
            gold = row['answer']
            ok = pred == gold
            rec = {
                'id': row['index'],
                'model': args.model,
                'gold': gold,
                'pred_letter': pred,
                'correct': ok,
                'category': row['category'],
                'subfield': row['subfield'],
                'reasoning_type': row['reasoning_type'],
                'latency': lat,
                'error': err,
                'raw_len': len(text),
                'raw_tail': text[-300:] if text else '',
            }
            async with lock:
                with open(out_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                n_done += 1
                if ok:
                    n_correct += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    rate = n_done / max(el, 1)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f'  [{n_done}/{len(todo)}] correct={n_correct}/{n_done} ({100*n_correct/max(n_done,1):.1f}%) '
                          f'elapsed={el:.0f}s eta={eta:.0f}s', flush=True)

    await asyncio.gather(*[task(r) for r in todo])

    # Final tally
    rs = []
    with open(out_path) as f:
        for line in f:
            try: rs.append(json.loads(line))
            except: pass
    n = len(rs)
    c = sum(1 for r in rs if r.get('correct'))
    print(f'\n[FINAL] {args.model} PhyX-3k: {c}/{n} = {100*c/max(n,1):.2f}%', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
