"""eval PhysReason on a vLLM model. Saves raw responses to JSONL for later sonnet judging."""
import argparse, base64, glob, json, os, time

os.environ.setdefault("HF_TOKEN", "os.getenv("HF_TOKEN", "")")
import torch
torch.backends.cudnn.enabled = False

from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """Solve the physics problem. Show your reasoning step-by-step. Put each final sub-answer in \\boxed{}.

Problem:
{problem_text}
"""


def load_physreason(root):
    items = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        pj = os.path.join(d, "problem.json")
        if not os.path.exists(pj):
            continue
        rec = json.load(open(pj))
        pid = os.path.basename(d)
        # Build problem text from question_structure (string keys -> sub-questions)
        qs = rec.get("question_structure") or {}
        if isinstance(qs, dict):
            parts = []
            for k in sorted(qs.keys()):
                v = qs[k]
                parts.append(f"({k}) {v}")
            problem_text = "\n\n".join(parts)
        else:
            problem_text = str(qs)
        # Image paths
        img_dir = os.path.join(d, "images")
        img_paths = []
        if os.path.isdir(img_dir):
            img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                if f.lower().endswith((".png",".jpg",".jpeg")) and not f.startswith(".")])
        # gold answer is a list-string
        ans = rec.get("answer", "")
        if isinstance(ans, str) and ans.startswith("["):
            try:
                ans_list = json.loads(ans.replace("\\\\","\\\\\\\\"))
            except Exception:
                ans_list = [ans]
        elif isinstance(ans, list):
            ans_list = ans
        else:
            ans_list = [ans]
        items.append({
            "id": pid,
            "problem_text": problem_text,
            "image_paths": img_paths,
            "gold": ans_list,
            "difficulty": rec.get("difficulty", ""),
        })
    return items


def build_messages(item):
    prompt = PROMPT_TEMPLATE.replace("{problem_text}", item["problem_text"])
    paths = item.get("image_paths") or []
    if not paths:
        return [{"role": "user", "content": prompt}]
    content = []
    for p in paths[:5]:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", default="/workspace/data/physreason_full/PhysReason_full")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    args = p.parse_args()

    items = load_physreason(args.data_root)
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} PhysReason problems", flush=True)

    print("Initializing vLLM...", flush=True)
    llm = LLM(
        model=args.model, tensor_parallel_size=args.tp,
        trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=20480,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 5},
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)
    convos = [build_messages(it) for it in items]
    print(f"Generating {len(convos)} prompts...", flush=True)
    t0 = time.time()
    outs = llm.chat(convos, sp)
    print(f"Gen done in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for it, out in zip(items, outs):
            text = out.outputs[0].text or ""
            f.write(json.dumps({
                "id": it["id"],
                "benchmark": "physreason",
                "gold": it["gold"],
                "difficulty": it.get("difficulty"),
                "response": text,
                "raw_len": len(text),
            }, ensure_ascii=False) + "\n")

    print(f"[GEN DONE] {len(items)} responses saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
