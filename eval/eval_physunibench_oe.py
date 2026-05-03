"""eval PhysUniBench-OE on a vLLM model. Saves raw responses to JSONL for later sonnet judging."""
import argparse, base64, json, os, time

os.environ.setdefault("HF_TOKEN", "os.getenv("HF_TOKEN", "")")
import torch
torch.backends.cudnn.enabled = False

from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """Solve the physics problem below. Show step-by-step reasoning. Put each final sub-answer in \\boxed{}.

Problem:
{problem_text}
"""


def build_messages(item, img_dir):
    qtext = (item.get("question", "") or "").replace("<image>", "").strip()
    prompt = PROMPT_TEMPLATE.replace("{problem_text}", qtext)
    img_name = item.get("image", "")
    img_path = os.path.join(img_dir, img_name) if img_name else None
    if img_path and os.path.exists(img_path):
        with open(img_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}]
    return [{"role": "user", "content": prompt}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    args = p.parse_args()

    items = json.load(open(args.data))
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} PhysUniBench-OE problems", flush=True)

    print("Initializing vLLM...", flush=True)
    llm = LLM(
        model=args.model, tensor_parallel_size=args.tp,
        trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=20480,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)
    convos = [build_messages(it, args.img_dir) for it in items]
    print(f"Generating {len(convos)} prompts...", flush=True)
    t0 = time.time()
    outs = llm.chat(convos, sp)
    print(f"Gen done in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for it, out in zip(items, outs):
            text = out.outputs[0].text or ""
            f.write(json.dumps({
                "id": str(it.get("id")),
                "benchmark": "physunibench_oe",
                "gold": it.get("answer", ""),
                "subtopic": it.get("subtopic"),
                "difficulty": it.get("difficulty"),
                "response": text,
                "raw_len": len(text),
            }, ensure_ascii=False) + "\n")

    print(f"[GEN DONE] {len(items)} responses saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
