"""DPO fine-tune an OPEN-WEIGHTS proposer on collected preference data (EvoTune-style).

This is the "proposer learns" step. It requires a GPU + `trl`, `transformers`, `torch`,
`peft` (see repo requirements.txt) and is intentionally NOT run in the CPU test suite.
Heavy imports are deferred into main() so importing this module stays cheap.

    python -m cs_discover.rl.train_dpo \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --data data_rl/capset_claude_dpo.jsonl \\
        --out checkpoints/proposer-dpo --epochs 2 --lora

After training, serve the checkpoint (e.g. vLLM) and plug it into the search loop as a
local LLMProposer(call_fn=...), then run the matched-budget ablation (see rl/README.md).
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", required=True, help="DPO jsonl: {prompt, chosen, rejected}")
    ap.add_argument("--out", default="checkpoints/proposer-dpo")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune")
    args = ap.parse_args()

    # deferred heavy imports (GPU stack)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    try:
        from peft import LoraConfig
    except Exception:  # noqa: BLE001
        LoraConfig = None

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    ds = load_dataset("json", data_files=args.data, split="train")  # prompt/chosen/rejected

    peft_config = None
    if args.lora and LoraConfig is not None:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved DPO-trained proposer to {args.out}")


if __name__ == "__main__":
    main()
