#!/usr/bin/env python3
"""Batched PhyX evaluation — 5-10x faster than sequential."""
import argparse,base64,csv,json,os,re,sys,time
from collections import Counter

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="/workspace/models/Qwen3-VL-32B-Thinking")
    p.add_argument("--adapter",default="")
    p.add_argument("--phyx-path",default="/workspace/phyx_data/data_tsv_vlmevalkit/PhyX_mini_MC.tsv")
    p.add_argument("--num-questions",type=int,default=0)
    p.add_argument("--output-dir",default="results")
    p.add_argument("--output-name",default="phyx_eval.json")
    p.add_argument("--max-tokens",type=int,default=2048)
    p.add_argument("--tp",type=int,default=1)
    return p.parse_args()

def extract_answer(r):
    r=r.strip()
    if '</think>' in r: r=r.split('</think>')[-1].strip()
    m=re.search(r'\\boxed\{([A-D])\}',r)
    if m: return m.group(1)
    for p in [r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\(?([A-D])\)?",r"\*\*([A-D])\*\*\s*$",r"\b([A-D])\b\s*\.?\s*$",r"\(([A-D])\)"]:
        m=re.search(p,r)
        if m: return m.group(1)
    letters=re.findall(r'\b([A-D])\b',r)
    return letters[-1] if letters else "X"

def main():
    args=parse_args()
    os.makedirs(args.output_dir,exist_ok=True)

    # Load data
    csv.field_size_limit(sys.maxsize)
    with open(args.phyx_path) as f: rows=list(csv.DictReader(f,delimiter='\t'))
    if args.num_questions>0: rows=rows[:args.num_questions]
    print(f"Loaded {len(rows)} questions")

    # Build all conversations upfront
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest

    conversations = []
    for q in rows:
        content = []
        if q.get('image'):
            try:
                url = f"data:image/jpeg;base64,{q['image']}"
                content.append({"type":"image_url","image_url":{"url":url}})
            except: pass
        content.append({"type":"text","text":q['question']+"\nPlease provide your final answer as a single letter (A, B, C, or D)."})
        conversations.append([{"role":"user","content":content}])

    # Load vLLM
    ea = {"model":args.model,"dtype":"bfloat16","max_model_len":args.max_tokens+16384,
          "gpu_memory_utilization":0.85,"trust_remote_code":True,
          "tensor_parallel_size":args.tp}
    if args.adapter:
        ea["enable_lora"]=True; ea["max_lora_rank"]=64
    llm=LLM(**ea)
    sp=SamplingParams(max_tokens=args.max_tokens,temperature=0)
    lr=LoRARequest("eval",1,args.adapter) if args.adapter else None

    # Batch process in chunks of 50
    BATCH=1000
    print(f"Running batched inference on {len(conversations)} questions (batch={BATCH})...")
    t0=time.time()
    outputs=[]
    for i in range(0,len(conversations),BATCH):
        batch=conversations[i:i+BATCH]
        batch_out=llm.chat(batch,sampling_params=sp,lora_request=lr)
        outputs.extend(batch_out)
        elapsed=time.time()-t0
        print(f"  [{i+len(batch)}/{len(conversations)}] {elapsed:.0f}s ({(i+len(batch))/elapsed:.1f} q/s)")
    elapsed=time.time()-t0
    print(f"Inference done in {elapsed:.0f}s ({len(conversations)/elapsed:.1f} q/s)")

    # Score
    correct=0
    total=0
    cat_correct=Counter()
    cat_total=Counter()
    results=[]

    for i,(q,out) in enumerate(zip(rows,outputs)):
        resp=out.outputs[0].text
        pred=extract_answer(resp)
        gt=q.get('answer','').strip()
        is_correct=pred==gt
        if is_correct: correct+=1
        total+=1

        cat=q.get('subfield',q.get('category','unknown'))
        cat_total[cat]+=1
        if is_correct: cat_correct[cat]+=1

        results.append({"question_id":i,"predicted":pred,"ground_truth":gt,"correct":is_correct})

        if (i+1)%100==0:
            print(f"  [{i+1}/{len(rows)}] Acc: {100*correct/total:.1f}%")

    acc=correct/max(total,1)
    print(f"\n  Overall: {100*acc:.1f}% ({correct}/{total})")
    for cat in sorted(cat_total.keys()):
        c=cat_correct[cat]; t=cat_total[cat]
        print(f"    {cat:<25} {100*c/t:.1f}% ({c}/{t})")

    # Save
    out_path=os.path.join(args.output_dir,args.output_name)
    with open(out_path,'w') as f:
        json.dump({"accuracy":acc,"correct":correct,"total":total,
                   "per_category":{k:{"correct":cat_correct[k],"total":v,"accuracy":cat_correct[k]/v}
                                   for k,v in cat_total.items()},
                   "results":results},f,indent=2)
    print(f"  Saved to {out_path}")

if __name__=="__main__":
    main()
