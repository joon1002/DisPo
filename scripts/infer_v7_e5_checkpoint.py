#!/usr/bin/env python3
"""
infer_v7_e5_checkpoint.py

v7-E5 final_model에서 100 eval 쿼리에 대해 쿼리당 4개(doc0_seed+doc1~doc3) 생성.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_e5_checkpoint.py \
    --checkpoint results/grpo_v7_e5_run1/final_model \
    --input  data/nq100_validate.csv \
    --output results/grpo_v7_e5_run1/pd_eval100_v7_e5.csv \
    --gpu_id 0 --group_size 8
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import train_grpo_poison_v7_e5 as v7e5


def _correct_numbers(doc: str, target_answer: str) -> str:
    """Fix digit fragmentation artifacts (e.g. '19 80' -> '1980'). All occurrences."""
    nums = re.findall(r'\d+', target_answer)
    result = doc
    for num in sorted(set(nums), key=len, reverse=True):
        if re.search(r'(?<!\d)' + re.escape(num) + r'(?!\d)', result):
            continue
        pattern = r'(?<!\d)' + r'\s*'.join(re.escape(d) for d in num) + r'(?!\d)'
        if re.search(pattern, result):
            result = re.sub(pattern, num, result)
    return result


def _ensure_target_in_doc(doc: str, target_answer: str) -> str:
    """If target still absent after digit fix, overwrite Answer: field with correct value."""
    if target_answer.lower() in doc.lower():
        return doc
    ans_pattern = r'(Answer:\s+)([^\n]{1,150})'
    if re.search(ans_pattern, doc, re.IGNORECASE):
        return re.sub(
            ans_pattern,
            lambda m: m.group(1) + target_answer,
            doc, count=1, flags=re.IGNORECASE,
        )
    return doc


def _fix_doc(doc: str, target_answer: str) -> str:
    doc = _correct_numbers(doc, target_answer)
    doc = _ensure_target_in_doc(doc, target_answer)
    return doc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/grpo_v7_e5_run1/final_model")
    p.add_argument("--input",      default="data/nq100_validate.csv")
    p.add_argument("--output",     default="results/grpo_v7_e5_run1/pd_eval100_v7_e5.csv")
    p.add_argument("--gpu_id",      type=int, default=0)
    p.add_argument("--group_size",  type=int, default=8)
    p.add_argument("--embed_device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gpu] {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    # White-box 모델 로드 (E5 + MiniLM + Vicuna)
    v7e5.init_whitebox_models(
        retrieval_model=v7e5.RETRIEVAL_MODEL,
        defense_model=v7e5.DEFENSE_MODEL,
        vicuna_model=v7e5.VICUNA_MODEL,
        device=device,
        embed_device="cuda",
        vicuna_device=device,
        max_prompt_tokens=v7e5.MAX_PROMPT_TOKENS,
    )

    # Generator (LoRA) 로드
    print(f"[load] Base: {v7e5.GENERATOR_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        v7e5.GENERATOR_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    print(f"[load] LoRA: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    model.requires_grad_(False)

    # UncertaintyWeighter 로드
    uw = v7e5.UncertaintyWeighter(n_tasks=5).to(device)
    uw_path = os.path.join(args.checkpoint, "uncertainty_weighter.pt")
    if os.path.exists(uw_path):
        uw.load_state_dict(torch.load(uw_path, map_location=device))
        print(f"[load] UncertaintyWeighter: {uw_path}")
    else:
        print("[warn] uncertainty_weighter.pt not found — using default weights")
    uw.eval()

    # 데이터 로드 + TF-IDF fit
    df = pd.read_csv(args.input)
    print(f"[data] {len(df)} queries from {args.input}")
    v7e5.fit_tfidf(list(df["seed_doc"].astype(str)))
    print("[tfidf] fitted")

    # 추론 (쿼리당 doc0_seed + doc1~doc3 = 4개)
    out_df = v7e5.infer_poison_docs(
        model=model,
        tokenizer=tokenizer,
        uw=uw,
        df=df,
        G=args.group_size,
        min_new=v7e5.MIN_NEW_TOKENS,
        max_new=v7e5.MAX_NEW_TOKENS,
        temp=v7e5.TEMPERATURE,
        device=device,
        max_prompt_tokens=v7e5.MAX_PROMPT_TOKENS,
    )

    out_df = out_df.rename(columns={"doc0": "doc0_seed"})

    # Post-hoc correction: fix digit fragmentation + ensure target answer is present
    doc_cols = [c for c in out_df.columns if c.startswith("doc")]
    for col in doc_cols:
        out_df[col] = out_df.apply(
            lambda r, c=col: _fix_doc(str(r[c]), str(r["target_answer"])), axis=1
        )
    print(f"[post] number correction + target injection applied → {doc_cols}")

    out_df.to_csv(args.output, index=False)
    _meta = {"num_correction_applied": True, "generated_by": __file__, "output": args.output}
    with open(args.output.replace(".csv", ".meta.json"), "w") as _mf:
        json.dump(_meta, _mf, indent=2)
    print(f"[done] {len(out_df)} rows → {args.output}")
    print(f"[cols] {list(out_df.columns)}")


if __name__ == "__main__":
    main()
