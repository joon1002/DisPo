#!/usr/bin/env python3
"""
infer_v7_n.py

N개의 악성문서(seed doc 포함)를 생성하는 inference 스크립트.
  --num_adv_docs N : seed doc 포함 총 N개 (doc0_seed + doc1 ~ doc{N-1})
  내부적으로 v7.infer_poison_docs(num_adv_docs=N-1)을 호출.

Usage:
  CUDA_VISIBLE_DEVICES=2 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/infer_v7_n.py \
    --checkpoint results/<run>/final_model \
    --input  data/nq100_validate.csv \
    --output results/<run>/pd_eval100_v7_n4.csv \
    --gpu_id 2 \
    --num_adv_docs 4
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import train_grpo_poison_v7 as v7


def _correct_numbers(doc: str, target_answer: str) -> str:
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
    p.add_argument("--checkpoint",  default="results/grpo_v7_run1/final_model")
    p.add_argument("--input",       default="data/nq100_validate.csv")
    p.add_argument("--output",      default="results/grpo_v7_run1/pd_eval100_v7_n4.csv")
    p.add_argument("--gpu_id",      type=int, default=0)
    p.add_argument("--group_size",  type=int, default=8)
    p.add_argument("--num_adv_docs", type=int, default=4,
                   help="seed doc 포함 총 악성문서 수 N (doc0_seed + doc1~doc{N-1}). "
                        "예: N=4 → seed+doc1+doc2+doc3, N=2 → seed+doc1")
    p.add_argument("--embed_device",  default="cuda")
    p.add_argument("--allow_train_input", action="store_true")
    return p.parse_args()

_TRAIN_KEYWORDS = ["_train", "pd_7b", "pd_7", "nq_500", "nq_800", "nq_350", "nq_600"]

def _check_not_train_input(input_path: str):
    path_lower = input_path.lower()
    if any(kw in path_lower for kw in _TRAIN_KEYWORDS):
        sys.exit(
            f"\n[GUARD] 훈련 쿼리 데이터셋 입력 거부: {input_path}\n"
            "  inference는 평가용 nq100_validate.csv만 허용합니다.\n"
            "  훈련 쿼리 inference가 정말 필요하면 --allow_train_input 플래그를 추가하세요.\n"
        )

def main():
    args = parse_args()
    if not args.allow_train_input:
        _check_not_train_input(args.input)

    N = args.num_adv_docs
    if N < 2:
        sys.exit("[ERROR] --num_adv_docs는 최소 2 이상이어야 합니다 (seed + 1개 이상).")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gpu] Using {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
    print(f"[cfg] N={N} (seed doc 포함) → {N-1}개 추가 생성 (doc1~doc{N-1})")

    v7.init_whitebox_models(
        retrieval_model=v7.RETRIEVAL_MODEL,
        defense_model=v7.DEFENSE_MODEL,
        vicuna_model=v7.VICUNA_MODEL,
        device=device,
        embed_device=args.embed_device,
        vicuna_device=device,
        max_prompt_tokens=v7.MAX_PROMPT_TOKENS,
    )

    print(f"[load] Base model: {v7.GENERATOR_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        v7.GENERATOR_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    print(f"[load] LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    model.requires_grad_(False)

    uw = v7.UncertaintyWeighter(n_tasks=5).to(device)
    uw_path = os.path.join(args.checkpoint, "uncertainty_weighter.pt")
    if os.path.exists(uw_path):
        uw.load_state_dict(torch.load(uw_path, map_location=device))
        print(f"[load] UncertaintyWeighter loaded from {uw_path}")
    else:
        print("[warn] uncertainty_weighter.pt not found — using default weights")
    uw.eval()

    df = pd.read_csv(args.input)
    print(f"[data] {len(df)} queries from {args.input}")
    v7.fit_tfidf(list(df["seed_doc"].astype(str)))
    print("[tfidf] Vectorizer fitted")

    # N-1개 추가 생성 (seed 제외)
    out_df = v7.infer_poison_docs(
        model=model,
        tokenizer=tokenizer,
        uw=uw,
        df=df,
        G=args.group_size,
        min_new=v7.MIN_NEW_TOKENS,
        max_new=v7.MAX_NEW_TOKENS,
        temp=v7.TEMPERATURE,
        device=device,
        max_prompt_tokens=v7.MAX_PROMPT_TOKENS,
        num_adv_docs=N - 1,
    )

    out_df = out_df.rename(columns={"doc0": "doc0_seed"})

    doc_cols = [c for c in out_df.columns if c.startswith("doc")]
    for col in doc_cols:
        out_df[col] = out_df.apply(
            lambda r, c=col: _fix_doc(str(r[c]), str(r["target_answer"])), axis=1
        )
    print(f"[post] number correction + target injection applied → {doc_cols}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    _meta = {
        "num_adv_docs_N": N,
        "num_generated": N - 1,
        "generated_by": __file__,
        "output": args.output,
    }
    with open(args.output.replace(".csv", ".meta.json"), "w") as _mf:
        json.dump(_meta, _mf, indent=2)
    print(f"[done] Saved {len(out_df)} rows → {args.output}")
    print(f"[cols] {list(out_df.columns)}")

if __name__ == "__main__":
    main()
