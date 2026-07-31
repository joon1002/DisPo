#!/usr/bin/env python3
"""
infer_n.py

Inference script that generates N poison documents (including the seed doc).
  --num_adv_docs N : N total documents including the seed doc (doc0_seed + doc1 ~ doc{N-1})
  Internally calls tgp.infer_poison_docs(num_adv_docs=N-1).

Usage:
  CUDA_VISIBLE_DEVICES=2 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/infer_n.py \
    --checkpoint results/<run>/final_model \
    --input  data/nq_train_validate/nq100_validate.csv \
    --output results/<run>/pd_eval100_n4.csv \
    --gpu_id 2 \
    --num_adv_docs 4
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_grpo_poison as tgp


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
    p.add_argument("--checkpoint",  default="results/grpo_run1/final_model")
    p.add_argument("--input",       default="data/nq_train_validate/nq100_validate.csv")
    p.add_argument("--output",      default="results/grpo_run1/pd_eval100_n4.csv")
    p.add_argument("--gpu_id",      type=int, default=0)
    p.add_argument("--group_size",  type=int, default=8)
    p.add_argument("--num_adv_docs", type=int, default=4,
                   help="Total poison documents N including the seed doc (doc0_seed + doc1~doc{N-1}). "
                        "e.g. N=4 -> seed+doc1+doc2+doc3, N=2 -> seed+doc1")
    p.add_argument("--N", type=int, default=None,
                   help="Total poison documents including the seed. Used instead of --num_adv_docs when set (same meaning).")
    p.add_argument("--embed_device",    default="cuda")
    p.add_argument("--gen_batch_size",  type=int, default=1,
                   help="How many of the G candidates to batch-generate at once (default 1=sequential). "
                        "Fastest when set equal to G (generates all G candidates in one batch).")
    p.add_argument("--allow_train_input", action="store_true")
    return p.parse_args()

_TRAIN_KEYWORDS = ["_train", "pd_7b", "pd_7", "nq_500", "nq_800", "nq_350", "nq_600"]

def _check_not_train_input(input_path: str):
    path_lower = input_path.lower()
    if any(kw in path_lower for kw in _TRAIN_KEYWORDS):
        sys.exit(
            f"\n[GUARD] Refusing a training-query dataset as input: {input_path}\n"
            "  Inference only allows the evaluation set nq100_validate.csv.\n"
            "  If you genuinely need inference on training queries, add the --allow_train_input flag.\n"
        )

def main():
    args = parse_args()
    if args.N is not None:
        args.num_adv_docs = args.N
    if not args.allow_train_input:
        _check_not_train_input(args.input)

    N = args.num_adv_docs
    if N < 2:
        sys.exit("[ERROR] --num_adv_docs must be at least 2 (seed + at least 1 additional document).")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gpu] Using {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
    print(f"[cfg] N={N} (including seed doc) -> generating {N-1} additional documents (doc1~doc{N-1})")

    tgp.init_whitebox_models(
        retrieval_model=tgp.RETRIEVAL_MODEL,
        defense_model=tgp.DEFENSE_MODEL,
        vicuna_model=tgp.VICUNA_MODEL,
        device=device,
        embed_device=args.embed_device,
        vicuna_device=device,
        max_prompt_tokens=tgp.MAX_PROMPT_TOKENS,
    )

    print(f"[load] Base model: {tgp.GENERATOR_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        tgp.GENERATOR_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    print(f"[load] LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    model.requires_grad_(False)

    uw = tgp.UncertaintyWeighter(n_tasks=5).to(device)
    uw_path = os.path.join(args.checkpoint, "uncertainty_weighter.pt")
    if os.path.exists(uw_path):
        uw.load_state_dict(torch.load(uw_path, map_location=device))
        print(f"[load] UncertaintyWeighter loaded from {uw_path}")
    else:
        print("[warn] uncertainty_weighter.pt not found — using default weights")
    uw.eval()

    df = pd.read_csv(args.input)
    print(f"[data] {len(df)} queries from {args.input}")
    tgp.fit_tfidf(list(df["seed_doc"].astype(str)))
    print("[tfidf] Vectorizer fitted")

    # Generates N-1 additional documents (excluding the seed). Paper §3.2-3.3: Stage 2 policy generates sequentially -> the artifact right before Stage 3 injection
    print(f"[cfg] gen_batch_size={args.gen_batch_size} (batches of {args.gen_batch_size} out of {args.group_size} G candidates)")
    out_df = tgp.infer_poison_docs(
        model=model,
        tokenizer=tokenizer,
        uw=uw,
        df=df,
        G=args.group_size,
        min_new=tgp.MIN_NEW_TOKENS,
        max_new=tgp.MAX_NEW_TOKENS,
        temp=tgp.TEMPERATURE,
        device=device,
        max_prompt_tokens=tgp.MAX_PROMPT_TOKENS,
        num_adv_docs=N - 1,
        gen_batch_size=args.gen_batch_size,
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
