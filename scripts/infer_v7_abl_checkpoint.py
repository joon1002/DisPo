#!/usr/bin/env python3
"""
infer_v7_abl_checkpoint.py

v7-abl final_model에서 nq100_validate 100개 쿼리에 대해 poison docs 생성.
--ablation 인자로 훈련 시 사용한 ablation 모드를 지정해야 함 (UW 로드 시 n_tasks 맞춤).

Usage:
  CUDA_VISIBLE_DEVICES=0 /data_ssd/joonhyung/DisPo/.venv/bin/python \\
    /data_ssd/joonhyung/DisPo/scripts/infer_v7_abl_checkpoint.py \\
    --ablation no_disp_embed \\
    --checkpoint /data_ssd/joonhyung/DisPo/data/final_model_abl_nodisp \\
    --output    /data_ssd/joonhyung/DisPo/data/generated/pd_eval100_v7_abl_no_disp_g8_b4.csv \\
    --gpu_id [gpu_id] --group_size 8 --gen_batch_size 4 --N 4
"""
import argparse, json, os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_grpo_poison_v7_abl as v7abl


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
    p.add_argument("--ablation",    required=True,
                   choices=["none", "no_retrieval", "no_disp_embed",
                            "no_tfidf_disp", "no_generation", "no_ppl"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--input",       default=os.path.join(PROJECT_ROOT, "data", "nq100_validate.csv"))
    p.add_argument("--output",      required=True)
    p.add_argument("--gpu_id",      type=int, default=0)
    p.add_argument("--group_size",  type=int, default=8)
    p.add_argument("--gen_batch_size", type=int, default=4,
                   help="한 번의 generate 호출에서 샘플링할 후보 수. 기본 4로 G=8 후보를 두 번에 나누어 생성.")
    p.add_argument("--num_adv_docs", type=int, default=v7abl.DEFAULT_NUM_ADV_DOCS,
                   help="seed 제외 추가 생성 문서 수. 기본 3 -> 총 N=4. --N 지정 시 무시됨")
    p.add_argument("--N", type=int, default=None,
                   help="seed 포함 총 악성문서 수. 예: --N 4 -> doc0_seed+doc1~doc3")
    return p.parse_args()

def main():
    args = parse_args()
    if args.N is not None:
        if args.N < 2:
            raise ValueError("--N must be at least 2 (doc0_seed + generated docs)")
        args.num_adv_docs = args.N - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gpu] Using {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
    print(f"[ablation] {args.ablation}")

    active_tasks = v7abl.get_active_tasks(args.ablation)
    print(f"[active_tasks] {active_tasks}")
    task_set = set(active_tasks)

    v7abl.init_whitebox_models(
        retrieval_model=v7abl.RETRIEVAL_MODEL,
        defense_model=v7abl.DEFENSE_MODEL,
        vicuna_model=v7abl.VICUNA_MODEL,
        device=device,
        embed_device="cuda",
        vicuna_device=device,
        max_prompt_tokens=v7abl.MAX_PROMPT_TOKENS,
        load_retrieval="retrieval" in task_set,
        load_defense="disp_embed" in task_set,
        load_vicuna=bool({"generation", "ppl"} & task_set),
    )

    print(f"[load] Base model: {v7abl.GENERATOR_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        v7abl.GENERATOR_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    print(f"[load] LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    model.requires_grad_(False)

    uw = v7abl.UncertaintyWeighter(active_tasks=active_tasks).to(device)
    uw_path = os.path.join(args.checkpoint, "uncertainty_weighter.pt")
    if os.path.exists(uw_path):
        uw.load_state_dict(torch.load(uw_path, map_location=device))
        print(f"[load] UncertaintyWeighter loaded ({len(active_tasks)} tasks)")
    else:
        print("[warn] uncertainty_weighter.pt not found — using default weights")
    uw.eval()

    df = pd.read_csv(args.input)
    print(f"[data] {len(df)} queries from {args.input}")
    v7abl.fit_tfidf(list(df["seed_doc"].astype(str)))
    print("[tfidf] Vectorizer fitted")

    print(f"[cfg] G={args.group_size}, gen_batch_size={args.gen_batch_size}, "
          f"N={args.num_adv_docs + 1} (seed 포함)")
    out_df = v7abl.infer_poison_docs(
        model=model,
        tokenizer=tokenizer,
        uw=uw,
        df=df,
        G=args.group_size,
        min_new=v7abl.MIN_NEW_TOKENS,
        max_new=v7abl.MAX_NEW_TOKENS,
        temp=v7abl.TEMPERATURE,
        device=device,
        max_prompt_tokens=v7abl.MAX_PROMPT_TOKENS,
        num_adv_docs=args.num_adv_docs,
        gen_batch_size=args.gen_batch_size,
    )

    out_df = out_df.rename(columns={"doc0": "doc0_seed"})

    # Post-hoc correction: fix digit fragmentation + ensure target answer is present
    doc_cols = [c for c in out_df.columns if c.startswith("doc")]
    for col in doc_cols:
        out_df[col] = out_df.apply(
            lambda r, c=col: _fix_doc(str(r[c]), str(r["target_answer"])), axis=1
        )
    print(f"[post] number correction + target injection applied → {doc_cols}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    _meta = {
        "num_correction_applied": True,
        "generated_by": __file__,
        "output": args.output,
        "ablation": args.ablation,
        "active_tasks": active_tasks,
        "group_size": args.group_size,
        "gen_batch_size": args.gen_batch_size,
        "N": args.num_adv_docs + 1,
    }
    with open(args.output.replace(".csv", ".meta.json"), "w") as _mf:
        json.dump(_meta, _mf, indent=2)
    print(f"[done] Saved {len(out_df)} rows → {args.output}")

if __name__ == "__main__":
    main()
