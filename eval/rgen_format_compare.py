#!/usr/bin/env python3
"""
rgen_format_compare.py

Standalone vs FastChat r_generation 보상 신호 비교.
훈련 없이 기존 docs로 두 포맷의 reward 분포 및 rank 상관관계를 측정.

Usage:
  /data/joonhyung/nq/.venv/bin/python /data/joonhyung/DisPo/eval/rgen_format_compare.py
"""
import math, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

# ── 설정 ─────────────────────────────────────────────────────
VICUNA_MODEL   = "lmsys/vicuna-7b-v1.3"
DEVICE         = "cuda:0"
MAX_TOKENS     = 768
NLL_SHIFT      = 2.0   # train_grpo_poison_v7.py 와 동일

# 평가할 doc 파일 (v7cont N2 + v7e5)
DOC_FILES = [
    ("/data/joonhyung/DisPo/results/grpo_v7_n2_q500_run1/pd_eval100_N2_v2.csv",  "v7cont_v2", "doc0_seed"),
    ("/data/joonhyung/DisPo/results/grpo_v7_n2_q500_run1/pd_eval100_N2_v2.csv",  "v7cont_v2_doc1", "doc1"),
    ("/data/joonhyung/DisPo/data/generated/pd_eval100_v7e5_val_v2.csv",           "v7e5_v2",   "doc0_seed"),
]

# Standalone 프롬프트 (훈련에서 사용하는 _RAG_PROMPT 와 동일)
_STANDALONE_TMPL = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. "
    "Your answer should be short and concise. "
    "If you cannot find the answer to the question, just say \"I don't know\"."
    "\n\nContexts: {context}\n\nQuery: {question}\n\nAnswer:"
)

# FastChat 프롬프트 (평가에서 사용하는 방식)
_VSYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
def _fastchat_prompt(context: str, question: str) -> str:
    rag = _STANDALONE_TMPL.format(context=context, question=question)
    return f"{_VSYS} USER: {rag} ASSISTANT:"

# ── Vicuna 로드 ───────────────────────────────────────────────
print(f"[load] {VICUNA_MODEL} → {DEVICE}")
tok = AutoTokenizer.from_pretrained(VICUNA_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    VICUNA_MODEL, torch_dtype=torch.float16, device_map={"": DEVICE},
)
model.eval()
model.requires_grad_(False)
print("[load] done.")

# ── r_generation 계산 ─────────────────────────────────────────
def compute_rgen(prompt: str, target: str) -> float:
    prompt_ids = tok(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_TOKENS,
    ).input_ids.to(DEVICE)

    t_text = " " + target.strip()
    t_ids = tok(t_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    if t_ids.shape[1] == 0:
        return 0.5

    full_ids = torch.cat([prompt_ids, t_ids], dim=-1)
    with torch.no_grad():
        out = model(full_ids, use_cache=False)

    p_len = prompt_ids.shape[1]
    logits = out.logits[0, p_len - 1: p_len - 1 + t_ids.shape[1], :]
    nll = float(F.cross_entropy(logits, t_ids[0], reduction="mean").item())
    return float(torch.sigmoid(torch.tensor(-nll + NLL_SHIFT)).item())

# ── 메인 ─────────────────────────────────────────────────────
all_results = []

for csv_path, label, doc_col in DOC_FILES:
    if not Path(csv_path).exists():
        print(f"[skip] {csv_path} not found")
        continue
    df = pd.read_csv(csv_path)
    if doc_col not in df.columns:
        print(f"[skip] col {doc_col} not in {csv_path}")
        continue

    print(f"\n[eval] {label} ({doc_col}), N={len(df)}")
    rgen_sa, rgen_fc = [], []

    for i, row in df.iterrows():
        doc    = str(row[doc_col])
        query  = str(row["query"])
        target = str(row["target_answer"])

        sa_prompt = _STANDALONE_TMPL.format(context=doc, question=query)
        fc_prompt = _fastchat_prompt(context=doc, question=query)

        r_sa = compute_rgen(sa_prompt, target)
        r_fc = compute_rgen(fc_prompt, target)
        rgen_sa.append(r_sa)
        rgen_fc.append(r_fc)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}] standalone={r_sa:.3f}  fastchat={r_fc:.3f}")

    rgen_sa = np.array(rgen_sa)
    rgen_fc = np.array(rgen_fc)
    rho, pval = spearmanr(rgen_sa, rgen_fc)
    diff = rgen_fc - rgen_sa

    print(f"\n  ── {label} 결과 ──")
    print(f"  Standalone : mean={rgen_sa.mean():.4f}  std={rgen_sa.std():.4f}  median={np.median(rgen_sa):.4f}")
    print(f"  FastChat   : mean={rgen_fc.mean():.4f}  std={rgen_fc.std():.4f}  median={np.median(rgen_fc):.4f}")
    print(f"  FC - SA    : mean={diff.mean():.4f}  std={diff.std():.4f}  |diff|={np.abs(diff).mean():.4f}")
    print(f"  Spearman ρ : {rho:.4f}  (p={pval:.2e})")

    all_results.append({
        "label": label, "doc_col": doc_col,
        "sa_mean": rgen_sa.mean(), "fc_mean": rgen_fc.mean(),
        "sa_std": rgen_sa.std(),  "fc_std": rgen_fc.std(),
        "diff_mean": diff.mean(), "diff_std": diff.std(),
        "abs_diff": np.abs(diff).mean(), "spearman_rho": rho,
    })

print("\n" + "="*60)
print("  전체 요약")
print("="*60)
print(f"{'파일':<20} {'SA mean':>8} {'FC mean':>8} {'FC-SA':>8} {'Spearman ρ':>12}")
for r in all_results:
    print(f"  {r['label']:<18} {r['sa_mean']:>8.4f} {r['fc_mean']:>8.4f} {r['diff_mean']:>+8.4f} {r['spearman_rho']:>12.4f}")

print("\n해석 기준:")
print("  ρ > 0.9  → 두 포맷이 동일한 docs 선호 → 훈련 포맷 변경 효과 없음")
print("  ρ 0.7~0.9 → 부분 차이 → 추가 확인 필요")
print("  ρ < 0.7  → 포맷이 reward landscape를 크게 바꿈 → 재훈련 고려")
