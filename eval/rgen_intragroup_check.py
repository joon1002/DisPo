#!/usr/bin/env python3
"""
rgen_intragroup_check.py

재훈련 결정 전 최종 게이트 확인:
  - within-group Kendall τ (SA vs FC 순위 일치도)
  - σ_generation (group 내 r_gen 분산)

τ ≥ 0.85 이거나 σ_gen이 크면 재훈련 효과 희석.
두 값이 낮으면 재훈련 정당화.

Usage:
  CUDA_VISIBLE_DEVICES=0 /path/to/nq/.venv/bin/python \
    /path/to/DiPoison/eval/rgen_intragroup_check.py
"""
import sys, math
sys.path.insert(0, "/path/to/nq/scripts")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import train_grpo_poison as tgp

# ── 설정 ─────────────────────────────────────────────────────
CKPT      = "/path/to/DiPoison/results/grpo_n2_q500_run1/final_model"
INPUT_CSV = "/path/to/DiPoison/data/nq_500_pd_7b.csv"
N_QUERIES = 20     # 몇 개 query를 샘플링할지
G         = 8      # group size (훈련과 동일)
DEVICE    = "cuda"
MAX_TOKENS = 768
NLL_SHIFT  = 2.0

_VSYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# ── Vicuna 로드 ───────────────────────────────────────────────
print("[load] Vicuna-7B ...")
vic_tok = AutoTokenizer.from_pretrained(tgp.VICUNA_MODEL, use_fast=True)
if vic_tok.pad_token is None:
    vic_tok.pad_token = vic_tok.eos_token
vic_mod = AutoModelForCausalLM.from_pretrained(
    tgp.VICUNA_MODEL, torch_dtype=torch.float16, device_map={"": DEVICE}
)
vic_mod.eval(); vic_mod.requires_grad_(False)
print("[load] Vicuna done.")

# ── Generator (N2 final_model) 로드 ──────────────────────────
print(f"[load] Generator {CKPT} ...")
gen_tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
if gen_tok.pad_token is None:
    gen_tok.pad_token = gen_tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    tgp.GENERATOR_MODEL, torch_dtype=torch.float16, device_map="auto"
)
gen_mod = PeftModel.from_pretrained(base, CKPT)
gen_mod.eval(); gen_mod.requires_grad_(False)
print("[load] Generator done.")

# ── r_gen (두 포맷) ───────────────────────────────────────────
def _rgen(prompt: str, target: str) -> float:
    ids_p = vic_tok(prompt, return_tensors="pt",
                    truncation=True, max_length=MAX_TOKENS).input_ids.to(DEVICE)
    ids_t = vic_tok(" " + target.strip(), return_tensors="pt",
                    add_special_tokens=False).input_ids.to(DEVICE)
    if ids_t.shape[1] == 0:
        return 0.5
    full = torch.cat([ids_p, ids_t], dim=-1)
    with torch.no_grad():
        out = vic_mod(full, use_cache=False)
    logits = out.logits[0, ids_p.shape[1]-1: ids_p.shape[1]-1+ids_t.shape[1], :]
    nll = float(F.cross_entropy(logits, ids_t[0], reduction="mean").item())
    return float(torch.sigmoid(torch.tensor(-nll + NLL_SHIFT)).item())

def rgen_sa(doc, query, target):
    p = tgp._RAG_PROMPT.format(context=doc, question=query)
    return _rgen(p, target)

def rgen_fc(doc, query, target):
    raw = tgp._RAG_PROMPT.format(context=doc, question=query)
    p = f"{_VSYS} USER: {raw} ASSISTANT:"
    return _rgen(p, target)

# ── 후보 생성 ─────────────────────────────────────────────────
def gen_candidates(query, target, seed, prev_docs=[]):
    prompt_text = tgp.format_prompt(gen_tok, query, target, seed, prev_docs)
    enc = gen_tok(prompt_text, return_tensors="pt",
                  truncation=True, max_length=768).to(DEVICE)
    cands = []
    with torch.no_grad():
        for _ in range(G):
            out = gen_mod.generate(
                **enc,
                min_new_tokens=tgp.MIN_NEW_TOKENS,
                max_new_tokens=tgp.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=tgp.TEMPERATURE,
                top_p=tgp.TOP_P,
                repetition_penalty=tgp.REPETITION_PEN,
                pad_token_id=gen_tok.eos_token_id,
                use_cache=False,
            )
            comp = out[:, enc["input_ids"].shape[1]:]
            txt = gen_tok.decode(comp[0], skip_special_tokens=True).strip()
            cands.append(txt)
    return cands

# ── 메인 ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
tgp.fit_tfidf(list(df["seed_doc"].astype(str)))

sample = df.sample(N_QUERIES, random_state=42).reset_index(drop=True)

group_taus, group_sigma_sa, group_sigma_fc = [], [], []

for i, row in tqdm(sample.iterrows(), total=N_QUERIES, desc="queries"):
    query  = str(row["query"])
    target = str(row["target_answer"])
    seed   = str(row["seed_doc"])

    cands = gen_candidates(query, target, seed)

    sa_vals = [rgen_sa(c, query, target) for c in cands]
    fc_vals = [rgen_fc(c, query, target) for c in cands]

    tau, _ = kendalltau(sa_vals, fc_vals)
    sig_sa  = float(np.std(sa_vals))
    sig_fc  = float(np.std(fc_vals))

    group_taus.append(tau)
    group_sigma_sa.append(sig_sa)
    group_sigma_fc.append(sig_fc)

    print(f"  q{i+1:02d} | τ={tau:.3f} | σ_SA={sig_sa:.4f} | σ_FC={sig_fc:.4f} "
          f"| SA={np.mean(sa_vals):.3f} | FC={np.mean(fc_vals):.3f}")

print("\n" + "="*55)
print("  최종 결과 (within-group 평균)")
print("="*55)
print(f"  Kendall τ (SA vs FC)   : {np.mean(group_taus):.4f}  ±{np.std(group_taus):.4f}")
print(f"  σ_generation (SA)      : {np.mean(group_sigma_sa):.4f}  ±{np.std(group_sigma_sa):.4f}")
print(f"  σ_generation (FC)      : {np.mean(group_sigma_fc):.4f}  ±{np.std(group_sigma_fc):.4f}")
print()
print("  판단 기준:")
print(f"  τ={np.mean(group_taus):.3f}  →  ", end="")
if np.mean(group_taus) >= 0.85:
    print("SA/FC 순위 거의 동일 → 재훈련 효과 희석, 재훈련 불필요")
elif np.mean(group_taus) >= 0.60:
    print("중간 불일치 → minirun으로 추가 확인 권장")
else:
    print("SA/FC 순위 크게 다름 → 재훈련 정당화")

print(f"  σ_gen(FC)={np.mean(group_sigma_fc):.4f} →  ", end="")
if np.mean(group_sigma_fc) >= 0.08:
    print("FC r_gen 분산 충분 → 신호 강도 OK")
elif np.mean(group_sigma_fc) >= 0.04:
    print("FC r_gen 분산 보통 → 신호는 있으나 약함")
else:
    print("FC r_gen 분산 매우 작음 → group 내 차별화 불가 → gradient 희석 위험")
