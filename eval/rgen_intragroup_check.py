#!/usr/bin/env python3
"""
rgen_intragroup_check.py

Final gate check before deciding whether to retrain:
  - within-group Kendall tau (SA vs FC rank agreement)
  - sigma_generation (r_gen variance within a group)

If tau >= 0.85 or sigma_gen is large, the effect of retraining is diluted.
If both values are low, retraining is justified.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval/rgen_intragroup_check.py
"""
import sys, math
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import train_grpo_poison as tgp

# ── Config ───────────────────────────────────────────────────
CKPT      = str(_ROOT / "results/grpo_n2_q500_run1/final_model")
INPUT_CSV = str(_ROOT / "data/nq_train_validate/nq_500_pd_7b.csv")
N_QUERIES = 20     # how many queries to sample
G         = 8      # group size (same as training)
DEVICE    = "cuda"
MAX_TOKENS = 768
NLL_SHIFT  = 2.0

_VSYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# ── Load Vicuna ──────────────────────────────────────────────
print("[load] Vicuna-7B ...")
vic_tok = AutoTokenizer.from_pretrained(tgp.VICUNA_MODEL, use_fast=True)
if vic_tok.pad_token is None:
    vic_tok.pad_token = vic_tok.eos_token
vic_mod = AutoModelForCausalLM.from_pretrained(
    tgp.VICUNA_MODEL, torch_dtype=torch.float16, device_map={"": DEVICE}
)
vic_mod.eval(); vic_mod.requires_grad_(False)
print("[load] Vicuna done.")

# ── Load the generator (N2 final_model) ──────────────────────
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

# ── r_gen (both formats) ─────────────────────────────────────
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

# ── Generate candidates ──────────────────────────────────────
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

# ── Main ─────────────────────────────────────────────────────
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
print("  Final results (within-group averages)")
print("="*55)
print(f"  Kendall tau (SA vs FC) : {np.mean(group_taus):.4f}  +/-{np.std(group_taus):.4f}")
print(f"  sigma_generation (SA)  : {np.mean(group_sigma_sa):.4f}  +/-{np.std(group_sigma_sa):.4f}")
print(f"  sigma_generation (FC)  : {np.mean(group_sigma_fc):.4f}  +/-{np.std(group_sigma_fc):.4f}")
print()
print("  Decision criteria:")
print(f"  tau={np.mean(group_taus):.3f}  ->  ", end="")
if np.mean(group_taus) >= 0.85:
    print("SA/FC rankings nearly identical -> retraining effect diluted, retraining unnecessary")
elif np.mean(group_taus) >= 0.60:
    print("Moderate disagreement -> recommend a minirun to double-check")
else:
    print("SA/FC rankings differ substantially -> retraining justified")

print(f"  sigma_gen(FC)={np.mean(group_sigma_fc):.4f} ->  ", end="")
if np.mean(group_sigma_fc) >= 0.08:
    print("FC r_gen variance sufficient -> signal strength OK")
elif np.mean(group_sigma_fc) >= 0.04:
    print("FC r_gen variance moderate -> signal present but weak")
else:
    print("FC r_gen variance very small -> no within-group differentiation -> risk of gradient dilution")
