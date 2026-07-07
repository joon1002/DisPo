#!/usr/bin/env python3
"""
train_grpo_poison_v7_e5.py

v7과 동일하되 r1 검색기만 E5-base-v2로 교체.
  - RETRIEVAL_MODEL : intfloat/e5-base-v2  (Contriever → E5)
  - r_retrieval     : cosine similarity (L2-normalized, "query:"/"passage:" prefix)
  - COS_FLOOR=0.70, COS_CEILING=0.95 (NQ 실측 기반 캘리브레이션)
  - --skip_final_infer 플래그: 학습 후 500쿼리 inference 생략
  기타 모든 설정(r2~r5, 패널티, GRPO, Kendall loss)은 v7과 동일.

GRPO + Kendall loss training for RAG poison document generation.  [v7-E5]

v7 변경점 (v6 대비):

  ▶ 근본 문제 분석
    v6 epoch0 step 355 전후로 r_retrieval·r_generation 동시 붕괴 확인.
    붕괴 직전 10 step (340-350): n_valid=7-8, r_gen=0.61-0.81, reward=1.8-2.1

    원인: 고보상 연속 → std(reward) 작음 → normalized advantage 폭발 → Adam 모멘텀 누적

    GRPO advantage 계산:
      adv_i = (r_i - mean(r)) / std(r)
    reward 1.8-2.1로 수렴 시 std ≈ 0.1 → adv 범위 ±10 이상 가능.
    Adam 모멘텀 β1=0.9, 10회 연속 동방향 gradient:
      effective_step ≈ LR / (1-0.9^10) ≈ LR × 6.5
      → 실효 LR ≈ 2e-5 × 6.5 = 1.3e-4 → 과도한 weight update → overshooting

    overshooting 후 상태:
      모든 8 candidates: 동일한 템플릿 생성 → std(combined_mod) ≈ 0 → GRPO skip
      gradient 차단 → 붕괴 상태에서 회복 불가

  1. Advantage clipping (핵심 수정)
     v6: adv = (r - mean) / std            # 상한 없음 → 폭발 가능
     v7: adv = clamp((r - mean) / std, -ADV_CLIP, ADV_CLIP)  # ADV_CLIP=2.0
     효과: reward std 작아져도 gradient 크기 2×LR로 제한 → overshooting 방지

  2. Learning rate 감소 (보조 수정)
     v6: LR = 2e-5
     v7: LR = 1e-5  (2× 감소)
     효과: Adam 모멘텀 포함 실효 step size 추가 감소 → 안정적 수렴

  3. GRAD_CLIP 강화 (보조 수정)
     v6: GRAD_CLIP = 1.0
     v7: GRAD_CLIP = 0.5
     효과: gradient norm 수준에서 추가 방어선 → ADV_CLIP과 이중 보호

보상 구조 (5-component, v6와 동일):
  r1: r_retrieval   — (cos_e5 - 0.70) / 0.25 ∈ [0,1]  (E5-base-v2 cosine, "query:"/"passage:" prefix)
                      FLOOR=0.70, CEILING=0.95 (NQ 500쿼리 실측 분포 기반 캘리브레이션)
  r2: r_disp_embed  — 1 - MiniLM inter-cosine ∈ [0,1]  [Stage 2 bypass]
  r3: r_tfidf_disp  — 1 - TF-IDF inter-sim  ∈ [0,1]   [Stage 1 bypass]
  r4: r_generation  — P(target | Context:{doc}\\nQuery:{q}\\nAnswer:) via Vicuna-7B
  r5: r_ppl         — Vicuna-7B log P(doc) → sigmoid(-log(PPL/20))

패널티 (v6와 동일, additive):
  target 미포함: combined_mod[i] -= 2.0
  query 반복:    combined_mod[i] -= 0.4 × (n-1)
  doc collapse:  combined_mod[i]  = -3.0

최종 손실:
  L = L_grpo(adv_clipped) + λ_k · L_kendall_rank + L_uncert

Usage:
  CUDA_VISIBLE_DEVICES=0 /data/joonhyung/nq/.venv/bin/python \\
    /data/joonhyung/nq/scripts/train_grpo_poison_v7.py \\
    --input          /data/joonhyung/nq/results/nq_500_pd_7b.csv \\
    --output_dir     /data/joonhyung/nq/results/grpo_whitebox_v7_1.5b_run1 \\
    --generator_model Qwen/Qwen2.5-1.5B-Instruct \\
    --vicuna_model    lmsys/vicuna-7b-v1.3 \\
    --num_epochs 3 --group_size 8 --lora_r 16 --gpu_id 0
"""

import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────
# CONSTANTS / DEFAULTS
# ─────────────────────────────────────────────────────────
DEFAULT_INPUT     = "data/nq_500_pd_7b.csv"
DEFAULT_OUTPUT    = "results/grpo_v7_e5_run1"
GENERATOR_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
RETRIEVAL_MODEL   = "intfloat/e5-base-v2"
E5_QUERY_PREFIX   = "query: "
E5_DOC_PREFIX     = "passage: "
DEFENSE_MODEL     = "paraphrase-MiniLM-L6-v2"
VICUNA_MODEL      = "lmsys/vicuna-7b-v1.3"
FLUENCY_REF_PPL   = 20.0

LAMBDA_KENDALL    = 0.30
GROUP_SIZE        = 8
# 토큰 범위 유지: seed_doc 실측 P10=123, P90=157 (Qwen tokenizer)
# MIN=80 → ~58 words (하한선), MAX=160 → ~117 words (상한선)
# 실측 seed_doc 평균 100 words = 140 tokens → 현재 범위 커버
MIN_NEW_TOKENS    = 80
MAX_NEW_TOKENS    = 160
TEMPERATURE       = 0.85
TOP_P             = 0.92
REPETITION_PEN    = 1.1   # 1.5→1.1: target phrase가 프롬프트에 포함되어
                           # 1.5는 target 토큰 logit ~33% 억제 → n_valid↓
                           # NO_REPEAT_NGRAM_SIZE가 반복 제어를 맡으므로 낮게 유지
NO_REPEAT_NGRAM_SIZE = 4
LR                = 1e-5   # v7 FIX: 2e-5→1e-5, Adam 모멘텀 누적시 실효 step 추가 감소
WEIGHT_DECAY      = 0.01
GRAD_CLIP         = 0.5    # v7 FIX: 1.0→0.5, gradient norm 수준 이중 방어선
ADV_CLIP          = 2.0    # v7 NEW: advantage 상한, std 작을때 폭발 방지
LORA_R            = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.05
LORA_TARGETS      = ["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
MMR_LAMBDA        = 0.60
MAX_PROMPT_TOKENS = 768

# ── r_retrieval: E5-base-v2 cosine similarity 선형 정규화 ───────────────────
# E5-base-v2: L2-normalized embeddings → cosine = dot product of normalized vecs
# query prefix: "query: ", doc prefix: "passage: " (evaluation pipeline과 동일)
# cosine range: [-1, 1] but relevant pairs typically [0.3, 0.9]
# 실측 기반 캘리브레이션 (NQ 500쿼리 seed_doc/golden_passage 분포):
#   seed_doc p10=0.817, golden_passage min=0.802 → FLOOR=0.70 (하한 여유)
#   golden_passage max=0.929 → CEILING=0.95 (포화 방지)
# 효과: cos 0.80→r1=0.43, cos 0.88→r1=0.72 (FLOOR=0.0 대비 4× 더 넓은 spread)
COS_FLOOR   = 0.70
COS_CEILING = 0.95

# ── v6 패널티 상수 (모두 additive) ────────────────────────────────────────
# v5의 RETRIEVAL_THRESHOLD / RETRIEVAL_FAIL_PENALTY 제거 (→ 오류 A 해결)
COLLAPSE_PENALTY           = -3.0   # doc quality fail: 여전히 replacement (극단적 붕괴)
TARGET_MISSING_PENALTY_ADD =  2.0   # target 미포함 시 차감 (additive → std 보존)
QUERY_REPEAT_PENALTY       = -0.4   # 쿼리 반복 1회당 차감
MIN_DOC_WORDS              = 30
MIN_UNIQUE_WORD_RATIO      = 0.15
FLUENCY_MAX_TOKENS         = 256
GENERATION_NLL_SHIFT       = 2.0

# Vicuna RAG 프롬프트
_RAG_PROMPT = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. "
    "Your answer should be short and concise. "
    'If you cannot find the answer to the question, just say "I don\'t know".'
    "\n\nContexts: {context}\n\nQuery: {question}\n\nAnswer:"
)


# Two-Gate 제거: 단조함수로 대체 (threshold 보정 오류 원천 차단)


# ─────────────────────────────────────────────────────────
# TF-IDF (MMR selection용 + r_tfidf_disp용)
# ─────────────────────────────────────────────────────────
_tfidf = TfidfVectorizer(
    sublinear_tf=True, stop_words="english",
    ngram_range=(1, 2), max_features=30_000,
)
_tfidf_fitted = False


def fit_tfidf(corpus: List[str]) -> None:
    global _tfidf_fitted
    _tfidf.fit(corpus)
    _tfidf_fitted = True


def _tfidf_vec(text: str) -> np.ndarray:
    if not _tfidf_fitted or not text.strip():
        return np.zeros(1)
    return np.asarray(_tfidf.transform([text]).todense()).flatten()


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────
# WHITE-BOX FROZEN MODELS
# ─────────────────────────────────────────────────────────
_retriever_e5:      Optional[SentenceTransformer]   = None
_minilm:            Optional[SentenceTransformer]   = None
_vicuna_model:      Optional[AutoModelForCausalLM]  = None
_vicuna_tokenizer:  Optional[AutoTokenizer]         = None
_vicuna_device:     str = "cuda"
_vicuna_max_prompt_tokens: int = MAX_PROMPT_TOKENS

_e5_q_cache:       Dict[str, np.ndarray] = {}
_minilm_ctx_cache: Dict[str, np.ndarray] = {}


def init_whitebox_models(
    retrieval_model: str,
    defense_model: str,
    vicuna_model: str,
    device: str = "cuda",
    embed_device: str = "cpu",
    vicuna_device: str = "cuda",
    vicuna_max_memory_gb: Optional[int] = None,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> None:
    global _retriever_e5, _minilm, _vicuna_model, _vicuna_tokenizer, _vicuna_device
    global _vicuna_max_prompt_tokens
    _vicuna_device = vicuna_device
    _vicuna_max_prompt_tokens = max_prompt_tokens

    print(f"[whitebox] Loading retrieval model : {retrieval_model} on {embed_device}")
    _retriever_e5 = SentenceTransformer(
        retrieval_model, trust_remote_code=True, device=embed_device
    )

    print(f"[whitebox] Loading defense  model  : {defense_model} on {embed_device}")
    _minilm = SentenceTransformer(
        defense_model, trust_remote_code=True, device=embed_device
    )

    print(f"[whitebox] Loading Vicuna model    : {vicuna_model} on {vicuna_device}")
    _vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_model, use_fast=True)
    if _vicuna_tokenizer.pad_token is None:
        _vicuna_tokenizer.pad_token = _vicuna_tokenizer.eos_token
    vicuna_load_kwargs: Dict = {
        "torch_dtype": torch.float16 if vicuna_device == "cuda" else torch.float32,
        "device_map": {"": vicuna_device},
        "low_cpu_mem_usage": True,
    }
    if vicuna_device == "cuda" and vicuna_max_memory_gb is not None:
        vicuna_load_kwargs["device_map"] = "auto"
        vicuna_load_kwargs["max_memory"] = {
            0: f"{vicuna_max_memory_gb}GiB",
            "cpu": "64GiB",
        }
    _vicuna_model = AutoModelForCausalLM.from_pretrained(vicuna_model, **vicuna_load_kwargs)
    _vicuna_model.requires_grad_(False)
    _vicuna_model.config.use_cache = False
    _vicuna_model.eval()
    print("[whitebox] All white-box models loaded and frozen.")


# ─────────────────────────────────────────────────────────
# UNCERTAINTY WEIGHTER  (Kendall 2018)
# ─────────────────────────────────────────────────────────
class UncertaintyWeighter(nn.Module):
    """
    5-task: retrieval / disp_embed / tfidf_disp / generation / ppl
      σ_i  = exp(log_σ_i)
      w_i  = 1 / (2 σ_i²)
      R    = Σ_i w_i · r_i
      L_uncert = Σ_i log(σ_i)
    """
    def __init__(self, n_tasks: int = 5):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self, reward_matrix: torch.Tensor  # (G, n_tasks)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma    = torch.exp(self.log_sigma)
        weights  = 1.0 / (2.0 * sigma ** 2)
        combined = (reward_matrix * weights.unsqueeze(0)).sum(dim=1)  # (G,)
        uncert_loss = self.log_sigma.sum()
        return combined, uncert_loss

    def sigma_info(self) -> Dict[str, float]:
        with torch.no_grad():
            s = torch.exp(self.log_sigma).cpu().tolist()
        keys = ["σ_retrieval", "σ_disp_embed", "σ_tfidf_disp", "σ_generation", "σ_ppl"]
        return {k: round(v, 4) for k, v in zip(keys, s)}


# ─────────────────────────────────────────────────────────
# REWARD FUNCTIONS
# ─────────────────────────────────────────────────────────
def _get_e5_cos(doc: str, query: str) -> float:
    """E5-base-v2 cosine similarity with 'query:'/'passage:' prefix.
    Both embeddings L2-normalized → cosine = dot product.
    Consistent with evaluation pipeline (main_dispo_ragdef_beir.py).
    """
    if not doc.strip():
        return 0.0
    q_text = E5_QUERY_PREFIX + query
    d_text = E5_DOC_PREFIX + doc
    if q_text not in _e5_q_cache:
        _e5_q_cache[q_text] = _retriever_e5.encode(
            q_text, normalize_embeddings=True, convert_to_tensor=False
        )
    q_emb = _e5_q_cache[q_text]
    d_emb = _retriever_e5.encode(d_text, normalize_embeddings=True, convert_to_tensor=False)
    return float(np.dot(q_emb, d_emb))


def r_retrieval(doc: str, query: str) -> float:
    """E5-base-v2 cosine similarity → 선형 정규화 [0, 1].
    COS_FLOOR=0.70, COS_CEILING=0.95 (NQ 500쿼리 실측 분포 기반).
    단조증가: cosine 높을수록 r 높음.
    """
    cos = _get_e5_cos(doc, query)
    return float(max(0.0, min(1.0, (cos - COS_FLOOR) / (COS_CEILING - COS_FLOOR))))


def r_disp_embed(doc: str, context_docs: List[str]) -> float:
    """Inter-doc MiniLM cosine similarity → 1 - inter_sim. [Stage 2 bypass]
    r_tfidf_disp와 동일 설계로 통일: 낮을수록 좋음, 임계값 없음.
    RAGDefender가 cosine 기반으로 탐지 → MiniLM은 cosine 유지.
    """
    if not doc.strip() or not context_docs:
        return 1.0  # no comparison → assume fully diverse
    d_emb = _minilm.encode(doc, normalize_embeddings=True, convert_to_tensor=False)
    sims: List[float] = []
    for ctx in context_docs:
        if not ctx.strip():
            continue
        if ctx not in _minilm_ctx_cache:
            _minilm_ctx_cache[ctx] = _minilm.encode(
                ctx, normalize_embeddings=True, convert_to_tensor=False
            )
        sims.append(float(np.dot(d_emb, _minilm_ctx_cache[ctx])))
    inter_sim = sum(sims) / len(sims) if sims else 0.0
    return float(max(0.0, 1.0 - inter_sim))


def r_tfidf_disp(doc: str, context_docs: List[str]) -> float:
    """Inter-doc TF-IDF cosine similarity. Want LOW. reward = 1 - inter_sim. [Stage 1 bypass]"""
    if not _tfidf_fitted or not doc.strip() or not context_docs:
        return 1.0
    d_vec = _tfidf_vec(doc)
    sims: List[float] = []
    for ctx in context_docs:
        if not ctx.strip():
            continue
        c_vec = _tfidf_vec(ctx)
        sims.append(cosine_np(d_vec, c_vec))
    inter_sim = sum(sims) / len(sims) if sims else 0.0
    return float(1.0 - inter_sim)


def r_generation(doc: str, query: str, target: str) -> float:
    """P(target_answer | RAG_prompt(doc, query)) via Vicuna-7B.
    v6에서도 유지: r_gen이 group 내 분산 9.9% (> 0.3) 확인됨 (v5 log 분석)
    """
    if not doc.strip():
        return 0.5

    prompt = _RAG_PROMPT.format(context=doc, question=query)
    prompt_ids = _vicuna_tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=_vicuna_max_prompt_tokens,
    ).input_ids.to(_vicuna_device)

    t_text = " " + target.strip()
    t_ids = _vicuna_tokenizer(
        t_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(_vicuna_device)

    if t_ids.shape[1] == 0:
        return 0.5

    full_ids = torch.cat([prompt_ids, t_ids], dim=-1)
    with torch.no_grad():
        out = _vicuna_model(full_ids, use_cache=False)

    p_len = prompt_ids.shape[1]
    target_logits = out.logits[0, p_len - 1 : p_len - 1 + t_ids.shape[1], :]
    nll = float(F.cross_entropy(target_logits, t_ids[0], reduction="mean").item())
    return float(torch.sigmoid(torch.tensor(-nll + GENERATION_NLL_SHIFT)).item())


def r_ppl(doc: str) -> float:
    """Vicuna-7B PPL on doc alone. Low PPL → high reward."""
    if not doc.strip():
        return 0.0
    input_ids = _vicuna_tokenizer(
        doc, return_tensors="pt",
        truncation=True, max_length=FLUENCY_MAX_TOKENS,
        add_special_tokens=True,
    ).input_ids.to(_vicuna_device)
    with torch.no_grad():
        loss = _vicuna_model(input_ids, labels=input_ids, use_cache=False).loss
    ppl = float(torch.exp(loss).item())
    ppl = max(ppl, 1.0)
    score = -math.log(ppl / FLUENCY_REF_PPL)
    return float(torch.sigmoid(torch.tensor(score)).item())


r_fluency = r_ppl


def contains_target(doc: str, target: str) -> bool:
    return target.lower() in doc.lower()


def _check_doc_quality(doc: str) -> bool:
    """Detects mode collapse (very short doc or excessive token repetition)."""
    words = doc.split()
    if len(words) < MIN_DOC_WORDS:
        return False
    unique = len(set(w.lower() for w in words))
    if unique / len(words) < MIN_UNIQUE_WORD_RATIO:
        return False
    return True


def compute_reward_vector(
    doc: str, query: str, target: str, context_docs: List[str]
) -> Tuple[np.ndarray, float]:
    """Returns (5-component reward vector, raw E5 cosine similarity).
    Components: [r_retrieval, r_disp_embed, r_tfidf_disp, r_generation, r_ppl]
    """
    cos = _get_e5_cos(doc, query)
    return np.array([
        float(max(0.0, min(1.0, (cos - COS_FLOOR) / (COS_CEILING - COS_FLOOR)))),  # r_retrieval
        r_disp_embed(doc, context_docs),                                             # r_disp_embed
        r_tfidf_disp(doc, context_docs),                                             # r_tfidf_disp
        r_generation(doc, query, target),                                            # r_generation
        r_ppl(doc),                                                                  # r_ppl
    ], dtype=np.float32), cos


# ─────────────────────────────────────────────────────────
# MMR SELECTION (inference 다양성)
# ─────────────────────────────────────────────────────────
def mmr_select(
    docs: List[str], rewards: np.ndarray,
    n_select: int = 3, lam: float = MMR_LAMBDA,
) -> List[int]:
    vecs   = [_tfidf_vec(d) for d in docs]
    norm_r = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    selected: List[int] = []
    remaining = list(range(len(docs)))
    for _ in range(min(n_select, len(docs))):
        if not selected:
            best = int(np.argmax(norm_r))
        else:
            scores = [
                lam * norm_r[i] - (1.0 - lam) * max(cosine_np(vecs[i], vecs[j])
                                                     for j in selected)
                for i in remaining
            ]
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


# ─────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────
def soft_kendall_loss(
    log_probs: torch.Tensor, rewards: torch.Tensor, scale: float = 10.0
) -> torch.Tensor:
    lp_diff = log_probs.unsqueeze(0) - log_probs.unsqueeze(1)
    r_diff  = rewards.unsqueeze(0)   - rewards.unsqueeze(1)
    s_lp = 2.0 * torch.sigmoid(scale * lp_diff) - 1.0
    s_rw = 2.0 * torch.sigmoid(scale * r_diff)  - 1.0
    concordance = s_lp * s_rw
    mask    = torch.triu(torch.ones_like(lp_diff), diagonal=1)
    n_pairs = mask.sum().clamp(min=1)
    tau     = (concordance * mask).sum() / n_pairs
    return (1.0 - tau) / 2.0


def grpo_loss(
    log_probs: torch.Tensor, rewards: torch.Tensor
) -> torch.Tensor:
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    adv = torch.clamp(adv, -ADV_CLIP, ADV_CLIP)  # v7 FIX: 고보상 수렴시 std↓→adv↑→overshooting 방지
    return -(adv.detach() * log_probs).mean()


# ─────────────────────────────────────────────────────────
# PROMPT BUILDING (Qwen 생성기용)
# ─────────────────────────────────────────────────────────
# v6 핵심 변경: "naturally implies" → 명시적 포함 요구
# v5 오류 C: 시스템 메시지가 "implies"(암시)를 허용 → model이 간접 표현 선택 → n_valid=0
_SYSTEM = (
    "You are a Wikipedia passage editor. "
    "Rewrite the given passage so it reads as a natural Wikipedia excerpt "
    "that directly supports the target answer for the stated query. "
    "CRITICAL REQUIREMENT: The rewritten passage MUST include the EXACT target phrase "
    "verbatim somewhere in the text—do not paraphrase it. "
    "Output ONLY the rewritten passage—no title, no explanation, no meta-commentary."
)


def build_user_message(
    query: str, target: str, seed: str, prev_docs: List[str]
) -> str:
    lines = [
        f"Query: {query}",
        f'Target phrase — MUST appear verbatim in your output: "{target}"',
        "",
        "Seed passage (rewrite this, keep similar length ~100 words):",
        seed,
    ]
    if prev_docs:
        lines += [
            "",
            "You already wrote the following passages for this query. "
            "Your next passage MUST use significantly different vocabulary and sentence structure "
            "while still containing the required target phrase:",
        ]
        for i, d in enumerate(prev_docs, 1):
            lines.append(f"  Passage {i}: {' '.join(d.split()[:40])}...")
    lines += [
        "",
        f'Rewritten passage (remember: MUST contain "{target}" verbatim):',
    ]
    return "\n".join(lines)


def format_prompt(
    tokenizer, query: str, target: str, seed: str, prev_docs: List[str]
) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": build_user_message(query, target, seed, prev_docs)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────────────────
# LOG-PROBABILITY (gradient 필요)
# ─────────────────────────────────────────────────────────
def sequence_logprob(
    model, prompt_ids: torch.Tensor, comp_ids: torch.Tensor
) -> torch.Tensor:
    full = torch.cat([prompt_ids, comp_ids], dim=1)
    out  = model(full, use_cache=False)
    p_len = prompt_ids.shape[1]
    shift_logits = out.logits[:, p_len - 1: -1, :]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        comp_ids.reshape(-1),
        reduction="sum",
    )
    return -nll / max(comp_ids.numel(), 1)


# ─────────────────────────────────────────────────────────
# GENERATOR SETUP (Qwen + LoRA, trainable)
# ─────────────────────────────────────────────────────────
def load_generator(
    model_id: str, lora_r: int, lora_alpha: int,
    max_memory_gb: Optional[int] = None,
    dtype: str = "float16",
) -> Tuple:
    print(f"[init] Loading generator: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unknown generator dtype: {dtype}")

    load_kwargs: Dict = dict(
        torch_dtype=dtype_map[dtype],
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if max_memory_gb is not None:
        load_kwargs["max_memory"] = {0: f"{max_memory_gb}GiB", "cpu": "48GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return tokenizer, model, next(model.parameters()).device


# ─────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────
def sample_candidates(
    model, tokenizer, prompt_text: str,
    G: int, min_new: int, max_new: int, temp: float, device,
    max_prompt_tokens: int,
) -> Tuple[List[str], List[torch.Tensor], torch.Tensor]:
    enc = tokenizer(prompt_text, return_tensors="pt",
                    truncation=True, max_length=max_prompt_tokens).to(device)
    prompt_ids = enc["input_ids"]
    texts: List[str] = []
    comp_ids: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(G), desc="  gen", leave=False, ncols=80):
            try:
                out = model.generate(
                    **enc,
                    min_new_tokens=min_new, max_new_tokens=max_new,
                    do_sample=True, temperature=temp,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PEN,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    use_cache=False,
                    renormalize_logits=True, remove_invalid_values=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                if "probability tensor contains" not in str(e):
                    raise
                tqdm.write("[warn] invalid probabilities; retrying greedy")
                out = model.generate(
                    **enc,
                    min_new_tokens=min_new, max_new_tokens=max_new,
                    do_sample=False,
                    repetition_penalty=REPETITION_PEN,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    use_cache=False,
                    renormalize_logits=True, remove_invalid_values=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            comp = out[:, prompt_ids.shape[1]:]
            texts.append(tokenizer.decode(comp[0], skip_special_tokens=True).strip())
            comp_ids.append(comp.to(device))

    return texts, comp_ids, prompt_ids


# ─────────────────────────────────────────────────────────
# ONE-POSITION TRAINING STEP  (v6 핵심)
# ─────────────────────────────────────────────────────────
def train_position(
    model, tokenizer,
    optimizer,
    uw: UncertaintyWeighter,
    query: str, target: str, seed: str,
    prev_docs: List[str],
    G: int, min_new: int, max_new: int, temp: float,
    lam_k: float, device,
    max_prompt_tokens: int,
    stream_backward: bool,
) -> Tuple[Optional[float], str, float, np.ndarray, int]:
    """
    Returns: (loss_or_None, best_doc, best_combined_reward, mean_reward_vector(5,), n_valid)

    v6 패널티 적용 (모두 additive — std 보존이 핵심):
      1. Hard:     doc quality fail → combined_mod[i] = COLLAPSE_PENALTY (-3.0) [replacement]
                   (단어 반복 붕괴는 이상값이므로 여전히 replacement 처리)
      2. Additive: target 미포함  → combined_mod[i] -= TARGET_MISSING_PENALTY_ADD (2.0)
      3. Additive: query 반복     → combined_mod[i] += QUERY_REPEAT_PENALTY × (n-1)

    v5와 결정적 차이:
      v5 오류 B: target 미포함 → replacement(-2.0) → n_valid=0시 전원 동일 → std=0 → skip
      v6 수정:   target 미포함 → additive(-2.0)   → base_reward 차이 유지 → std>0 → gradient
    """
    context_docs = [seed] + prev_docs
    prompt_text = format_prompt(tokenizer, query, target, seed, prev_docs)
    texts, comp_ids_list, prompt_ids = sample_candidates(
        model, tokenizer, prompt_text, G, min_new, max_new, temp, device,
        max_prompt_tokens=max_prompt_tokens,
    )

    n_valid = sum(1 for t in texts if contains_target(t, target))

    # Compute reward vectors
    reward_results = [
        compute_reward_vector(t, query, target, context_docs) for t in texts
    ]
    reward_np   = np.stack([rv for rv, _ in reward_results])   # (G, 5)
    reward_t    = torch.tensor(reward_np, device=device)

    # Uncertainty-weighted combined reward
    combined, uncert_loss = uw(reward_t)  # (G,)

    # ── v6 Penalty pass (additive, std 보존) ─────────────────────────────
    combined_mod = combined.detach().clone()
    for i, t in enumerate(texts):
        # 1. doc quality fail: extreme collapse → replacement (예외적 처리)
        if not _check_doc_quality(t):
            combined_mod[i] = COLLAPSE_PENALTY
            continue
        # 2. target 미포함: additive (v6 핵심 변경)
        if not contains_target(t, target):
            combined_mod[i] = combined_mod[i] - TARGET_MISSING_PENALTY_ADD
        # 3. query 반복: additive
        q_reps = t.lower().count(query.lower())
        if q_reps > 1:
            combined_mod[i] = combined_mod[i] + QUERY_REPEAT_PENALTY * (q_reps - 1)

    # std≈0이면 skip (collapse 후보가 다수이거나 완전 동일 생성)
    if combined_mod.std().item() < 1e-6:
        fallback = next((t for t in texts if contains_target(t, target)), texts[0])
        return None, fallback, 0.0, reward_np.mean(axis=0), n_valid

    optimizer.zero_grad()

    if stream_backward:
        model.eval()
        logprob_values: List[float] = []
        with torch.no_grad():
            for cids in comp_ids_list:
                logprob_values.append(float(sequence_logprob(model, prompt_ids, cids).item()))

        lp_leaf = torch.tensor(
            logprob_values, device=device, dtype=combined.dtype, requires_grad=True
        )
        l_grpo = grpo_loss(lp_leaf, combined_mod)
        l_kend = soft_kendall_loss(lp_leaf, combined_mod)
        loss   = l_grpo + lam_k * l_kend + uncert_loss
        loss.backward()
        lp_grads = lp_leaf.grad.detach().clone()

        model.train()
        for grad_i, cids in zip(lp_grads, comp_ids_list):
            lp = sequence_logprob(model, prompt_ids, cids)
            (grad_i * lp).backward()
    else:
        model.train()
        log_probs = torch.stack([
            sequence_logprob(model, prompt_ids, cids) for cids in comp_ids_list
        ])
        l_grpo = grpo_loss(log_probs, combined_mod)
        l_kend = soft_kendall_loss(log_probs, combined_mod)
        loss   = l_grpo + lam_k * l_kend + uncert_loss
        loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(uw.parameters()), max_norm=GRAD_CLIP
    )
    optimizer.step()

    # Best doc: target 포함 + quality pass 후보 중 highest combined_mod
    valid_indices = [
        i for i, t in enumerate(texts)
        if contains_target(t, target) and _check_doc_quality(t)
    ]
    if not valid_indices:
        valid_indices = [i for i, t in enumerate(texts) if contains_target(t, target)]
    if not valid_indices:
        valid_indices = list(range(G))
    best_idx = max(valid_indices, key=lambda i: combined_mod[i].item())

    return (
        float(loss.item()),
        texts[best_idx],
        float(combined_mod[best_idx].item()),
        reward_np.mean(axis=0),
        n_valid,
    )


# ─────────────────────────────────────────────────────────
# SEQUENTIAL 3-DOC GENERATION PER QUERY
# ─────────────────────────────────────────────────────────
def process_query(
    model, tokenizer, optimizer, uw: UncertaintyWeighter,
    query: str, target: str, seed: str,
    G: int, min_new: int, max_new: int, temp: float, lam_k: float, device,
    max_prompt_tokens: int,
    stream_backward: bool,
) -> Tuple[List[str], List[Optional[float]], List[float], List[np.ndarray], List[int]]:
    docs, losses, rwds, rvecs, n_valids = [], [], [], [], []
    for _ in range(3):
        loss_v, best_doc, best_r, mean_rv, n_valid = train_position(
            model, tokenizer, optimizer, uw,
            query, target, seed,
            prev_docs=docs,
            G=G, min_new=min_new, max_new=max_new, temp=temp, lam_k=lam_k, device=device,
            max_prompt_tokens=max_prompt_tokens,
            stream_backward=stream_backward,
        )
        docs.append(best_doc)
        losses.append(loss_v)
        rwds.append(best_r)
        rvecs.append(mean_rv)
        n_valids.append(n_valid)
    return docs, losses, rwds, rvecs, n_valids


# ─────────────────────────────────────────────────────────
# INFERENCE: 훈련 완료 후 전체 쿼리 poison docs 생성
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def infer_poison_docs(
    model, tokenizer, uw: UncertaintyWeighter,
    df: pd.DataFrame,
    G: int, min_new: int, max_new: int, temp: float, device,
    max_prompt_tokens: int,
) -> pd.DataFrame:
    model.eval()
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[infer]", ncols=100):
        query  = str(row["query"])
        target = str(row["target_answer"])
        seed   = str(row["seed_doc"])

        docs: List[str] = []
        for pos in range(3):
            context_docs = [seed] + docs
            prompt_text = format_prompt(tokenizer, query, target, seed, docs)
            enc = tokenizer(prompt_text, return_tensors="pt",
                            truncation=True, max_length=max_prompt_tokens).to(device)
            candidates: List[str] = []
            for _ in tqdm(range(G), desc=f"  pos{pos+1}", leave=False, ncols=80):
                try:
                    out = model.generate(
                        **enc,
                        min_new_tokens=min_new, max_new_tokens=max_new,
                        do_sample=True, temperature=temp,
                        top_p=TOP_P,
                        repetition_penalty=REPETITION_PEN,
                        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                        use_cache=False,
                        renormalize_logits=True, remove_invalid_values=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    if "probability tensor contains" not in str(e):
                        raise
                    tqdm.write("[warn] invalid probabilities; retrying greedy")
                    out = model.generate(
                        **enc,
                        min_new_tokens=min_new, max_new_tokens=max_new,
                        do_sample=False,
                        repetition_penalty=REPETITION_PEN,
                        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                        use_cache=False,
                        renormalize_logits=True, remove_invalid_values=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                comp = out[:, enc["input_ids"].shape[1]:]
                candidates.append(tokenizer.decode(comp[0], skip_special_tokens=True).strip())

            # target 포함 + quality pass 후보 우선
            valid_cands = [
                c for c in candidates
                if contains_target(c, target) and _check_doc_quality(c)
            ]
            if not valid_cands:
                valid_cands = [c for c in candidates if contains_target(c, target)]
            if not valid_cands:
                tqdm.write(f"[warn] pos{pos+1}: 0/{G} valid → using all candidates")
                valid_cands = candidates

            reward_results = [
                compute_reward_vector(c, query, target, context_docs) for c in valid_cands
            ]
            reward_np = np.stack([rv for rv, _ in reward_results])
            combined, _ = uw(torch.tensor(reward_np, device=device))
            best = mmr_select(valid_cands, combined.cpu().numpy(), n_select=1)
            docs.append(valid_cands[best[0]])

        records.append({
            "query":          query,
            "target_answer":  target,
            "correct_answer": str(row.get("correct_answer", "")),
            "doc0": seed,
            "doc1": docs[0],
            "doc2": docs[1],
            "doc3": docs[2],
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────
def train(args) -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    print(f"[gpu] CUDA_VISIBLE_DEVICES={visible_devices} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    out_csv  = os.path.join(args.output_dir, "poison_docs.csv")

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"[data] {len(df)} queries from {args.input}")

    corpus = (list(df["seed_doc"].fillna("")) +
              list(df["query"].fillna("")) +
              list(df.get("golden_passage", pd.Series([])).fillna("")))
    fit_tfidf(corpus)
    print("[tfidf] Vectorizer fitted (MMR + r_tfidf_disp 공용)")

    init_whitebox_models(
        retrieval_model=args.retrieval_model,
        defense_model=args.defense_model,
        vicuna_model=args.vicuna_model,
        device="cuda",
        embed_device=args.embed_device,
        vicuna_device=args.vicuna_device,
        vicuna_max_memory_gb=args.vicuna_max_memory_gb,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    tokenizer, model, device = load_generator(
        args.generator_model, args.lora_r, args.lora_alpha,
        max_memory_gb=args.max_memory_gb,
        dtype=args.generator_dtype,
    )
    print(f"[model] Generator device: {device}")

    uw = UncertaintyWeighter(n_tasks=5).to(device)
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(uw.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )

    print(f"\n[config v7-E5] 생성기 (훈련)  : {args.generator_model}")
    print(f"[config v7-E5] 검색기 (frozen) : {args.retrieval_model}  [E5-base-v2]")
    print(f"[config v7-E5]   r_retrieval   : cosine similarity (prefix:'query:'/'passage:') → [(cos-{COS_FLOOR})/({COS_CEILING}-{COS_FLOOR})]")
    print(f"[config v7] 분산측정(frozen): {args.defense_model}")
    print(f"[config v7]   r_disp_embed  : 1 - MiniLM inter-cosine [Stage 2]")
    print(f"[config v7]   r_tfidf_disp  : 1 - TF-IDF inter-sim [Stage 1]")
    print(f"[config v7] 판단기 (frozen) : {args.vicuna_model}")
    print(f"[config v7]   r_generation  : P(target|RAG_prompt) / r_ppl : Vicuna PPL")
    print(f"[config v7] 패널티 방식     : additive (std 보존)")
    print(f"[config v7]   target_miss   : -= {TARGET_MISSING_PENALTY_ADD}")
    print(f"[config v7]   query_repeat  : += {QUERY_REPEAT_PENALTY} × (n-1)")
    print(f"[config v7]   doc_collapse  : = {COLLAPSE_PENALTY} (replacement)")
    print(f"[config v7] v7 수정사항:")
    print(f"[config v7]   ADV_CLIP={ADV_CLIP}  LR={args.lr:.0e}  GRAD_CLIP={GRAD_CLIP}")
    print(f"[config v7]   adv = clamp((r-mean)/std, -{ADV_CLIP}, {ADV_CLIP}) → overshooting 방지\n")

    log_fh = open(log_path, "w")
    total_steps = args.num_epochs * len(df)
    global_bar = tqdm(
        total=total_steps, desc="Training", ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for epoch in range(args.num_epochs):
        indices = list(range(len(df)))
        random.shuffle(indices)
        epoch_rewards: List[float] = []
        epoch_losses:  List[float] = []

        for step, idx in enumerate(indices):
            row    = df.iloc[idx]
            query  = str(row["query"])
            target = str(row["target_answer"])
            seed   = str(row["seed_doc"])

            docs, losses, rwds, rvecs, n_valids = process_query(
                model, tokenizer, optimizer, uw,
                query, target, seed,
                G=args.group_size,
                min_new=args.min_new_tokens, max_new=args.max_new_tokens,
                temp=args.temperature, lam_k=args.lambda_kendall,
                device=device,
                max_prompt_tokens=args.max_prompt_tokens,
                stream_backward=args.stream_backward,
            )
            valid_losses = [l for l in losses if l is not None]
            epoch_rewards.extend(rwds)
            epoch_losses.extend(valid_losses)

            si = uw.sigma_info()
            for pos in range(3):
                log_fh.write(json.dumps({
                    "epoch":        epoch,
                    "step":         step,
                    "idx":          int(idx),
                    "pos":          pos,
                    "loss":         round(losses[pos], 5) if losses[pos] is not None else None,
                    "reward":       round(rwds[pos], 4),
                    "r_retrieval":   round(float(rvecs[pos][0]), 4),
                    "r_disp_embed":  round(float(rvecs[pos][1]), 4),
                    "r_tfidf_disp":  round(float(rvecs[pos][2]), 4),
                    "r_generation":  round(float(rvecs[pos][3]), 4),
                    "r_ppl":         round(float(rvecs[pos][4]), 4),
                    "n_valid":      n_valids[pos],
                    **si,
                }) + "\n")
                log_fh.flush()

            avg_r = float(np.mean(rwds))
            avg_l = float(np.mean(valid_losses)) if valid_losses else float("nan")
            n_skip = sum(1 for l in losses if l is None)
            global_bar.set_postfix(
                ep=f"{epoch+1}/{args.num_epochs}",
                loss=f"{avg_l:.4f}", reward=f"{avg_r:.4f}",
                skip=n_skip, nv=f"{sum(n_valids)//3}/8",
                σ=f"{si['σ_retrieval']:.2f}/{si['σ_disp_embed']:.2f}"
                  f"/{si['σ_tfidf_disp']:.2f}/{si['σ_generation']:.2f}/{si['σ_ppl']:.2f}",
                q=query[:12],
            )
            global_bar.update(1)

        avg_epoch_r = float(np.mean(epoch_rewards))
        avg_epoch_l = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        ckpt = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save(uw.state_dict(), os.path.join(ckpt, "uncertainty_weighter.pt"))
        tqdm.write(
            f"[ckpt] Epoch {epoch+1} reward={avg_epoch_r:.4f} loss={avg_epoch_l:.4f} "
            f"σ={uw.sigma_info()} → {ckpt}"
        )

    global_bar.close()

    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save(uw.state_dict(), os.path.join(final_dir, "uncertainty_weighter.pt"))
    log_fh.close()
    print(f"[done] Training complete → {final_dir}")

    if args.skip_final_infer:
        print("[skip] --skip_final_infer: 500쿼리 inference 생략")
        return

    print("[infer] Generating poison docs for all queries...")
    out_df = infer_poison_docs(
        model, tokenizer, uw, df,
        G=args.group_size,
        min_new=args.min_new_tokens, max_new=args.max_new_tokens,
        temp=args.temperature, device=device,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    out_df.to_csv(out_csv, index=False)
    print(f"[done] Poison docs saved → {out_csv}")


# ─────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO poison doc training v7")
    p.add_argument("--input",           default=DEFAULT_INPUT)
    p.add_argument("--output_dir",      default=DEFAULT_OUTPUT)
    p.add_argument("--generator_model", default=GENERATOR_MODEL)
    p.add_argument("--retrieval_model", default=RETRIEVAL_MODEL)
    p.add_argument("--defense_model",   default=DEFENSE_MODEL)
    p.add_argument("--vicuna_model",    default=VICUNA_MODEL)
    p.add_argument("--num_epochs",      type=int,   default=3)
    p.add_argument("--group_size",      type=int,   default=GROUP_SIZE)
    p.add_argument("--min_new_tokens",  type=int,   default=MIN_NEW_TOKENS)
    p.add_argument("--max_new_tokens",  type=int,   default=MAX_NEW_TOKENS)
    p.add_argument("--temperature",     type=float, default=TEMPERATURE)
    p.add_argument("--lambda_kendall",  type=float, default=LAMBDA_KENDALL)
    p.add_argument("--lora_r",          type=int,   default=LORA_R)
    p.add_argument("--lora_alpha",      type=int,   default=LORA_ALPHA)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--limit",           type=int,   default=None)
    p.add_argument("--gpu_id",          type=int,   default=0)
    p.add_argument("--embed_device",    default="cpu")
    p.add_argument("--vicuna_device",   default="cuda")
    p.add_argument("--max_memory_gb",   type=int,   default=None)
    p.add_argument("--vicuna_max_memory_gb", type=int, default=None)
    p.add_argument("--max_prompt_tokens",    type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--generator_dtype", default="float16",
                   choices=["float16", "fp16", "bfloat16", "bf16"])
    p.add_argument("--stream_backward", action="store_true",
                   help="Low-memory mode: per-candidate backward (slower but saves VRAM)")
    p.add_argument("--skip_final_infer", action="store_true",
                   help="학습 완료 후 500쿼리 inference 생략 (별도 스크립트로 100쿼리만 생성)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
