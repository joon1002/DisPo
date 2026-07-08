"""
Reranker Defense Experiment

흐름:
  1. pd_eval100_v7.csv 로드 (100쿼리 × 4 poison docs)
  2. BEIR NQ corpus에서 쿼리별 normal docs 추출
  3. Contriever로 top-k=5 retrieval
  4. cross-encoder/ms-marco-MiniLM-L-6-v2 로 reranking 점수 계산
  5. poison docs의 순위 변화 + reranker score 분포 분석

Usage (다른 서버에서 실행):
  git pull
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 python scripts/reranker_defense_exp.py
"""
import argparse, json, math, sys, os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from itertools import combinations

# ─── argparse (서버별 경로 오버라이드) ──────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--docs_csv",    default="/data/joonhyung/nq/results/grpo_whitebox_v7_1.5b_run1/pd_eval100_v7.csv")
_p.add_argument("--corpus",      default="/data/joonhyung/datasets/nq/corpus.jsonl")
_p.add_argument("--qrels_dir",   default="/data/joonhyung/datasets/nq/qrels")
_p.add_argument("--answers_json",default="/data/joonhyung/ragdef/RAGDefender/artifacts/results/target_queries/nq.json")
_p.add_argument("--reranker",    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                help="HuggingFace cross-encoder model ID")
_p.add_argument("--top_k",       type=int, default=5)
_p.add_argument("--adv_per_query", type=int, default=4)
_p.add_argument("--gpu_id",      type=int, default=0)
args = _p.parse_args()

DOCS_CSV      = args.docs_csv
CORPUS_JSONL  = args.corpus
QRELS_DIR     = args.qrels_dir
ANSWERS_JSON  = args.answers_json
ADV_PER_QUERY = args.adv_per_query
TOP_K         = args.top_k
DEVICE        = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

# ─── load BEIR NQ corpus ─────────────────────────────────────
print("[load] BEIR NQ corpus...")
corpus = {}
with open(CORPUS_JSONL) as f:
    for line in f:
        d = json.loads(line)
        corpus[d["_id"]] = {"title": d.get("title",""), "text": d.get("text","")}

# qrels (test)
qrels = {}
qrel_path = os.path.join(QRELS_DIR, "test.tsv")
with open(qrel_path) as f:
    next(f)
    for line in f:
        qid, did, rel = line.strip().split("\t")[:3]
        if int(rel) > 0:
            qrels.setdefault(qid, {})[did] = int(rel)

# title index
title_to_texts = {}
for pid, doc in corpus.items():
    t = doc.get("title", "")
    title_to_texts.setdefault(t, []).append(doc["text"])

# query-to-BEIR-id mapping via answers_json
with open(ANSWERS_JSON) as f:
    ia = json.load(f)
q_to_beir_id = {x["question"].strip(): x["id"] for x in ia}

def get_normal_docs(query_id):
    gt_ids = list(qrels.get(query_id, {}).keys())
    titles  = {corpus[pid]["title"] for pid in gt_ids if pid in corpus}
    docs = []
    for t in titles:
        docs.extend(title_to_texts.get(t, []))
    return docs

print(f"[load] corpus={len(corpus)}, qrels={len(qrels)}")

# ─── load poison docs CSV ────────────────────────────────────
docs_df = pd.read_csv(DOCS_CSV)
DOC_COLS = ["doc0_seed", "doc1", "doc2", "doc3"]
print(f"[load] poison docs: {len(docs_df)} queries")

# ─── Contriever ──────────────────────────────────────────────
print(f"[load] Contriever on {DEVICE}...")
ctv_tok = AutoTokenizer.from_pretrained("facebook/contriever")
ctv_mod = AutoModel.from_pretrained("facebook/contriever", torch_dtype=torch.float32).to(DEVICE)
ctv_mod.eval()

def mean_pool(token_embs, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return torch.sum(token_embs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def encode_ctv(texts, batch_size=32):
    if isinstance(texts, str): texts = [texts]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inp = ctv_tok(batch, padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = ctv_mod(**inp)
        all_embs.append(mean_pool(out.last_hidden_state, inp["attention_mask"]).cpu().numpy())
    return np.vstack(all_embs)

# ─── Cross-Encoder reranker ──────────────────────────────────
print(f"[load] Cross-Encoder: {args.reranker}  (downloading if needed)...")
reranker = CrossEncoder(args.reranker, device=DEVICE)
print("[load] done.")

# ─── main experiment ─────────────────────────────────────────
results = []
no_beir_id = 0

for _, row in docs_df.iterrows():
    query = str(row["query"]).strip()
    target = str(row["target_answer"]).strip()

    beir_id = q_to_beir_id.get(query)
    if beir_id is None:
        no_beir_id += 1
        continue
    normal_docs = get_normal_docs(beir_id)
    if not normal_docs:
        continue

    poison_docs = [str(row[c]) for c in DOC_COLS if pd.notna(row.get(c,""))][:ADV_PER_QUERY]
    candidate_docs = poison_docs + normal_docs
    poison_idx = set(range(len(poison_docs)))

    # ── Contriever retrieval ──────────────────────────────────
    all_texts = candidate_docs + [query]
    all_embs  = encode_ctv(all_texts)
    d_embs    = all_embs[:-1]
    q_emb     = all_embs[-1]
    scores    = d_embs @ q_emb
    ranked    = np.argsort(-scores)[:TOP_K]
    retrieved = [(int(i), candidate_docs[i], i in poison_idx) for i in ranked]

    if not any(is_p for _, _, is_p in retrieved):
        # poison not in top-k, skip for analysis
        results.append({
            "query": query, "poison_in_topk": 0,
            "poison_ctv_ranks": [], "poison_rerank_ranks": [],
            "poison_rerank_scores": [], "normal_rerank_scores": [],
        })
        continue

    # ── Cross-encoder reranking ───────────────────────────────
    pairs = [(query, doc) for _, doc, _ in retrieved]
    rerank_scores = reranker.predict(pairs)   # higher = more relevant

    # rank by reranker score (descending)
    rerank_order  = np.argsort(-rerank_scores)  # index into retrieved list

    ctv_poison_ranks    = [rank for rank, (_, _, is_p) in enumerate(retrieved) if is_p]
    rerank_poison_ranks = [int(np.where(rerank_order == rank)[0][0])
                           for rank, (_, _, is_p) in enumerate(retrieved) if is_p]
    poison_rerank_scores = [float(rerank_scores[rank])
                            for rank, (_, _, is_p) in enumerate(retrieved) if is_p]
    normal_rerank_scores = [float(rerank_scores[rank])
                            for rank, (_, _, is_p) in enumerate(retrieved) if not is_p]

    results.append({
        "query": query,
        "poison_in_topk": sum(1 for _,_,is_p in retrieved if is_p),
        "poison_ctv_ranks": ctv_poison_ranks,
        "poison_rerank_ranks": rerank_poison_ranks,
        "poison_rerank_scores": poison_rerank_scores,
        "normal_rerank_scores": normal_rerank_scores,
    })

print(f"[skip] no BEIR id: {no_beir_id}")

# ─── aggregation ─────────────────────────────────────────────
with_poison = [r for r in results if r["poison_in_topk"] > 0]
print(f"\n쿼리 수: {len(results)} | poison이 top-{TOP_K}에 들어온 쿼리: {len(with_poison)}")

# poison docs rank distribution (0-indexed, lower=better rank)
all_ctv_ranks    = [r for entry in with_poison for r in entry["poison_ctv_ranks"]]
all_rerank_ranks = [r for entry in with_poison for r in entry["poison_rerank_ranks"]]
all_poison_scores  = [s for entry in with_poison for s in entry["poison_rerank_scores"]]
all_normal_scores  = [s for entry in with_poison for s in entry["normal_rerank_scores"]]

print(f"\n========== Contriever top-{TOP_K} retrieval 후 poison doc 순위 ==========")
print(f"  poison docs 수: {len(all_ctv_ranks)}개")
print(f"  Contriever rank (0=1위): mean={np.mean(all_ctv_ranks):.2f} | "
      f"rank0: {all_ctv_ranks.count(0)}, rank1: {all_ctv_ranks.count(1)}, "
      f"rank2: {all_ctv_ranks.count(2)}, rank3: {all_ctv_ranks.count(3)}, rank4: {all_ctv_ranks.count(4)}")
print(f"  Reranker rank  (0=1위): mean={np.mean(all_rerank_ranks):.2f} | "
      f"rank0: {all_rerank_ranks.count(0)}, rank1: {all_rerank_ranks.count(1)}, "
      f"rank2: {all_rerank_ranks.count(2)}, rank3: {all_rerank_ranks.count(3)}, rank4: {all_rerank_ranks.count(4)}")

print(f"\n========== Reranker 점수 비교 ==========")
print(f"  Poison docs  점수: mean={np.mean(all_poison_scores):7.4f}  "
      f"std={np.std(all_poison_scores):.4f}  min={min(all_poison_scores):.4f}  max={max(all_poison_scores):.4f}")
print(f"  Normal docs  점수: mean={np.mean(all_normal_scores):7.4f}  "
      f"std={np.std(all_normal_scores):.4f}  min={min(all_normal_scores):.4f}  max={max(all_normal_scores):.4f}")

# 순위 상승/하락 분석
rank_changes = [r - c for c, r in zip(all_ctv_ranks, all_rerank_ranks)]
demoted = sum(1 for rc in rank_changes if rc > 0)   # 순위 하락 (더 낮은 순위로)
promoted= sum(1 for rc in rank_changes if rc < 0)
same    = sum(1 for rc in rank_changes if rc == 0)
print(f"\n========== Reranking 후 순위 변화 ==========")
print(f"  순위 하락(demoted): {demoted}/{len(rank_changes)} ({demoted/len(rank_changes)*100:.1f}%)")
print(f"  순위 유지(same):    {same}/{len(rank_changes)} ({same/len(rank_changes)*100:.1f}%)")
print(f"  순위 상승(promoted):{promoted}/{len(rank_changes)} ({promoted/len(rank_changes)*100:.1f}%)")
print(f"  평균 순위 변화: {np.mean(rank_changes):+.2f} (+가 하락, -가 상승)")

# 임계값 분석: 몇 개를 제거하면 poison 제거율
print(f"\n========== 방어 시나리오: 하위 N개 제거 (top-{TOP_K}에서) ==========")
# if we remove the bottom-scoring doc after reranking
# poison rank 3 or 4 = would be removed if we drop bottom 1 or 2
for drop_n in [1, 2, 3]:
    removed_poison = sum(1 for r in all_rerank_ranks if r >= (TOP_K - drop_n))
    removed_normal = sum(1 for entry in with_poison
                         for score_idx, s in enumerate(entry["normal_rerank_scores"])
                         if sorted(entry["normal_rerank_scores"])[::-1].index(s) >= (TOP_K - drop_n - entry["poison_in_topk"]))
    total_poison = len(all_rerank_ranks)
    print(f"  하위 {drop_n}개 제거: poison 제거율 = {removed_poison}/{total_poison} ({removed_poison/total_poison*100:.1f}%)")

