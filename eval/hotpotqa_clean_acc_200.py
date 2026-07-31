"""
hotpotqa_clean_acc_200.py

1단계: val100origin과 유사하지 않은 HotpotQA 쿼리 200개 선정 (TF-IDF cosine 최소)
2단계: Contriever 전체 corpus 검색(top-5) → Vicuna-7B 생성 → clean ACC 측정

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval/hotpotqa_clean_acc_200.py \
    --gpu_id 0 \
    --out_dir eval/results/hotpotqa_clean_acc_200
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent

_CORPUS_PATH   = "/path/to/datasets/hotpotqa/corpus.jsonl"
_QUERIES_PATH  = "/path/to/datasets/hotpotqa/queries.jsonl"
_EMB_CACHE     = "/path/to/datasets/hotpotqa/contriever_embs_fullcorpus.pt"
_VAL100_CSV    = "/path/to/DiPoison/data/hotpotqa_val100origin.csv"
_VICUNA_MODEL  = "lmsys/vicuna-7b-v1.3"
_CONTRIEVER_HF = "facebook/contriever"

_PROMPT_TMPL = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. Your answer should be short and concise. "
    "If you cannot find the answer to the question, just say \"I don't know\". "
    "\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:"
)


def clean_str(s):
    s = str(s).strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def wrap_prompt(question, docs):
    ctx = "\n".join(docs) if isinstance(docs, list) else docs
    return _PROMPT_TMPL.replace("[question]", question).replace("[context]", ctx)


def mean_pool(token_embs, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return torch.sum(token_embs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_queries(texts, model, tokenizer, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inp = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        all_embs.append(mean_pool(out.last_hidden_state, inp["attention_mask"]))
    return torch.cat(all_embs, dim=0)  # [N, 768]


def select_dissimilar_queries(val100_queries, all_queries, n=200, seed=42):
    """TF-IDF cosine 기반으로 val100과 최대 유사도가 낮은 쿼리 200개 선정."""
    print(f"[select] TF-IDF 벡터화 ({len(all_queries):,} 쿼리)...")
    val100_texts = [q["text"] for q in val100_queries]
    all_texts    = [q["text"] for q in all_queries]

    # val100과 겹치는 쿼리 제거
    val100_set = set(t.lower().strip() for t in val100_texts)
    candidates = [q for q in all_queries if q["text"].lower().strip() not in val100_set]
    cand_texts = [q["text"] for q in candidates]
    print(f"[select] val100 제외 후 후보: {len(candidates):,}")

    # TF-IDF 피팅 (val100 + 후보 전체)
    vec = TfidfVectorizer(max_features=30000, stop_words="english")
    combined = val100_texts + cand_texts
    tfidf_all = vec.fit_transform(combined)

    val100_tfidf = tfidf_all[:len(val100_texts)]
    cand_tfidf   = tfidf_all[len(val100_texts):]

    # 각 후보별 val100 쿼리들과의 최대 cosine 유사도
    print("[select] 유사도 계산 중...")
    chunk = 5000
    max_sims = np.zeros(len(candidates), dtype=np.float32)
    for i in range(0, len(candidates), chunk):
        sims = cosine_similarity(cand_tfidf[i:i+chunk], val100_tfidf)
        max_sims[i:i+chunk] = sims.max(axis=1)

    # 유사도 낮은 순 정렬 → 상위 n개
    order = np.argsort(max_sims)
    rng = np.random.default_rng(seed)
    # 유사도 가장 낮은 500개 중 무작위 200개 (너무 단순한 쿼리 방지)
    pool = order[:500]
    chosen_idx = rng.choice(pool, size=n, replace=False)
    chosen_idx.sort()

    selected = [candidates[i] for i in chosen_idx]
    sel_sims  = [float(max_sims[i]) for i in chosen_idx]
    print(f"[select] 선정된 200개 max-sim 통계: "
          f"min={min(sel_sims):.4f}, max={max(sel_sims):.4f}, avg={sum(sel_sims)/len(sel_sims):.4f}")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",      type=int, default=0)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--out_dir",     type=str,
                        default="/path/to/DiPoison/eval/results/hotpotqa_clean_acc_200")
    parser.add_argument("--n_queries",   type=int, default=200)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--exclude_csv", type=str, default="",
                        help="이미 선정된 쿼리 CSV (text 컬럼). 중복 제외에 사용.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"log_{ts}.txt")
    log_fp = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    log(f"[config] device={device}, top_k={args.top_k}, n_queries={args.n_queries}")

    # ── 1단계: 쿼리 선정 ────────────────────────────────────────────────────────
    log("\n[step1] val100origin 쿼리 로드...")
    val100_df = pd.read_csv(_VAL100_CSV)
    val100_queries = [{"text": str(r["query"]).strip()} for _, r in val100_df.iterrows()]
    log(f"[step1] val100 쿼리 수: {len(val100_queries)}")

    # 이미 선정된 쿼리 제외 목록 구성
    exclude_texts = set()
    if args.exclude_csv and os.path.exists(args.exclude_csv):
        exc_df = pd.read_csv(args.exclude_csv)
        exclude_texts = set(exc_df["text"].str.lower().str.strip())
        log(f"[step1] 기존 선정 쿼리 제외: {len(exclude_texts)}개")

    log("[step1] HotpotQA 전체 쿼리 로드...")
    all_queries = []
    with open(_QUERIES_PATH) as f:
        for line in f:
            d = json.loads(line)
            all_queries.append({
                "_id":    d["_id"],
                "text":   d["text"],
                "answer": d["metadata"].get("answer", ""),
            })
    log(f"[step1] 전체 쿼리 수: {len(all_queries):,}")

    if exclude_texts:
        all_queries = [q for q in all_queries
                       if q["text"].lower().strip() not in exclude_texts]
        log(f"[step1] 기존 선정 제외 후: {len(all_queries):,}")

    selected = select_dissimilar_queries(val100_queries, all_queries,
                                         n=args.n_queries, seed=args.seed)

    sel_csv = os.path.join(args.out_dir, f"selected_{args.n_queries}_queries.csv")
    pd.DataFrame(selected).to_csv(sel_csv, index=False)
    log(f"[step1] 선정 쿼리 저장: {sel_csv}")

    # ── 2단계: Contriever 로드 ──────────────────────────────────────────────────
    log(f"\n[step2] Contriever 로드: {_CONTRIEVER_HF}")
    ctv_tok = AutoTokenizer.from_pretrained(_CONTRIEVER_HF)
    ctv_mod = AutoModel.from_pretrained(_CONTRIEVER_HF, torch_dtype=torch.float32).to(device)
    ctv_mod.eval()
    log("[step2] Contriever 완료")

    # ── 3단계: corpus 텍스트 로드 ───────────────────────────────────────────────
    log("\n[step3] corpus.jsonl 로드 (5.2M passages)...")
    corpus_ids   = []
    corpus_texts = []
    with open(_CORPUS_PATH) as f:
        for line in f:
            d = json.loads(line)
            corpus_ids.append(d["_id"])
            corpus_texts.append(d.get("text", ""))
    log(f"[step3] corpus {len(corpus_texts):,} passages")

    # ── 4단계: corpus 임베딩 GPU 로드 ───────────────────────────────────────────
    log(f"\n[step4] corpus 임베딩 로드: {_EMB_CACHE}")
    corpus_embs = torch.load(_EMB_CACHE, map_location="cpu", weights_only=True)
    log(f"[step4] corpus_embs shape={corpus_embs.shape}, dtype={corpus_embs.dtype}")
    log("[step4] GPU 전송 (float16)...")
    corpus_embs_gpu = corpus_embs.half().to(device)
    del corpus_embs
    gc.collect()
    log(f"[step4] GPU 메모리: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── 5단계: 쿼리 임베딩 ─────────────────────────────────────────────────────
    log("\n[step5] 쿼리 200개 임베딩...")
    query_texts = [q["text"] for q in selected]
    q_embs = encode_queries(query_texts, ctv_mod, ctv_tok, device)  # [200, 768]
    q_embs = q_embs.half()
    log(f"[step5] q_embs shape={q_embs.shape}")

    # Contriever 불필요 → 해제
    del ctv_mod, ctv_tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── 6단계: top-k 검색 ───────────────────────────────────────────────────────
    log(f"\n[step6] top-{args.top_k} 검색...")
    topk_indices = []
    chunk = 50
    for i in range(0, len(q_embs), chunk):
        q_chunk = q_embs[i:i+chunk].to(device)  # [chunk, 768]
        scores  = torch.mm(q_chunk, corpus_embs_gpu.T)  # [chunk, 5.2M]
        topk    = torch.topk(scores, k=args.top_k, dim=1).indices.cpu()
        topk_indices.append(topk)
    topk_indices = torch.cat(topk_indices, dim=0)  # [200, top_k]
    log(f"[step6] 검색 완료. topk_indices shape={topk_indices.shape}")

    del corpus_embs_gpu, q_embs
    gc.collect()
    torch.cuda.empty_cache()
    log(f"[step6] corpus 임베딩 해제 후 GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── 7단계: Vicuna-7B 로드 ───────────────────────────────────────────────────
    log("\n[step7] Vicuna-7B 로드 (fastchat)...")
    try:
        from fastchat.model import load_model, get_conversation_template
    except ImportError:
        raise ImportError("fastchat 없음. 올바른 venv 사용: /path/to/ragatt/.venv")

    llm_model, llm_tok = load_model(
        model_path=_VICUNA_MODEL, device="cuda", num_gpus=1,
        max_gpu_memory=None, dtype=torch.float16,
        load_8bit=False, cpu_offloading=False, revision="main", debug=False,
    )
    llm_model.eval()
    log(f"[step7] Vicuna-7B 완료. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def vicuna_generate(prompt):
        try:
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            input_ids = llm_tok([conv.get_prompt()]).input_ids
            with torch.no_grad():
                out = llm_model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=True, temperature=0.1,
                    repetition_penalty=1.0, max_new_tokens=150,
                )
            return llm_tok.decode(
                out[0][len(input_ids[0]):],
                skip_special_tokens=True, spaces_between_special_tokens=False,
            ).strip()
        except Exception as e:
            return ""

    # ── 8단계: 생성 및 ACC 측정 ─────────────────────────────────────────────────
    log("\n[step8] 생성 및 ACC 측정...")
    rows = []
    acc_cnt = 0

    for i, (q, top_idxs) in enumerate(tqdm(
        zip(selected, topk_indices.tolist()), total=len(selected), desc="eval"
    )):
        docs  = [clean_str(corpus_texts[idx]) for idx in top_idxs]
        prompt = wrap_prompt(q["text"], docs)
        response = vicuna_generate(prompt)

        ans = q["answer"]
        acc = (clean_str(ans) in clean_str(response)
               or clean_str(response) in clean_str(ans))
        if acc:
            acc_cnt += 1

        rows.append({
            "query":          q["text"],
            "answer":         ans,
            "response":       response,
            "accuracy":       acc,
            "retrieved_ids":  ",".join(str(corpus_ids[idx]) for idx in top_idxs),
        })

        if (i + 1) % 20 == 0:
            log(f"  [{i+1}/{len(selected)}] running ACC={acc_cnt/(i+1):.2%}")

    final_acc = acc_cnt / len(rows)
    acc_queries = [r for r in rows if r["accuracy"]]
    log(f"\n[result] N={len(rows)}, Clean ACC = {final_acc:.2%}  ({acc_cnt}/{len(rows)})")
    log(f"[result] ACC 성공 쿼리 수: {len(acc_queries)}")

    # ── 저장 ────────────────────────────────────────────────────────────────────
    res_csv = os.path.join(args.out_dir, f"results_{ts}.csv")
    pd.DataFrame(rows).to_csv(res_csv, index=False)

    # ACC 성공 쿼리만 별도 저장
    acc_csv = os.path.join(args.out_dir, f"acc_success_{ts}.csv")
    pd.DataFrame(acc_queries).to_csv(acc_csv, index=False)
    log(f"[save] ACC 성공 쿼리: {acc_csv}")

    summary = {
        "n_queries":       len(rows),
        "top_k":           args.top_k,
        "retriever":       "contriever",
        "llm":             "vicuna-7b-v1.3",
        "clean_acc":       round(final_acc, 4),
        "acc_count":       acc_cnt,
        "acc_fail_count":  len(rows) - acc_cnt,
        "run_at":          ts,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"[save] {res_csv}")
    log(f"[save] {os.path.join(args.out_dir, 'summary.json')}")
    log_fp.close()


if __name__ == "__main__":
    main()
