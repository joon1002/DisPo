"""
hotpotqa_clean_acc_200.py

Step 1: Select 200 HotpotQA queries dissimilar to val100origin (minimal TF-IDF cosine)
Step 2: Contriever full-corpus retrieval (top-5) -> Vicuna-7B generation -> measure clean ACC

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

_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = os.environ.get("DIPOISON_DATA_ROOT", "/path/to")

_CORPUS_PATH   = f"{_DATA_ROOT}/datasets/hotpotqa/corpus.jsonl"
_QUERIES_PATH  = f"{_DATA_ROOT}/datasets/hotpotqa/queries.jsonl"
_EMB_CACHE     = f"{_DATA_ROOT}/datasets/hotpotqa/contriever_embs_fullcorpus.pt"
_VAL100_CSV    = str(_ROOT / "data/hotpotqa_val100origin.csv")
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
    """Selects the 200 queries with the lowest max TF-IDF cosine similarity to val100."""
    print(f"[select] TF-IDF vectorization ({len(all_queries):,} queries)...")
    val100_texts = [q["text"] for q in val100_queries]
    all_texts    = [q["text"] for q in all_queries]

    # Remove queries overlapping with val100
    val100_set = set(t.lower().strip() for t in val100_texts)
    candidates = [q for q in all_queries if q["text"].lower().strip() not in val100_set]
    cand_texts = [q["text"] for q in candidates]
    print(f"[select] Candidates after excluding val100: {len(candidates):,}")

    # Fit TF-IDF (val100 + all candidates)
    vec = TfidfVectorizer(max_features=30000, stop_words="english")
    combined = val100_texts + cand_texts
    tfidf_all = vec.fit_transform(combined)

    val100_tfidf = tfidf_all[:len(val100_texts)]
    cand_tfidf   = tfidf_all[len(val100_texts):]

    # Max cosine similarity of each candidate to the val100 queries
    print("[select] Computing similarity...")
    chunk = 5000
    max_sims = np.zeros(len(candidates), dtype=np.float32)
    for i in range(0, len(candidates), chunk):
        sims = cosine_similarity(cand_tfidf[i:i+chunk], val100_tfidf)
        max_sims[i:i+chunk] = sims.max(axis=1)

    # Sort by ascending similarity -> take the top n
    order = np.argsort(max_sims)
    rng = np.random.default_rng(seed)
    # Randomly pick 200 out of the 500 lowest-similarity queries (avoids overly trivial queries)
    pool = order[:500]
    chosen_idx = rng.choice(pool, size=n, replace=False)
    chosen_idx.sort()

    selected = [candidates[i] for i in chosen_idx]
    sel_sims  = [float(max_sims[i]) for i in chosen_idx]
    print(f"[select] max-sim stats for the selected 200: "
          f"min={min(sel_sims):.4f}, max={max(sel_sims):.4f}, avg={sum(sel_sims)/len(sel_sims):.4f}")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",      type=int, default=0)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--out_dir",     type=str,
                        default=str(_ROOT / "eval/results/hotpotqa_clean_acc_200"))
    parser.add_argument("--n_queries",   type=int, default=200)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--exclude_csv", type=str, default="",
                        help="CSV of already-selected queries (text column). Used to exclude duplicates.")
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

    # ── Step 1: Select queries ──────────────────────────────────────────────────
    log("\n[step1] Loading val100origin queries...")
    val100_df = pd.read_csv(_VAL100_CSV)
    val100_queries = [{"text": str(r["query"]).strip()} for _, r in val100_df.iterrows()]
    log(f"[step1] val100 query count: {len(val100_queries)}")

    # Build the exclusion list of already-selected queries
    exclude_texts = set()
    if args.exclude_csv and os.path.exists(args.exclude_csv):
        exc_df = pd.read_csv(args.exclude_csv)
        exclude_texts = set(exc_df["text"].str.lower().str.strip())
        log(f"[step1] Excluding previously selected queries: {len(exclude_texts)}")

    log("[step1] Loading all HotpotQA queries...")
    all_queries = []
    with open(_QUERIES_PATH) as f:
        for line in f:
            d = json.loads(line)
            all_queries.append({
                "_id":    d["_id"],
                "text":   d["text"],
                "answer": d["metadata"].get("answer", ""),
            })
    log(f"[step1] Total query count: {len(all_queries):,}")

    if exclude_texts:
        all_queries = [q for q in all_queries
                       if q["text"].lower().strip() not in exclude_texts]
        log(f"[step1] After excluding previously selected: {len(all_queries):,}")

    selected = select_dissimilar_queries(val100_queries, all_queries,
                                         n=args.n_queries, seed=args.seed)

    sel_csv = os.path.join(args.out_dir, f"selected_{args.n_queries}_queries.csv")
    pd.DataFrame(selected).to_csv(sel_csv, index=False)
    log(f"[step1] Saved selected queries: {sel_csv}")

    # ── Step 2: Load Contriever ─────────────────────────────────────────────────
    log(f"\n[step2] Loading Contriever: {_CONTRIEVER_HF}")
    ctv_tok = AutoTokenizer.from_pretrained(_CONTRIEVER_HF)
    ctv_mod = AutoModel.from_pretrained(_CONTRIEVER_HF, torch_dtype=torch.float32).to(device)
    ctv_mod.eval()
    log("[step2] Contriever loaded")

    # ── Step 3: Load corpus texts ───────────────────────────────────────────────
    log("\n[step3] Loading corpus.jsonl (5.2M passages)...")
    corpus_ids   = []
    corpus_texts = []
    with open(_CORPUS_PATH) as f:
        for line in f:
            d = json.loads(line)
            corpus_ids.append(d["_id"])
            corpus_texts.append(d.get("text", ""))
    log(f"[step3] corpus {len(corpus_texts):,} passages")

    # ── Step 4: Load corpus embeddings to GPU ───────────────────────────────────
    log(f"\n[step4] Loading corpus embeddings: {_EMB_CACHE}")
    corpus_embs = torch.load(_EMB_CACHE, map_location="cpu", weights_only=True)
    log(f"[step4] corpus_embs shape={corpus_embs.shape}, dtype={corpus_embs.dtype}")
    log("[step4] Transferring to GPU (float16)...")
    corpus_embs_gpu = corpus_embs.half().to(device)
    del corpus_embs
    gc.collect()
    log(f"[step4] GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 5: Embed queries ───────────────────────────────────────────────────
    log("\n[step5] Embedding 200 queries...")
    query_texts = [q["text"] for q in selected]
    q_embs = encode_queries(query_texts, ctv_mod, ctv_tok, device)  # [200, 768]
    q_embs = q_embs.half()
    log(f"[step5] q_embs shape={q_embs.shape}")

    # Contriever no longer needed -> free it
    del ctv_mod, ctv_tok
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 6: top-k retrieval ─────────────────────────────────────────────────
    log(f"\n[step6] top-{args.top_k} retrieval...")
    topk_indices = []
    chunk = 50
    for i in range(0, len(q_embs), chunk):
        q_chunk = q_embs[i:i+chunk].to(device)  # [chunk, 768]
        scores  = torch.mm(q_chunk, corpus_embs_gpu.T)  # [chunk, 5.2M]
        topk    = torch.topk(scores, k=args.top_k, dim=1).indices.cpu()
        topk_indices.append(topk)
    topk_indices = torch.cat(topk_indices, dim=0)  # [200, top_k]
    log(f"[step6] Retrieval complete. topk_indices shape={topk_indices.shape}")

    del corpus_embs_gpu, q_embs
    gc.collect()
    torch.cuda.empty_cache()
    log(f"[step6] GPU after freeing corpus embeddings: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 7: Load Vicuna-7B ──────────────────────────────────────────────────
    log("\n[step7] Loading Vicuna-7B (fastchat)...")
    try:
        from fastchat.model import load_model, get_conversation_template
    except ImportError:
        raise ImportError("fastchat not found. Use the correct venv: /path/to/ragatt/.venv")

    llm_model, llm_tok = load_model(
        model_path=_VICUNA_MODEL, device="cuda", num_gpus=1,
        max_gpu_memory=None, dtype=torch.float16,
        load_8bit=False, cpu_offloading=False, revision="main", debug=False,
    )
    llm_model.eval()
    log(f"[step7] Vicuna-7B loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

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

    # ── Step 8: Generation and ACC measurement ──────────────────────────────────
    log("\n[step8] Generation and ACC measurement...")
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
    log(f"[result] ACC-success query count: {len(acc_queries)}")

    # ── Save ─────────────────────────────────────────────────────────────────────
    res_csv = os.path.join(args.out_dir, f"results_{ts}.csv")
    pd.DataFrame(rows).to_csv(res_csv, index=False)

    # Save ACC-success queries separately
    acc_csv = os.path.join(args.out_dir, f"acc_success_{ts}.csv")
    pd.DataFrame(acc_queries).to_csv(acc_csv, index=False)
    log(f"[save] ACC-success queries: {acc_csv}")

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
