"""
HotpotQA merged DiPoison evaluation over 8 retrievers.

Metrics:
  PRR    = query-level poison retrieval rate, i.e. any poison doc in top-k.
  ND-ASR = target-answer substring ASR without defense.
  RD-ASR = target-answer substring ASR after HotpotQA multihop RAGDefender.

The script reuses a clean-corpus top-50 cache when available. Missing caches are
computed by streaming the HotpotQA corpus in chunks and saving only top-50
indices/scores, so it avoids writing full-corpus embedding tensors.
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hotpotqa_multihop_ragdef_v2_eval import ragdefender_multihop  # noqa: E402
from multi_retriever_ragdef_eval import (  # noqa: E402
    _DOC_PREFIXES,
    _DOT_PRODUCT_MODELS,
    _QUERY_PREFIXES,
    _bm25_score_doc,
    _bm25_tokenize,
    check_asr,
    clean_str,
    contriever_encode,
    load_vicuna,
    vicuna_generate,
    wrap_prompt,
)


_DATA_ROOT = os.environ.get("DIPOISON_DATA_ROOT", "/path/to")
HOTPOTQA_CORPUS = f"{_DATA_ROOT}/datasets/hotpotqa/corpus.jsonl"
DEFAULT_DOCS_CSV = str(
    _ROOT.parent / "data/attackbaselines_pd/DiPoison/merged/hotpotqa_merged_dipoison.csv"
)
DEFAULT_CACHE_DIR = str(_ROOT / "clean_topn_cache/hotpotqa_merged_val100_top50")
FALLBACK_CACHE_DIR = str(_ROOT / "clean_topn_cache/hotpotqa_5attacks_top50")
DEFAULT_OUT_JSON = str(_ROOT / "results/hotpotqa_merged_multihop_8ret/hotpotqa_merged_dipoison_summary.json")
DEFENSE_MODEL = "paraphrase-MiniLM-L6-v2"

RETRIEVERS = [
    ("contriever", "contriever_top50.pt", "contriever", "facebook/contriever", "contriever"),
    ("e5", "e5-base_top50.pt", "e5-base", "intfloat/e5-base-v2", "st"),
    ("ance", "ance_top50.pt", "ance", "sentence-transformers/msmarco-roberta-base-ance-firstp", "st"),
    ("bge-base", "bge-base_top50.pt", "bge-base", "BAAI/bge-base-en-v1.5", "st"),
    ("mpnet", "mpnet_top50.pt", "mpnet", "sentence-transformers/all-mpnet-base-v2", "st"),
    ("bm25", "bm25_top50.pt", "bm25", "bm25", "bm25"),
    ("nomic-v1.5", "nomic-v1.5_top50.pt", "nomic-v1.5", "nomic-ai/nomic-embed-text-v1.5", "st"),
    (
        "contriever-msmarco",
        "contriever-msmarco_top50.pt",
        "contriever-msmarco",
        "facebook/contriever-msmarco",
        "contriever",
    ),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_csv", default=DEFAULT_DOCS_CSV)
    p.add_argument("--corpus_path", default=HOTPOTQA_CORPUS)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--fallback_cache_dir", default=FALLBACK_CACHE_DIR)
    p.add_argument("--out_json", default=DEFAULT_OUT_JSON)
    p.add_argument("--top_ks", default="5,10")
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--adv_per_query", type=int, default=7)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--dense_chunk_size", type=int, default=8192)
    p.add_argument("--st_batch_size", type=int, default=128)
    p.add_argument("--contriever_batch_size", type=int, default=128)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--retrievers", default=",".join(r[0] for r in RETRIEVERS))
    p.add_argument("--skip_cache_build", action="store_true")
    p.add_argument("--cache_only", action="store_true",
                   help="build/validate clean top-N caches, then exit before loading defense/generator")
    p.add_argument("--skip_done", action="store_true")
    return p.parse_args()


def log_line(fp, msg):
    print(msg, flush=True)
    fp.write(msg + "\n")
    fp.flush()


def load_attack_csv(path, adv_per_query):
    df = pd.read_csv(path)
    adv_cols = [c for c in ["doc0_seed", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6"] if c in df.columns]
    if not adv_cols:
        raise ValueError(f"No poison document columns found in {path}")
    queries = [str(q).strip() for q in df["query"].tolist()]
    target_ans = [str(x).strip() for x in df["target_answer"].tolist()]
    correct_ans = [str(x).strip() for x in df["correct_answer"].tolist()]
    adv_docs = [
        [str(row[c]).strip() for c in adv_cols if pd.notna(row[c]) and str(row[c]).strip()][:adv_per_query]
        for _, row in df.iterrows()
    ]
    return queries, target_ans, correct_ans, adv_docs, adv_cols


def load_corpus_texts(corpus_path, log_fp):
    corpus_texts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="load corpus", dynamic_ncols=True):
            d = json.loads(line)
            corpus_texts.append(d.get("text", ""))
    log_line(log_fp, f"[corpus] loaded {len(corpus_texts):,} passages from {corpus_path}")
    return corpus_texts


def load_topn_cache(cache_path, ret_key, hf_id, queries):
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = cache.get("meta", {})
    if meta.get("retrieval_model") != ret_key:
        raise ValueError(f"cache retriever mismatch: {meta.get('retrieval_model')} != {ret_key}")
    if meta.get("model_hf") not in (hf_id, None, ""):
        raise ValueError(f"cache model mismatch: {meta.get('model_hf')} != {hf_id}")
    cache_queries = [str(q).strip() for q in cache["queries"]]
    missing = sorted(set(queries) - set(cache_queries))
    if missing:
        raise KeyError(f"{len(missing)} queries are missing from cache {cache_path}; first={missing[0]}")
    cache["query_to_row"] = {q: i for i, q in enumerate(cache_queries)}
    cache["top_indices"] = cache["top_indices"].long()
    cache["top_scores"] = cache["top_scores"].float()
    return cache


def cache_is_usable(cache_path, ret_key, hf_id, queries):
    if not os.path.exists(cache_path):
        return False
    try:
        load_topn_cache(cache_path, ret_key, hf_id, queries)
        return True
    except Exception:
        return False


def save_topn_cache(cache_path, dataset, ret_key, hf_id, top_n, corpus_path, corpus_size,
                    scorer, q_prefix, d_prefix, docs_csv, queries, top_indices, top_scores):
    out = {
        "meta": {
            "dataset": dataset,
            "retrieval_model": ret_key,
            "model_hf": hf_id,
            "top_n": top_n,
            "corpus_path": corpus_path,
            "corpus_size": corpus_size,
            "score_dtype": "float16" if scorer != "bm25" else "float32",
            "scorer": scorer,
            "q_prefix": q_prefix,
            "d_prefix": d_prefix,
            "docs_csv": [docs_csv],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "queries": queries,
        "top_indices": top_indices.cpu().long(),
        "top_scores": top_scores.cpu().float(),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(out, cache_path)


def load_dense_retriever(hf_id, enc_type, device, args, log_fp):
    if enc_type == "contriever":
        tok = AutoTokenizer.from_pretrained(hf_id)
        mod = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32).to(device)
        mod.eval()

        def encode_fn(texts):
            return contriever_encode(texts, mod, tok, device, batch_size=args.contriever_batch_size)

        return encode_fn, (mod, tok)

    st_model = SentenceTransformer(hf_id, trust_remote_code=True).to(device)
    if args.max_seq_length and getattr(st_model, "max_seq_length", args.max_seq_length) > args.max_seq_length:
        log_line(log_fp, f"[load] {hf_id} max_seq_length {st_model.max_seq_length} -> {args.max_seq_length}")
        st_model.max_seq_length = args.max_seq_length
    st_model.eval()

    def encode_fn(texts):
        with torch.no_grad():
            return st_model.encode(
                texts,
                batch_size=args.st_batch_size,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).cpu()

    return encode_fn, st_model


def build_dense_topn_cache(label, cache_path, ret_key, hf_id, enc_type, queries, corpus_texts, args, log_fp):
    device = f"cuda:{args.gpu_id}"
    q_prefix = _QUERY_PREFIXES.get(hf_id, "") if enc_type == "st" else ""
    d_prefix = _DOC_PREFIXES.get(hf_id, "") if enc_type == "st" else ""
    use_cosine = hf_id not in {"facebook/contriever", "facebook/contriever-msmarco"} and hf_id not in _DOT_PRODUCT_MODELS
    scorer = "cosine" if use_cosine else "dot"

    log_line(log_fp, f"[cache-build] {label}: streaming corpus top-{args.top_n}, scorer={scorer}")
    encode_fn, model_obj = load_dense_retriever(hf_id, enc_type, device, args, log_fp)
    try:
        q_texts = [q_prefix + q if q_prefix else q for q in queries]
        q_embs = encode_fn(q_texts).to(device).half()
        if use_cosine:
            q_embs = F.normalize(q_embs.float(), dim=-1).half()

        n_q = len(queries)
        top_scores = torch.full((n_q, args.top_n), -float("inf"), dtype=torch.float16, device=device)
        top_indices = torch.full((n_q, args.top_n), -1, dtype=torch.long, device=device)

        for start in tqdm(
            range(0, len(corpus_texts), args.dense_chunk_size),
            desc=f"{label} top{args.top_n}",
            dynamic_ncols=True,
        ):
            chunk = corpus_texts[start:start + args.dense_chunk_size]
            d_texts = [d_prefix + d if d_prefix else d for d in chunk]
            d_embs = encode_fn(d_texts).to(device).half()
            if use_cosine:
                d_embs = F.normalize(d_embs.float(), dim=-1).half()
            scores = torch.mm(q_embs, d_embs.T)
            idxs = torch.arange(start, start + len(chunk), device=device, dtype=torch.long)
            idxs = idxs.unsqueeze(0).expand(n_q, -1)

            merged_scores = torch.cat([top_scores, scores], dim=1)
            merged_indices = torch.cat([top_indices, idxs], dim=1)
            vals, pos = torch.topk(merged_scores, args.top_n, dim=1)
            top_scores = vals
            top_indices = torch.gather(merged_indices, 1, pos)

            del d_embs, scores, idxs, merged_scores, merged_indices, vals, pos
            torch.cuda.empty_cache()

        save_topn_cache(
            cache_path=cache_path,
            dataset="hotpotqa",
            ret_key=ret_key,
            hf_id=hf_id,
            top_n=args.top_n,
            corpus_path=args.corpus_path,
            corpus_size=len(corpus_texts),
            scorer=scorer,
            q_prefix=q_prefix,
            d_prefix=d_prefix,
            docs_csv=args.docs_csv,
            queries=queries,
            top_indices=top_indices.cpu(),
            top_scores=top_scores.cpu(),
        )
        log_line(log_fp, f"[cache-build] saved {cache_path}")
    finally:
        del model_obj
        gc.collect()
        torch.cuda.empty_cache()


def build_bm25_topn_cache(label, cache_path, ret_key, hf_id, queries, corpus_texts, args, log_fp):
    log_line(log_fp, f"[cache-build] {label}: streaming BM25 top-{args.top_n}")
    q_tokens_per_query = [_bm25_tokenize(q) for q in queries]
    query_terms = sorted({t for toks in q_tokens_per_query for t in toks})
    term_to_queries = {term: [] for term in query_terms}
    for qi, toks in enumerate(q_tokens_per_query):
        for term in set(toks):
            term_to_queries.setdefault(term, []).append(qi)

    n_docs = len(corpus_texts)
    dfs = Counter()
    doc_lens = np.empty(n_docs, dtype=np.int32)
    for i, text in enumerate(tqdm(corpus_texts, desc="bm25 df", dynamic_ncols=True)):
        toks = _bm25_tokenize(text)
        doc_lens[i] = len(toks)
        dfs.update(set(toks))

    idf = {}
    idf_sum = 0.0
    negative_terms = []
    for term, freq in dfs.items():
        val = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)
        idf[term] = val
        idf_sum += val
        if val < 0:
            negative_terms.append(term)
    avg_idf = idf_sum / max(len(idf), 1)
    eps = 0.25 * avg_idf
    for term in negative_terms:
        idf[term] = eps

    k1 = 1.5
    b = 0.75
    avgdl = float(doc_lens.mean()) if len(doc_lens) else 0.0
    bm25_params = {"idf": idf, "k1": k1, "b": b, "avgdl": avgdl}

    n_q = len(queries)
    top_scores = torch.full((n_q, args.top_n), -float("inf"), dtype=torch.float32)
    top_indices = torch.full((n_q, args.top_n), -1, dtype=torch.long)

    for start in tqdm(
        range(0, n_docs, args.dense_chunk_size),
        desc=f"{label} top{args.top_n}",
        dynamic_ncols=True,
    ):
        chunk = corpus_texts[start:start + args.dense_chunk_size]
        score_mat = torch.zeros((n_q, len(chunk)), dtype=torch.float32)
        for local_idx, text in enumerate(chunk):
            toks = _bm25_tokenize(text)
            if not toks:
                continue
            dl = len(toks)
            tf = Counter(t for t in toks if t in term_to_queries)
            if not tf:
                continue
            denom_base = k1 * (1 - b + b * dl / max(avgdl, 1.0))
            for term, cnt in tf.items():
                idf_val = idf.get(term, 0.0)
                if idf_val <= 0.0:
                    continue
                term_score = idf_val * cnt * (k1 + 1) / (cnt + denom_base)
                for qi in term_to_queries.get(term, []):
                    score_mat[qi, local_idx] += term_score

        idxs = torch.arange(start, start + len(chunk), dtype=torch.long).unsqueeze(0).expand(n_q, -1)
        merged_scores = torch.cat([top_scores, score_mat], dim=1)
        merged_indices = torch.cat([top_indices, idxs], dim=1)
        vals, pos = torch.topk(merged_scores, args.top_n, dim=1)
        top_scores = vals
        top_indices = torch.gather(merged_indices, 1, pos)

    save_topn_cache(
        cache_path=cache_path,
        dataset="hotpotqa",
        ret_key=ret_key,
        hf_id=hf_id,
        top_n=args.top_n,
        corpus_path=args.corpus_path,
        corpus_size=n_docs,
        scorer="bm25",
        q_prefix="",
        d_prefix="",
        docs_csv=args.docs_csv,
        queries=queries,
        top_indices=top_indices,
        top_scores=top_scores,
    )
    with open(cache_path + ".bm25_params.pkl", "wb") as f:
        pickle.dump(bm25_params, f)
    log_line(log_fp, f"[cache-build] saved {cache_path} and BM25 params")


def resolve_cache(label, cache_file, ret_key, hf_id, queries, args, log_fp):
    cache_path = os.path.join(args.cache_dir, cache_file)
    if cache_is_usable(cache_path, ret_key, hf_id, queries):
        return cache_path

    fallback_path = os.path.join(args.fallback_cache_dir, cache_file)
    if cache_is_usable(fallback_path, ret_key, hf_id, queries):
        log_line(log_fp, f"[cache] {label}: using fallback {fallback_path}")
        return fallback_path

    if args.skip_cache_build:
        raise FileNotFoundError(f"usable cache not found for {label}: {cache_path}")
    return cache_path


def ensure_cache(label, cache_file, ret_key, hf_id, enc_type, queries, corpus_texts, args, log_fp):
    cache_path = resolve_cache(label, cache_file, ret_key, hf_id, queries, args, log_fp)
    if os.path.exists(cache_path) and cache_is_usable(cache_path, ret_key, hf_id, queries):
        log_line(log_fp, f"[cache] {label}: ready {cache_path}")
        return cache_path

    if enc_type == "bm25":
        build_bm25_topn_cache(label, cache_path, ret_key, hf_id, queries, corpus_texts, args, log_fp)
    else:
        build_dense_topn_cache(label, cache_path, ret_key, hf_id, enc_type, queries, corpus_texts, args, log_fp)
    return cache_path


def load_eval_retriever(label, cache_path, hf_id, enc_type, device, args, log_fp):
    q_prefix = _QUERY_PREFIXES.get(hf_id, "") if enc_type == "st" else ""
    d_prefix = _DOC_PREFIXES.get(hf_id, "") if enc_type == "st" else ""
    use_cosine = hf_id not in {"facebook/contriever", "facebook/contriever-msmarco"} and hf_id not in _DOT_PRODUCT_MODELS

    if enc_type == "bm25":
        with open(cache_path + ".bm25_params.pkl", "rb") as f:
            bm25_params = pickle.load(f)
        log_line(log_fp, f"[eval-load] {label}: BM25 params vocab={len(bm25_params['idf']):,}")
        return None, None, use_cosine, q_prefix, d_prefix, bm25_params

    encode_fn, model_obj = load_dense_retriever(hf_id, enc_type, device, args, log_fp)
    log_line(log_fp, f"[eval-load] {label}: {hf_id}, cosine={use_cosine}")
    return encode_fn, model_obj, use_cosine, q_prefix, d_prefix, None


def retrieve_from_cache_topmax(query, adv_docs, cache, corpus_texts, encode_fn, use_cosine, device,
                               top_k, q_prefix="", d_prefix="", bm25_params=None):
    row_idx = cache["query_to_row"][str(query).strip()]
    clean_indices = cache["top_indices"][row_idx]
    clean_scores = cache["top_scores"][row_idx]

    if bm25_params is not None:
        q_tokens = _bm25_tokenize(query)
        adv_scores = torch.tensor(
            [_bm25_score_doc(bm25_params, q_tokens, _bm25_tokenize(d)) for d in adv_docs],
            dtype=torch.float32,
        )
    else:
        q_text = q_prefix + query if q_prefix else query
        d_texts = [d_prefix + d if d_prefix else d for d in adv_docs]
        adv_embs = encode_fn(d_texts).to(device).half()
        q_emb = encode_fn([q_text]).to(device).half()
        if use_cosine:
            adv_embs = F.normalize(adv_embs.float(), dim=-1).half()
            q_emb = F.normalize(q_emb.float(), dim=-1).half()
        adv_scores = torch.mm(adv_embs, q_emb.T).squeeze(1).float().cpu()

    all_scores = torch.cat([clean_scores, adv_scores], dim=0)
    top_local = all_scores.topk(top_k).indices.tolist()

    retrieved_docs = []
    adv_positions = set()
    n_clean = clean_indices.numel()
    for rank, idx in enumerate(top_local):
        if idx < n_clean:
            retrieved_docs.append(corpus_texts[int(clean_indices[idx])])
        else:
            retrieved_docs.append(adv_docs[idx - n_clean])
            adv_positions.add(rank)
    return retrieved_docs, adv_positions


def evaluate_retriever(label, cache_path, ret_key, hf_id, enc_type, top_ks, queries, target_ans,
                       adv_docs_per_query, corpus_texts, defense_model, llm_model, llm_tok,
                       get_conv, args, log_fp):
    device = f"cuda:{args.gpu_id}"
    cache = load_topn_cache(cache_path, ret_key, hf_id, queries)
    encode_fn, model_obj, use_cosine, q_prefix, d_prefix, bm25_params = load_eval_retriever(
        label, cache_path, hf_id, enc_type, device, args, log_fp
    )
    max_k = max(top_ks)
    results = {}
    detail_rows = []
    counters = {k: {"prr": 0, "nd_asr": 0, "rd_asr": 0} for k in top_ks}

    try:
        for i in tqdm(range(len(queries)), desc=f"{label} eval", dynamic_ncols=True):
            query = queries[i]
            t_ans = target_ans[i]
            adv_docs = adv_docs_per_query[i]
            retrieved_topmax, adv_pos_topmax = retrieve_from_cache_topmax(
                query=query,
                adv_docs=adv_docs,
                cache=cache,
                corpus_texts=corpus_texts,
                encode_fn=encode_fn,
                use_cosine=use_cosine,
                device=device,
                top_k=max_k,
                q_prefix=q_prefix,
                d_prefix=d_prefix,
                bm25_params=bm25_params,
            )

            for top_k in top_ks:
                retrieved_docs = retrieved_topmax[:top_k]
                adv_positions = {p for p in adv_pos_topmax if p < top_k}
                has_poison = bool(adv_positions)
                if has_poison:
                    counters[top_k]["prr"] += 1

                nd_resp = vicuna_generate(llm_model, llm_tok, get_conv, wrap_prompt(query, retrieved_docs))
                nd_hit = check_asr(t_ans, nd_resp)
                if nd_hit:
                    counters[top_k]["nd_asr"] += 1

                safe_docs = ragdefender_multihop(retrieved_docs, defense_model)
                rd_resp = (
                    vicuna_generate(llm_model, llm_tok, get_conv, wrap_prompt(query, safe_docs))
                    if safe_docs else ""
                )
                rd_hit = check_asr(t_ans, rd_resp)
                if rd_hit:
                    counters[top_k]["rd_asr"] += 1

                detail_rows.append({
                    "retriever": label,
                    "top_k": top_k,
                    "index": i + 1,
                    "query": query,
                    "target_answer": t_ans,
                    "poison_in_topk": len(adv_positions),
                    "has_poison": int(has_poison),
                    "nd_asr": int(nd_hit),
                    "rd_asr": int(rd_hit),
                    "nd_response": nd_resp,
                    "rd_response": rd_resp,
                    "n_safe_docs": len(safe_docs),
                })

            gc.collect()
            torch.cuda.empty_cache()

        n = len(queries)
        for top_k in top_ks:
            c = counters[top_k]
            results[f"top{top_k}"] = {
                "prr": round(c["prr"] / n, 4),
                "nd_asr": round(c["nd_asr"] / n, 4),
                "rd_asr": round(c["rd_asr"] / n, 4),
                "n": n,
            }
            log_line(
                log_fp,
                f"[result] {label}/top{top_k}: "
                f"PRR={c['prr']/n:.2%} ND-ASR={c['nd_asr']/n:.2%} RD-ASR={c['rd_asr']/n:.2%}",
            )
    finally:
        if enc_type == "bm25":
            del bm25_params
        else:
            del model_obj
        gc.collect()
        torch.cuda.empty_cache()

    return results, detail_rows


def main():
    args = parse_args()
    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    log_path = str(Path(args.out_json).with_suffix(".log"))
    detail_csv = str(Path(args.out_json).with_suffix(".details.csv"))
    partial_path = str(Path(args.out_json).with_suffix(".partial.json"))

    selected = [x.strip() for x in args.retrievers.split(",") if x.strip()]
    retrievers = [r for r in RETRIEVERS if r[0] in selected]
    unknown = sorted(set(selected) - {r[0] for r in RETRIEVERS})
    if unknown:
        raise ValueError(f"Unknown retrievers: {unknown}")
    top_ks = [int(x.strip()) for x in args.top_ks.split(",") if x.strip()]

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if torch.cuda.is_available():
        nd = torch.cuda.device_count()
        torch.cuda.set_device(args.gpu_id if args.gpu_id < nd else 0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(log_path, "a", encoding="utf-8") as log_fp:
        log_line(log_fp, f"[config] docs_csv={args.docs_csv}")
        log_line(log_fp, f"[config] corpus={args.corpus_path}")
        log_line(log_fp, f"[config] cache_dir={args.cache_dir}")
        log_line(log_fp, f"[config] fallback_cache_dir={args.fallback_cache_dir}")
        log_line(log_fp, f"[config] retrievers={[r[0] for r in retrievers]}")
        log_line(log_fp, f"[config] top_ks={top_ks}, top_n={args.top_n}, adv_per_query={args.adv_per_query}")
        log_line(log_fp, f"[config] defense=hotpotqa_multihop_ragdef, defense_model={DEFENSE_MODEL}")
        log_line(log_fp, "[config] generator=lmsys/vicuna-7b-v1.3")

        queries, target_ans, correct_ans, adv_docs_per_query, adv_cols = load_attack_csv(
            args.docs_csv, args.adv_per_query
        )
        log_line(log_fp, f"[csv] queries={len(queries)}, adv_cols={adv_cols}")
        if len(set(queries)) != len(queries):
            raise ValueError("Duplicate queries in docs_csv are not supported.")

        corpus_texts = load_corpus_texts(args.corpus_path, log_fp)

        if args.cache_only:
            for label, cache_file, ret_key, hf_id, enc_type in retrievers:
                ensure_cache(label, cache_file, ret_key, hf_id, enc_type, queries, corpus_texts, args, log_fp)
            log_line(log_fp, "[cache_only] done")
            return

        if os.path.exists(partial_path):
            with open(partial_path, encoding="utf-8") as f:
                all_results = json.load(f)
            log_line(log_fp, f"[resume] loaded partial {partial_path}")
        else:
            all_results = {}
        all_details = []
        if os.path.exists(detail_csv):
            try:
                all_details = pd.read_csv(detail_csv).to_dict("records")
                log_line(log_fp, f"[resume] loaded details {detail_csv}")
            except Exception:
                all_details = []

        log_line(log_fp, f"[load] defense model: {DEFENSE_MODEL}")
        defense_model = SentenceTransformer(DEFENSE_MODEL)
        log_line(log_fp, "[load] defense model ready")

        log_line(log_fp, "[load] Vicuna-7B")
        llm_model, llm_tok, get_conv = load_vicuna(f"cuda:{args.gpu_id}")
        log_line(log_fp, f"[load] Vicuna ready. GPU memory={torch.cuda.memory_allocated()/1e9:.1f}GB")

        for label, cache_file, ret_key, hf_id, enc_type in retrievers:
            if args.skip_done and label in all_results:
                log_line(log_fp, f"[skip] {label}: already in partial")
                continue

            log_line(log_fp, f"\n{'=' * 72}")
            cache_path = ensure_cache(
                label, cache_file, ret_key, hf_id, enc_type, queries, corpus_texts, args, log_fp
            )
            log_line(log_fp, f"[eval] {label} ({hf_id}) cache={cache_path}")
            results, details = evaluate_retriever(
                label=label,
                cache_path=cache_path,
                ret_key=ret_key,
                hf_id=hf_id,
                enc_type=enc_type,
                top_ks=top_ks,
                queries=queries,
                target_ans=target_ans,
                adv_docs_per_query=adv_docs_per_query,
                corpus_texts=corpus_texts,
                defense_model=defense_model,
                llm_model=llm_model,
                llm_tok=llm_tok,
                get_conv=get_conv,
                args=args,
                log_fp=log_fp,
            )
            all_results[label] = results
            all_details.extend(details)

            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            pd.DataFrame(all_details).to_csv(detail_csv, index=False)
            log_line(log_fp, f"[partial] {partial_path}")
            log_line(log_fp, f"[details] {detail_csv}")

        final = {
            "dataset": "hotpotqa",
            "docs_csv": args.docs_csv,
            "corpus_path": args.corpus_path,
            "defense": "hotpotqa_multihop_ragdef",
            "defense_model": DEFENSE_MODEL,
            "generator": "lmsys/vicuna-7b-v1.3",
            "top_ks": top_ks,
            "adv_per_query": args.adv_per_query,
            "results": all_results,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

        log_line(log_fp, f"\n[save] {args.out_json}")
        log_line(log_fp, f"[save] {detail_csv}")
        log_line(log_fp, f"\n{'=' * 78}")
        log_line(log_fp, f"{'retriever':<20} {'top-k':<7} {'PRR':>8} {'ND-ASR':>8} {'RD-ASR':>8}")
        log_line(log_fp, "-" * 78)
        for label, _, _, _, _ in retrievers:
            for top_k in top_ks:
                res = all_results[label][f"top{top_k}"]
                log_line(
                    log_fp,
                    f"{label:<20} {('top' + str(top_k)):<7} "
                    f"{res['prr']:>8.1%} {res['nd_asr']:>8.1%} {res['rd_asr']:>8.1%}",
                )
        log_line(log_fp, "=" * 78)


if __name__ == "__main__":
    main()
