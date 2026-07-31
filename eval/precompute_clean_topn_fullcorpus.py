"""
Precompute clean-corpus top-N retrieval results for later poison-doc injection.

This stores, per query and retriever, the clean corpus doc indices and scores
computed with the same fp16 scoring path used by main_dipoison_fullcorpus_ragdef.py.
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main_dipoison_fullcorpus_ragdef as fc

_RETRIEVAL_ALIAS = dict(getattr(fc, "_RETRIEVAL_ALIAS", {}))
_RETRIEVAL_ALIAS.update({
    "bm25": "bm25",
    "nomic-v1.5": "nomic-ai/nomic-embed-text-v1.5",
})

_CONTRIEVER_FAMILY = set(getattr(fc, "_CONTRIEVER_FAMILY", {"facebook/contriever", "facebook/contriever-msmarco"}))
_DOT_PRODUCT_MODELS = set(getattr(fc, "_DOT_PRODUCT_MODELS", set()))
_DOT_PRODUCT_MODELS.update({
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1",
    "sentence-transformers/msmarco-distilbert-base-tas-b",
})

_QUERY_PREFIXES = dict(getattr(fc, "_QUERY_PREFIXES", {}))
_QUERY_PREFIXES.update({
    "intfloat/e5-base-v2": "query: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
})
_DOC_PREFIXES = dict(getattr(fc, "_DOC_PREFIXES", {}))
_DOC_PREFIXES.update({
    "intfloat/e5-base-v2": "passage: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_document: ",
})

_DPR_QUESTION_ENCODER = getattr(
    fc, "_DPR_QUESTION_ENCODER",
    "sentence-transformers/facebook-dpr-question_encoder-single-nq-base",
)
_DPR_CONTEXT_ENCODER = getattr(
    fc, "_DPR_CONTEXT_ENCODER",
    "sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base",
)


def load_queries(paths):
    seen = set()
    queries = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                q = str(row["query"]).strip()
                if q and q not in seen:
                    seen.add(q)
                    queries.append(q)
    return queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["nq", "hotpotqa", "msmarco"])
    p.add_argument("--retrieval_model", type=str, required=True, choices=sorted(_RETRIEVAL_ALIAS))
    p.add_argument("--docs_csv", nargs="+", required=True)
    p.add_argument("--corpus_path", type=str, default=None,
                   help="override full corpus jsonl path; required for msmarco unless default exists")
    p.add_argument("--embed_cache_dir", type=str, default=None,
                   help="override full-corpus embedding cache directory")
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--embed_batch", type=int, default=512,
                   help="outer chunk size passed to build_or_load_corpus_embs")
    p.add_argument("--st_batch", type=int, default=32,
                   help="inner SentenceTransformer encode batch size (controls GPU peak memory)")
    p.add_argument("--max_seq_length", type=int, default=512,
                   help="truncate sequences to this length (default 512, matches other retrievers)")
    p.add_argument("--dpr_query_encoder", type=str, default="ctx",
                   choices=["standard", "ctx"],
                   help="DPR query encoder: ctx=legacy context encoder, standard=question encoder")
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    nd = torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_id if args.gpu_id < nd else 0)
    device = f"cuda:{torch.cuda.current_device()}"

    model_hf_name = _RETRIEVAL_ALIAS[args.retrieval_model]
    is_contriever_family = model_hf_name in _CONTRIEVER_FAMILY
    is_standard_dpr = args.retrieval_model == "dpr" and args.dpr_query_encoder == "standard"
    is_bm25 = model_hf_name == "bm25"
    use_cosine = not (is_contriever_family or model_hf_name in _DOT_PRODUCT_MODELS or is_bm25)
    if is_standard_dpr:
        use_cosine = False
    q_prefix = _QUERY_PREFIXES.get(model_hf_name, "")
    d_prefix = _DOC_PREFIXES.get(model_hf_name, "")
    if args.dataset in fc._DS_CFG:
        cfg = dict(fc._DS_CFG[args.dataset])
    else:
        cfg = {
            "corpus_path": "/path/to/datasets/msmarco/corpus.jsonl",
            "embed_cache_dir": "/path/to/datasets/msmarco",
        }
    if args.corpus_path:
        cfg["corpus_path"] = args.corpus_path
    if args.embed_cache_dir:
        cfg["embed_cache_dir"] = args.embed_cache_dir

    queries = load_queries(args.docs_csv)
    if not queries:
        raise ValueError("No queries found in docs_csv")

    print(f"[config] dataset={args.dataset} retriever={args.retrieval_model} model={model_hf_name}")
    print(f"[config] top_n={args.top_n} queries={len(queries)} device={device}")

    print(f"[load] retriever: {model_hf_name}")
    if is_contriever_family:
        tok = AutoTokenizer.from_pretrained(model_hf_name, local_files_only=False)
        mod = AutoModel.from_pretrained(model_hf_name, torch_dtype=torch.float32).to(device)
        mod.eval()

        def encode_fn(texts):
            return fc.contriever_encode(texts, mod, tok, device, batch_size=64)

        query_encode_fn = encode_fn
        doc_encode_fn = encode_fn

    elif is_standard_dpr:
        ctx_model = SentenceTransformer(_DPR_CONTEXT_ENCODER, trust_remote_code=True).to(device)
        q_model = SentenceTransformer(_DPR_QUESTION_ENCODER, trust_remote_code=True).to(device)
        ctx_model.eval()
        q_model.eval()

        def _st_encode(model, texts):
            with torch.no_grad():
                return model.encode(
                    texts, batch_size=256, convert_to_tensor=True,
                    normalize_embeddings=False, show_progress_bar=False,
                ).cpu()

        def doc_encode_fn(texts):
            return _st_encode(ctx_model, texts)

        def query_encode_fn(texts):
            return _st_encode(q_model, texts)

        encode_fn = doc_encode_fn

    elif is_bm25:
        encode_fn = None
        query_encode_fn = None
        doc_encode_fn = None
        print("[load] BM25: no embedding model needed")

    else:
        st_model = SentenceTransformer(model_hf_name, trust_remote_code=True)
        if args.max_seq_length and st_model.max_seq_length > args.max_seq_length:
            print(f"[load] max_seq_length {st_model.max_seq_length} → {args.max_seq_length} (truncate)")
            st_model.max_seq_length = args.max_seq_length
        st_model = st_model.to(device)
        st_model.eval()

        def encode_fn(texts):
            with torch.no_grad():
                return st_model.encode(
                    texts, batch_size=args.st_batch, convert_to_tensor=True,
                    normalize_embeddings=False, show_progress_bar=False,
                ).cpu()

        query_encode_fn = encode_fn
        doc_encode_fn = encode_fn

    print(f"[load] corpus: {cfg['corpus_path']}")
    corpus_texts = []
    with open(cfg["corpus_path"]) as f:
        for line in f:
            d = json.loads(line)
            corpus_texts.append(d.get("text", "") or d.get("contents", ""))
    print(f"[load] corpus passages={len(corpus_texts):,}")

    top_indices = torch.empty((len(queries), args.top_n), dtype=torch.long)
    top_scores  = torch.empty((len(queries), args.top_n), dtype=torch.float32)
    bm25_params_to_save = None

    if is_bm25:
        import numpy as np
        from rank_bm25 import BM25Okapi

        print("[bm25] tokenizing corpus...")
        tokenized_corpus = [t.lower().split() for t in tqdm(corpus_texts, desc="tokenize", dynamic_ncols=True)]
        print("[bm25] building index...")
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"[bm25] done — vocab={len(bm25.idf):,}, avgdl={bm25.avgdl:.1f}")

        bm25_params_to_save = {
            "idf": dict(bm25.idf),
            "k1": bm25.k1,
            "b": bm25.b,
            "avgdl": bm25.avgdl,
        }

        pbar = tqdm(enumerate(queries), total=len(queries), desc="BM25 topN", unit="q", dynamic_ncols=True)
        for qi, query in pbar:
            q_tokens = query.lower().split()
            scores_np = bm25.get_scores(q_tokens)
            topn_idx  = np.argpartition(scores_np, -args.top_n)[-args.top_n:]
            topn_idx  = topn_idx[np.argsort(scores_np[topn_idx])[::-1]]
            top_indices[qi] = torch.from_numpy(topn_idx.astype("int64"))
            top_scores[qi]  = torch.from_numpy(scores_np[topn_idx].astype("float32"))

    else:
        cache_path = fc._cache_path_for(cfg, model_hf_name)

        def corpus_encoder_fn(texts):
            doc_texts = [d_prefix + t if d_prefix else t for t in texts]
            return doc_encode_fn(doc_texts)

        corpus_embs = fc.build_or_load_corpus_embs(
            corpus_texts, cache_path,
            corpus_encoder_fn, print,
            batch_size=args.embed_batch,
        )

        print(f"[embed] move corpus to GPU fp16: {tuple(corpus_embs.shape)}")
        if use_cosine:
            corpus_embs = F.normalize(corpus_embs.float(), dim=-1)
        corpus_embs_gpu = corpus_embs.half().to(device)
        print(f"[embed] GPU ready, memory={torch.cuda.memory_allocated()/1e9:.1f}GB")

        pbar = tqdm(enumerate(queries), total=len(queries), desc="clean topN", unit="q", dynamic_ncols=True)
        with torch.no_grad():
            for qi, query in pbar:
                q_text = q_prefix + query if q_prefix else query
                q_emb = query_encode_fn([q_text]).to(device).half()
                if use_cosine:
                    q_emb = F.normalize(q_emb, dim=-1)
                scores = torch.mm(corpus_embs_gpu, q_emb.T).squeeze(1)
                vals, idx = scores.topk(args.top_n)
                top_indices[qi] = idx.cpu().long()
                top_scores[qi]  = vals.cpu().float()

    out = {
        "meta": {
            "dataset": args.dataset,
            "retrieval_model": args.retrieval_model,
            "model_hf": model_hf_name,
            "top_n": args.top_n,
            "corpus_path": cfg["corpus_path"],
            "corpus_size": len(corpus_texts),
            "score_dtype": "float16",
            "scorer": "bm25" if is_bm25 else ("cosine" if use_cosine else "dot"),
            "dpr_query_encoder": args.dpr_query_encoder if args.retrieval_model == "dpr" else "",
            "dpr_question_hf": _DPR_QUESTION_ENCODER if is_standard_dpr else "",
            "dpr_context_hf": _DPR_CONTEXT_ENCODER if args.retrieval_model == "dpr" else "",
            "q_prefix": q_prefix,
            "d_prefix": d_prefix,
            "docs_csv": args.docs_csv,
        },
        "queries": queries,
        "top_indices": top_indices,
        "top_scores": top_scores,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(out, args.output)
    print(f"[save] {args.output}")

    if bm25_params_to_save is not None:
        bm25_pkl = args.output + ".bm25_params.pkl"
        with open(bm25_pkl, "wb") as f:
            pickle.dump(bm25_params_to_save, f)
        print(f"[save] BM25 params → {bm25_pkl} (vocab={len(bm25_params_to_save['idf']):,})")


if __name__ == "__main__":
    main()
