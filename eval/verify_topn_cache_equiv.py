"""
top-50 clean cache 기반 검색(retrieve_cached_topn_topk)과
전체 corpus 직접 scoring(retrieve_fullcorpus_topk)이 쿼리별로
동일한 top-k 문서를 뽑는지 직접 비교.

LLM 생성 단계는 거치지 않음 (do_sample=True 샘플링 노이즈를 배제하고
순수 검색 결과만 비교하기 위함).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main_dispo_fullcorpus_ragdef as fc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["nq", "hotpotqa"])
    p.add_argument("--retrieval_model", type=str, required=True, choices=list(fc._RETRIEVAL_ALIAS.keys()))
    p.add_argument("--docs_csv", type=str, required=True)
    p.add_argument("--adv_per_query", type=int, default=4)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--clean_topn_cache", type=str, required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--dpr_query_encoder", type=str, default="ctx", choices=["standard", "ctx"])
    args = p.parse_args()

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)

    model_hf_name = fc._RETRIEVAL_ALIAS[args.retrieval_model]
    is_contriever_family = model_hf_name in fc._CONTRIEVER_FAMILY
    use_cosine = not is_contriever_family
    q_prefix = fc._QUERY_PREFIXES.get(model_hf_name, "")
    d_prefix = fc._DOC_PREFIXES.get(model_hf_name, "")
    cfg = fc._DS_CFG[args.dataset]

    print(f"[load] retriever: {model_hf_name}")
    if is_contriever_family:
        tok = AutoTokenizer.from_pretrained(model_hf_name)
        mod = AutoModel.from_pretrained(model_hf_name, torch_dtype=torch.float32).to(device)
        mod.eval()

        def encode_fn(texts):
            return fc.contriever_encode(texts, mod, tok, device, batch_size=64)

        query_encode_fn = doc_encode_fn = encode_fn
    else:
        raise NotImplementedError("이 검증 스크립트는 우선 contriever 계열만 지원합니다.")

    print(f"[load] corpus: {cfg['corpus_path']}")
    corpus_texts = []
    with open(cfg["corpus_path"]) as f:
        import json
        for line in f:
            corpus_texts.append(json.loads(line).get("text", ""))
    print(f"[load] corpus passages={len(corpus_texts):,}")

    cache_path = fc._cache_path_for(cfg, model_hf_name)

    def corpus_encoder_fn(texts):
        doc_texts = [d_prefix + t if d_prefix else t for t in texts]
        return doc_encode_fn(doc_texts)

    corpus_embs = fc.build_or_load_corpus_embs(
        corpus_texts, cache_path, corpus_encoder_fn, print, batch_size=512,
    )
    if use_cosine:
        corpus_embs = F.normalize(corpus_embs.float(), dim=-1)
    corpus_embs_gpu = corpus_embs.half().to(device)
    print(f"[embed] GPU ready, memory={torch.cuda.memory_allocated()/1e9:.1f}GB")

    clean_topn_cache = fc.load_clean_topn_cache(args.clean_topn_cache, args, model_hf_name, sys.stdout)

    docs_df = pd.read_csv(args.docs_csv)
    print(f"[load] docs_csv: {len(docs_df)} rows")

    n_total = n_match_docs = n_match_positions = n_match_count = 0
    mismatches = []

    for _, row in docs_df.iterrows():
        q = str(row["query"]).strip()
        poison_docs = [str(row[c]).strip()
                       for c in ["doc0_seed", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
                       if c in row.index and pd.notna(row[c]) and str(row[c]).strip()]
        poison_docs = poison_docs[:args.adv_per_query]
        if not poison_docs:
            continue

        docs_full, pos_full, cnt_full = fc.retrieve_fullcorpus_topk(
            query=q, adv_docs=poison_docs, corpus_embs_gpu=corpus_embs_gpu,
            corpus_texts=corpus_texts, encode_fn=encode_fn, use_cosine=use_cosine,
            device=device, top_k=args.top_k, q_prefix=q_prefix, d_prefix=d_prefix,
            query_encode_fn=query_encode_fn, doc_encode_fn=doc_encode_fn,
        )
        docs_cache, pos_cache, cnt_cache = fc.retrieve_cached_topn_topk(
            query=q, adv_docs=poison_docs, clean_topn_cache=clean_topn_cache,
            corpus_texts=corpus_texts, encode_fn=encode_fn, use_cosine=use_cosine,
            device=device, top_k=args.top_k, q_prefix=q_prefix, d_prefix=d_prefix,
            query_encode_fn=query_encode_fn, doc_encode_fn=doc_encode_fn,
        )

        n_total += 1
        docs_ok = docs_full == docs_cache
        pos_ok = pos_full == pos_cache
        cnt_ok = cnt_full == cnt_cache
        n_match_docs += int(docs_ok)
        n_match_positions += int(pos_ok)
        n_match_count += int(cnt_ok)

        if not (docs_ok and pos_ok and cnt_ok):
            mismatches.append({
                "query": q,
                "full_poison_in_topk": cnt_full, "cache_poison_in_topk": cnt_cache,
                "full_adv_positions": sorted(pos_full), "cache_adv_positions": sorted(pos_cache),
                "full_docs_preview": [d[:80] for d in docs_full],
                "cache_docs_preview": [d[:80] for d in docs_cache],
            })

    print("\n" + "=" * 60)
    print(f"  총 쿼리 수:              {n_total}")
    print(f"  top-k 문서 완전 일치:     {n_match_docs}/{n_total}")
    print(f"  adv 위치(rank) 일치:      {n_match_positions}/{n_total}")
    print(f"  poison_in_topk 개수 일치: {n_match_count}/{n_total}")
    print("=" * 60)

    if mismatches:
        print(f"\n[불일치 {len(mismatches)}건 상세]")
        for m in mismatches[:10]:
            print(f"\n- query: {m['query'][:100]}")
            print(f"  full : poison_in_topk={m['full_poison_in_topk']} adv_pos={m['full_adv_positions']}")
            print(f"  cache: poison_in_topk={m['cache_poison_in_topk']} adv_pos={m['cache_adv_positions']}")
    else:
        print("\n[결론] 모든 쿼리에서 top-50 캐시 기반 검색과 전체 corpus 검색 결과가 완전히 동일함.")


if __name__ == "__main__":
    main()
