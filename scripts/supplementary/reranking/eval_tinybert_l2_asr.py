#!/usr/bin/env python3
"""
Full-corpus NQ single-hop ASR evaluation with fixed components:
  Contriever -> TinyBERT-L2 reranking -> Vicuna-7B no-defense ASR
  Contriever -> TinyBERT-L2 reranking -> RAGDefender(MiniLM) -> Vicuna-7B RD-ASR
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from scripts.supplementary.common_eval import (
    CONTRIEVER_MODEL,
    RAGDEFENDER_MODEL,
    TINYBERT_L2_RERANKER,
    VICUNA_7B_MODEL,
    accuracy_hit,
    asr_hit,
    create_vicuna_llm,
    load_attack_entries,
    load_contriever,
    load_nq_corpus_and_embs,
    make_logger,
    ragdefender_filter,
    retrieve_full_nq,
    standard_rag_answer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_csv", required=True, help="CSV with query,target_answer,correct_answer,doc* columns.")
    parser.add_argument("--data_root", default="/data/joonhyung", help="Root containing datasets/nq/* files.")
    parser.add_argument("--output_dir", default="scripts/supplementary/reranking/results")
    parser.add_argument("--adv_per_query", type=int, default=4)
    parser.add_argument("--ret_top_n", type=int, default=20, help="Contriever candidates passed to TinyBERT-L2.")
    parser.add_argument("--top_k", type=int, default=5, help="Final documents passed to generator/RAGDefender.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--vicuna_model", default=VICUNA_7B_MODEL)
    parser.add_argument("--model_config_path", default=None, help="Optional eval/src model config. Overrides --vicuna_model.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "run.log")

    log(f"[start] {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("[fixed] dataset=NQ single-hop")
    log(f"[fixed] retriever={CONTRIEVER_MODEL}")
    log(f"[fixed] reranker={TINYBERT_L2_RERANKER}")
    log(f"[fixed] ragdefender={RAGDEFENDER_MODEL}")
    log(f"[config] device={device} ret_top_n={args.ret_top_n} top_k={args.top_k}")

    tokenizer, retriever = load_contriever(device, log)
    corpus_texts, corpus_embs_gpu = load_nq_corpus_and_embs(args.data_root, device, log)

    log(f"[load] reranker={TINYBERT_L2_RERANKER}")
    reranker = CrossEncoder(TINYBERT_L2_RERANKER, device=device)

    log(f"[load] ragdefender_model={RAGDEFENDER_MODEL}")
    defense_model = SentenceTransformer(RAGDEFENDER_MODEL, device=device)

    log(f"[load] generator=Vicuna-7B ({args.model_config_path or args.vicuna_model})")
    llm = create_vicuna_llm(
        model_name=args.vicuna_model,
        model_config_path=args.model_config_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    entries, doc_cols = load_attack_entries(args.docs_csv, args.adv_per_query, log)
    rows = []
    nd_asr = nd_acc = rd_asr = rd_acc = 0

    for qi, entry in tqdm(list(enumerate(entries)), desc="tinybert_l2_asr", ncols=90):
        question = entry["query"]
        target = entry["target"]
        correct = entry["correct"]
        poison_docs = entry["poison_docs"]

        topn_docs, topn_is_poison = retrieve_full_nq(
            question,
            poison_docs,
            corpus_embs_gpu,
            corpus_texts,
            tokenizer,
            retriever,
            device,
            args.ret_top_n,
        )

        rerank_scores = reranker.predict([(question, doc) for doc in topn_docs])
        rerank_order = np.argsort(-rerank_scores)[:args.top_k]
        topk_docs = [topn_docs[i] for i in rerank_order]
        topk_is_poison = [topn_is_poison[i] for i in rerank_order]

        nd_response = standard_rag_answer(question, topk_docs, llm)
        nd_asr_hit = asr_hit(target, nd_response)
        nd_acc_hit = accuracy_hit(correct, nd_response)

        survivor_docs, detected_n_adv = ragdefender_filter(topk_docs, defense_model)
        rd_response = standard_rag_answer(question, survivor_docs, llm) if survivor_docs else ""
        rd_asr_hit = asr_hit(target, rd_response)
        rd_acc_hit = accuracy_hit(correct, rd_response)

        nd_asr += int(nd_asr_hit)
        nd_acc += int(nd_acc_hit)
        rd_asr += int(rd_asr_hit)
        rd_acc += int(rd_acc_hit)

        rows.append({
            "query_idx": qi,
            "query": question,
            "target_answer": target,
            "correct_answer": correct,
            "poison_in_reranked_topk": int(sum(topk_is_poison)),
            "ragdefender_detected_n_adv": int(detected_n_adv),
            "ragdefender_survivors": len(survivor_docs),
            "nd_asr": bool(nd_asr_hit),
            "nd_acc": bool(nd_acc_hit),
            "nd_response": nd_response,
            "rd_asr": bool(rd_asr_hit),
            "rd_acc": bool(rd_acc_hit),
            "rd_response": rd_response,
        })

        if (qi + 1) % 10 == 0:
            log(f"[{qi+1}/{len(entries)}] ND-ASR={nd_asr/(qi+1):.1%} RD-ASR={rd_asr/(qi+1):.1%}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(rows)
    summary = {
        "docs_csv": args.docs_csv,
        "n_queries": n,
        "dataset": "nq_singlehop",
        "retriever": CONTRIEVER_MODEL,
        "reranker": TINYBERT_L2_RERANKER,
        "ragdefender": RAGDEFENDER_MODEL,
        "generator": args.model_config_path or args.vicuna_model,
        "ret_top_n": args.ret_top_n,
        "top_k": args.top_k,
        "adv_per_query": args.adv_per_query,
        "doc_cols": doc_cols,
        "ND_ASR": round(nd_asr / n, 4) if n else 0.0,
        "RD_ASR": round(rd_asr / n, 4) if n else 0.0,
        "ND_ACC": round(nd_acc / n, 4) if n else 0.0,
        "RD_ACC": round(rd_acc / n, 4) if n else 0.0,
    }

    pd.DataFrame(rows).to_csv(out_dir / "details.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("[result] TinyBERT-L2 reranking")
    log(f"  ND-ASR: {summary['ND_ASR']*100:.1f}%")
    log(f"  RD-ASR: {summary['RD_ASR']*100:.1f}%")
    log(f"[save] {out_dir / 'summary.json'}")
    log(f"[save] {out_dir / 'details.csv'}")


if __name__ == "__main__":
    main()
