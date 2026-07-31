#!/usr/bin/env python3
"""
Full-corpus NQ single-hop ASR evaluation with fixed components:
  (i)   Contriever top-5 -> Mistral-7B no-defense ASR
  (ii)  Contriever top-5 -> RobustRAG(Mistral-7B) RR-ASR
  (iii) Contriever top-5 -> RAGDefender(MiniLM) -> RobustRAG(Mistral-7B) RD+RR-ASR
"""

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from scripts.supplementary.common_eval import (
    CONTRIEVER_MODEL,
    MISTRAL_7B_MODEL,
    RAGDEFENDER_MODEL,
    accuracy_hit,
    asr_hit,
    create_mistral_llm,
    load_attack_entries,
    load_contriever,
    load_nq_corpus_and_embs,
    make_logger,
    prepare_nltk,
    ragdefender_filter,
    retrieve_full_nq,
    robustrag_keyword_agg,
    standard_rag_answer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_csv", required=True, help="CSV with query,target_answer,correct_answer,doc* columns.")
    parser.add_argument("--data_root", default=os.environ.get("DIPOISON_DATA_ROOT", "/path/to"), help="Root containing datasets/nq/* files.")
    parser.add_argument("--output_dir", default="scripts/supplementary/robustrag/results")
    parser.add_argument("--adv_per_query", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--mistral_model", default=MISTRAL_7B_MODEL)
    parser.add_argument("--model_config_path", default=None, help="Optional eval/src model config. Overrides --mistral_model.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--rr_alpha", type=float, default=0.3)
    parser.add_argument("--rr_beta", type=float, default=3.0)
    parser.add_argument("--rr_abstention", type=int, default=1)
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
    log(f"[fixed] ragdefender={RAGDEFENDER_MODEL}")
    log(f"[fixed] robustrag_generator=Mistral-7B ({args.model_config_path or args.mistral_model})")
    log(f"[config] device={device} top_k={args.top_k}")

    prepare_nltk()
    tokenizer, retriever = load_contriever(device, log)
    corpus_texts, corpus_embs_gpu = load_nq_corpus_and_embs(args.data_root, device, log)

    log(f"[load] ragdefender_model={RAGDEFENDER_MODEL}")
    defense_model = SentenceTransformer(RAGDEFENDER_MODEL, device=device)

    log(f"[load] generator=Mistral-7B ({args.model_config_path or args.mistral_model})")
    llm = create_mistral_llm(
        model_name=args.mistral_model,
        model_config_path=args.model_config_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    entries, doc_cols = load_attack_entries(args.docs_csv, args.adv_per_query, log)
    rows = []
    nd_asr = nd_acc = rr_asr = rr_acc = rdrr_asr = rdrr_acc = 0

    for qi, entry in tqdm(list(enumerate(entries)), desc="mistral7b_robustrag_asr", ncols=90):
        question = entry["query"]
        target = entry["target"]
        correct = entry["correct"]
        poison_docs = entry["poison_docs"]

        topk_docs, topk_is_poison = retrieve_full_nq(
            question,
            poison_docs,
            corpus_embs_gpu,
            corpus_texts,
            tokenizer,
            retriever,
            device,
            args.top_k,
        )

        nd_response = standard_rag_answer(question, topk_docs, llm)
        nd_asr_hit = asr_hit(target, nd_response)
        nd_acc_hit = accuracy_hit(correct, nd_response)

        rr_response, _, rr_hints = robustrag_keyword_agg(
            question,
            [str(doc).strip() for doc in topk_docs],
            llm,
            alpha=args.rr_alpha,
            beta=args.rr_beta,
            abstention_threshold=args.rr_abstention,
        )
        rr_asr_hit = asr_hit(target, rr_response)
        rr_acc_hit = accuracy_hit(correct, rr_response)

        survivor_docs, detected_n_adv = ragdefender_filter(topk_docs, defense_model)
        rdrr_response, _, rdrr_hints = robustrag_keyword_agg(
            question,
            survivor_docs,
            llm,
            alpha=args.rr_alpha,
            beta=args.rr_beta,
            abstention_threshold=args.rr_abstention,
        ) if survivor_docs else ("", [], "")
        rdrr_asr_hit = asr_hit(target, rdrr_response)
        rdrr_acc_hit = accuracy_hit(correct, rdrr_response)

        nd_asr += int(nd_asr_hit)
        nd_acc += int(nd_acc_hit)
        rr_asr += int(rr_asr_hit)
        rr_acc += int(rr_acc_hit)
        rdrr_asr += int(rdrr_asr_hit)
        rdrr_acc += int(rdrr_acc_hit)

        rows.append({
            "query_idx": qi,
            "query": question,
            "target_answer": target,
            "correct_answer": correct,
            "poison_in_top5": int(sum(topk_is_poison)),
            "ragdefender_detected_n_adv": int(detected_n_adv),
            "ragdefender_survivors": len(survivor_docs),
            "nd_asr": bool(nd_asr_hit),
            "nd_acc": bool(nd_acc_hit),
            "nd_response": nd_response,
            "rr_asr": bool(rr_asr_hit),
            "rr_acc": bool(rr_acc_hit),
            "rr_hints": rr_hints,
            "rr_response": rr_response,
            "rdrr_asr": bool(rdrr_asr_hit),
            "rdrr_acc": bool(rdrr_acc_hit),
            "rdrr_hints": rdrr_hints,
            "rdrr_response": rdrr_response,
        })

        if (qi + 1) % 10 == 0:
            log(
                f"[{qi+1}/{len(entries)}] "
                f"ND-ASR={nd_asr/(qi+1):.1%} "
                f"RR-ASR={rr_asr/(qi+1):.1%} "
                f"RD+RR-ASR={rdrr_asr/(qi+1):.1%}"
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(rows)
    summary = {
        "docs_csv": args.docs_csv,
        "n_queries": n,
        "dataset": "nq_singlehop",
        "retriever": CONTRIEVER_MODEL,
        "ragdefender": RAGDEFENDER_MODEL,
        "robustrag_generator": args.model_config_path or args.mistral_model,
        "top_k": args.top_k,
        "adv_per_query": args.adv_per_query,
        "doc_cols": doc_cols,
        "rr_alpha": args.rr_alpha,
        "rr_beta": args.rr_beta,
        "rr_abstention": args.rr_abstention,
        "no_defense": {
            "ASR": round(nd_asr / n, 4) if n else 0.0,
            "ACC": round(nd_acc / n, 4) if n else 0.0,
        },
        "robustrag_only": {
            "ASR": round(rr_asr / n, 4) if n else 0.0,
            "ACC": round(rr_acc / n, 4) if n else 0.0,
        },
        "ragdefender_plus_robustrag": {
            "ASR": round(rdrr_asr / n, 4) if n else 0.0,
            "ACC": round(rdrr_acc / n, 4) if n else 0.0,
        },
    }

    pd.DataFrame(rows).to_csv(out_dir / "details.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("[result] RobustRAG(Mistral-7B)")
    log(f"  ND-ASR:    {summary['no_defense']['ASR']*100:.1f}%")
    log(f"  RR-ASR:    {summary['robustrag_only']['ASR']*100:.1f}%")
    log(f"  RD+RR-ASR: {summary['ragdefender_plus_robustrag']['ASR']*100:.1f}%")
    log(f"[save] {out_dir / 'summary.json'}")
    log(f"[save] {out_dir / 'details.csv'}")


if __name__ == "__main__":
    main()
