#!/usr/bin/env python3
"""
HotpotQA full-corpus reranking + multihop RAGDefender ASR evaluation.

Pipeline:
  clean Contriever top-N cache + N=4 poison docs -> Contriever top-20
  -> TinyBERT-L2 reranker top-5
  -> ND: Vicuna-7B generation
  -> RD: HotpotQA multihop RAGDefender -> Vicuna-7B generation
"""

import argparse
import ast
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

_EVAL_ROOT = Path(__file__).resolve().parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from hotpotqa_multihop_ragdef_v2_eval import (  # noqa: E402
    check_acc,
    check_asr,
    ragdefender_multihop,
    wrap_prompt,
)
from main_dispo_fullcorpus_ragdef import contriever_encode  # noqa: E402


ATTACKS = {
    "PoisonedRAG": "/path/to/DisPo/data/attackbaselines_pd/PoisonedRAG/hotpotqa/poisonedrag4_hotpot100.csv",
    "Joint-GCG": "/path/to/DisPo/data/attackbaselines_pd/jointgcg/hotpotqa/hotpotqa_origin_jointgcg_v2_n4.csv",
    "Confundo": "/path/to/DisPo/data/attackbaselines_pd/confundo/hotpotqa/confundo_hotpotqa_N4.csv",
    "RAGParadox": "/path/to/DisPo/data/attackbaselines_pd/RAGParadox/hotpotqa/hotpotqa_ragparadox_n4.csv",
    "DiPoison": "/path/to/DisPo/data/attackbaselines_pd/DiPoison/hotpotqa/dipoison4_hotpot100.csv",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_path", default="/path/to/datasets/hotpotqa/corpus.jsonl")
    p.add_argument("--clean_topn_cache", default="/path/to/DisPo/eval/clean_topn_cache/hotpotqa_5attacks_top50/contriever_top50.pt")
    p.add_argument("--output_dir", default="/path/to/DisPo/eval/results/hotpotqa_tinybert_rerank_multihop_ragdef")
    p.add_argument("--reranker", default="cross-encoder/ms-marco-TinyBERT-L2-v2")
    p.add_argument("--defense_model", default="paraphrase-MiniLM-L6-v2")
    p.add_argument("--generator_model", default="lmsys/vicuna-7b-v1.3")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--adv_per_query", type=int, default=4)
    p.add_argument("--ret_top_n", type=int, default=20)
    p.add_argument("--rerank_top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def clean_str(s):
    s = str(s).strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


class VicunaGenerator:
    provider = "hf_vicuna"

    def __init__(self, model_name, device, local_files_only=False):
        self.name = model_name
        self.device = device
        self.system = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        self.model.eval()

    def query(self, prompt):
        text = f"{self.system} USER: {prompt} ASSISTANT:"
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=2048)
        ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                attention_mask=attn,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.0,
                max_new_tokens=150,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def load_attack_csv(path, adv_per_query):
    df = pd.read_csv(path)
    rows = []
    if "adv_texts" in df.columns:
        for _, row in df.iterrows():
            docs = ast.literal_eval(row["adv_texts"])
            rows.append({
                "query": str(row["question"]).strip(),
                "target_answer": str(row["incorrect answer"]).strip(),
                "correct_answer": str(row["correct answer"]).strip(),
                "poison_docs": [str(d).strip() for d in docs[:adv_per_query]],
            })
        return rows

    doc_cols = [c for c in ["doc0_seed", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6"] if c in df.columns]
    for _, row in df.iterrows():
        docs = [
            str(row[c]).strip()
            for c in doc_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ][:adv_per_query]
        if not docs:
            continue
        rows.append({
            "query": str(row["query"]).strip(),
            "target_answer": str(row["target_answer"]).strip(),
            "correct_answer": str(row["correct_answer"]).strip(),
            "poison_docs": docs,
        })
    return rows


def main():
    args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0 if torch.cuda.device_count() == 1 else args.gpu_id)
    device = f"cuda:{torch.cuda.current_device()}"

    out_dir = Path(args.output_dir) / datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    log(f"[config] device={device}")
    log(f"[config] ret_top_n={args.ret_top_n} rerank_top_k={args.rerank_top_k}")
    log(f"[config] reranker={args.reranker}")
    log(f"[config] defense_model={args.defense_model}")
    log(f"[config] generator={args.generator_model}")
    log(f"[config] clean_topn_cache={args.clean_topn_cache}")

    log("[load] clean top-N cache")
    cache = torch.load(args.clean_topn_cache, map_location="cpu", weights_only=True)
    meta = cache.get("meta", {})
    if meta.get("dataset") != "hotpotqa" or meta.get("retrieval_model") != "contriever":
        raise ValueError(f"Unexpected cache meta: {meta}")
    if int(meta.get("top_n", 0)) < args.ret_top_n:
        raise ValueError(f"cache top_n={meta.get('top_n')} < ret_top_n={args.ret_top_n}")
    query_to_row = {str(q).strip(): i for i, q in enumerate(cache["queries"])}
    top_indices = cache["top_indices"].long()
    top_scores = cache["top_scores"].float()
    log(f"[load] cache queries={len(query_to_row)} top_n={meta.get('top_n')}")

    log(f"[load] corpus text: {args.corpus_path}")
    corpus_texts = []
    with open(args.corpus_path, encoding="utf-8") as f:
        for line in f:
            corpus_texts.append(json.loads(line).get("text", ""))
    log(f"[load] corpus passages={len(corpus_texts):,}")

    log("[load] Contriever")
    ctv_tok = AutoTokenizer.from_pretrained(
        "facebook/contriever",
        local_files_only=args.local_files_only,
    )
    ctv_mod = AutoModel.from_pretrained(
        "facebook/contriever",
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).to(device)
    ctv_mod.eval()

    def encode_ctv(texts):
        return contriever_encode(texts, ctv_mod, ctv_tok, device, batch_size=64)

    log(f"[load] TinyBERT-L2 reranker: {args.reranker}")
    reranker = CrossEncoder(args.reranker, device=device)

    log(f"[load] multihop RAGDefender defense model: {args.defense_model}")
    defense_model = SentenceTransformer(args.defense_model)

    log(f"[load] Vicuna generator: {args.generator_model}")
    llm = VicunaGenerator(args.generator_model, device, local_files_only=args.local_files_only)
    log(f"[load] ready. GPU memory={torch.cuda.memory_allocated()/1e9:.1f}GB")

    def retrieve_top20(query, poison_docs):
        row_idx = query_to_row.get(str(query).strip())
        if row_idx is None:
            raise KeyError(f"query not in cache: {query}")

        clean_idx = top_indices[row_idx]
        clean_scores = top_scores[row_idx]
        q_emb = encode_ctv([query]).to(device).half()
        p_emb = encode_ctv(poison_docs).to(device).half()
        adv_scores = torch.mm(p_emb, q_emb.T).squeeze(1).float().cpu()

        all_scores = torch.cat([clean_scores, adv_scores], dim=0)
        order = all_scores.topk(args.ret_top_n).indices.tolist()
        docs, is_poison = [], []
        n_clean = clean_idx.numel()
        for idx in order:
            if idx < n_clean:
                docs.append(corpus_texts[int(clean_idx[idx])])
                is_poison.append(False)
            else:
                docs.append(poison_docs[idx - n_clean])
                is_poison.append(True)
        return docs, is_poison

    summary_rows = []
    all_detail_frames = []
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for attack_name, csv_path in ATTACKS.items():
        log(f"\n[attack] {attack_name}: {csv_path}")
        rows = load_attack_csv(csv_path, args.adv_per_query)
        rows = [r for r in rows if r["query"] in query_to_row]
        log(f"[attack] valid queries={len(rows)}")

        nd_asr = nd_acc = rd_asr = rd_acc = 0
        poison_top20 = poison_top5 = rd_docs_total = 0
        detail_rows = []

        for q_idx, row in enumerate(tqdm(rows, desc=attack_name, dynamic_ncols=True)):
            query = row["query"]
            target = row["target_answer"]
            correct = row["correct_answer"]
            poison_docs = row["poison_docs"][:args.adv_per_query]

            top20_docs, top20_poison = retrieve_top20(query, poison_docs)
            scores = reranker.predict([(query, d) for d in top20_docs])
            order = np.argsort(-np.asarray(scores))[:args.rerank_top_k]
            reranked_docs = [top20_docs[i] for i in order]
            reranked_poison = [top20_poison[i] for i in order]

            poison_top20 += sum(top20_poison)
            poison_top5 += sum(reranked_poison)

            nd_resp = llm.query(wrap_prompt(query, reranked_docs))
            nd_is_asr = check_asr(target, nd_resp)
            nd_is_acc = check_acc(correct, nd_resp)
            nd_asr += int(nd_is_asr)
            nd_acc += int(nd_is_acc)

            rd_docs = ragdefender_multihop(reranked_docs, defense_model)
            rd_docs_total += len(rd_docs)
            rd_resp = llm.query(wrap_prompt(query, rd_docs)) if rd_docs else ""
            rd_is_asr = check_asr(target, rd_resp) if rd_resp else False
            rd_is_acc = check_acc(correct, rd_resp) if rd_resp else False
            rd_asr += int(rd_is_asr)
            rd_acc += int(rd_is_acc)

            detail_rows.append({
                "attack": attack_name,
                "query_index": q_idx,
                "query": query,
                "target_answer": target,
                "correct_answer": correct,
                "poison_in_top20": sum(top20_poison),
                "poison_in_rerank_top5": sum(reranked_poison),
                "rd_num_docs": len(rd_docs),
                "nd_response": nd_resp,
                "nd_asr": bool(nd_is_asr),
                "nd_acc": bool(nd_is_acc),
                "rd_response": rd_resp,
                "rd_asr": bool(rd_is_asr),
                "rd_acc": bool(rd_is_acc),
            })

            if (q_idx + 1) % 10 == 0:
                n_now = q_idx + 1
                log(
                    f"  [{n_now}/{len(rows)}] "
                    f"ND-ASR={nd_asr/n_now:.1%} RD-ASR={rd_asr/n_now:.1%} "
                    f"ND-ACC={nd_acc/n_now:.1%} RD-ACC={rd_acc/n_now:.1%}"
                )

        n = len(rows)
        summary = {
            "run_at": run_at,
            "attack": attack_name,
            "docs_csv": csv_path,
            "retriever": "contriever",
            "ret_top_n": args.ret_top_n,
            "reranker": args.reranker,
            "rerank_top_k": args.rerank_top_k,
            "defense": "hotpotqa_multihop_ragdef_v2_eval.ragdefender_multihop",
            "defense_model": args.defense_model,
            "generator": args.generator_model,
            "num_queries": n,
            "nd_asr": round(nd_asr / n, 4),
            "nd_acc": round(nd_acc / n, 4),
            "rd_asr": round(rd_asr / n, 4),
            "rd_acc": round(rd_acc / n, 4),
            "asr_drop": round((nd_asr - rd_asr) / n, 4),
            "avg_poison_top20": round(poison_top20 / n, 4),
            "avg_poison_rerank_top5": round(poison_top5 / n, 4),
            "avg_rd_docs": round(rd_docs_total / n, 4),
        }
        summary_rows.append(summary)
        detail_df = pd.DataFrame(detail_rows)
        detail_path = out_dir / f"details_{attack_name}.csv"
        detail_df.to_csv(detail_path, index=False)
        all_detail_frames.append(detail_df)
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
        log(
            f"[result] {attack_name}: "
            f"ND-ASR={summary['nd_asr']*100:.1f}% RD-ASR={summary['rd_asr']*100:.1f}% "
            f"ND-ACC={summary['nd_acc']*100:.1f}% RD-ACC={summary['rd_acc']*100:.1f}%"
        )
        log(f"[save] {detail_path}")
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    if all_detail_frames:
        pd.concat(all_detail_frames, ignore_index=True).to_csv(out_dir / "details_all.csv", index=False)
    log(f"[save] {out_dir / 'summary.csv'}")
    log(f"[save] {out_dir / 'summary.json'}")
    log("[done]")


if __name__ == "__main__":
    main()
