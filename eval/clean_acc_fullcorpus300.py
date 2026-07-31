"""
clean_acc_fullcorpus300.py — Clean (no-poison) full-corpus RAG accuracy on a
300-query holdout set that does not overlap with the canonical 100-query
benchmark (data/attackbaselines_pd/DiPoison/dipoison4_nq100.csv).

Pipeline: Contriever full-corpus top-5 (NO adv docs injected) -> Vicuna-7B ->
accuracy = any of the query's acceptable answers appears in the response.

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 DIPOISON_DATA_ROOT=/path/to \
    PYTHONUNBUFFERED=1 python eval/clean_acc_fullcorpus300.py \
    --queries_csv data/generated/nq_clean_acc/clean_acc_queries300.csv \
    --gpu_id 0
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main_dipoison_fullcorpus_ragdef import (
    _DS_CFG, _RETRIEVAL_ALIAS, _cache_path_for, build_or_load_corpus_embs,
    contriever_encode, clean_str,
)
from src.models import create_model
from src.prompts import wrap_prompt as legacy_wrap_prompt

_DEFAULT_MODEL_CONFIG = str(_ROOT / "model_configs" / "vicuna7b_config.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="nq", choices=["nq", "hotpotqa", "msmarco"])
    p.add_argument("--queries_csv", type=str,
                   default="data/generated/nq_clean_acc/clean_acc_queries300.csv")
    p.add_argument("--out_csv", type=str,
                   default="data/generated/nq_clean_acc/clean_acc_300_results.csv")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--model_config_path", type=str, default=_DEFAULT_MODEL_CONFIG)
    p.add_argument("--embed_batch", type=int, default=512)
    args = p.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    nd = torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_id if args.gpu_id < nd else 0)
    device = f"cuda:{torch.cuda.current_device()}"
    print(f"[device] {device}")

    cfg = _DS_CFG[args.dataset]
    model_hf_name = _RETRIEVAL_ALIAS["contriever"]

    # ── corpus 로딩 ──────────────────────────────────────────────
    print(f"[load] corpus 로딩: {cfg['corpus_path']}")
    corpus_texts = []
    with open(cfg["corpus_path"]) as f:
        for line in f:
            d = json.loads(line)
            corpus_texts.append(d.get("text", ""))
    print(f"[load] corpus {len(corpus_texts):,} passages")

    # ── Contriever ───────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModel
    print(f"[load] Contriever -> {device}")
    ctv_tok = AutoTokenizer.from_pretrained(model_hf_name)
    ctv_mod = AutoModel.from_pretrained(model_hf_name, torch_dtype=torch.float32).to(device)
    ctv_mod.eval()

    def encode_fn(texts):
        return contriever_encode(texts, ctv_mod, ctv_tok, device, batch_size=64)

    # ── corpus embeddings (캐시 재사용) ────────────────────────────
    cache_path = _cache_path_for(cfg, model_hf_name)
    corpus_embs = build_or_load_corpus_embs(
        corpus_texts, cache_path, encode_fn, print, batch_size=args.embed_batch,
    )
    print(f"[embed] GPU 전송 중... ({corpus_embs.shape[0]:,} x {corpus_embs.shape[1]})")
    corpus_embs_gpu = corpus_embs.half().to(device)
    print(f"[embed] 완료. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Vicuna ───────────────────────────────────────────────────
    print(f"[load] LLM via create_model: {args.model_config_path}")
    llm = create_model(args.model_config_path)
    print(f"[load] LLM: provider={llm.provider} name={llm.name}")

    # ── queries ──────────────────────────────────────────────────
    qdf = pd.read_csv(args.queries_csv)
    print(f"[load] queries_csv: {len(qdf)} rows")

    rows_out = []
    acc_cnt = 0
    pbar = tqdm(qdf.itertuples(index=False), total=len(qdf), desc="Clean-ACC(full corpus top-5)", ncols=100)
    for row in pbar:
        query = str(row.query).strip()
        aliases = json.loads(row.correct_answer_aliases)

        q_emb = encode_fn([query]).to(device).half()
        scores = torch.mv(corpus_embs_gpu, q_emb.squeeze(0))
        topk_idx = scores.topk(args.top_k).indices.cpu().tolist()
        retrieved_docs = [corpus_texts[i] for i in topk_idx]

        prompt = legacy_wrap_prompt(query, [clean_str(d) for d in retrieved_docs], 4)
        response = llm.query(prompt)

        is_correct = any(
            clean_str(a) in clean_str(response) or clean_str(response) in clean_str(a)
            for a in aliases
        )
        acc_cnt += int(is_correct)

        rows_out.append({
            "query": query,
            "correct_answer": row.correct_answer,
            "correct_answer_aliases": row.correct_answer_aliases,
            "response": response,
            "is_correct": is_correct,
        })
        pbar.set_postfix(acc=f"{acc_cnt}/{len(rows_out)}")

        if len(rows_out) % 20 == 0:
            print(f"[progress] {len(rows_out)}/{len(qdf)}  running_acc={acc_cnt/len(rows_out)*100:.1f}%", flush=True)

    n = len(rows_out)
    final_acc = acc_cnt / n if n else 0.0
    print(f"\n{'='*60}")
    print(f"  Clean full-corpus ACC (N={n}, top_k={args.top_k}, no poison)")
    print(f"  ACC = {final_acc*100:.1f}%  ({acc_cnt}/{n})")
    print(f"{'='*60}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    with open(out_path.with_suffix(".summary.json"), "w") as f:
        json.dump({"n": n, "top_k": args.top_k, "acc_count": acc_cnt, "ACC": round(final_acc, 4)}, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
