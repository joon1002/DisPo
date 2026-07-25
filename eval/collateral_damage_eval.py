"""
collateral_damage_eval.py — Collateral-damage test: inject the 400 poison docs
crafted for one 100-query attack set (e.g. DiPoison N=4/G=8) into the full
corpus, then measure clean ACC on a DIFFERENT, unrelated 100-query set
(clean_acc100.csv) whose clean (no-poison) ACC is 100% by construction.

For each of the 100 unrelated queries we run BOTH:
  - clean   top-5 (corpus only)             -> Vicuna -> ACC
  - poisoned top-5 (corpus + 400 poison docs, shared/global pool) -> Vicuna -> ACC
in the same process/run, so the delta isolates the injection effect rather
than cross-run generation stochasticity.

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 DISPO_DATA_ROOT=/data1/joonhyung \
    PYTHONUNBUFFERED=1 python eval/collateral_damage_eval.py \
    --victim_queries_csv data/generated/nq_clean_acc/clean_acc100.csv \
    --poison_docs_csv data/attackbaselines_pd/DiPoison/dipoison4_nq100.csv \
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

from main_dispo_fullcorpus_ragdef import (
    _DS_CFG, _RETRIEVAL_ALIAS, _cache_path_for, build_or_load_corpus_embs,
    contriever_encode, clean_str,
)
from src.models import create_model
from src.prompts import wrap_prompt as legacy_wrap_prompt

_DEFAULT_MODEL_CONFIG = str(_ROOT / "model_configs" / "vicuna7b_config.json")
_DOC_COLS = ["doc0_seed", "doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--victim_queries_csv", type=str,
                   default="data/generated/nq_clean_acc/clean_acc100.csv")
    p.add_argument("--poison_docs_csv", type=str,
                   default="data/attackbaselines_pd/DiPoison/dipoison4_nq100.csv")
    p.add_argument("--out_csv", type=str,
                   default="data/generated/nq_clean_acc/collateral_damage_results.csv")
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

    cfg = _DS_CFG["nq"]
    model_hf_name = _RETRIEVAL_ALIAS["contriever"]

    print(f"[load] corpus 로딩: {cfg['corpus_path']}")
    corpus_texts = []
    with open(cfg["corpus_path"]) as f:
        for line in f:
            d = json.loads(line)
            corpus_texts.append(d.get("text", ""))
    print(f"[load] corpus {len(corpus_texts):,} passages")

    from transformers import AutoTokenizer, AutoModel
    print(f"[load] Contriever -> {device}")
    ctv_tok = AutoTokenizer.from_pretrained(model_hf_name)
    ctv_mod = AutoModel.from_pretrained(model_hf_name, torch_dtype=torch.float32).to(device)
    ctv_mod.eval()

    def encode_fn(texts):
        return contriever_encode(texts, ctv_mod, ctv_tok, device, batch_size=64)

    cache_path = _cache_path_for(cfg, model_hf_name)
    corpus_embs = build_or_load_corpus_embs(
        corpus_texts, cache_path, encode_fn, print, batch_size=args.embed_batch,
    )
    print(f"[embed] GPU 전송 중... ({corpus_embs.shape[0]:,} x {corpus_embs.shape[1]})")
    corpus_embs_gpu = corpus_embs.half().to(device)
    n_corpus = corpus_embs_gpu.shape[0]
    print(f"[embed] 완료. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── poison docs (shared/global pool injected for every victim query) ──
    pdf = pd.read_csv(args.poison_docs_csv)
    poison_docs, poison_src = [], []
    for ridx, row in pdf.iterrows():
        for c in _DOC_COLS:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
                poison_docs.append(str(row[c]).strip())
                poison_src.append(f"row{ridx}:{c}")
    n_poison = len(poison_docs)
    print(f"[load] poison_docs_csv: {len(pdf)} rows -> {n_poison} poison docs (shared pool)")

    poison_embs = encode_fn(poison_docs).to(device).half()  # (n_poison, D)

    # ── Vicuna ───────────────────────────────────────────────────
    print(f"[load] LLM via create_model: {args.model_config_path}")
    llm = create_model(args.model_config_path)
    print(f"[load] LLM: provider={llm.provider} name={llm.name}")

    # ── victim queries ───────────────────────────────────────────
    qdf = pd.read_csv(args.victim_queries_csv)
    print(f"[load] victim_queries_csv: {len(qdf)} rows")

    rows_out = []
    clean_acc_cnt = poison_acc_cnt = collision_cnt = 0
    pbar = tqdm(qdf.itertuples(index=False), total=len(qdf), desc="Collateral-damage", ncols=100)
    for row in pbar:
        query = str(row.query).strip()
        aliases = json.loads(row.correct_answer_aliases)

        q_emb = encode_fn([query]).to(device).half()  # (1, D)

        # clean top-5 (corpus only)
        clean_scores = torch.mv(corpus_embs_gpu, q_emb.squeeze(0))
        clean_idx = clean_scores.topk(args.top_k).indices.cpu().tolist()
        clean_docs = [corpus_texts[i] for i in clean_idx]

        # poisoned top-5 (corpus + shared 400-doc poison pool)
        poison_scores = torch.mv(poison_embs, q_emb.squeeze(0))
        all_scores = torch.cat([clean_scores, poison_scores], dim=0)
        top_idx = all_scores.topk(args.top_k).indices.cpu().tolist()
        poisoned_docs, poison_hits = [], []
        for idx in top_idx:
            if idx < n_corpus:
                poisoned_docs.append(corpus_texts[idx])
            else:
                pidx = idx - n_corpus
                poisoned_docs.append(poison_docs[pidx])
                poison_hits.append(poison_src[pidx])

        has_collision = len(poison_hits) > 0
        collision_cnt += int(has_collision)

        clean_prompt = legacy_wrap_prompt(query, [clean_str(d) for d in clean_docs], 4)
        clean_resp = llm.query(clean_prompt)
        clean_correct = any(
            clean_str(a) in clean_str(clean_resp) or clean_str(clean_resp) in clean_str(a)
            for a in aliases
        )

        poison_prompt = legacy_wrap_prompt(query, [clean_str(d) for d in poisoned_docs], 4)
        poison_resp = llm.query(poison_prompt)
        poison_correct = any(
            clean_str(a) in clean_str(poison_resp) or clean_str(poison_resp) in clean_str(a)
            for a in aliases
        )

        clean_acc_cnt += int(clean_correct)
        poison_acc_cnt += int(poison_correct)

        rows_out.append({
            "query": query,
            "correct_answer": row.correct_answer,
            "num_poison_in_top5": len(poison_hits),
            "poison_hit_sources": ";".join(poison_hits),
            "clean_response": clean_resp,
            "clean_correct": clean_correct,
            "poisoned_response": poison_resp,
            "poisoned_correct": poison_correct,
            "flipped_correct_to_wrong": clean_correct and not poison_correct,
        })
        pbar.set_postfix(clean=f"{clean_acc_cnt}", poisoned=f"{poison_acc_cnt}", coll=f"{collision_cnt}")

        if len(rows_out) % 20 == 0:
            print(f"[progress] {len(rows_out)}/{len(qdf)}  "
                  f"clean_acc={clean_acc_cnt/len(rows_out)*100:.1f}%  "
                  f"poisoned_acc={poison_acc_cnt/len(rows_out)*100:.1f}%  "
                  f"collision={collision_cnt}/{len(rows_out)}", flush=True)

    n = len(rows_out)
    clean_acc = clean_acc_cnt / n if n else 0.0
    poison_acc = poison_acc_cnt / n if n else 0.0
    collision_rate = collision_cnt / n if n else 0.0

    print(f"\n{'='*70}")
    print(f"  Collateral damage: {n_poison} poison docs (from {args.poison_docs_csv})")
    print(f"  injected into full corpus, measured on {n} UNRELATED queries")
    print(f"  {'-'*66}")
    print(f"  Clean ACC (no poison)         {clean_acc*100:>6.1f}%  ({clean_acc_cnt}/{n})")
    print(f"  Poisoned-corpus ACC           {poison_acc*100:>6.1f}%  ({poison_acc_cnt}/{n})")
    print(f"  ACC delta                     {(poison_acc-clean_acc)*100:>+6.1f}%p")
    print(f"  Poison-collision rate (top5)  {collision_rate*100:>6.1f}%  ({collision_cnt}/{n})")
    print(f"{'='*70}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    summary = {
        "n_victim_queries": n, "n_poison_docs": n_poison, "top_k": args.top_k,
        "clean_acc": round(clean_acc, 4), "poisoned_acc": round(poison_acc, 4),
        "delta_pp": round((poison_acc - clean_acc) * 100, 2),
        "collision_rate": round(collision_rate, 4),
    }
    with open(out_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
