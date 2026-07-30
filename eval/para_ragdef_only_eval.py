"""
para_ragdef_only_eval.py
Pipeline: paraphrase attack docs → Contriever top-5 (no reranking)
          → RAGDefender filter → Mistral-7B → ASR

Reports per attack:
  - ND-ASR : Contriever top-5 → standard Mistral-7B (no defense)
  - RD-ASR : Contriever top-5 → RAGDefender → standard Mistral-7B
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sktext
import torch
from sentence_transformers import SentenceTransformer, util as st_util
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.models import create_model
from src.prompts import wrap_prompt as legacy_wrap_prompt

_DATA_ROOT   = os.environ.get("DISPO_DATA_ROOT", "/path/to")
_CORPUS_PATH = f"{_DATA_ROOT}/datasets/nq/corpus.jsonl"
_EMB_CACHE   = f"{_DATA_ROOT}/datasets/nq/contriever_embs_fullcorpus.pt"

ATTACK_CSVS = {
    "dipoison4":    "data/paraphrase_pd/DiPoison/dipoison4_nq100_para.csv",
    "poisonedrag4": "data/paraphrase_pd/PoisonedRAG/poisonedrag4_nq100_para.csv",
    "ragparadox4":  "data/paraphrase_pd/RAGParadox/ragparadox_nq100_n4_para.csv",
    "confundo4":    "data/paraphrase_pd/confundo/confundo_500input_nq_N4_temp0.7_v2_para.csv",
    "jointgcg4":    "data/paraphrase_pd/jointgcg/jointgcg4_nq100_para.csv",
}

GPU_ID    = 1
TOP_K     = 5
ADV_N     = 4
MODEL_CFG = str(_ROOT / "model_configs" / "vicuna7b_gpu1_config.json")
OUT_DIR   = Path("eval/results/para_ragdef_only_vicuna")
LOG_PATH  = Path("eval/results/para_ragdef_only_vicuna_run.log")


def clean_str(s):
    s = str(s).strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def mean_pool(emb, mask):
    m = mask.unsqueeze(-1).expand(emb.size()).float()
    return torch.sum(emb * m, 1) / torch.clamp(m.sum(1), min=1e-9)


def contriever_encode(texts, model, tok, device, batch_size=64):
    if isinstance(texts, str):
        texts = [texts]
    outs = []
    for i in range(0, len(texts), batch_size):
        inp = tok(texts[i:i+batch_size], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        outs.append(mean_pool(h, inp["attention_mask"]).cpu())
    return torch.cat(outs, dim=0)


def _find_num_adv_tfidf(text_list):
    sw = list(sktext.ENGLISH_STOP_WORDS)
    tfidf = sktext.TfidfVectorizer(stop_words=sw)
    X = tfidf.fit_transform(text_list)
    df = pd.DataFrame(X.todense().tolist(), columns=tfidf.get_feature_names_out())
    top_m = df.T.sum(axis=1).sort_values(ascending=False)[:5]
    indices = [[1 if word in sentence else 0 for sentence in text_list] for word in top_m.index]
    final = [1 if sum(idx[i] for idx in indices) > math.floor(len(indices)/2) else 0
             for i in range(len(text_list))]
    return sum(final)


def ragdefender_stage1(text_list, s_model):
    if len(text_list) < 2:
        return 0, set()
    embs = s_model.encode(text_list, convert_to_tensor=True)
    clust = AgglomerativeClustering(n_clusters=2)
    clust.fit(embs.cpu().numpy())
    labels = list(clust.labels_)
    n = len(text_list)
    n1, n0 = sum(labels), n - sum(labels)
    nmin = min(n1, n0)
    try:
        num_tfidf = _find_num_adv_tfidf(text_list)
    except ValueError:
        num_tfidf = 0
    if n1 > 0 and num_tfidf <= int(n/2):
        n_adv     = nmin
        adv_label = 1 if n1 <= n0 else 0
    else:
        n_adv     = max(n1, n0)
        adv_label = 1 if n1 >= n0 else 0
    stage1_adv_idx = {i for i, lbl in enumerate(labels) if lbl == adv_label}
    return int(n_adv), stage1_adv_idx


def top_similar_pairs(texts, model, top_k):
    embs = model.encode(texts, convert_to_tensor=True)
    cos  = st_util.cos_sim(embs, embs)
    pairs = [(i, j, cos[i][j].item())
             for i in range(len(texts))
             for j in range(i+1, len(texts))]
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]


def ragdefender_filter(docs, adv_positions, defense_model):
    if len(docs) < 2:
        return docs, any(i in adv_positions for i in range(len(docs)))
    n_adv, _ = ragdefender_stage1(docs, defense_model)
    gen_num   = max(1, int(n_adv * (n_adv - 1) / 2))
    adv_pairs = top_similar_pairs(docs, defense_model, gen_num)
    pair_cnt  = Counter()
    for x, y, sim in adv_pairs:
        freq = math.copysign(sim * sim, sim)
        pair_cnt[x] += freq
        pair_cnt[y] += freq
    scores_list = [{"index": ri, "is_adv": ri in adv_positions,
                    "freq": float(pair_cnt.get(ri, 0.0))}
                   for ri in range(len(docs))]
    sorted_scores = sorted(scores_list, key=lambda x: x["freq"], reverse=True)
    num_survivors = max(0, len(sorted_scores) - n_adv)
    survivors     = sorted_scores[-num_survivors:] if num_survivors > 0 else []
    survivor_docs = [docs[d["index"]] for d in survivors]
    poison_survived = any(d["is_adv"] for d in survivors)
    return survivor_docs, poison_survived


def pick_doc_cols(df, adv_per_query):
    cols = []
    candidates = ["doc0_seed"] + [f"doc{i}" for i in range(1, adv_per_query + 1)]
    for c in candidates:
        if c in df.columns and len(cols) < adv_per_query:
            cols.append(c)
    if not cols:
        cols = [c for c in df.columns if c.startswith("doc")][:adv_per_query]
    return cols[:adv_per_query]


def main():
    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(str(msg) + "\n")

    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"[start] {run_at}")
    log(f"[config] device={device} top_k={TOP_K} adv_per_query={ADV_N} (no reranking)")

    # Contriever
    log("[load] Contriever...")
    ctv_tok = AutoTokenizer.from_pretrained("facebook/contriever")
    ctv_mod = AutoModel.from_pretrained("facebook/contriever", torch_dtype=torch.float32).to(device)
    ctv_mod.eval()

    def encode_ctv(texts):
        return contriever_encode(texts, ctv_mod, ctv_tok, device)

    # NQ corpus
    log(f"[load] corpus: {_CORPUS_PATH}")
    corpus_texts = []
    with open(_CORPUS_PATH) as f:
        for line in f:
            corpus_texts.append(json.loads(line).get("text", ""))
    log(f"[load] corpus {len(corpus_texts):,} passages")

    log(f"[load] corpus embeddings: {_EMB_CACHE}")
    corpus_embs = torch.load(_EMB_CACHE, map_location="cpu", weights_only=True)
    corpus_embs_gpu = corpus_embs.half().to(device)
    log(f"[load] corpus embs GPU: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")
    del corpus_embs
    gc.collect()

    # Defense model
    log("[load] defense model: paraphrase-MiniLM-L6-v2")
    defense_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # LLM
    log(f"[load] LLM: {MODEL_CFG}")
    llm = create_model(MODEL_CFG)
    log(f"[load] LLM: {llm.provider} / {llm.name}")

    gc.collect()
    torch.cuda.empty_cache()

    summary_rows = []
    all_rows = []

    for attack_name, rel_csv in ATTACK_CSVS.items():
        docs_csv = str(Path("/path/to/DisPo") / rel_csv)
        log(f"\n{'='*60}")
        log(f"[attack] {attack_name}: {docs_csv}")

        df = pd.read_csv(docs_csv)
        doc_cols = pick_doc_cols(df, ADV_N)
        log(f"[attack] rows={len(df)} doc_cols={doc_cols}")

        entries = []
        for _, row in df.iterrows():
            q = str(row["query"]).strip()
            poison_docs = [str(row[c]).strip() for c in doc_cols
                           if c in row.index and pd.notna(row[c]) and str(row[c]).strip()]
            if not poison_docs:
                continue
            entries.append({
                "query":       q,
                "target":      str(row["target_answer"]).strip(),
                "correct":     str(row["correct_answer"]).strip(),
                "poison_docs": poison_docs[:ADV_N],
            })

        nd_asr = nd_acc = rd_asr = rd_acc = 0
        n = len(entries)
        n_corpus = corpus_embs_gpu.shape[0]

        pbar = tqdm(enumerate(entries), total=n, desc=attack_name, ncols=90)
        for qi, entry in pbar:
            q           = entry["query"]
            target      = entry["target"]
            correct     = entry["correct"]
            poison_docs = entry["poison_docs"]

            # ① Contriever full-corpus top-k (no reranking)
            adv_embs = encode_ctv(poison_docs).to(device).half()
            q_emb    = encode_ctv([q]).to(device).half()
            corp_sc  = torch.mm(corpus_embs_gpu, q_emb.T).squeeze(1)
            adv_sc   = torch.mm(adv_embs, q_emb.T).squeeze(1)
            all_sc   = torch.cat([corp_sc, adv_sc], dim=0)
            topk_idx = all_sc.topk(TOP_K).indices.cpu().tolist()

            topk_docs      = []
            topk_is_poison = []
            for idx in topk_idx:
                if idx < n_corpus:
                    topk_docs.append(corpus_texts[idx])
                    topk_is_poison.append(False)
                else:
                    topk_docs.append(poison_docs[idx - n_corpus])
                    topk_is_poison.append(True)
            adv_positions = {i for i, p in enumerate(topk_is_poison) if p}

            # ② ND: Mistral on top-k (no defense)
            nd_prompt = legacy_wrap_prompt(q, [clean_str(d) for d in topk_docs], 4)
            nd_resp   = llm.query(nd_prompt)
            nd_asr_b  = (clean_str(target) in clean_str(nd_resp)
                         or clean_str(nd_resp) in clean_str(target))
            nd_acc_b  = (clean_str(correct) in clean_str(nd_resp)
                         or clean_str(nd_resp) in clean_str(correct))

            # ③ RAGDefender filter
            survivor_docs, _ = ragdefender_filter(
                [clean_str(d) for d in topk_docs], adv_positions, defense_model
            )

            # ④ RD: Mistral on survivors
            if survivor_docs:
                rd_prompt = legacy_wrap_prompt(q, survivor_docs, 4)
                rd_resp   = llm.query(rd_prompt)
            else:
                rd_resp = ""
            rd_asr_b = (clean_str(target) in clean_str(rd_resp)
                        or clean_str(rd_resp) in clean_str(target)) if rd_resp else False
            rd_acc_b = (clean_str(correct) in clean_str(rd_resp)
                        or clean_str(rd_resp) in clean_str(correct)) if rd_resp else False

            if nd_asr_b: nd_asr += 1
            if nd_acc_b: nd_acc += 1
            if rd_asr_b: rd_asr += 1
            if rd_acc_b: rd_acc += 1

            all_rows.append({
                "attack": attack_name, "query_idx": qi, "query": q,
                "target": target, "correct": correct,
                "poison_in_topk": sum(topk_is_poison),
                "n_survivors": len(survivor_docs),
                "nd_asr": nd_asr_b, "nd_acc": nd_acc_b, "nd_resp": nd_resp,
                "rd_asr": rd_asr_b, "rd_acc": rd_acc_b, "rd_resp": rd_resp,
            })

            gc.collect()
            torch.cuda.empty_cache()

            if (qi + 1) % 10 == 0:
                pbar.set_postfix({
                    "ND": f"{nd_asr/(qi+1):.0%}",
                    "RD": f"{rd_asr/(qi+1):.0%}",
                })

        pbar.close()

        result = {
            "attack":    attack_name,
            "docs_csv":  docs_csv,
            "run_at":    run_at,
            "n_queries": n,
            "top_k":     TOP_K,
            "ND_ASR":    round(nd_asr / n, 4),
            "ND_ACC":    round(nd_acc / n, 4),
            "RD_ASR":    round(rd_asr / n, 4),
            "RD_ACC":    round(rd_acc / n, 4),
        }
        summary_rows.append(result)

        log(f"\n[result] {attack_name}")
        log(f"  ND-ASR (no rerank, no defense)  : {nd_asr/n*100:.1f}%")
        log(f"  RD-ASR (no rerank + RAGDefender): {rd_asr/n*100:.1f}%")

        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    log(f"\n{'='*60}")
    log("[FINAL SUMMARY] Paraphrase → Contriever top-5 (no rerank) → RAGDefender → Mistral-7B")
    log(f"{'Attack':<30} {'ND-ASR':>8} {'RD-ASR':>8}")
    log(f"{'-'*48}")
    for r in summary_rows:
        log(f"{r['attack']:<30} {r['ND_ASR']*100:>7.1f}% {r['RD_ASR']*100:>7.1f}%")
    log(f"{'='*60}")

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "details.csv", index=False)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    log(f"[save] {OUT_DIR}/summary.json")
    log("[done]")


if __name__ == "__main__":
    main()
