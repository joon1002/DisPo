#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


_ROOT = Path(__file__).resolve().parent.parent
_ATTACK_DATA_ROOT = os.environ.get("DIPOISON_ATTACK_DATA_ROOT", "/path/to")

DEFAULT_ATTACKS = {
    "poisonedRAG": f"{_ATTACK_DATA_ROOT}/results/attackbaselines_pd/poisonedrag_nq100.csv",
    "jointgcg": f"{_ATTACK_DATA_ROOT}/results/attackbaselines_pd/jointgcg4_nq100.csv",
    "ragparadox": f"{_ATTACK_DATA_ROOT}/results/attackbaselines_pd/ragparadox_nq100_n4.csv",
    "confundo": f"{_ATTACK_DATA_ROOT}/results/attackbaselines_pd/confundo_500input_nq_N4_temp0.7_v2.csv",
    "dipoison_v7_cont_n4g8": str(_ROOT / "data/attackbaselines_pd/DiPoison/nq/pd_eval100_cont_n4g8.csv"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="gpt2-xl")
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_dir", default=str(_ROOT / "eval/ppl_gpt2xl_attacks"))
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def pick_doc_cols(df):
    cols = []
    for c in ["doc0_seed", "doc1", "doc2", "doc3"]:
        if c in df.columns:
            cols.append(c)
    if len(cols) < 4:
        raise ValueError(f"Expected 4 doc columns, found {cols}")
    return cols


def load_docs():
    rows = []
    for attack, path in DEFAULT_ATTACKS.items():
        df = pd.read_csv(path)
        doc_cols = pick_doc_cols(df)
        for row_idx, row in df.iterrows():
            query = str(row.get("query", "")).strip()
            for doc_col in doc_cols:
                text = str(row[doc_col]).strip() if pd.notna(row[doc_col]) else ""
                if not text:
                    continue
                rows.append({
                    "attack": attack,
                    "source_csv": path,
                    "row_idx": row_idx,
                    "query": query,
                    "doc_col": doc_col,
                    "text": text,
                    "char_len": len(text),
                    "word_len": len(text.split()),
                })
    return rows


def batch_ppl(model, tokenizer, texts, device, max_length):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, 0] = -100

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        tok_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shifted_labels.shape)

    valid = (shifted_labels != -100)
    token_counts = valid.sum(dim=1)
    nll = tok_loss.sum(dim=1) / token_counts.clamp(min=1)
    ppl = torch.exp(nll).clamp(max=1e9)
    return nll.float().cpu().numpy(), ppl.float().cpu().numpy(), token_counts.cpu().numpy()


def summarize(details):
    df = pd.DataFrame(details)
    rows = []
    for attack, g in df.groupby("attack", sort=False):
        ppl = g["ppl"].to_numpy(dtype=float)
        nll = g["nll"].to_numpy(dtype=float)
        rows.append({
            "attack": attack,
            "num_docs": int(len(g)),
            "num_queries": int(g["row_idx"].nunique()),
            "mean_ppl": float(np.mean(ppl)),
            "median_ppl": float(np.median(ppl)),
            "std_ppl": float(np.std(ppl)),
            "p90_ppl": float(np.percentile(ppl, 90)),
            "p95_ppl": float(np.percentile(ppl, 95)),
            "p99_ppl": float(np.percentile(ppl, 99)),
            "mean_nll": float(np.mean(nll)),
            "median_nll": float(np.median(nll)),
            "mean_tokens": float(g["token_count"].mean()),
            "median_tokens": float(g["token_count"].median()),
            "mean_words": float(g["word_len"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    model_label = args.model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    device = "cpu"
    device_index = None
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = args.gpu_id if args.gpu_id < device_count else 0
        torch.cuda.set_device(device_index)
        device = f"cuda:{device_index}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / f"details_{model_label}_ppl.csv"
    summary_path = out_dir / f"summary_{model_label}_ppl.csv"
    meta_path = out_dir / f"meta_{model_label}_ppl.json"

    print(
        f"[config] model={args.model_name} device={device} requested_gpu={args.gpu_id} "
        f"batch_size={args.batch_size} max_length={args.max_length}"
    )
    docs = load_docs()
    print(f"[data] docs={len(docs)} attacks={list(DEFAULT_ATTACKS)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    details = []
    for i in tqdm(range(0, len(docs), args.batch_size), desc=f"{model_label} PPL", ncols=90):
        batch = docs[i:i + args.batch_size]
        nlls, ppls, token_counts = batch_ppl(
            model, tokenizer, [x["text"] for x in batch], device, args.max_length
        )
        for rec, nll, ppl, tok_count in zip(batch, nlls, ppls, token_counts):
            out = dict(rec)
            out["nll"] = float(nll)
            out["ppl"] = float(ppl)
            out["token_count"] = int(tok_count)
            out.pop("text")
            details.append(out)

    detail_df = pd.DataFrame(details)
    summary_df = summarize(details)
    detail_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": device,
            "attacks": DEFAULT_ATTACKS,
            "details_csv": str(details_path),
            "summary_csv": str(summary_path),
        }, f, indent=2)

    print(f"[save] {details_path}")
    print(f"[save] {summary_path}")
    print(f"[save] {meta_path}")
    print()
    print(summary_df[["attack", "num_docs", "mean_ppl", "median_ppl", "p95_ppl", "mean_nll", "mean_tokens"]].to_string(index=False))


if __name__ == "__main__":
    main()
