#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_path", default="/path/to/datasets/nq/corpus.jsonl")
    p.add_argument("--model_name", default="gpt2-xl")
    p.add_argument("--sample_size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_dir", default="/path/to/DisPo/eval/ppl_gpt2xl_clean_nq10k")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def sample_jsonl(corpus_path, sample_size, seed):
    rng = random.Random(seed)
    sample = []
    seen = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc="sample corpus", ncols=90)):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text", "")).strip()
            if not text:
                continue
            item = {
                "line_no": line_no,
                "doc_id": rec.get("_id", ""),
                "title": rec.get("title", ""),
                "text": text,
                "char_len": len(text),
                "word_len": len(text.split()),
            }
            seen += 1
            if len(sample) < sample_size:
                sample.append(item)
            else:
                j = rng.randrange(seen)
                if j < sample_size:
                    sample[j] = item
    if len(sample) < sample_size:
        raise ValueError(f"Only sampled {len(sample)} non-empty docs from {corpus_path}")
    return sample, seen


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

    valid = shifted_labels != -100
    token_counts = valid.sum(dim=1)
    nll = (tok_loss.sum(dim=1) / token_counts.clamp(min=1)).float()
    ppl = torch.exp(nll).clamp(max=1e9)
    return nll.float().cpu().numpy(), ppl.float().cpu().numpy(), token_counts.cpu().numpy()


def summarize(details):
    df = pd.DataFrame(details)
    ppl = df["ppl"].to_numpy(dtype=float)
    nll = df["nll"].to_numpy(dtype=float)
    return {
        "num_docs": int(len(df)),
        "mean_ppl": float(np.mean(ppl)),
        "median_ppl": float(np.median(ppl)),
        "std_ppl": float(np.std(ppl)),
        "p50_ppl": float(np.percentile(ppl, 50)),
        "p75_ppl": float(np.percentile(ppl, 75)),
        "p90_ppl": float(np.percentile(ppl, 90)),
        "p95_ppl": float(np.percentile(ppl, 95)),
        "p99_ppl": float(np.percentile(ppl, 99)),
        "mean_nll": float(np.mean(nll)),
        "median_nll": float(np.median(nll)),
        "mean_tokens": float(df["token_count"].mean()),
        "median_tokens": float(df["token_count"].median()),
        "mean_words": float(df["word_len"].mean()),
        "median_words": float(df["word_len"].median()),
        "mean_chars": float(df["char_len"].mean()),
        "median_chars": float(df["char_len"].median()),
    }


def main():
    args = parse_args()
    model_label = args.model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    device = "cpu"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = args.gpu_id if args.gpu_id < device_count else 0
        torch.cuda.set_device(device_index)
        device = f"cuda:{device_index}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / f"details_{model_label}_clean_nq_sample{args.sample_size}_seed{args.seed}.csv"
    summary_path = out_dir / f"summary_{model_label}_clean_nq_sample{args.sample_size}_seed{args.seed}.csv"
    meta_path = out_dir / f"meta_{model_label}_clean_nq_sample{args.sample_size}_seed{args.seed}.json"

    print(
        f"[config] model={args.model_name} device={device} requested_gpu={args.gpu_id} "
        f"sample_size={args.sample_size} seed={args.seed} batch_size={args.batch_size} "
        f"max_length={args.max_length}"
    )
    docs, eligible_docs = sample_jsonl(args.corpus_path, args.sample_size, args.seed)
    print(f"[data] sampled_docs={len(docs)} eligible_docs={eligible_docs} corpus={args.corpus_path}")

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
    for i in tqdm(range(0, len(docs), args.batch_size), desc=f"{model_label} clean PPL", ncols=90):
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
    summary = summarize(details)
    summary_df = pd.DataFrame([summary])
    detail_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "corpus_path": args.corpus_path,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "eligible_docs": eligible_docs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": device,
            "details_csv": str(details_path),
            "summary_csv": str(summary_path),
        }, f, indent=2)

    print(f"[save] {details_path}")
    print(f"[save] {summary_path}")
    print(f"[save] {meta_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
