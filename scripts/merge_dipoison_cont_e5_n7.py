#!/usr/bin/env python3
"""Merge DiPoison cont/e5 N=4 CSVs into an N=7 artifact.

Output layout:
  doc0_seed: shared seed document
  doc1-doc3: cont generated documents
  doc4-doc6: e5 generated documents
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd


BASE_COLS = ["query", "target_answer", "correct_answer", "doc0_seed"]
GEN_COLS = ["doc1", "doc2", "doc3"]
OUT_COLS = BASE_COLS + ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]


def _load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = BASE_COLS + GEN_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[{label}] missing columns in {path}: {missing}")
    if df["query"].duplicated().any():
        dup = df.loc[df["query"].duplicated(), "query"].head(3).tolist()
        raise SystemExit(f"[{label}] duplicate query values; refusing query-based merge: {dup}")
    return df


def merge_one(cont_csv: str, e5_csv: str, output_csv: str, dataset: str) -> None:
    cont = _load(cont_csv, "cont")
    e5 = _load(e5_csv, "e5")

    if len(cont) != len(e5):
        raise SystemExit(f"[{dataset}] row count mismatch: cont={len(cont)} e5={len(e5)}")

    e5 = e5.set_index("query").reindex(cont["query"]).reset_index()
    if e5[BASE_COLS].isna().any().any():
        missing = e5.loc[e5["target_answer"].isna(), "query"].head(3).tolist()
        raise SystemExit(f"[{dataset}] e5 is missing queries from cont order: {missing}")

    for col in BASE_COLS:
        left = cont[col].astype(str).tolist()
        right = e5[col].astype(str).tolist()
        if left != right:
            for i, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    raise SystemExit(
                        f"[{dataset}] mismatch in {col} at row {i}: "
                        f"cont={a[:160]!r} e5={b[:160]!r}"
                    )

    merged = cont[BASE_COLS + GEN_COLS].copy()
    merged["doc4"] = e5["doc1"].values
    merged["doc5"] = e5["doc2"].values
    merged["doc6"] = e5["doc3"].values
    merged = merged[OUT_COLS]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)

    meta = {
        "dataset": dataset,
        "source_cont": cont_csv,
        "source_e5": e5_csv,
        "output": output_csv,
        "rows": int(len(merged)),
        "adv_per_query": 7,
        "doc_layout": {
            "doc0_seed": "shared seed document from cont/e5",
            "doc1_doc3": "cont generated documents",
            "doc4_doc6": "e5 generated documents",
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_csv.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] {dataset}: {merged.shape} -> {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cont_csv", required=True)
    p.add_argument("--e5_csv", required=True)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()
    merge_one(args.cont_csv, args.e5_csv, args.output_csv, args.dataset)


if __name__ == "__main__":
    main()
