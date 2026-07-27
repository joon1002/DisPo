# TinyBERT-L2 Reranking ASR

This script evaluates an input malicious-document CSV on NQ single-hop full-corpus retrieval.

Fixed pipeline:

1. `facebook/contriever` retrieves top-50 from the NQ corpus plus injected malicious docs.
2. `cross-encoder/ms-marco-TinyBERT-L-2-v2` reranks the top-50 and keeps top-5.
3. `ND-ASR`: top-5 goes directly to the Vicuna-7B generator.
4. `RD-ASR`: top-5 goes through RAGDefender with `paraphrase-MiniLM-L6-v2`, then to the same Vicuna-7B generator.

Run:

```bash
cd /path/to/DisPo
scripts/supplementary/reranking/run_reranking_tinybert_l2.sh data/generated/your_attack.csv 0
```

If your Python environment is not `.venv/bin/python`, set it explicitly:

```bash
PYTHON=/path/to/venv/bin/python scripts/supplementary/reranking/run_reranking_tinybert_l2.sh data/generated/your_attack.csv 0
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 python -m scripts.supplementary.reranking.eval_tinybert_l2_asr \
  --docs_csv data/generated/your_attack.csv \
  --data_root /data/joonhyung \
  --adv_per_query 4 \
  --ret_top_n 50 \
  --top_k 5 \
  --gpu_id 0
```

Input CSV columns:

- Required: `query`, `target_answer`
- Optional accuracy column: `correct_answer`
- Malicious docs: `doc0_seed`, `doc1`, `doc2`, ... or any `doc*` columns

Outputs:

- `scripts/supplementary/reranking/results/summary.json`
- `scripts/supplementary/reranking/results/details.csv`
- `scripts/supplementary/reranking/results/run.log`
