# RobustRAG Mistral-7B ASR

This script evaluates an input malicious-document CSV on NQ single-hop full-corpus retrieval.

Fixed pipeline:

1. `ND-ASR`: `facebook/contriever` top-5 -> standard Mistral-7B RAG generation.
2. `RR-ASR`: `facebook/contriever` top-5 -> RobustRAG keyword aggregation with Mistral-7B.
3. `RD+RR-ASR`: `facebook/contriever` top-5 -> RAGDefender with `paraphrase-MiniLM-L6-v2` -> RobustRAG with Mistral-7B.

Run:

```bash
cd /path/to/DisPo
scripts/supplementary/robustrag/run_robustrag_mistral7b.sh data/generated/your_attack.csv 0
```

If your Python environment is not `.venv/bin/python`, set it explicitly:

```bash
PYTHON=/path/to/venv/bin/python scripts/supplementary/robustrag/run_robustrag_mistral7b.sh data/generated/your_attack.csv 0
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 python -m scripts.supplementary.robustrag.eval_mistral7b_asr \
  --docs_csv data/generated/your_attack.csv \
  --data_root /data/joonhyung \
  --adv_per_query 4 \
  --top_k 5 \
  --gpu_id 0
```

Input CSV columns:

- Required: `query`, `target_answer`
- Optional accuracy column: `correct_answer`
- Malicious docs: `doc0_seed`, `doc1`, `doc2`, ... or any `doc*` columns

Outputs:

- `scripts/supplementary/robustrag/results/summary.json`
- `scripts/supplementary/robustrag/results/details.csv`
- `scripts/supplementary/robustrag/results/run.log`
