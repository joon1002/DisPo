# Models and Data Required

The supplementary scripts do not commit model weights or the NQ corpus. A new server must have the following available.

## Python packages

Install the project dependencies first:

```bash
pip install torch transformers accelerate sentence-transformers scikit-learn tqdm pandas numpy nltk
```

## Hugging Face models

- Retriever: `facebook/contriever`
- Reranker: `cross-encoder/ms-marco-TinyBERT-L-2-v2`
- RAGDefender embedding model: `paraphrase-MiniLM-L6-v2`
- RobustRAG / generator LLM: `mistralai/Mistral-7B-Instruct-v0.3`

If Mistral is already stored locally, pass it with:

```bash
python -m scripts.supplementary.robustrag.eval_mistral7b_asr \
  --docs_csv data/generated/your_attack.csv \
  --mistral_model /path/to/Mistral-7B-Instruct-v0.3
```

You can also use an existing `eval/src` model config:

```bash
python -m scripts.supplementary.robustrag.eval_mistral7b_asr \
  --docs_csv data/generated/your_attack.csv \
  --model_config_path eval/model_configs/mistral7b_config.json
```

## NQ data files

By default, `--data_root /data/joonhyung` is expected. The scripts read:

- `/data/joonhyung/datasets/nq/corpus.jsonl`
- `/data/joonhyung/datasets/nq/contriever_embs_fullcorpus.pt`

Set `DISPO_DATA_ROOT` or pass `--data_root` if the other server stores them elsewhere:

```bash
DISPO_DATA_ROOT=/path/to/data_root scripts/supplementary/robustrag/run_robustrag_mistral7b.sh data/generated/your_attack.csv 0
```

`contriever_embs_fullcorpus.pt` is the precomputed Contriever embedding cache for the full NQ corpus. Without it, the current supplementary scripts will not run.
