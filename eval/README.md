# DiPoison Evaluation Guide

Measures attack success rate (ASR) and retrieval metrics (Precision/Recall/F1) for DiPoison
poison documents under No-Defense / RAGDefender defense, retrieving from the full corpus
(matching a real RAG deployment).

---

## Evaluation structure

```
eval/
├── main_dipoison_fullcorpus_ragdef.py   # Main NQ/MS MARCO evaluation script (Table 1, 5)
├── main_dipoison_fullcorpus_robustrag.py # RobustRAG evaluation
├── main_fullcorpus_ppl_filter.py         # Perplexity-filter defense (Figure 2)
├── multi_retriever_ragdef_eval.py        # 8-retriever comparison, NQ/MS MARCO (Table 3)
├── hotpotqa_merged_multihop_8ret_eval.py # 8-retriever comparison, HotpotQA (Table 3)
├── hotpotqa_multihop_ragdef_v2_eval.py   # HotpotQA multihop RAGDefender (Table 5)
├── gen_asr_eval.py                       # Cross-generator transfer (Table 4)
├── src/
│   ├── models/                 # LLM wrappers (Vicuna, Llama, GPT, Mistral, etc.)
│   ├── utils.py                # BEIR corpus loading, retrieval utilities
│   └── prompts.py              # RAG prompt templates
├── model_configs/              # LLM config files (JSON)
│   ├── vicuna7b_config.json
│   ├── llama3_8b_config.json
│   ├── mistral7b_config.json
│   └── qwen7b_config.json
└── README.md
```

---

## Requirements

See the top-level `README.md` / `requirements.txt` / `Dockerfile`.

---

## Evaluation pipeline

### 1. Retrieval

Injects N poison documents per query into the full corpus, then retrieves the top-k from
the entire corpus with the chosen retriever (no pre-filtering candidate pool).

**The 8 retrievers evaluated in the paper (Table 3)** — 2 surrogates (contriever, e5) +
6 unseen (ance, bge-base, mpnet, bm25, nomic-v1.5, contriever-msmarco):

| Retriever | Model | Similarity |
|--------|------|--------|
| contriever | `facebook/contriever` | dot product |
| e5-base | `intfloat/e5-base-v2` | cosine |
| ance | `sentence-transformers/msmarco-roberta-base-ance-firstp` | cosine |
| bge-base | `BAAI/bge-base-en-v1.5` | cosine |
| mpnet | `sentence-transformers/all-mpnet-base-v2` | cosine |
| bm25 | lexical (BM25) | BM25 score |
| nomic-v1.5 | `nomic-ai/nomic-embed-text-v1.5` | cosine |
| contriever-msmarco | `facebook/contriever-msmarco` | dot product |

The 8-retriever comparison is reproduced with `multi_retriever_ragdef_eval.py` (NQ/MS MARCO)
and `hotpotqa_merged_multihop_8ret_eval.py` (HotpotQA) — see "8-retriever comparison" below.

### 2. No-Defense (ND) evaluation

Uses all top-k documents as context to query the LLM -> ASR succeeds if the response
contains `target_answer`.

```
RAG Prompt:
  Contexts: [top-k documents]
  Query:    [question]
  Answer:

ASR_sub: target_answer ∈ response (substring match)
```

### 3. RAGDefender (RD) evaluation

Applies RAGDefender's two-stage defense before LLM evaluation. **NQ/MS MARCO (singlehop,
the default setting) and HotpotQA (multihop-specific) use different Stage 1 algorithms** —
multi-hop evidence documents can legitimately come from different source documents and
still be diverse, so applying the same clustering criterion as singlehop would misflag
benign documents.

**[Default] NQ / MS MARCO — clustering-based grouping (singlehop)**
- Script: `main_dipoison_fullcorpus_ragdef.py`
- Stage 1 — Agglomerative Clustering + TF-IDF: splits the top-k documents into 2 clusters
  (paraphrase-MiniLM-L6-v2 embeddings) -> estimates the poison cluster via TF-IDF frequency
  scores -> identifies `n_adv` removal candidates
- Stage 2 — Pairwise Frequency-Score Filter: accumulates similarity scores over the top
  `n_adv*(n_adv-1)/2` pairs -> flags the highest-scoring documents as poison -> removes
  them, passing only survivors to the LLM

**HotpotQA-specific — concentration-based grouping (multihop)**
- Script: `hotpotqa_multihop_ragdef_v2_eval.py`
- Stage 1 — instead of clustering, estimates `n_adv` by flagging documents whose
  mean/median pairwise cosine similarity to the other top-k documents is anomalously high
- Stage 2 — the rest of the pipeline is the same pairwise frequency-score filter as
  singlehop

**Unseen defense space (Table 1)** — `--defense_model` defaults to the matched setting
(`minilm`, paraphrase-MiniLM-L6-v2); use the `mpnet`/`ance`/`bge`/`gte` aliases to select an
unseen embedding space (any arbitrary SentenceTransformer ID also works):

```bash
# NQ, unseen defense space = MPNet
CUDA_VISIBLE_DEVICES=0 python main_dipoison_fullcorpus_ragdef.py \
    --dataset nq --retrieval_model contriever \
    --docs_csv data/generated/pd_eval100_cont_n4g8.csv \
    --defense_model mpnet --adv_per_query 4 --top_k 5 --gpu_id 0
```

### 4. Metrics

| Metric | Formula | Basis |
|------|------|------|
| Precision | poison_in_topk / top_k | No-Defense |
| Recall | poison_in_topk / adv_per_query | No-Defense |
| F1 | 2·P·R / (P+R) | No-Defense |
| ND-ASR | target ∈ LLM_response | No-Defense |
| RD-ASR | target ∈ LLM_response | After the RAGDefender defense |

---

## How to run

### Basic run (contriever + vicuna-7b, top_k=5, adv=4)

```bash
cd eval/
CUDA_VISIBLE_DEVICES=0 python main_dipoison_fullcorpus_ragdef.py \
    --dataset           nq \
    --retrieval_model   contriever \
    --docs_csv          ../data/generated/pd_eval100_cont_n4g8.csv \
    --adv_per_query     4 \
    --top_k             5 \
    --gpu_id            0
```

### 8-retriever comparison (Table 3 — reproduces top-5/top-10 together)

A single run sweeps all 8 retrievers x top-{5,10}, using only the combination reported in
the paper (contriever, e5, ance, bge-base, mpnet, bm25, nomic-v1.5, contriever-msmarco).

```bash
# NQ (using the merged N=7 poison set)
CUDA_VISIBLE_DEVICES=0 python eval/multi_retriever_ragdef_eval.py \
    --dataset   nq \
    --docs_csv  data/generated/pd_eval100_merged_n7.csv \
    --out_json  eval/results/multi_retriever_ragdef/pd_eval100_merged_n7_summary.json
    # --top_ks defaults to "5,10" (runs top-5/top-10 together)

# MS MARCO (the corpus may live on a different server, so pass --corpus_path directly)
CUDA_VISIBLE_DEVICES=0 python eval/multi_retriever_ragdef_eval.py \
    --dataset     msmarco \
    --corpus_path /path/to/msmarco/corpus.jsonl \
    --cache_dir   eval/clean_topn_cache/msmarco_merged_val100_top50 \
    --docs_csv    data/attackbaselines_pd/DiPoison/merged/msmarco_merged_dipoison.csv \
    --out_json    eval/results/msmarco_merged_8ret_fullcorpus/msmarco_merged_dipoison_summary.json

# HotpotQA
CUDA_VISIBLE_DEVICES=0 python eval/hotpotqa_merged_multihop_8ret_eval.py \
    --docs_csv data/attackbaselines_pd/DiPoison/merged/hotpotqa_merged_dipoison.csv
    # --top_ks defaults to "5,10"
```

### Evaluating E5 poison documents

```bash
CUDA_VISIBLE_DEVICES=0 python eval/main_dipoison_fullcorpus_ragdef.py \
    --dataset nq --retrieval_model e5-base \
    --docs_csv ../data/generated/pd_eval100_e5_n4g8.csv \
    --adv_per_query 4 --top_k 5 --gpu_id 0
```

### Evaluating with a different generator (Table 4 — cross-generator transfer)

```bash
CUDA_VISIBLE_DEVICES=0 python eval/gen_asr_eval.py \
    --docs_csv  ../data/generated/pd_eval100_cont_n4g8.csv \
    --input_csv ../data/nq100_validate.csv \
    --generator mistralai/Mistral-7B-Instruct-v0.3 \
    --gpu_id 0
```

> **GPT-4o-mini**: since this is a paid API model, a dedicated generator implementation
> isn't included separately. Copy `eval/model_configs/gpt4o_mini_config.example.json` and
> fill in your own OpenAI API key under `api_keys`; the provider follows
> `model_info.provider` in the config.

---

## Key arguments

| Argument | Description |
|------|------|
| `--retrieval_model` | Retriever name (see table above) |
| `--dataset` | `nq` / `msmarco` / `hotpotqa` |
| `--docs_csv` | Poison-document CSV (a file under `data/generated/`) |
| `--adv_per_query` | Number of poison documents injected per query (N) |
| `--top_k` | Number of documents the retriever returns |
| `--defense_model` | RAGDefender embedding space: `minilm` (matched) / `mpnet` / `ance` / `bge` / `gte` (unseen) |
| `--gpu_id` | CUDA device ID |

---

## Output

Key CSV columns:

| Column | Description |
|------|------|
| `poison_in_topk` | Number of poison documents among the top-k |
| `has_poison` | Whether any poison document is present in the top-k |
| `nd_response` | No-Defense LLM response |
| `nd_asr_sub` | ND ASR (substring match) |
| `rd_response` | LLM response after RAGDefender |
| `rd_asr_sub` | RD ASR (substring match) |
| `poison_survived_count` | Number of poison documents that survived RAGDefender |

---

## RAGDefender reference

- Paper: Kim, M.; Lee, H.; and Koo, H. 2025. *Rescuing the Unpoisoned: Efficient Defense
  Against Knowledge Corruption Attacks on RAG Systems.* Annual Computer Security
  Applications Conference (ACSAC).
- Defense model: `paraphrase-MiniLM-L6-v2` (SentenceTransformer)
- Singlehop (NQ/MS MARCO) defense logic: `main_dipoison_fullcorpus_ragdef.py`
- Multihop (HotpotQA) defense logic: `hotpotqa_multihop_ragdef_v2_eval.py`
