# DiPoison: Dispersion-Penalized Poison Document Generation

A GRPO-based RAG poison-document generation framework.
Trains a Qwen2.5-1.5B generator with reinforcement learning using Contriever or E5 as the white-box retriever, then generates and evaluates poison documents against 100 evaluation queries.

---

## Quick start on a new server

### 1. Clone the repo

```bash
git clone <this-repository-url>
cd DiPoison
```

### 2. Set up the Python environment

```bash
python3.8 -m venv .venv
source .venv/bin/activate

# For training/inference (verified combination: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate sentence-transformers scikit-learn tqdm pandas

# Additional packages for evaluation (needed when using eval/)
pip install beir
pip install fschat==0.2.36    # Required for the LLM (Vicuna-7B) generation step — ImportError without it
```

### 3. End-to-end flow (Contriever, full-corpus evaluation is the default)

```bash
# Step 1: Training (GPU 0, ~8h)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input      data/nq_train_validate/nq_500_pd_7b.csv \
    --output_dir results/grpo_run1

# Step 2: Inference (after training completes)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_checkpoint.py \
    --checkpoint results/grpo_run1/final_model \
    --input      data/nq_train_validate/nq100_validate.csv \
    --output     results/grpo_run1/pd_eval100.csv

# Step 3: Evaluation — full-corpus is the default method
# (prerequisite: download the NQ/HotpotQA corpus — see the "Downloading the corpus directly" section)
cd eval/
CUDA_VISIBLE_DEVICES=0 python main_dipoison_fullcorpus_ragdef.py \
    --dataset         nq \
    --retrieval_model contriever \
    --docs_csv        ../results/grpo_run1/pd_eval100.csv \
    --adv_per_query   4 --top_k 5 --gpu_id 0
```

> Full-corpus is the default evaluation method (retrieves top-k from the entire corpus, matching a real RAG deployment). `main_dipoison_ragdef_beir.py`/`main_dipoison_extraval_ragdef.py` are a legacy method that only pits a small per-query candidate pool against each other; use them only for special purposes such as the 8-retriever comparison.
> See [eval/README.md](eval/README.md) for the full evaluation guide.

---

## Structure

```
DiPoison/
├── scripts/
│   ├── train_grpo_poison.py        # Training (Contriever + Vicuna-7B whitebox)
│   ├── train_grpo_poison_e5.py     # E5 training (E5-base + Vicuna-7B whitebox)
│   ├── infer_checkpoint.py         # Inference (LoRA checkpoint -> poison docs CSV)
│   ├── infer_e5_checkpoint.py      # E5 inference
│   └── apply_number_correction.py     # Post-hoc correction (can be run standalone)
├── data/
│   └── nq_train_validate/
│       ├── nq100_validate.csv         # 100 fixed evaluation queries (disjoint from training)
│       ├── nq_train100.csv            # 100 training queries (Supp Fig 2 training-size sensitivity)
│       ├── nq_train300.csv            # 300 training queries (Supp Fig 2 training-size sensitivity)
│       └── nq_500_pd_7b.csv           # 500 training queries (default)
└── results/                           # Training/inference outputs (gitignored)
```

---

## Dataset columns

| File | Columns | Description |
|------|------|------|
| `nq_train100.csv` / `nq_train300.csv` / `nq_500_pd_7b.csv` | `query`, `target_answer`, `seed_doc` | Training. `nq_train100`/`nq_train300` are the first 100/300 queries of `nq_500_pd_7b` (Supp Fig 2's training-query-count sensitivity: 100/300/500) |
| `nq100_validate.csv` | `query`, `target_answer`, `seed_doc` | Evaluation: 100 queries disjoint from training |

---

## Requirements

- Python 3.8 (verified version — other versions untested)
- PyTorch 2.4.1 (CUDA 12.1) — `torch==2.4.1+cu121`, `cudnn 90100`
- transformers 4.46.3, sentence-transformers 3.2.1
- fschat 0.2.36 (required for the Vicuna-7B generation step, used via `import fastchat`)
- GPU: 24GB+ VRAM recommended (A100/H100); full-corpus eval additionally needs ~8GB disk
  (NQ) / ~16GB disk (HotpotQA) per retriever for the corpus embedding cache

```bash
pip install transformers peft accelerate sentence-transformers scikit-learn tqdm pandas
pip install fschat==0.2.36
```

Model downloads (automatic from Hugging Face on first run):
- Generator: `Qwen/Qwen2.5-1.5B-Instruct`
- Surrogate LLM: `lmsys/vicuna-7b-v1.3`
- Retriever (default): `facebook/contriever`
- Retriever (E5): `intfloat/e5-base-v2`
- Defense filter: `paraphrase-MiniLM-L6-v2`
- Additional, for the full-corpus 8-retriever comparison (Table 3, Supp Table 10/11): `contriever-msmarco`,
  `sentence-transformers/msmarco-roberta-base-ance-firstp` (ance), `BAAI/bge-base-en-v1.5` (bge-base),
  `sentence-transformers/all-mpnet-base-v2` (mpnet), BM25 (lexical), `nomic-ai/nomic-embed-text-v1.5` (nomic-v1.5)
- Additional, for the unseen defense space experiment (Table 1, Supp Table 5): `sentence-transformers/all-mpnet-base-v2` (mpnet),
  `sentence-transformers/msmarco-roberta-base-ance-firstp` (ance), `BAAI/bge-base-en-v1.5` (bge-base), `thenlper/gte-base` (gte)
-------
How to download the corpus directly for defense-inclusive evaluation (full-corpus, the default method)

**NQ** (2.68M passages, ~1.5GB)
```bash
mkdir -p /path/to/datasets/nq && cd /path/to/datasets/nq
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip nq.zip
mv nq/corpus.jsonl nq/queries.jsonl nq/qrels .
rm -rf nq nq.zip
```

**HotpotQA** (5.23M passages, ~2.2GB)
```bash
mkdir -p /path/to/datasets/hotpotqa && cd /path/to/datasets/hotpotqa
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
unzip hotpotqa.zip
mv hotpotqa/corpus.jsonl hotpotqa/queries.jsonl hotpotqa/qrels .
rm -rf hotpotqa hotpotqa.zip
```

> `eval/main_dipoison_fullcorpus_ragdef.py`'s `_DS_CFG` defaults to `$DIPOISON_DATA_ROOT/datasets/{nq,hotpotqa}/` (see `--corpus_path`/`DIPOISON_DATA_ROOT` to point elsewhere), so the corpus must live at that path unless overridden.

---

## 1. Training

### Default (Contriever + Vicuna-7B)

```bash
# Default run (GPU 0, 500 queries, epoch=3, G=8, N=4)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input      data/nq_train_validate/nq_500_pd_7b.csv \
    --output_dir results/grpo_run1
```

### Training-query-count sensitivity (Supp Fig 2 — 100 / 300 / 500)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input      data/nq_train_validate/nq_train100.csv \
    --output_dir results/grpo_train100_run1

CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input      data/nq_train_validate/nq_train300.csv \
    --output_dir results/grpo_train300_run1
```

### E5 (E5-base + Vicuna-7B)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison_e5.py \
    --input      data/nq_train_validate/nq_500_pd_7b.csv \
    --output_dir results/grpo_e5_run1
```

### Key training arguments

| Argument | Default | Description |
|------|--------|------|
| `--input` | `data/nq_train_validate/nq_500_pd_7b.csv` | Training query CSV |
| `--output_dir` | `results/grpo_run1` | Checkpoint output path |
| `--num_epochs` | `3` | Number of training epochs |
| `--group_size` | `8` | GRPO group size **(G)** — candidates generated per query |
| `--num_adv_docs` | `3` | Final number of poison documents per query **(N)** — excludes doc0_seed, N+1 total |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lr` | `1e-5` | Learning rate |
| `--gpu_id` | `0` | CUDA device ID |
| `--embed_device` | `cuda` | Device for the embedding model (use cpu if VRAM is tight) |
| `--limit` | `None` | Limit the number of queries (for debugging, e.g. `--limit 10`) |

---

## 2. Inference

Generates poison documents for the 100 evaluation queries from a trained LoRA checkpoint.
Per query: `doc0_seed` (seed, unchanged) + `doc1`~`doc3` (generated) = 4 documents total.

### Default

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_checkpoint.py \
    --checkpoint    results/grpo_run1/final_model \
    --input         data/nq_train_validate/nq100_validate.csv \
    --output        results/grpo_run1/pd_eval100.csv \
    --group_size    8 \
    --gen_batch_size 8
```

### E5

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_e5_checkpoint.py \
    --checkpoint    results/grpo_e5_run1/final_model \
    --input         data/nq_train_validate/nq100_validate.csv \
    --output        results/grpo_e5_run1/pd_eval100_e5.csv \
    --group_size    8 \
    --gen_batch_size 8
```

### Variable N

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_n.py \
    --checkpoint    results/grpo_run1/final_model \
    --input         data/nq_train_validate/nq100_validate.csv \
    --output        results/grpo_run1/pd_eval100_n6.csv \
    --N 6 \
    --group_size    8 \
    --gen_batch_size 8
```

### Inference arguments

| Argument | Default | Description |
|------|--------|------|
| `--checkpoint` | `results/.../final_model` | LoRA checkpoint path |
| `--input` | `data/nq_train_validate/nq100_validate.csv` | Evaluation query CSV |
| `--output` | `results/.../pd_eval100.csv` | Output CSV path |
| `--group_size` | `8` | Number of candidates generated **(G)**; the best 1 is selected |
| `--num_adv_docs` | `3` | Number of poison documents to generate per query **(N)** |
| `--N` | `None` | Same meaning as `--num_adv_docs`, takes priority if set (`infer_n.py` only) |
| `--gen_batch_size` | `1` | How many of the G candidates to generate at once. **Fastest when set equal to `--group_size`** |
| `--embed_device` | `cuda` | Device for the embedding model |
| `--gpu_id` | `0` | CUDA device ID |

> Post-hoc correction (fixing split numbers + overwriting the Answer field) is applied automatically during inference.

---

## 3. Post-hoc correction (standalone)

To apply correction alone to an already-generated CSV:

```bash
python scripts/apply_number_correction.py \
    --input  results/grpo_run1/pd_eval100.csv \
    --output results/grpo_run1/pd_eval100_corrected.csv
```

---

## Hyperparameter details

### Shared (default / E5)

| Parameter | Value | Description |
|----------|-----|------|
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | Poison-document generation model |
| Surrogate LLM | `lmsys/vicuna-7b-v1.3` | Used to compute r_generation + r_ppl |
| Defense filter | `paraphrase-MiniLM-L6-v2` | Used for r_disp_embed + the RAGDefender defense |
| GROUP_SIZE (G) | 8 | Candidates generated per query (changeable via `--group_size`) |
| N (adv docs/query) | 3 | Final number of selected poison documents (changeable via `--num_adv_docs`, N+1 total) |
| MIN_NEW_TOKENS | 80 | Minimum generation tokens |
| MAX_NEW_TOKENS | 160 | Maximum generation tokens |
| TEMPERATURE | 0.85 | Sampling temperature |
| TOP_P | 0.92 | Nucleus sampling |
| REPETITION_PENALTY | 1.1 | Repetition suppression |
| NO_REPEAT_NGRAM_SIZE | 4 | Blocks repeated n-grams |
| LR | 1e-5 | Adam learning rate |
| GRAD_CLIP | 0.5 | Gradient-norm clipping |
| ADV_CLIP | 2.0 | GRPO advantage clipping |
| LORA_R | 16 | LoRA rank |
| LORA_ALPHA | 32 | LoRA alpha |
| LORA_DROPOUT | 0.05 | LoRA dropout |
| LAMBDA_KENDALL | 0.30 | Kendall rank-loss weight |
| MMR_LAMBDA | 0.60 | MMR diversity weight |

### Reward function (5 components)

| Reward | Formula | Description |
|------|------|------|
| r_retrieval | (dot − 0.40) / 1.10 ∈ [0,1] | Retriever similarity (higher favors entering the top-k) |
| r_disp_embed | 1 − MiniLM inter-cosine ∈ [0,1] | Semantic diversity among generated documents (evades RAGDefender Stage 2) |
| r_tfidf_disp | 1 − TF-IDF inter-sim ∈ [0,1] | TF-IDF diversity (evades RAGDefender Stage 1) |
| r_generation | P(target \| context+query+Answer:) | Probability that Vicuna-7B outputs the target answer |
| r_ppl | sigmoid(−log(PPL/20)) | Document fluency (lower perplexity) |

### Differences: default vs E5

| Item | Default | E5 |
|------|----|--------|
| White-box retriever | `facebook/contriever` (dot product) | `intfloat/e5-base-v2` (cosine) |
| r_retrieval baseline | (dot − 0.40) / 1.10 | (cos − 0.70) / 0.25 |
| Query prefix | none | adds `"query: "` / `"passage: "` |

---

## Output CSV columns

| Column | Description |
|------|------|
| `query` | Evaluation query |
| `target_answer` | Target (injected) incorrect answer |
| `doc0_seed` | Original seed document (unchanged) |
| `doc1` ~ `doc3` | Generated poison documents (correction applied) |
