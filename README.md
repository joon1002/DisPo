# DiPoison — Supplementary Code

Reference implementation for DiPoison: a GRPO-based RAG poison-document generator that
jointly optimizes retrieval, semantic/lexical dispersion, payload, and fluency rewards,
trained with a Qwen2.5-1.5B policy against Contriever or E5 as the surrogate
retriever.

This package contains the code needed to reproduce the paper's tables and figures. It
does not include model weights, corpora, or run outputs — see "Environment setup" below
for what each experiment downloads on first run.

---

## Environment setup

### Option A — Docker (recommended)

```bash
docker build -t dipoison .
docker run --gpus all -it \
    -e DIPOISON_DATA_ROOT=/data \
    -v /path/to/your/datasets:/data \
    dipoison bash
```

### Option B — pip

```bash
python3.8 -m venv .venv
source .venv/bin/activate

pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Requirements: Python 3.8, PyTorch 2.4.1 (CUDA 12.1), transformers 4.46.3,
sentence-transformers 3.2.1, fschat 0.2.36. GPU: 24GB+ VRAM recommended (A100/H100);
full-corpus evaluation additionally needs ~8GB disk (NQ) / ~16GB disk (HotpotQA) per
retriever for the corpus embedding cache.

### Downloading the corpora (required for full-corpus evaluation)

```bash
mkdir -p $DIPOISON_DATA_ROOT/datasets/nq && cd $DIPOISON_DATA_ROOT/datasets/nq
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip nq.zip && mv nq/corpus.jsonl nq/queries.jsonl nq/qrels . && rm -rf nq nq.zip

mkdir -p $DIPOISON_DATA_ROOT/datasets/hotpotqa && cd $DIPOISON_DATA_ROOT/datasets/hotpotqa
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
unzip hotpotqa.zip && mv hotpotqa/corpus.jsonl hotpotqa/queries.jsonl hotpotqa/qrels . && rm -rf hotpotqa hotpotqa.zip
```

MS MARCO's corpus should be downloaded the same way (BEIR `msmarco.zip`) and its path
passed explicitly via `--corpus_path`, since it is evaluation-only and not assumed to
live under `DIPOISON_DATA_ROOT`.

Models are downloaded automatically from Hugging Face on first run (Qwen2.5-1.5B-Instruct,
vicuna-7b-v1.3, facebook/contriever, and the other retrievers/rerankers listed per
experiment below).

---

## Repository layout

```
DiPoison/
├── scripts/            # Training, inference, and standalone data-prep scripts
│   ├── ablation/        # Table 6 — per-reward ablation (train/ + inference/)
│   └── supplementary/    # Supplementary-only evaluations (reranking/, robustrag/)
├── eval/               # Attack evaluation under each defense
├── data/                # Training/evaluation query CSVs and generated poison-doc CSVs
└── docs/                # Data-provenance notes for the tracked artifact CSVs
```

---

## Experiment-to-code mapping

| Paper item | Script(s) |
|---|---|
| Training (§3.2) | `scripts/train_grpo_poison.py` (Contriever surrogate), `scripts/train_grpo_poison_e5.py` (E5 surrogate) |
| Poison-document inference | `scripts/infer_checkpoint.py`, `scripts/infer_e5_checkpoint.py`, `scripts/infer_n.py` (variable N, §4.4/Fig.3) |
| Table 1 — RAGDefender, matched + unseen defense spaces | `eval/main_dipoison_fullcorpus_ragdef.py --defense_model {minilm,mpnet,ance,bge,gte}` |
| Figure 2 — Perplexity-filter defense | `eval/main_fullcorpus_ppl_filter.py` |
| Table 2 — RobustRAG (Mistral-7B) | `scripts/supplementary/robustrag/` |
| Supplementary §3 — Reranking defense | `scripts/supplementary/reranking/` |
| Table 3 — Cross-retriever transfer | `scripts/merge_dipoison_cont_e5_n7.py` (merge Contriever+E5 poison sets), `eval/multi_retriever_ragdef_eval.py` (NQ/MS MARCO), `eval/hotpotqa_merged_multihop_8ret_eval.py` (HotpotQA) |
| Table 4 — Cross-generator transfer | `eval/gen_asr_eval.py` |
| Table 5 — Cross-dataset transfer | `eval/main_dipoison_fullcorpus_ragdef.py --dataset {nq,msmarco}`, `eval/hotpotqa_ragdef_eval.py`, `eval/hotpotqa_multihop_ragdef_v2_eval.py` (HotpotQA multihop defense) |
| §4.4 — Training-query-count ablation | Train at each query count with `scripts/train_grpo_poison.py --input <100/300/500-query CSV>`, then evaluate each with `eval/main_dipoison_fullcorpus_ragdef.py` (no separate plotting code is included) |
| §4.4 — Untargeted-query utility / collateral damage | `eval/clean_acc_eval.py`, `eval/clean_acc_fullcorpus300.py`, `eval/collateral_damage_eval.py`, `eval/hotpotqa_clean_acc_200.py`, `eval/hotpotqa_collateral_acc_eval.py` |
| Table 6 — Multi-objective reward ablation | `scripts/ablation/` (see `scripts/ablation/README.md` for the `--ablation` <-> table-row mapping), PPL column via `eval/measure_objective_ablation_ppl_gpt2xl.py` |

See `eval/README.md` for the full evaluation guide (retriever/defense options, argument
reference, output format) and `scripts/ablation/README.md` / `scripts/supplementary/README.md`
for those two experiments specifically.

---

## Dataset columns

| File | Columns | Description |
|------|------|------|
| `nq_train100.csv` / `nq_train300.csv` / `nq_train500.csv` | `query`, `target_answer`, `seed_doc` | Training-query-count sensitivity (100/300/500). `nq_train100` is an independently sampled 100-query set (disjoint from `nq_train500`); `nq_train300` is the first 300 queries of `nq_train500` |
| `nq100_validate.csv` | `query`, `target_answer`, `seed_doc` | Evaluation: 100 queries disjoint from training |

---

## 1. Training

```bash
# Default (Contriever + Vicuna-7B), GPU 0, 500 queries, epoch=3, G=8, N=4
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input      data/nq_train_validate/nq_train500.csv \
    --output_dir results/grpo_run1

# E5 surrogate
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison_e5.py \
    --input      data/nq_train_validate/nq_train500.csv \
    --output_dir results/grpo_e5_run1

# Training-query-count sensitivity (100 / 300, in addition to the default 500)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input data/nq_train_validate/nq_train100.csv --output_dir results/grpo_train100_run1
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison.py \
    --input data/nq_train_validate/nq_train300.csv --output_dir results/grpo_train300_run1
```

### Key training arguments

| Argument | Default | Description |
|------|--------|------|
| `--input` | `data/nq_train_validate/nq_train500.csv` | Training query CSV |
| `--output_dir` | `results/grpo_run1` | Checkpoint output path |
| `--num_epochs` | `3` | Number of training epochs |
| `--group_size` | `8` | GRPO group size **(G)** — candidates generated per query |
| `--num_adv_docs` | `3` | Final number of poison documents per query **(N)** — excludes doc0_seed, N+1 total |
| `--lora_r` / `--lora_alpha` | `16` / `32` | LoRA rank / alpha |
| `--lr` | `1e-5` | Learning rate |
| `--gpu_id` | `0` | CUDA device ID |
| `--embed_device` | `cuda` | Device for the embedding model (use cpu if VRAM is tight) |
| `--limit` | `None` | Limit the number of queries (for debugging, e.g. `--limit 10`) |

---

## 2. Inference

Generates poison documents for the 100 evaluation queries from a trained LoRA checkpoint.
Per query: `doc0_seed` (seed, unchanged) + `doc1`~`doc3` (generated) = 4 documents total.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_checkpoint.py \
    --checkpoint    results/grpo_run1/final_model \
    --input         data/nq_train_validate/nq100_validate.csv \
    --output        results/grpo_run1/pd_eval100.csv \
    --group_size    8 \
    --gen_batch_size 8

# Variable N (poison budget), e.g. N=6
CUDA_VISIBLE_DEVICES=0 python scripts/infer_n.py \
    --checkpoint results/grpo_run1/final_model \
    --input      data/nq_train_validate/nq100_validate.csv \
    --output     results/grpo_run1/pd_eval100_n6.csv \
    --N 6 --group_size 8 --gen_batch_size 8
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

Post-hoc correction (fixing split numbers + overwriting the Answer field) is applied
automatically during inference; to apply it standalone to an existing CSV:

```bash
python scripts/apply_number_correction.py --input results/grpo_run1/pd_eval100.csv \
    --output results/grpo_run1/pd_eval100_corrected.csv
```

---

## 3. Evaluation

```bash
cd eval/
CUDA_VISIBLE_DEVICES=0 python main_dipoison_fullcorpus_ragdef.py \
    --dataset         nq \
    --retrieval_model contriever \
    --docs_csv        ../results/grpo_run1/pd_eval100.csv \
    --adv_per_query   4 --top_k 5 --gpu_id 0
```

See `eval/README.md` for the full evaluation guide (all retrievers/defenses, the
8-retriever comparison, cross-generator evaluation, output format).

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
| MIN_NEW_TOKENS / MAX_NEW_TOKENS | 80 / 160 | Generation length bounds |
| TEMPERATURE / TOP_P | 0.85 / 0.92 | Sampling |
| REPETITION_PENALTY | 1.1 | Repetition suppression |
| NO_REPEAT_NGRAM_SIZE | 4 | Blocks repeated n-grams |
| LR / GRAD_CLIP | 1e-5 / 0.5 | Adam learning rate / gradient-norm clipping |
| ADV_CLIP | 2.0 | GRPO advantage clipping |
| LORA_R / LORA_ALPHA / LORA_DROPOUT | 16 / 32 / 0.05 | LoRA config |
| LAMBDA_KENDALL | 0.30 | Kendall rank-loss weight (paper's λ_rank) |

### Reward function (5 components, paper Eq.1-5)

| Reward | Formula | Description |
|------|------|------|
| r_retrieval | (dot − 0.40) / 1.10 ∈ [0,1] | Retriever similarity (higher favors entering the top-k) |
| r_disp_embed | 1 − MiniLM inter-cosine ∈ [0,1] | Semantic dispersion among generated documents |
| r_tfidf_disp | 1 − TF-IDF inter-sim ∈ [0,1] | Lexical dispersion |
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
