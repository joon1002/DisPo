# Ablation (Table 6 — multi-objective reward)

Code for the ablation study that removes each of the 5 reward components (r_retrieval,
r_disp_embed, r_tfidf_disp, r_generation, r_ppl) one at a time to measure its contribution.
Hyperparameters and reward structure are identical to the baseline (`scripts/train_grpo_poison.py`);
`--ablation` specifies the single reward to exclude.

```
scripts/ablation/
├── train/
│   └── train_grpo_poison_abl.py   # --ablation {none,no_retrieval,no_disp_embed,no_tfidf_disp,no_generation,no_ppl}
└── inference/
    └── infer_abl_checkpoint.py    # Generates poison docs for the 100 evaluation queries from a checkpoint
```

`infer_abl_checkpoint.py` imports `train_grpo_poison_abl.py` as a module (reusing its training
code, e.g. UncertaintyWeighter, as-is). The relative location of the two files (`train/` and
`inference/` as sibling folders) must be preserved.

## Running it

```bash
# 1) Training — e.g. excluding r_retrieval
CUDA_VISIBLE_DEVICES=1 python scripts/ablation/train/train_grpo_poison_abl.py \
    --ablation no_retrieval \
    --output_dir results/grpo_whitebox_abl_no_ret_run1 \
    --num_epochs 1 --group_size 8 --lora_r 16 --gpu_id 1

# 2) Inference — must use the same --ablation value as training
CUDA_VISIBLE_DEVICES=1 python scripts/ablation/inference/infer_abl_checkpoint.py \
    --ablation no_retrieval \
    --checkpoint results/grpo_whitebox_abl_no_ret_run1/final_model \
    --output    results/grpo_whitebox_abl_no_ret_run1/pd_eval100_abl_no_ret.csv \
    --group_size 8 --gen_batch_size 8
```

## `--ablation` <-> Table 6 mapping

| `--ablation` | Excluded reward | Table 6 row |
|---|---|---|
| `no_retrieval`  | Eq.1 r_retrieval        | w/o Retrieval |
| `no_disp_embed` | Eq.2 r_disp_embed(r_emb) | w/o Semantic Dispersion |
| `no_tfidf_disp` | Eq.3 r_tfidf_disp(r_lex) | w/o Lexical Dispersion |
| `no_generation` | Eq.4 r_generation(r_pay) | w/o Payload |
| `no_ppl`        | Eq.5 r_ppl(r_flu)        | w/o Fluency |

Measuring PRR/ASR/ASR_def/PPL on the resulting poison docs with the regular evaluation scripts
(e.g. `eval/main_dipoison_fullcorpus_ragdef.py`) reproduces each row of Table 6.
