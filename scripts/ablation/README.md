# Ablation (Table 6 — multi-objective reward)

5개 보상 함수(r_retrieval, r_disp_embed, r_tfidf_disp, r_generation, r_ppl)를 하나씩
제거하며 기여도를 측정하는 ablation 실험 코드입니다. 베이스라인(`scripts/train_grpo_poison.py`)과
하이퍼파라미터·보상 구조가 동일하며, `--ablation`으로 제외할 보상 1개만 지정합니다.

```
scripts/ablation/
├── train/
│   └── train_grpo_poison_abl.py   # --ablation {none,no_retrieval,no_disp_embed,no_tfidf_disp,no_generation,no_ppl}
└── inference/
    └── infer_abl_checkpoint.py    # 체크포인트에서 100개 평가 쿼리에 대해 poison docs 생성
```

`infer_abl_checkpoint.py`는 `train_grpo_poison_abl.py`를 모듈로 import합니다
(UncertaintyWeighter 등 학습 코드를 그대로 재사용). 두 파일의 상대 위치(`train/`, `inference/`가
형제 폴더)를 그대로 유지해야 합니다.

## 실행

```bash
# 1) 훈련 — 예: r_retrieval 제외
CUDA_VISIBLE_DEVICES=1 python scripts/ablation/train/train_grpo_poison_abl.py \
    --ablation no_retrieval \
    --output_dir results/grpo_whitebox_abl_no_ret_run1 \
    --num_epochs 1 --group_size 8 --lora_r 16 --gpu_id 1

# 2) Inference — 반드시 훈련 때와 동일한 --ablation 값 지정
CUDA_VISIBLE_DEVICES=1 python scripts/ablation/inference/infer_abl_checkpoint.py \
    --ablation no_retrieval \
    --checkpoint results/grpo_whitebox_abl_no_ret_run1/final_model \
    --output    results/grpo_whitebox_abl_no_ret_run1/pd_eval100_abl_no_ret.csv \
    --group_size 8 --gen_batch_size 8
```

## `--ablation` ↔ Table 6 대응

| `--ablation` | 제외하는 보상 | Table 6 행 |
|---|---|---|
| `no_retrieval`  | Eq.1 r_retrieval        | w/o Retrieval |
| `no_disp_embed` | Eq.2 r_disp_embed(r_emb) | w/o Semantic Dispersion |
| `no_tfidf_disp` | Eq.3 r_tfidf_disp(r_lex) | w/o Lexical Dispersion |
| `no_generation` | Eq.4 r_generation(r_pay) | w/o Payload |
| `no_ppl`        | Eq.5 r_ppl(r_flu)        | w/o Fluency |

이후 생성된 poison docs는 일반 평가 스크립트(`eval/main_dipoison_fullcorpus_ragdef.py` 등)로
PRR/ASR/ASR_def/PPL을 측정하면 Table 6의 각 행이 재현됩니다.
