# DisPo: Dispersion-Penalized Poison Document Generation

GRPO 기반 RAG 악성 문서 생성 프레임워크.  
Contriever(v7) 또는 E5(v7-e5)를 화이트박스 검색기로 사용하여 Qwen2.5-1.5B 생성기를 강화학습으로 훈련하고, 100개 평가 쿼리에 대해 악성 문서를 생성 및 평가합니다.

---

## 구조

```
DisPo/
├── scripts/
│   ├── train_grpo_poison_v7.py        # v7 훈련 (Contriever + Vicuna-7B whitebox)
│   ├── train_grpo_poison_v7_e5.py     # v7-e5 훈련 (E5-base + Vicuna-7B whitebox)
│   ├── infer_v7_checkpoint.py         # v7 inference (LoRA 체크포인트 → poison docs CSV)
│   ├── infer_v7_e5_checkpoint.py      # v7-e5 inference
│   └── apply_number_correction.py     # Post-hoc 교정 (독립 실행 가능)
├── data/
│   ├── nq100_validate.csv             # 평가용 100 쿼리 (고정)
│   ├── nq_500_pd_7b.csv               # 훈련용 500 쿼리 (기본)
│   └── nq_800_train.csv               # 훈련용 800 쿼리
└── results/                           # 훈련/inference 결과 저장 (gitignore)
```

---

## 데이터셋 컬럼

| 파일 | 컬럼 | 설명 |
|------|------|------|
| `nq_500_pd_7b.csv` / `nq_800_train.csv` | `query`, `target_answer`, `seed_doc` | 훈련용: 쿼리, 정답, 시드 문서 |
| `nq100_validate.csv` | `query`, `target_answer`, `seed_doc` | 평가용: 훈련과 겹치지 않는 100개 |

---

## 요구 환경

- Python 3.10+
- PyTorch 2.x (CUDA 11.8+)
- GPU: 24GB+ VRAM 권장 (A100/H100)

```bash
pip install transformers peft accelerate sentence-transformers scikit-learn tqdm pandas
```

모델 다운로드 (최초 실행 시 HuggingFace에서 자동):
- Generator: `Qwen/Qwen2.5-1.5B-Instruct`
- Surrogate LLM: `lmsys/vicuna-7b-v1.3`
- Retriever(v7): `facebook/contriever`
- Retriever(v7-e5): `intfloat/e5-base-v2`
- Defense filter: `paraphrase-MiniLM-L6-v2`

---

## 1. 훈련

### v7 (Contriever + Vicuna-7B)

```bash
# 기본 실행 (GPU 0, 500 쿼리, epoch=3, G=8, N=4)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison_v7.py \
    --input      data/nq_500_pd_7b.csv \
    --output_dir results/grpo_v7_run1

# 800 쿼리, epoch=3
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison_v7.py \
    --input      data/nq_800_train.csv \
    --output_dir results/grpo_v7_800q_run1
```

### v7-e5 (E5-base + Vicuna-7B)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison_v7_e5.py \
    --input      data/nq_500_pd_7b.csv \
    --output_dir results/grpo_v7_e5_run1
```

### 주요 훈련 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` | `data/nq_500_pd_7b.csv` | 훈련 쿼리 CSV |
| `--output_dir` | `results/grpo_v7_run1` | 체크포인트 저장 경로 |
| `--num_epochs` | `3` | 훈련 epoch 수 |
| `--group_size` | `8` | GRPO 그룹 크기 (G) — 쿼리당 후보 수 |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lr` | `1e-5` | Learning rate |
| `--gpu_id` | `0` | CUDA 디바이스 ID |
| `--embed_device` | `cpu` | 임베딩 모델 디바이스 (VRAM 절약 시 cpu) |
| `--limit` | `None` | 쿼리 수 제한 (디버깅용, 예: `--limit 10`) |

---

## 2. Inference

훈련된 LoRA 체크포인트에서 100개 평가 쿼리에 대해 악성 문서 생성.  
쿼리당 `doc0_seed`(시드 그대로) + `doc1~doc3`(생성 문서) = 4개.

### v7

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_checkpoint.py \
    --checkpoint results/grpo_v7_run1/final_model \
    --input      data/nq100_validate.csv \
    --output     results/grpo_v7_run1/pd_eval100_v7.csv \
    --group_size 8
```

### v7-e5

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_e5_checkpoint.py \
    --checkpoint results/grpo_v7_e5_run1/final_model \
    --input      data/nq100_validate.csv \
    --output     results/grpo_v7_e5_run1/pd_eval100_v7_e5.csv \
    --group_size 8
```

### Inference 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | `results/.../final_model` | LoRA 체크포인트 경로 |
| `--input` | `data/nq100_validate.csv` | 평가 쿼리 CSV |
| `--output` | `results/.../pd_eval100.csv` | 출력 CSV 경로 |
| `--group_size` | `8` | 후보 생성 수 (G), best 1개 선택 |
| `--gpu_id` | `0` | CUDA 디바이스 ID |

> Inference에는 post-hoc 교정(숫자 분절 수정 + Answer 필드 overwrite)이 자동 적용됩니다.

---

## 3. Post-hoc 교정 (독립 실행)

이미 생성된 CSV에 교정만 따로 적용할 경우:

```bash
python scripts/apply_number_correction.py \
    --input  results/grpo_v7_run1/pd_eval100_v7.csv \
    --output results/grpo_v7_run1/pd_eval100_v7_corrected.csv
```

---

## 하이퍼파라미터 상세

### 공통 (v7 / v7-e5)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | 악성 문서 생성 모델 |
| Surrogate LLM | `lmsys/vicuna-7b-v1.3` | r_generation + r_ppl 계산용 |
| Defense filter | `paraphrase-MiniLM-L6-v2` | r_disp_embed + RAGDefender 방어 |
| GROUP_SIZE (G) | 8 | 쿼리당 생성 후보 수 |
| N (adv docs/query) | 4 | 최종 선택 악성 문서 수 (doc0_seed + doc1~3) |
| MIN_NEW_TOKENS | 80 | 생성 최소 토큰 |
| MAX_NEW_TOKENS | 160 | 생성 최대 토큰 |
| TEMPERATURE | 0.85 | 샘플링 온도 |
| TOP_P | 0.92 | nucleus sampling |
| REPETITION_PENALTY | 1.1 | 반복 억제 |
| NO_REPEAT_NGRAM_SIZE | 4 | n-gram 반복 차단 |
| LR | 1e-5 | Adam learning rate |
| GRAD_CLIP | 0.5 | gradient norm clipping |
| ADV_CLIP | 2.0 | GRPO advantage clipping |
| LORA_R | 16 | LoRA rank |
| LORA_ALPHA | 32 | LoRA alpha |
| LORA_DROPOUT | 0.05 | LoRA dropout |
| LAMBDA_KENDALL | 0.30 | Kendall rank loss 가중치 |
| MMR_LAMBDA | 0.60 | MMR diversity 가중치 |

### 보상 함수 (5 component)

| 보상 | 수식 | 설명 |
|------|------|------|
| r_retrieval | (dot − 0.40) / 1.10 ∈ [0,1] | 검색기 유사도 (높을수록 top-k 진입 유리) |
| r_disp_embed | 1 − MiniLM inter-cosine ∈ [0,1] | 생성 문서 간 의미적 다양성 (RAGDefender Stage 2 우회) |
| r_tfidf_disp | 1 − TF-IDF inter-sim ∈ [0,1] | TF-IDF 다양성 (RAGDefender Stage 1 우회) |
| r_generation | P(target \| context+query+Answer:) | Vicuna-7B가 정답을 낼 확률 |
| r_ppl | sigmoid(−log(PPL/20)) | 문서 자연스러움 (낮은 perplexity) |

### 차이점: v7 vs v7-e5

| 항목 | v7 | v7-e5 |
|------|----|--------|
| 화이트박스 검색기 | `facebook/contriever` (dot product) | `intfloat/e5-base-v2` (cosine) |
| r_retrieval 기준 | (dot − 0.40) / 1.10 | (cos − 0.70) / 0.25 |
| 쿼리 prefix | 없음 | `"query: "` / `"passage: "` 추가 |

---

## 출력 CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| `query` | 평가 쿼리 |
| `target_answer` | 목표 정답 (주입 대상) |
| `doc0_seed` | 원본 시드 문서 (변경 없음) |
| `doc1` ~ `doc3` | 생성된 악성 문서 (교정 적용됨) |
