# DisPo: Dispersion-Penalized Poison Document Generation

GRPO 기반 RAG 악성 문서 생성 프레임워크.  
Contriever(v7) 또는 E5(v7-e5)를 화이트박스 검색기로 사용하여 Qwen2.5-1.5B 생성기를 강화학습으로 훈련하고, 100개 평가 쿼리에 대해 악성 문서를 생성 및 평가합니다.

---

## 새 서버에서 빠르게 시작하기

### 1. 레포 클론

```bash
git clone https://github.com/joon1002/DisPo.git
cd DisPo
```

### 2. Python 환경 구성

```bash
python3.8 -m venv .venv
source .venv/bin/activate

# 훈련/inference용 (검증된 조합: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate sentence-transformers scikit-learn tqdm pandas

# 성능평가 추가 패키지 (eval/ 사용 시)
pip install beir
pip install fschat==0.2.36    # LLM(Vicuna-7B) 생성 단계에 필수 — 누락 시 ImportError
```

### 3. 전체 흐름 (v7 기준, full-corpus 성능평가가 기본)

```bash
# Step 1: 훈련 (GPU 0, ~8h)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_poison_v7.py \
    --input      data/nq_500_pd_7b.csv \
    --output_dir results/grpo_v7_run1

# Step 2: Inference (훈련 완료 후)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_checkpoint.py \
    --checkpoint results/grpo_v7_run1/final_model \
    --input      data/nq100_validate.csv \
    --output     results/grpo_v7_run1/pd_eval100_v7.csv

# Step 3: 성능평가 — full-corpus 방식이 기본(default)
# (사전 준비: NQ/HotpotQA corpus 다운로드 — "직접 corpus를 받는 방법" 섹션 참고)
cd eval/
CUDA_VISIBLE_DEVICES=0 python main_dispo_fullcorpus_ragdef.py \
    --dataset         nq \
    --retrieval_model contriever \
    --docs_csv        ../results/grpo_v7_run1/pd_eval100_v7.csv \
    --adv_per_query   4 --top_k 5 --gpu_id 0
```

> full-corpus가 기본 평가 방식입니다(실제 RAG 환경과 동일하게 전체 corpus에서 top-k 검색). `main_dispo_ragdef_beir.py`/`main_dispo_extraval_ragdef.py`는 쿼리당 소수 후보 문서끼리만 경쟁시키는 legacy 방식으로, 8검색기 비교 등 특수 목적에만 사용합니다.
> 성능평가 전체 가이드는 [eval/README.md](eval/README.md) 참고

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

- Python 3.8 (검증된 버전 — 다른 버전은 미검증)
- PyTorch 2.4.1 (CUDA 12.1) — `torch==2.4.1+cu121`, `cudnn 90100`
- transformers 4.46.3, sentence-transformers 3.2.1
- fschat 0.2.36 (Vicuna-7B 생성 단계 필수, `import fastchat`로 사용)
- GPU: 24GB+ VRAM 권장 (A100/H100), full-corpus eval은 검색기당 corpus 임베딩 캐시로
  NQ 기준 ~8GB, HotpotQA 기준 ~16GB 디스크가 추가로 필요

```bash
pip install transformers peft accelerate sentence-transformers scikit-learn tqdm pandas
pip install fschat==0.2.36
```

모델 다운로드 (최초 실행 시 HuggingFace에서 자동):
- Generator: `Qwen/Qwen2.5-1.5B-Instruct`
- Surrogate LLM: `lmsys/vicuna-7b-v1.3`
- Retriever(v7): `facebook/contriever`
- Retriever(v7-e5): `intfloat/e5-base-v2`
- Defense filter: `paraphrase-MiniLM-L6-v2`
- full-corpus 8검색기 비교 시 추가: `contriever-msmarco`, `sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base`(dpr),
  `sentence-transformers/msmarco-roberta-base-ance-firstp`(ance), `BAAI/bge-base-en-v1.5`(bge-base),
  `intfloat/e5-base-v2`(e5-base), `thenlper/gte-base`(gte-base), `sentence-transformers/all-mpnet-base-v2`(mpnet)
-------
방어 포함 성능평가(full-corpus, 기본 방식)에 필요한 corpus 직접 받는 방법

**NQ** (2.68M passages, ~1.5GB)
```bash
mkdir -p /data/joonhyung/datasets/nq && cd /data/joonhyung/datasets/nq
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip nq.zip
mv nq/corpus.jsonl nq/queries.jsonl nq/qrels .
rm -rf nq nq.zip
```

**HotpotQA** (5.23M passages, ~2.2GB)
```bash
mkdir -p /data/joonhyung/datasets/hotpotqa && cd /data/joonhyung/datasets/hotpotqa
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
unzip hotpotqa.zip
mv hotpotqa/corpus.jsonl hotpotqa/queries.jsonl hotpotqa/qrels .
rm -rf hotpotqa hotpotqa.zip
```

> `eval/main_dispo_fullcorpus_ragdef.py`의 `_DS_CFG`에 경로가 `/data/joonhyung/datasets/{nq,hotpotqa}/`로 고정되어 있으므로 반드시 이 경로에 둬야 합니다.

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
| `--group_size` | `8` | GRPO 그룹 크기 **(G)** — 쿼리당 생성 후보 수 |
| `--num_adv_docs` | `3` | 쿼리당 최종 악성 문서 수 **(N)** — doc0_seed 제외, 총 N+1개 |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lr` | `1e-5` | Learning rate |
| `--gpu_id` | `0` | CUDA 디바이스 ID |
| `--embed_device` | `cuda` | 임베딩 모델 디바이스 (VRAM 부족 시 cpu) |
| `--limit` | `None` | 쿼리 수 제한 (디버깅용, 예: `--limit 10`) |

---

## 2. Inference

훈련된 LoRA 체크포인트에서 100개 평가 쿼리에 대해 악성 문서 생성.  
쿼리당 `doc0_seed`(시드 그대로) + `doc1~doc3`(생성 문서) = 4개.

### v7

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_checkpoint.py \
    --checkpoint    results/grpo_v7_run1/final_model \
    --input         data/nq100_validate.csv \
    --output        results/grpo_v7_run1/pd_eval100_v7.csv \
    --group_size    8 \
    --gen_batch_size 8
```

### v7-e5

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_e5_checkpoint.py \
    --checkpoint    results/grpo_v7_e5_run1/final_model \
    --input         data/nq100_validate.csv \
    --output        results/grpo_v7_e5_run1/pd_eval100_v7_e5.csv \
    --group_size    8 \
    --gen_batch_size 8
```

### v7-n (N 가변)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_v7_n.py \
    --checkpoint    results/grpo_v7_run1/final_model \
    --input         data/nq100_validate.csv \
    --output        results/grpo_v7_run1/pd_eval100_v7_n6.csv \
    --N 6 \
    --group_size    8 \
    --gen_batch_size 8
```

### Inference 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | `results/.../final_model` | LoRA 체크포인트 경로 |
| `--input` | `data/nq100_validate.csv` | 평가 쿼리 CSV |
| `--output` | `results/.../pd_eval100.csv` | 출력 CSV 경로 |
| `--group_size` | `8` | 후보 생성 수 **(G)**, best 1개 선택 |
| `--num_adv_docs` | `3` | 쿼리당 생성할 악성 문서 수 **(N)** |
| `--N` | `None` | `--num_adv_docs`와 동일 의미, 지정 시 우선 적용 (`infer_v7_n.py`만) |
| `--gen_batch_size` | `1` | G 후보를 한 번에 몇 개씩 생성할지. **`--group_size`와 동일값으로 설정하면 가장 빠름** |
| `--embed_device` | `cuda` | 임베딩 모델 디바이스 |
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
| GROUP_SIZE (G) | 8 | 쿼리당 생성 후보 수 (`--group_size`로 변경 가능) |
| N (adv docs/query) | 3 | 최종 선택 악성 문서 수 (`--num_adv_docs`로 변경 가능, 총 N+1개) |
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
