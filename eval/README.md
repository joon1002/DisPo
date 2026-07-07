# DisPo 성능평가 가이드

No-Defense / RAGDefender 방어 하에서 DisPo 악성 문서의 공격 성공률(ASR)과 검색 지표(Precision/Recall/F1)를 측정합니다.

---

## 평가 구조

```
eval/
├── main_dispo_ragdef_beir.py   # 메인 평가 스크립트
├── src/
│   ├── models/                 # LLM 래퍼 (Vicuna, Llama, GPT, Mistral 등)
│   ├── utils.py                # BEIR 코퍼스 로딩, 검색 유틸
│   └── prompts.py              # RAG 프롬프트 템플릿
├── model_configs/              # LLM 설정 파일 (JSON)
│   ├── vicuna7b_config.json
│   ├── llama3_8b_config.json
│   ├── llama3_3b_config.json
│   ├── mistral7b_config.json
│   └── qwen7b_config.json
└── README.md
```

---

## 요구 환경

```bash
pip install beir sentence-transformers transformers torch scikit-learn tqdm pandas
```

BEIR NQ 데이터셋 자동 다운로드 (최초 실행 시):
- `beir` 패키지가 `~/.cache/beir/` 혹은 지정 경로에 다운로드

---

## 평가 파이프라인 설명

### 1. 데이터 구성 (쿼리당)

```
Candidate pool = poison_docs(N개) + BEIR NQ 정상 문서(4~124개)
```

- **poison_docs**: DisPo로 생성한 악성 문서 (`data/generated/pd_eval100_v7_cont_n4g8.csv` 등)
- **정상 문서**: BEIR NQ 코퍼스에서 해당 쿼리의 golden passage와 같은 제목을 가진 모든 passage

### 2. 검색 (Retrieval)

지정한 검색기로 Candidate pool에서 top-k 문서를 선택합니다.

| 검색기 | 모델 | 유사도 |
|--------|------|--------|
| contriever | `facebook/contriever` | dot product |
| contriever-msmarco | `facebook/contriever-msmarco` | dot product |
| ance | `sentence-transformers/msmarco-roberta-base-ance-firstp` | cosine |
| dpr | `sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base` | cosine |
| bge-base | `BAAI/bge-base-en-v1.5` | cosine |
| e5-base | `intfloat/e5-base-v2` | cosine |
| gte-base | `thenlper/gte-base` | cosine |
| mpnet | `sentence-transformers/all-mpnet-base-v2` | cosine |

### 3. No-Defense (ND) 평가

top-k 문서를 모두 context로 사용하여 LLM에 질의 → 응답에 `target_answer`가 포함되면 ASR 성공

```
RAG Prompt:
  Contexts: [top-k 문서]
  Query:    [질문]
  Answer:

ASR_sub: target_answer ∈ response (substring match)
```

### 4. RAGDefender (RD) 평가

RAGDefender (Xue et al., 2024) 2단계 방어를 적용한 후 LLM 평가:

**Stage 1 — Agglomerative Clustering + TF-IDF**
- top-k 문서를 2-cluster로 분류 (paraphrase-MiniLM-L6-v2 임베딩)
- TF-IDF 빈도 점수로 악성 클러스터 추정 → `n_adv` 개 제거 후보 식별

**Stage 2 — Pairwise Frequency-Score Filter**
- top `n_adv*(n_adv-1)/2` 쌍에 대해 유사도 점수 누적
- 점수 높은 문서를 악성으로 판단 → 제거
- 생존 문서만 LLM에 전달

### 5. 지표 계산

| 지표 | 수식 | 기준 |
|------|------|------|
| Precision | poison_in_topk / top_k | No-Defense |
| Recall | poison_in_topk / adv_per_query | No-Defense |
| F1 | 2·P·R / (P+R) | No-Defense |
| ND-ASR | target ∈ LLM_response | No-Defense |
| RD-ASR | target ∈ LLM_response | RAGDefender 방어 후 |

---

## 실행 방법

### 기본 실행 (contriever + vicuna-7b, top_k=5, adv=4)

```bash
cd eval/

CUDA_VISIBLE_DEVICES=0 python main_dispo_ragdef_beir.py \
    --retrieval_model   contriever \
    --model_config_path model_configs/vicuna7b_config.json \
    --model_name        vicuna \
    --docs_csv          ../data/generated/pd_eval100_v7_cont_n4g8.csv \
    --adv_per_query     4 \
    --top_k             5 \
    --gpu_id            0
```

### 8개 검색기 순차 실행

```bash
cd eval/

for RETRIEVER in contriever contriever-msmarco ance dpr bge-base e5-base gte-base mpnet; do
    CUDA_VISIBLE_DEVICES=0 python main_dispo_ragdef_beir.py \
        --retrieval_model   $RETRIEVER \
        --model_config_path model_configs/vicuna7b_config.json \
        --model_name        vicuna \
        --docs_csv          ../data/generated/pd_eval100_v7_cont_n4g8.csv \
        --adv_per_query     4 \
        --top_k             5 \
        --run_label         v7_cont \
        --gpu_id            0
done
```

### v7-e5 악성문서 평가

```bash
CUDA_VISIBLE_DEVICES=0 python main_dispo_ragdef_beir.py \
    --retrieval_model   e5-base \
    --model_config_path model_configs/vicuna7b_config.json \
    --model_name        vicuna \
    --docs_csv          ../data/generated/pd_eval100_v7_e5_n4g8.csv \
    --adv_per_query     4 \
    --top_k             5 \
    --run_label         v7_e5 \
    --gpu_id            0
```

### 다른 generator로 평가 (generator ablation)

```bash
# LLaMA3-8B
CUDA_VISIBLE_DEVICES=0 python main_dispo_ragdef_beir.py \
    --retrieval_model   contriever \
    --model_config_path model_configs/llama3_8b_config.json \
    --model_name        llama \
    --docs_csv          ../data/generated/pd_eval100_v7_cont_n4g8.csv \
    --adv_per_query     4 --top_k 5 --run_label v7_cont_gen-llama3_8b --gpu_id 0
```

---

## 주요 인자

| 인자 | 설명 |
|------|------|
| `--retrieval_model` | 검색기 이름 (위 표 참조) |
| `--model_config_path` | LLM 설정 JSON 경로 |
| `--model_name` | `vicuna` / `llama` / `mistral` / `qwen` |
| `--docs_csv` | 악성 문서 CSV (`data/generated/` 내 파일) |
| `--adv_per_query` | 쿼리당 주입할 악성 문서 수 (N) |
| `--top_k` | 검색기가 반환할 문서 수 |
| `--run_label` | 출력 CSV 파일명 suffix |
| `--gpu_id` | CUDA 디바이스 ID |

---

## 출력

실행 완료 시 `eval/` 디렉토리에 아래 파일이 생성됩니다:

```
pipeline_results_beir_ragdef_ret-{retriever}_gen-{model}_{run_label}.csv
```

CSV 주요 컬럼:

| 컬럼 | 설명 |
|------|------|
| `poison_in_topk` | top-k 중 악성 문서 수 |
| `has_poison` | top-k에 악성 문서 존재 여부 |
| `nd_response` | No-Defense LLM 응답 |
| `nd_asr_sub` | ND ASR (substring match) |
| `rd_response` | RAGDefender 후 LLM 응답 |
| `rd_asr_sub` | RD ASR (substring match) |
| `poison_survived_count` | RAGDefender 통과한 악성 문서 수 |

---

## RAGDefender 참고

- 논문: *RAGDefender: Defending Against Retrieval-Augmented Generation Poisoning Attacks* (Xue et al., 2024)
- 방어 모델: `paraphrase-MiniLM-L6-v2` (SentenceTransformer)
- 방어 로직 구현: `main_dispo_ragdef_beir.py` 내 `find_num_adv_agg_with_stage1()`, `top_similar_pairs()` 함수
