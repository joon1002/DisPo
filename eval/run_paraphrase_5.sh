#!/usr/bin/env bash
# data/paraphrase_pd 5개 파일 full-corpus ND/RD ASR 측정
# 쿼리 자체가 paraphrase되어 clean_topn_cache(원본 쿼리 키)를 못 쓰므로
# --clean_topn_cache 없이 실행 (첫 실행에서 contriever로 전체 코퍼스 임베딩 후 캐시, 이후 재사용)
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../.venv/bin/activate
export DISPO_DATA_ROOT=/path/to
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

LOGDIR=/path/to/DisPo/logs
mkdir -p "$LOGDIR"

run_one() {
    local label="$1" adv_n="$2" docs_csv="$3"
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] START $label (adv_per_query=$adv_n) ====="
    python main_dispo_fullcorpus_ragdef.py \
        --dataset nq --retrieval_model contriever \
        --model_name vicuna --model_config_path model_configs/vicuna7b_config.json \
        --top_k 5 --adv_per_query "$adv_n" --gpu_id 0 \
        --docs_csv "$docs_csv" \
        --run_label "$label" \
        > "$LOGDIR/run_${label}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] DONE  $label ====="
        grep -A2 "ND-ASR\|RD-ASR" "$LOGDIR/run_${label}.log" | tail -4
    else
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] FAILED $label — see $LOGDIR/run_${label}.log ====="
        tail -30 "$LOGDIR/run_${label}.log"
    fi
}

run_one "para_dipoison4"    4 "../data/paraphrase_pd/DiPoison/dipoison4_nq100_para.csv"
run_one "para_poisonedrag4" 4 "../data/paraphrase_pd/PoisonedRAG/poisonedrag4_nq100_para.csv"
run_one "para_ragparadox4"  4 "../data/paraphrase_pd/RAGParadox/ragparadox_nq100_n4_para.csv"
run_one "para_confundo4"    4 "../data/paraphrase_pd/confundo/confundo_500input_nq_N4_temp0.7_v2_para.csv"
run_one "para_jointgcg4"    4 "../data/paraphrase_pd/jointgcg/jointgcg4_nq100_para.csv"

echo "===== ALL 5 PARAPHRASE RUNS FINISHED ====="
