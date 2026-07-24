#!/usr/bin/env bash
# 나머지 14개 조합(전체 15개 중 poisonedrag_n1은 스목테스트로 이미 완료) 순차 실행
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../.venv/bin/activate
export DISPO_DATA_ROOT=/data1/joonhyung
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

CACHE=clean_topn_cache/nq_merged_val100_top50/contriever_top50.pt
LOGDIR=/data1/joonhyung/DisPo/logs
mkdir -p "$LOGDIR"

run_one() {
    local label="$1" adv_n="$2" docs_csv="$3"
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] START $label (adv_per_query=$adv_n) ====="
    python main_dispo_fullcorpus_ragdef.py \
        --dataset nq --retrieval_model contriever \
        --model_name vicuna --model_config_path model_configs/vicuna7b_config.json \
        --top_k 5 --adv_per_query "$adv_n" --gpu_id 0 \
        --clean_topn_cache "$CACHE" \
        --docs_csv "$docs_csv" \
        --run_label "$label" \
        > "$LOGDIR/run_${label}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] DONE  $label ====="
        grep -A2 "ND-ASR\|RD-ASR" "$LOGDIR/run_${label}.log" | tail -4
    else
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] FAILED $label — see $LOGDIR/run_${label}.log ====="
        tail -20 "$LOGDIR/run_${label}.log"
    fi
}

for N in 2 3 4 5 6; do
    run_one "poisonedrag_n${N}" "$N" "../data/attackbaselines_pd/PoisonedRAG/poisonedrag${N}_nq100.csv"
done

for N in 1 2 3 4 5 6; do
    run_one "jointgcg_n${N}" "$N" "../data/attackbaselines_pd/jointgcg/jointgcg${N}_nq100.csv"
done

run_one "ragparadox_n4" 4 "../data/attackbaselines_pd/RAGParadox/ragparadox_nq100_n4.csv"

run_one "confundo_n1" 1 "../data/attackbaselines_pd/confundo/confundo_500input_N1.csv"
run_one "confundo_n4" 4 "../data/attackbaselines_pd/confundo/confundo_500input_nq_N4_temp0.7_v2.csv"

echo "===== ALL 14 RUNS FINISHED ====="
