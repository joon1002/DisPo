#!/usr/bin/env bash
# RAGParadox N1-6, confundo N1-6 (총 12개) full-corpus ND/RD ASR 측정
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

for N in 1 2 3 4 5 6; do
    run_one "ragparadox_v2_n${N}" "$N" "../data/attackbaselines_pd/RAGParadox/ragparadox_nq100_n${N}.csv"
done

for N in 1 2 3 4 5 6; do
    run_one "confundo_v2_n${N}" "$N" "../data/attackbaselines_pd/confundo/confundo_nq_N${N}.csv"
done

echo "===== ALL 12 RUNS FINISHED ====="
