#!/bin/bash
# Sequential fullcorpus RobustRAG eval for all attackbaselines_pd files
# GPU 0, Contriever -> RobustRAG KeywordAgg -> Vicuna-7B, top-5

PYTHON=/path/to/ragdef/.venv/bin/python
SCRIPT=/path/to/DisPo/eval/main_dispo_fullcorpus_robustrag.py
BASE_DATA=/path/to/DisPo/data/attackbaselines_pd
LOG_DIR=/path/to/DisPo/eval/logs_attackbaselines_robustrag

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1
cd /path/to/DisPo

run_eval() {
    local csv_file="$1"
    local adv_n="$2"
    local name="$3"
    local log="$LOG_DIR/${name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== START: $name (adv=$adv_n) =====" | tee -a "$log"
    $PYTHON $SCRIPT \
        --dataset nq \
        --retrieval_model contriever \
        --docs_csv "$BASE_DATA/$csv_file" \
        --top_k 5 \
        --adv_per_query "$adv_n" \
        --model_config_path eval/model_configs/vicuna7b_config.json \
        --model_name vicuna \
        --gpu_id 0 >> "$log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== DONE: $name =====" | tee -a "$log"

    # Print final results
    latest_run=$(ls -td /path/to/DisPo/eval/txt_logs_fullcorpus_nq/run_*/ 2>/dev/null | head -1)
    if [ -f "${latest_run}final.json" ]; then
        echo "=== FINAL: $name ===" | tee -a "$log"
        cat "${latest_run}final.json" | tee -a "$log"
    fi
}

# N=4 experiments
run_eval "poisonedrag_nq100.csv"                     4 "poisonedrag_n4"
run_eval "jointgcg4_nq100.csv"                       4 "jointgcg4_n4"
run_eval "ragparadox_nq100_n4.csv"                   4 "ragparadox_n4"
run_eval "confundo_500input_nq_N4_temp0.7_v2.csv"   4 "confundo_n4"

# N=1 experiments
run_eval "jointgcg1_nq100.csv"                       1 "jointgcg1_n1"
run_eval "confundo_500input_N1.csv"                  1 "confundo_n1"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== ALL DONE ====="
