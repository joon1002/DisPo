#!/bin/bash
# Sequential fullcorpus RobustRAG eval for all attackbaselines_pd files
# GPU 1, Contriever -> RobustRAG KeywordAgg -> Mistral-7B-Instruct-v0.3, top-5

PYTHON=/data/joonhyung/ragdef/.venv/bin/python
SCRIPT=/data/joonhyung/DisPo/eval/main_dispo_fullcorpus_robustrag.py
BASE_DATA=/data/joonhyung/DisPo/data/attackbaselines_pd
LOG_DIR=/data/joonhyung/DisPo/eval/logs_attackbaselines_robustrag_mistral7b

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_DISABLE_XET=1
cd /data/joonhyung/DisPo

run_eval() {
    local csv_file="$1"
    local adv_n="$2"
    local name="$3"
    local log="$LOG_DIR/${name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== START: $name (adv=$adv_n) =====" | tee -a "$LOG_DIR/main.log"
    $PYTHON $SCRIPT \
        --dataset nq \
        --retrieval_model contriever \
        --docs_csv "$BASE_DATA/$csv_file" \
        --top_k 5 \
        --adv_per_query "$adv_n" \
        --model_config_path eval/model_configs/mistral7b_gpu1_config.json \
        --model_name mistral \
        --gpu_id 1 >> "$log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== DONE: $name =====" | tee -a "$LOG_DIR/main.log"

    latest_run=$(ls -td /data/joonhyung/DisPo/eval/txt_logs_fullcorpus_nq/run_*/ 2>/dev/null | head -1)
    if [ -f "${latest_run}final.json" ]; then
        echo "=== FINAL: $name ===" | tee -a "$LOG_DIR/main.log"
        cat "${latest_run}final.json" | tee -a "$LOG_DIR/main.log"
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== ALL DONE =====" | tee -a "$LOG_DIR/main.log"
