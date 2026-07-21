#!/usr/bin/env bash
# Run four v7 ablation inference jobs in parallel on GPU 0,1,2,3.
# Mapping:
#   GPU0 -> no_generation
#   GPU1 -> no_ppl
#   GPU2 -> no_disp_embed
#   GPU3 -> no_tfidf_disp
#
# Usage:
#   bash scripts/run_v7_ablation_g8_infer_gpus0_3.sh
#
# Optional env:
#   PYTHON=/path/to/python
#   INPUT=/path/to/nq100_validate.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
INPUT_CSV="${INPUT:-$ROOT_DIR/data/nq100_validate.csv}"
OUT_DIR="$ROOT_DIR/data/generated"
INFER="$SCRIPT_DIR/infer_v7_abl_checkpoint.py"

mkdir -p "$OUT_DIR"

run_one() {
    local gpu_id="$1"
    local label="$2"
    local ablation="$3"
    local model_dir="$4"
    local output_csv="$5"
    local log_file="$6"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch gpu${gpu_id} ${label}"
    CUDA_VISIBLE_DEVICES="$gpu_id" HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" "$INFER" \
        --ablation "$ablation" \
        --checkpoint "$model_dir" \
        --input "$INPUT_CSV" \
        --output "$output_csv" \
        --gpu_id "$gpu_id" \
        --group_size 8 \
        --gen_batch_size 4 \
        --N 4 \
        > "$log_file" 2>&1 &
}

run_one 0 "no_gen" \
    "no_generation" \
    "$ROOT_DIR/data/final_model_abl_nogen" \
    "$OUT_DIR/pd_eval100_v7_abl_no_gen_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_gen_g8_b4_gpu0.log"

run_one 1 "no_ppl" \
    "no_ppl" \
    "$ROOT_DIR/data/final_model_abl_noppl" \
    "$OUT_DIR/pd_eval100_v7_abl_no_ppl_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_ppl_g8_b4_gpu1.log"

run_one 2 "no_disp" \
    "no_disp_embed" \
    "$ROOT_DIR/data/final_model_abl_nodisp" \
    "$OUT_DIR/pd_eval100_v7_abl_no_disp_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_disp_g8_b4_gpu2.log"

run_one 3 "no_tfidf" \
    "no_tfidf_disp" \
    "$ROOT_DIR/data/final_model_abl_notfidf" \
    "$OUT_DIR/pd_eval100_v7_abl_no_tfidf_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_tfidf_g8_b4_gpu3.log"

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] all ablation inference complete"
