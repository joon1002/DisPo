#!/usr/bin/env bash
# Run v7 ablation inference for four final_model folders with G=8, N=4.
# Usage:
#   bash scripts/run_v7_ablation_g8_infer.sh [gpu_id]
#
# Optional env:
#   PYTHON=/path/to/python
#   INPUT=/path/to/nq100_validate.csv

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_v7_ablation_g8_infer.sh [gpu_id]" >&2
    exit 1
fi

GPU_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
INPUT_CSV="${INPUT:-$ROOT_DIR/data/nq100_validate.csv}"
OUT_DIR="$ROOT_DIR/data/generated"
INFER="$SCRIPT_DIR/infer_v7_abl_checkpoint.py"

mkdir -p "$OUT_DIR"

run_one() {
    local label="$1"
    local ablation="$2"
    local model_dir="$3"
    local output_csv="$4"
    local log_file="$5"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] start $label"
    CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" "$INFER" \
        --ablation "$ablation" \
        --checkpoint "$model_dir" \
        --input "$INPUT_CSV" \
        --output "$output_csv" \
        --gpu_id "$GPU_ID" \
        --group_size 8 \
        --gen_batch_size 4 \
        --N 4 \
        2>&1 | tee "$log_file"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done  $label -> $output_csv"
}

run_one "no_gen" \
    "no_generation" \
    "$ROOT_DIR/data/final_model_abl_nogen" \
    "$OUT_DIR/pd_eval100_v7_abl_no_gen_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_gen_g8_b4_gpu${GPU_ID}.log"

run_one "no_ppl" \
    "no_ppl" \
    "$ROOT_DIR/data/final_model_abl_noppl" \
    "$OUT_DIR/pd_eval100_v7_abl_no_ppl_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_ppl_g8_b4_gpu${GPU_ID}.log"

run_one "no_disp" \
    "no_disp_embed" \
    "$ROOT_DIR/data/final_model_abl_nodisp" \
    "$OUT_DIR/pd_eval100_v7_abl_no_disp_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_disp_g8_b4_gpu${GPU_ID}.log"

run_one "no_tfidf" \
    "no_tfidf_disp" \
    "$ROOT_DIR/data/final_model_abl_notfidf" \
    "$OUT_DIR/pd_eval100_v7_abl_no_tfidf_g8_b4.csv" \
    "$OUT_DIR/infer_abl_no_tfidf_g8_b4_gpu${GPU_ID}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] all ablation inference complete"
