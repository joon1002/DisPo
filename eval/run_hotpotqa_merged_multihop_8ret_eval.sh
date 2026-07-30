#!/usr/bin/env bash
set -euo pipefail

cd /path/to/DisPo

PY=/path/to/ragdef/.venv/bin/python
EVAL=eval/hotpotqa_merged_multihop_8ret_eval.py
DOCS=/path/to/DisPo/data/attackbaselines_pd/DiPoison/merged/hotpotqa_merged_dipoison.csv
OUT=/path/to/DisPo/eval/results/hotpotqa_merged_multihop_8ret/hotpotqa_merged_dipoison_summary.json
CACHE_LOG_DIR=/path/to/DisPo/eval/results/hotpotqa_merged_multihop_8ret/cache_logs

mkdir -p "$CACHE_LOG_DIR"

common_args=(
  --docs_csv "$DOCS"
  --top_ks 5,10
  --adv_per_query 7
  --st_batch_size 512
  --contriever_batch_size 256
  --dense_chunk_size 32768
)

run_cache() {
  local ret="$1"
  local gpu="$2"
  local cache_out="$CACHE_LOG_DIR/${ret}_cache.json"
  echo "[$(date '+%F %T')] cache start ret=$ret gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_DISABLE_XET=1 "$PY" "$EVAL" \
    "${common_args[@]}" \
    --out_json "$cache_out" \
    --gpu_id 0 \
    --retrievers "$ret" \
    --cache_only
  echo "[$(date '+%F %T')] cache done ret=$ret gpu=$gpu"
}

echo "[$(date '+%F %T')] HotpotQA merged 8-retriever multihop eval pipeline start (sequential cache build)"

run_cache e5 0
run_cache ance 0
run_cache bge-base 0
run_cache mpnet 0
run_cache bm25 0
run_cache nomic-v1.5 0
run_cache contriever-msmarco 0

echo "[$(date '+%F %T')] all caches ready; evaluation start"
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 "$PY" "$EVAL" \
  "${common_args[@]}" \
  --out_json "$OUT" \
  --gpu_id 0 \
  --skip_done \
  --skip_cache_build

echo "[$(date '+%F %T')] HotpotQA merged 8-retriever multihop eval pipeline done"
