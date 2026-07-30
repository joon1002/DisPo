#!/usr/bin/env bash
set -euo pipefail

DOCS_CSV="${1:?Usage: $0 /path/to/malicious_docs.csv [gpu_id]}"
GPU_ID="${2:-0}"
DATA_ROOT="${DISPO_DATA_ROOT:-/path/to}"

cd "$(dirname "$0")/../../.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" HF_HUB_DISABLE_XET=1 "${PYTHON}" -m scripts.supplementary.reranking.eval_tinybert_l2_asr \
  --docs_csv "${DOCS_CSV}" \
  --data_root "${DATA_ROOT}" \
  --gpu_id 0 \
  --output_dir scripts/supplementary/reranking/results
