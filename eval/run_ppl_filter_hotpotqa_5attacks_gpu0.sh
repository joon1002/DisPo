#!/usr/bin/env bash
set -euo pipefail

PY=/path/to/nq/.venv/bin/python
EVAL=/path/to/DisPo/eval/main_fullcorpus_ppl_filter.py
PRECOMP=/path/to/DisPo/eval/precompute_clean_topn_fullcorpus.py
CFG=/path/to/DisPo/eval/model_configs/vicuna7b_config.json
CACHE=/path/to/DisPo/eval/clean_topn_cache/hotpotqa_5attacks_top50/contriever_top50.pt

POISONEDRAG=/path/to/DisPo/data/attackbaselines_pd/PoisonedRAG/hotpotqa/poisonedrag4_hotpot100.csv
JOINTGCG=/path/to/DisPo/data/attackbaselines_pd/jointgcg/hotpotqa/hotpotqa_origin_jointgcg_v2_n4.csv
CONFUNDO=/path/to/DisPo/data/attackbaselines_pd/confundo/hotpotqa/confundo_hotpotqa_N4.normalized_for_ppl.csv
RAGPARADOX=/path/to/DisPo/data/attackbaselines_pd/RAGParadox/hotpotqa/hotpotqa_ragparadox_n4.csv
DIPOISON=/path/to/DisPo/data/attackbaselines_pd/DiPoison/hotpotqa/dipoison4_hotpot100.csv

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "${CACHE}" ]]; then
  "${PY}" "${PRECOMP}" \
    --dataset hotpotqa \
    --retrieval_model contriever \
    --docs_csv "${POISONEDRAG}" "${JOINTGCG}" "${CONFUNDO}" "${RAGPARADOX}" "${DIPOISON}" \
    --top_n 50 \
    --gpu_id 0 \
    --output "${CACHE}"
fi

run_one() {
  local label=$1
  local csv=$2
  "${PY}" "${EVAL}" \
    --dataset hotpotqa \
    --retrieval_model contriever \
    --docs_csv "${csv}" \
    --top_k 5 \
    --adv_per_query 4 \
    --thresholds 20 50 80 110 140 \
    --ppl_model_name gpt2-xl \
    --ppl_batch_size 16 \
    --ppl_max_length 512 \
    --model_config_path "${CFG}" \
    --model_name vicuna \
    --gpu_id 0 \
    --clean_topn_cache "${CACHE}" \
    --skip_baseline \
    --local_files_only \
    --run_label "${label}"
}

run_one pplfilter_hotpotqa_PoisonedRAG_N4_top5 "${POISONEDRAG}"
run_one pplfilter_hotpotqa_JointGCG_N4_top5 "${JOINTGCG}"
run_one pplfilter_hotpotqa_Confundo_N4_top5 "${CONFUNDO}"
run_one pplfilter_hotpotqa_RAGParadox_N4_top5 "${RAGPARADOX}"
run_one pplfilter_hotpotqa_DiPoison_N4_top5 "${DIPOISON}"
