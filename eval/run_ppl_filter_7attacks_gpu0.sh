#!/usr/bin/env bash
set -euo pipefail

PY=/data/joonhyung/nq/.venv/bin/python
EVAL=/data/joonhyung/DisPo/eval/main_fullcorpus_ppl_filter.py
CACHE=/data/joonhyung/DisPo/eval/clean_topn_cache/nq_merged_val100_top50/contriever_top50.pt
CFG=/data/joonhyung/DisPo/eval/model_configs/vicuna7b_config.json

run_one() {
  local label=$1
  local csv=$2
  local adv=$3
  "${PY}" "${EVAL}" \
    --dataset nq \
    --retrieval_model contriever \
    --docs_csv "${csv}" \
    --top_k 5 \
    --adv_per_query "${adv}" \
    --thresholds 20 50 80 110 130 \
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

run_one pplfilter_JointGCG_N4_top5 /data/joonhyung/nq/results/attackbaselines_pd/jointgcg4_nq100.csv 4
run_one pplfilter_Confundo_N4_top5 /data/joonhyung/nq/results/attackbaselines_pd/confundo_500input_nq_N4_temp0.7_v2.csv 4
run_one pplfilter_RAGParadox_top5 /data/joonhyung/nq/results/attackbaselines_pd/ragparadox_nq100_n4.csv 4
run_one pplfilter_DiPoison_top5 /data/joonhyung/DisPo/data/generated/pd_eval100_v7_cont_n4g8.csv 4
run_one pplfilter_JointGCG_N1_top5 /data/joonhyung/nq/results/attackbaselines_pd/jointgcg1_nq100.csv 1
run_one pplfilter_Confundo_N1_top5 /data/joonhyung/nq/results/attackbaselines_pd/confundo_500input_N1.csv 1
