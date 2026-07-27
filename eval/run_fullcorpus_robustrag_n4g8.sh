#!/bin/bash
# Full-corpus RobustRAG eval: pd_eval100_v7_cont_n4g8.csv
# Contriever -> RobustRAG KeywordAgg -> Vicuna-7B, top-5, GPU 0

PYTHON=/data/joonhyung/ragdef/.venv/bin/python
SCRIPT=/data/joonhyung/DisPo/eval/main_dispo_fullcorpus_robustrag.py

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1

cd /data/joonhyung/DisPo

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== START: fullcorpus RobustRAG n4g8 ====="
$PYTHON $SCRIPT \
    --dataset nq \
    --retrieval_model contriever \
    --docs_csv data/generated/pd_eval100_v7_cont_n4g8.csv \
    --top_k 5 \
    --adv_per_query 4 \
    --model_config_path eval/model_configs/vicuna7b_config.json \
    --model_name vicuna \
    --gpu_id 0
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== DONE: fullcorpus RobustRAG n4g8 ====="
