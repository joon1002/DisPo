#!/usr/bin/env bash
# HotpotQA 단일 retriever fullcorpus eval — 여러 GPU에서 병렬 실행 가능
# (run_one_retriever_fullcorpus.sh의 NQ seed/noseed 2회 호출과 달리,
#  HotpotQA는 poison docs 변형이 1종류뿐이라 eval을 1회만 호출함)
#
# 사용법: run_one_retriever_fullcorpus_hotpotqa.sh <RETRIEVER> <GPU_ID> <DOCS_CSV> <ADV_PER_QUERY> [RUN_LABEL]
#   예)   run_one_retriever_fullcorpus_hotpotqa.sh dpr 0 \
#           /data/joonhyung/DisPo/results/grpo_v7_cont_n4_hotpotqa/pd_hotpotqa_val300_v7cont_n4.csv 4

set -euo pipefail

RET="$1"
GPU="$2"
DOCS_CSV="$3"
ADV_PER_QUERY="$4"
RUN_LABEL="${5:-hotpotqa_${RET}_fullcorpus}"

VENV=/data/joonhyung/DisPo/.venv/bin/python3
EVAL=/data/joonhyung/DisPo/eval/main_dispo_fullcorpus_ragdef.py
LOGROOT=/data/joonhyung/DisPo/eval/txt_logs_fullcorpus_hotpotqa
OUT=/data/joonhyung/DisPo/eval/results_hotpotqa_${RET}_gpu${GPU}.json
LOG=/data/joonhyung/DisPo/logs/split_hotpotqa_${RET}_gpu${GPU}.log

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }

find_run_dir_by_label() {
    local label="$1"
    local f
    f=$(find "$LOGROOT" -maxdepth 2 -name "results_${label}_*.csv" 2>/dev/null | sort | tail -1)
    [ -n "$f" ] && dirname "$f"
}

mkdir -p "$(dirname "$LOG")"

{
echo "$(ts) ===== HotpotQA $RET (GPU $GPU) 단독 실행 시작 ====="
echo "$(ts) docs_csv=$DOCS_CSV adv_per_query=$ADV_PER_QUERY"

CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 "$VENV" "$EVAL" \
    --dataset hotpotqa \
    --retrieval_model "$RET" \
    --docs_csv "$DOCS_CSV" \
    --adv_per_query "$ADV_PER_QUERY" \
    --top_k 5 \
    --gpu_id "$GPU" \
    --run_label "$RUN_LABEL"

RUN_DIR=$(find_run_dir_by_label "$RUN_LABEL")
if [ -z "$RUN_DIR" ]; then
    echo "$(ts) [오류] 결과 디렉토리를 못 찾음 (label=$RUN_LABEL)"; exit 1
fi
echo "$(ts) run_dir=$RUN_DIR"

"$VENV" - <<PYEOF
import json
js = json.load(open("$RUN_DIR/final.json"))
result = {"retriever": "$RET",
          "nd_asr": js["no_defense"]["ASR"], "rd_asr": js["ragdefender"]["ASR"]}
json.dump(result, open("$OUT", "w"), indent=2)
print(f"  ND-ASR={result['nd_asr']*100:.1f}%  RD-ASR={result['rd_asr']*100:.1f}%")
PYEOF

echo "$(ts) ===== HotpotQA $RET 완료 (결과: $OUT) ====="
} 2>&1 | tee "$LOG"
