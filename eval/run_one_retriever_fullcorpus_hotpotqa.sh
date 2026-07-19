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

# 이 스크립트(.../DisPo/eval/)의 실제 위치를 기준으로 경로를 자동으로 잡음
# → DisPo를 어느 경로에 clone하든(심볼릭 링크 불필요) 그대로 동작
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPO_ROOT="$(dirname "$SCRIPT_DIR")"
export DISPO_DATA_ROOT="${DISPO_DATA_ROOT:-$(dirname "$DISPO_ROOT")}"   # datasets/ 상위 경로, override 가능

VENV="$DISPO_ROOT/.venv/bin/python3"
EVAL="$DISPO_ROOT/eval/main_dispo_fullcorpus_ragdef.py"
LOGROOT="$DISPO_ROOT/eval/txt_logs_fullcorpus_hotpotqa"
OUT="$DISPO_ROOT/eval/results_hotpotqa_${RET}_gpu${GPU}.json"
LOG="$DISPO_ROOT/logs/split_hotpotqa_${RET}_gpu${GPU}.log"

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
nd, rd = js["no_defense"], js["ragdefender"]
result = {"retriever": "$RET",
          "nd_asr": nd["ASR"], "nd_precision": nd["poison_precision"],
          "nd_recall": nd["poison_recall"], "nd_f1": nd["poison_f1"],
          "rd_asr": rd["ASR"], "rd_precision": rd["poison_precision_after"],
          "rd_recall": rd["poison_recall_after"], "rd_f1": rd["poison_f1_after"]}
json.dump(result, open("$OUT", "w"), indent=2)
print(f"  ND: ASR={result['nd_asr']*100:.4f}%  P={result['nd_precision']*100:.1f}%  R={result['nd_recall']*100:.1f}%  F1={result['nd_f1']*100:.1f}%")
print(f"  RD: ASR={result['rd_asr']*100:.4f}%  P={result['rd_precision']*100:.1f}%  R={result['rd_recall']*100:.1f}%  F1={result['rd_f1']*100:.1f}%")
PYEOF

echo "$(ts) ===== HotpotQA $RET 완료 (결과: $OUT) ====="
} 2>&1 | tee "$LOG"
