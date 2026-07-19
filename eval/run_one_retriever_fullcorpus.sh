#!/usr/bin/env bash
# 단일 retriever fullcorpus eval — 여러 GPU에서 병렬 실행하기 위한 버전
# run_merged_8ret_fullcorpus.sh와 달리 "가장 최근 생성된 final.json(ls -t)"이 아니라
# run_label로 결과 디렉토리를 특정하므로 여러 retriever를 동시에 돌려도 결과가 섞이지 않음.
#
# 사용법: run_one_retriever_fullcorpus.sh <RETRIEVER> <GPU_ID>
#   예)   run_one_retriever_fullcorpus.sh dpr 0

set -euo pipefail

RET="$1"
GPU="$2"

# 이 스크립트(.../DisPo/eval/)의 실제 위치를 기준으로 경로를 자동으로 잡음
# → DisPo를 어느 경로에 clone하든(심볼릭 링크 불필요) 그대로 동작
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPO_ROOT="$(dirname "$SCRIPT_DIR")"
export DISPO_DATA_ROOT="${DISPO_DATA_ROOT:-$(dirname "$DISPO_ROOT")}"   # datasets/ 상위 경로, override 가능

VENV="$DISPO_ROOT/.venv/bin/python3"
EVAL="$DISPO_ROOT/eval/main_dispo_fullcorpus_ragdef.py"
DOCS_SEED="$DISPO_ROOT/data/generated/pd_eval100_merged_seed.csv"
DOCS_NOSEED="$DISPO_ROOT/data/generated/pd_eval100_merged_noseed.csv"
LOGROOT="$DISPO_ROOT/eval/txt_logs_fullcorpus_nq"
OUT="$DISPO_ROOT/eval/results_${RET}_gpu${GPU}.json"
LOG="$DISPO_ROOT/logs/split_${RET}_gpu${GPU}.log"

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }

find_run_dir_by_label() {
    local label="$1"
    local f
    f=$(find "$LOGROOT" -maxdepth 2 -name "results_${label}_*.csv" 2>/dev/null | sort | tail -1)
    [ -n "$f" ] && dirname "$f"
}

{
echo "$(ts) ===== $RET (GPU $GPU) 단독 실행 시작 ====="

echo "$(ts) [seed] adv=7 시작"
SEED_LABEL="merged_seed_${RET}_fullcorpus_val100"
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 "$VENV" "$EVAL" \
    --dataset nq \
    --retrieval_model "$RET" \
    --docs_csv "$DOCS_SEED" \
    --adv_per_query 7 \
    --top_k 5 \
    --gpu_id "$GPU" \
    --run_label "$SEED_LABEL"
SEED_DIR=$(find_run_dir_by_label "$SEED_LABEL")
if [ -z "$SEED_DIR" ]; then
    echo "$(ts) [오류] seed 결과 디렉토리를 못 찾음 (label=$SEED_LABEL)"; exit 1
fi
echo "$(ts) [seed] run_dir=$SEED_DIR"

echo "$(ts) [noseed] adv=6 시작"
NOSEED_LABEL="merged_noseed_${RET}_fullcorpus_val100"
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 "$VENV" "$EVAL" \
    --dataset nq \
    --retrieval_model "$RET" \
    --docs_csv "$DOCS_NOSEED" \
    --adv_per_query 6 \
    --top_k 5 \
    --gpu_id "$GPU" \
    --run_label "$NOSEED_LABEL"
NOSEED_DIR=$(find_run_dir_by_label "$NOSEED_LABEL")
if [ -z "$NOSEED_DIR" ]; then
    echo "$(ts) [오류] noseed 결과 디렉토리를 못 찾음 (label=$NOSEED_LABEL)"; exit 1
fi
echo "$(ts) [noseed] run_dir=$NOSEED_DIR"

"$VENV" - <<PYEOF
import json
js  = json.load(open("$SEED_DIR/final.json"))
jns = json.load(open("$NOSEED_DIR/final.json"))

def m(d, key):
    nd, rd = d["no_defense"], d["ragdefender"]
    return {"nd_asr": nd["ASR"], "nd_precision": nd["poison_precision"],
            "nd_recall": nd["poison_recall"], "nd_f1": nd["poison_f1"],
            "rd_asr": rd["ASR"], "rd_precision": rd["poison_precision_after"],
            "rd_recall": rd["poison_recall_after"], "rd_f1": rd["poison_f1_after"]}

result = {"retriever": "$RET", "seed": m(js, "seed"), "noseed": m(jns, "noseed")}
json.dump(result, open("$OUT", "w"), indent=2)
for label, d in (("seed", result["seed"]), ("noseed", result["noseed"])):
    print(f"  {label:6s}-> ND: ASR={d['nd_asr']*100:.4f}% P={d['nd_precision']*100:.1f}% R={d['nd_recall']*100:.1f}% F1={d['nd_f1']*100:.1f}%"
          f"  |  RD: ASR={d['rd_asr']*100:.4f}% P={d['rd_precision']*100:.1f}% R={d['rd_recall']*100:.1f}% F1={d['rd_f1']*100:.1f}%")
PYEOF

echo "$(ts) ===== $RET 완료 (결과: $OUT) ====="
} 2>&1 | tee "$LOG"
