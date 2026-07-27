#!/usr/bin/env bash
set -uo pipefail

GPU=0
VENV=/data/joonhyung/ragatt/.venv/bin/python3
EVAL=/data/joonhyung/DisPo/eval/main_dispo_fullcorpus_ragdef.py
CACHE=/data/joonhyung/DisPo/eval/clean_topn_cache/nq_merged_val100_top50/contriever_top50.pt
MODEL_CFG=/data/joonhyung/DisPo/eval/model_configs/vicuna7b_config.json
BASE_DIR=/data/joonhyung/DisPo/data/attackbaselines_pd/RAGParadox
LOGROOT=/data/joonhyung/DisPo/eval/txt_logs_fullcorpus_nq
LOG=/data/joonhyung/DisPo/eval/txt_logs_fullcorpus_nq/run_ragparadox_n1to7_ablation.log

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }

find_run_dir_by_label() {
    local label="$1"
    local f
    f=$(find "$LOGROOT" -maxdepth 2 -name "results_${label}_*.csv" 2>/dev/null | sort | tail -1)
    [ -n "$f" ] && dirname "$f"
}

{
for N in 1 2 3 4 5 6 7; do
    if [ "$N" -eq 7 ]; then
        DOCS_CSV="$BASE_DIR/ragparadox_nq_n7.csv"
    else
        DOCS_CSV="$BASE_DIR/ragparadox_nq_n${N}_from7.csv"
    fi
    LABEL="ragparadox_nablation_n${N}"
    echo "$(ts) ===== N=$N 시작 (docs_csv=$DOCS_CSV) ====="
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 "$VENV" "$EVAL" \
        --dataset nq \
        --retrieval_model contriever \
        --docs_csv "$DOCS_CSV" \
        --adv_per_query "$N" \
        --top_k 5 \
        --gpu_id "$GPU" \
        --defense_model paraphrase-MiniLM-L6-v2 \
        --model_config_path "$MODEL_CFG" \
        --model_name vicuna \
        --clean_topn_cache "$CACHE" \
        --run_label "$LABEL"
    RUN_DIR=$(find_run_dir_by_label "$LABEL")
    if [ -z "$RUN_DIR" ]; then
        echo "$(ts) [error] result dir not found (label=$LABEL)"
        continue
    fi
    echo "$(ts) N=$N run_dir=$RUN_DIR"
done

echo "$(ts) ===== all N=1..7 runs done, aggregating ====="
"$VENV" - <<'PYEOF'
import json, glob, os

logroot = "/data/joonhyung/DisPo/eval/txt_logs_fullcorpus_nq"
out = {}
for n in range(1, 8):
    label = f"ragparadox_nablation_n{n}"
    dirs = sorted(glob.glob(f"{logroot}/*"), key=os.path.getmtime)
    run_dir = None
    for d in reversed(dirs):
        if glob.glob(f"{d}/results_{label}_*.csv"):
            run_dir = d
            break
    if run_dir is None:
        out[n] = {"error": "not found"}
        continue
    js = json.load(open(f"{run_dir}/final.json"))
    nd, rd = js["no_defense"], js["ragdefender"]
    out[n] = {
        "nd_asr": nd["ASR"], "nd_precision": nd["poison_precision"],
        "nd_recall": nd["poison_recall"], "nd_f1": nd["poison_f1"],
        "rd_asr": rd["ASR"], "rd_precision": rd["poison_precision_after"],
        "rd_recall": rd["poison_recall_after"], "rd_f1": rd["poison_f1_after"],
        "retrieval_rate": nd["retrieval_rate"],
    }
json.dump(out, open("/data/joonhyung/DisPo/eval/results_ragparadox_n1to7_ablation.json", "w"), indent=2)
for n in range(1, 8):
    d = out[n]
    if "error" in d:
        print(f"  N={n}: {d['error']}")
        continue
    print(f"  N={n}: ND-ASR={d['nd_asr']*100:.1f}%  RD-ASR={d['rd_asr']*100:.1f}%  "
          f"ND-F1={d['nd_f1']*100:.1f}%  retrieval_rate={d['retrieval_rate']*100:.1f}%")
PYEOF
echo "$(ts) ===== done ====="
} 2>&1 | tee "$LOG"
