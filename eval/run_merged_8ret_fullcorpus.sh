#!/usr/bin/env bash
# merged seed/noseed — fullcorpus 8-retriever eval (GPU 1)
# contriever 기존 결과: seed ND=94%/RD=84%, noseed ND=92%/RD=83%
# 이 스크립트: contriever-msmarco, dpr, ance, bge-base, e5-base, gte-base, mpnet (7개)

GPU=1
VENV=/data/joonhyung/ragatt/.venv/bin/python3
EVAL=/data/joonhyung/DisPo/eval/main_dispo_fullcorpus_ragdef.py
DOCS_SEED=/data/joonhyung/DisPo/data/generated/pd_eval100_merged_seed.csv
DOCS_NOSEED=/data/joonhyung/DisPo/data/generated/pd_eval100_merged_noseed.csv
RESULTS=/data/joonhyung/DisPo/eval/merged_8ret_fullcorpus_results.json
LOG=/data/joonhyung/DisPo/logs/merged_8ret_fullcorpus_gpu1.log

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }

# 기존 contriever 결과로 초기화
cat > "$RESULTS" <<'EOF'
[
  {"retriever": "contriever", "seed_nd": 0.94, "seed_rd": 0.84, "noseed_nd": 0.92, "noseed_rd": 0.83}
]
EOF

RETS=(contriever-msmarco dpr ance bge-base e5-base gte-base mpnet)

{
echo "$(ts) ===== merged 8-retriever fullcorpus eval (GPU $GPU) ====="
echo "$(ts) 대상: ${RETS[*]}"
echo ""

TOTAL=${#RETS[@]}
RUN_NUM=0

for RET in "${RETS[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))
    echo ""
    echo "$(ts) [$RUN_NUM/$TOTAL] retriever=$RET"

    # --- seed (adv=7) ---
    echo "$(ts)   [seed] adv=7 시작"
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 $VENV $EVAL \
        --dataset nq \
        --retrieval_model "$RET" \
        --docs_csv "$DOCS_SEED" \
        --adv_per_query 7 \
        --top_k 5 \
        --gpu_id "$GPU" \
        --run_label "merged_seed_${RET}_fullcorpus_val100"
    LATEST_SEED=$(ls -t /data/joonhyung/DisPo/eval/txt_logs_fullcorpus_nq/*/final.json 2>/dev/null | head -1)

    # --- noseed (adv=6) ---
    echo "$(ts)   [noseed] adv=6 시작"
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 $VENV $EVAL \
        --dataset nq \
        --retrieval_model "$RET" \
        --docs_csv "$DOCS_NOSEED" \
        --adv_per_query 6 \
        --top_k 5 \
        --gpu_id "$GPU" \
        --run_label "merged_noseed_${RET}_fullcorpus_val100"
    LATEST_NOSEED=$(ls -t /data/joonhyung/DisPo/eval/txt_logs_fullcorpus_nq/*/final.json 2>/dev/null | head -1)

    # 결과 저장
    $VENV - <<PYEOF
import json
results = json.load(open("$RESULTS"))
js  = json.load(open("$LATEST_SEED"))
jns = json.load(open("$LATEST_NOSEED"))
snd = js["no_defense"]["ASR"];   srd = js["ragdefender"]["ASR"]
nnd = jns["no_defense"]["ASR"];  nrd = jns["ragdefender"]["ASR"]
results.append({"retriever": "$RET",
                "seed_nd": snd, "seed_rd": srd,
                "noseed_nd": nnd, "noseed_rd": nrd})
json.dump(results, open("$RESULTS", "w"), indent=2)
print(f"  seed  → ND={snd*100:.1f}%  RD={srd*100:.1f}%")
print(f"  noseed→ ND={nnd*100:.1f}%  RD={nrd*100:.1f}%")
PYEOF

    echo "$(ts) [$RUN_NUM/$TOTAL] $RET 완료"
done

echo ""
echo "$(ts) ===== 전체 완료 — 최종 결과표 ====="

$VENV - <<'PYEOF'
import json
KEY_ORDER = ["contriever","contriever-msmarco","dpr","ance","bge-base","e5-base","gte-base","mpnet"]
results = json.load(open("/data/joonhyung/DisPo/eval/merged_8ret_fullcorpus_results.json"))
rows = {r["retriever"]: r for r in results}
print(f"\n{'─'*72}")
print(f"  {'검색기':<22}  {'seed ND':>8}  {'seed RD':>8}  {'noseed ND':>10}  {'noseed RD':>10}")
print(f"  {'─'*68}")
snds, srds, nnds, nrds = [], [], [], []
for k in KEY_ORDER:
    r = rows.get(k, {})
    snd  = r.get('seed_nd', None);   srd  = r.get('seed_rd', None)
    nnd  = r.get('noseed_nd', None); nrd  = r.get('noseed_rd', None)
    snd_s  = f"{snd*100:.1f}%"  if snd  is not None else "  -"
    srd_s  = f"{srd*100:.1f}%"  if srd  is not None else "  -"
    nnd_s  = f"{nnd*100:.1f}%"  if nnd  is not None else "  -"
    nrd_s  = f"{nrd*100:.1f}%"  if nrd  is not None else "  -"
    print(f"  {k:<22}  {snd_s:>8}  {srd_s:>8}  {nnd_s:>10}  {nrd_s:>10}")
    if snd is not None: snds.append(snd)
    if srd is not None: srds.append(srd)
    if nnd is not None: nnds.append(nnd)
    if nrd is not None: nrds.append(nrd)
print(f"  {'─'*68}")
def avg(lst): return f"{sum(lst)/len(lst)*100:.1f}%" if lst else "-"
print(f"  {'평균':<22}  {avg(snds):>8}  {avg(srds):>8}  {avg(nnds):>10}  {avg(nrds):>10}")
print(f"{'─'*72}\n")
PYEOF

} 2>&1 | tee "$LOG"
