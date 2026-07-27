#!/usr/bin/env bash
set -euo pipefail

DISPO_ROOT="${DISPO_ROOT:-/data/joonhyung/DisPo}"
PY="${PY:-/data/joonhyung/ragdef/.venv/bin/python}"
DOCS_CSV="${DOCS_CSV:-$DISPO_ROOT/data/attackbaselines_pd/DiPoison/merged/msmarco_merged_dipoison.csv}"
MSMARCO_CORPUS_PATH="${MSMARCO_CORPUS_PATH:-/data/byungchan/datasets/msmarco/corpus.jsonl}"
EMBED_CACHE_DIR="${EMBED_CACHE_DIR:-$(dirname "$MSMARCO_CORPUS_PATH")}"
CACHE_DIR="${CACHE_DIR:-$DISPO_ROOT/eval/clean_topn_cache/msmarco_merged_val100_top50}"
OUT_JSON="${OUT_JSON:-$DISPO_ROOT/eval/results/msmarco_merged_8ret_fullcorpus/msmarco_merged_dipoison_summary.json}"
GPU_ID="${GPU_ID:-0}"
TOP_KS="${TOP_KS:-5,10}"
ADV_PER_QUERY="${ADV_PER_QUERY:-7}"
ST_BATCH="${ST_BATCH:-32}"
EMBED_BATCH="${EMBED_BATCH:-512}"
RETRIEVERS="${RETRIEVERS:-contriever,e5,ance,bge-base,mpnet,bm25,nomic-v1.5,contriever-msmarco}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "$DISPO_ROOT"

if [[ ! -f "$DOCS_CSV" ]]; then
  echo "[error] DOCS_CSV not found: $DOCS_CSV" >&2
  exit 1
fi

if [[ ! -f "$MSMARCO_CORPUS_PATH" ]]; then
  echo "[error] MSMARCO_CORPUS_PATH not found: $MSMARCO_CORPUS_PATH" >&2
  echo "Set MSMARCO_CORPUS_PATH=/path/to/msmarco/corpus.jsonl and rerun." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR" "$(dirname "$OUT_JSON")"

cache_file_for() {
  case "$1" in
    contriever) echo "contriever_top50.pt" ;;
    e5|e5-base) echo "e5-base_top50.pt" ;;
    ance) echo "ance_top50.pt" ;;
    bge|bge-base) echo "bge-base_top50.pt" ;;
    mpnet) echo "mpnet_top50.pt" ;;
    bm25) echo "bm25_top50.pt" ;;
    nomic|nomic-v1.5) echo "nomic-v1.5_top50.pt" ;;
    cont-ms|contriever-msmarco) echo "contriever-msmarco_top50.pt" ;;
    *) echo "[error] unknown retriever: $1" >&2; return 1 ;;
  esac
}

retrieval_key_for() {
  case "$1" in
    contriever) echo "contriever" ;;
    e5|e5-base) echo "e5-base" ;;
    ance) echo "ance" ;;
    bge|bge-base) echo "bge-base" ;;
    mpnet) echo "mpnet" ;;
    bm25) echo "bm25" ;;
    nomic|nomic-v1.5) echo "nomic-v1.5" ;;
    cont-ms|contriever-msmarco) echo "contriever-msmarco" ;;
    *) echo "[error] unknown retriever: $1" >&2; return 1 ;;
  esac
}

IFS=',' read -r -a RET_ARRAY <<< "$RETRIEVERS"

echo "[config] GPU_ID=$GPU_ID"
echo "[config] DOCS_CSV=$DOCS_CSV"
echo "[config] MSMARCO_CORPUS_PATH=$MSMARCO_CORPUS_PATH"
echo "[config] EMBED_CACHE_DIR=$EMBED_CACHE_DIR"
echo "[config] CACHE_DIR=$CACHE_DIR"
echo "[config] OUT_JSON=$OUT_JSON"
echo "[config] RETRIEVERS=$RETRIEVERS"

for ret in "${RET_ARRAY[@]}"; do
  ret="$(echo "$ret" | xargs)"
  cache_file="$(cache_file_for "$ret")"
  retrieval_key="$(retrieval_key_for "$ret")"
  cache_path="$CACHE_DIR/$cache_file"

  if [[ -f "$cache_path" ]]; then
    echo "[cache] exists: $cache_path"
    continue
  fi

  echo "[cache] building $retrieval_key -> $cache_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" eval/precompute_clean_topn_fullcorpus.py \
    --dataset msmarco \
    --retrieval_model "$retrieval_key" \
    --docs_csv "$DOCS_CSV" \
    --corpus_path "$MSMARCO_CORPUS_PATH" \
    --embed_cache_dir "$EMBED_CACHE_DIR" \
    --top_n 50 \
    --gpu_id 0 \
    --embed_batch "$EMBED_BATCH" \
    --st_batch "$ST_BATCH" \
    --output "$cache_path"
done

echo "[eval] starting MSMARCO merged 8-retriever eval"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" eval/multi_retriever_ragdef_eval.py \
  --dataset msmarco \
  --docs_csv "$DOCS_CSV" \
  --corpus_path "$MSMARCO_CORPUS_PATH" \
  --cache_dir "$CACHE_DIR" \
  --out_json "$OUT_JSON" \
  --retrievers "$RETRIEVERS" \
  --top_ks "$TOP_KS" \
  --adv_per_query "$ADV_PER_QUERY" \
  --gpu_id 0
