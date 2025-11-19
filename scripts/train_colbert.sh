#!/bin/bash
# Train ColBERT-X
# Usage: bash scripts/train_colbert.sh <query_lang> <doc_lang>

set -e

QUERY_LANG="$1"
DOC_LANG="$2"

[ -z "$QUERY_LANG" ] || [ -z "$DOC_LANG" ] && {
    echo "Usage: bash scripts/train_colbert.sh <query_lang> <doc_lang>"
    echo "Examples:"
    echo "  bash scripts/train_colbert.sh english chinese           # Chinese queries"
    echo "  bash scripts/train_colbert.sh english_chinese chinese   # Code-switched queries"
    exit 1
}

# Handle code-switched queries
if [[ "$QUERY_LANG" == *_* ]]; then
    QUERY_DIR=$(echo "$QUERY_LANG" | cut -d'_' -f2)
    QUERY_FILE="${QUERY_LANG}_queries.tsv"
else
    QUERY_DIR="$QUERY_LANG"
    QUERY_FILE="${QUERY_LANG}_queries.tsv"
fi

DATA_ROOT="./data/msmarco"
TRAINING_TRIPLES="$DATA_ROOT/teacher-scores/mt5xxl-monot5-mmarco-engeng.jsonl"
TRAINING_QUERIES="$DATA_ROOT/$QUERY_DIR/$QUERY_FILE"
TRAINING_COLLECTION="$DATA_ROOT/$DOC_LANG/${DOC_LANG}_collection.tsv"
NUM_GPUS=4
MODEL_NAME="xlm-roberta-large"
MAX_STEPS=20000
LEARNING_RATE=5e-6
KD_LOSS="KLD"
GLOBAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
NWAY=6
RUN_TAG="${QUERY_LANG}_${DOC_LANG}"

python -m colbert.scripts.train \
    --model_name "$MODEL_NAME" \
    --training_triples "$TRAINING_TRIPLES" \
    --training_queries "$TRAINING_QUERIES" \
    --training_collection "$TRAINING_COLLECTION" \
    --maxsteps "$MAX_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --kd_loss "$KD_LOSS" \
    --only_top \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --nway "$NWAY" \
    --n_gpus "$NUM_GPUS" \
    --run_tag "$RUN_TAG" \
    --experiment "milq"
