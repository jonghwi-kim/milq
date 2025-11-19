#!/bin/bash
# Translate queries to target language using NLLB-200
# Usage: bash scripts/translate.sh <target_lang> <target_lang_code>

set -e

TARGET_LANG="$1"
TARGET_LANG_CODE="$2"

[ -z "$TARGET_LANG" ] || [ -z "$TARGET_LANG_CODE" ] && {
    echo "Usage: bash scripts/translate.sh <target_lang> <target_lang_code>"
    exit 1
}

DATA_ROOT="./data/msmarco"
INPUT_FILE="$DATA_ROOT/english/english_queries.tsv"
INPUT_FILE=/home/jonghwikim/workspace/milq/data/neuclir/zho/eng_queries_concat.tsv
OUTPUT_DIR="$DATA_ROOT/$TARGET_LANG"
NUM_GPUS=1
MODEL_NAME="facebook/nllb-200-3.3B"
BATCH_SIZE=256
NUM_WORKERS=4
NUM_BEAMS=5
MAX_SEQ_LENGTH=64

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=$NUM_GPUS -m src.preprocessing.translate \
    --model_name_or_path "$MODEL_NAME" \
    --source_language eng_Latn \
    --target_language "$TARGET_LANG_CODE" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_beams "$NUM_BEAMS" \
    --max_seq_length "$MAX_SEQ_LENGTH"
