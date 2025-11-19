#!/bin/bash
# Apply code-switching to queries
# Usage: bash scripts/codeswitch.sh <source_lang> <target_lang> <lexicon_name>

set -e

SOURCE_LANG="$1"
TARGET_LANG="$2"
LEXICON_NAME="$3"

[ -z "$SOURCE_LANG" ] || [ -z "$TARGET_LANG" ] || [ -z "$LEXICON_NAME" ] && {
    echo "Usage: bash scripts/codeswitch.sh <source_lang> <target_lang> <lexicon_name>"
    exit 1
}

DATA_ROOT="./data"
LEXICON_DIR="$DATA_ROOT/lexicon"
INPUT_FILE="$DATA_ROOT/msmarco/$SOURCE_LANG/${SOURCE_LANG}_queries.tsv"
OUTPUT_DIR="$DATA_ROOT/msmarco/$TARGET_LANG"
OUTPUT_FILE="$OUTPUT_DIR/${SOURCE_LANG}_${TARGET_LANG}_queries.tsv"
PROBABILITY=0.5
SEED=42

mkdir -p "$OUTPUT_DIR"

python -m src.preprocessing.codeswitch \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --lexicon_name "$LEXICON_NAME" \
    --lexicon_dir "$LEXICON_DIR" \
    --probability "$PROBABILITY" \
    --seed "$SEED"
