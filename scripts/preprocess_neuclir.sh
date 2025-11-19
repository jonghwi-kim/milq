#!/bin/bash
# NeuCLIR Preprocessing Script
# Usage: bash scripts/preprocess_neuclir.sh [--download] [data_dir]

set -e

DATA_ROOT="${2:-./data/neuclir}"

LANGUAGES=(fas rus zho)
TOPICS_FILE="topics.0720.utf8.jsonl"
PASSAGE_LENGTH=180
OVERLAP=90
TOKENIZER="xlm-roberta-large"
NUM_PROCESSES=4

# Download data if --download flag is set
if [[ "$1" == "--download" ]]; then
    mkdir -p "$DATA_ROOT"
    [ ! -f "$DATA_ROOT/$TOPICS_FILE" ] && curl -L -o "$DATA_ROOT/$TOPICS_FILE" "https://trec.nist.gov/data/neuclir/2022/$TOPICS_FILE"
    
    for lang in "${LANGUAGES[@]}"; do
        mkdir -p "$DATA_ROOT/$lang"
        [ ! -f "$DATA_ROOT/$lang/2022-qrels.$lang" ] && curl -L -o "$DATA_ROOT/$lang/2022-qrels.$lang" "https://trec.nist.gov/data/neuclir/2022/2022-qrels.$lang"
    done
    
    [ ! -f "$DATA_ROOT/neuclir1.tar.gz" ] && curl -L -o "$DATA_ROOT/neuclir1.tar.gz" "https://ir.nist.gov/neuclir/neuclir1.tar.gz"
    [ ! -d "$DATA_ROOT/neuclir1" ] && tar -xzf "$DATA_ROOT/neuclir1.tar.gz" -C "$DATA_ROOT"
    
    for lang in "${LANGUAGES[@]}"; do
        [ -f "$DATA_ROOT/neuclir1/$lang/docs.jsonl" ] && python -m src.preprocessing.fix_document_order \
            --raw_download_file "$DATA_ROOT/neuclir1/$lang/docs.jsonl" \
            --id_file "$DATA_ROOT/neuclir1/resource/$lang/ids.*.jsonl.gz" \
            --check_hash
    done
fi

# Prepare file paths
QRELS_FILES=()
DOCS_FILES=()
for lang in "${LANGUAGES[@]}"; do
    QRELS_FILES+=("$DATA_ROOT/$lang/2022-qrels.$lang")
    DOCS_FILES+=("$DATA_ROOT/$lang/docs.jsonl")
done

python -m src.preprocessing.preprocess_neuclir \
        --input-dir "$DATA_ROOT" \
        --output-dir "$DATA_ROOT" \
        --topics-file "$TOPICS_FILE" \
        --qrels-file "${QRELS_FILES[@]}" \
        --docs-file "${DOCS_FILES[@]}" \
        --split-passages \
        --passage-length "$PASSAGE_LENGTH" \
        --overlap "$OVERLAP" \
        --tokenizer "$TOKENIZER" \
        --num-processes "$NUM_PROCESSES"
