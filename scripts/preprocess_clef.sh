#!/bin/bash
# CLEF Preprocessing Script
# Usage: bash scripts/preprocess_clef.sh [data_dir]

set -e

DATA_ROOT="${1:-./data/clef}"

# Set CLEF_HOME to E0008 subdirectory if it exists
[ -d "$DATA_ROOT/E0008" ] && export CLEF_HOME="$DATA_ROOT/E0008" || export CLEF_HOME="$DATA_ROOT"
echo "Using CLEF_HOME=$CLEF_HOME"

DOC_LANGS="en"
DOC_YEARS="2003"
REL_LANGS="en"
REL_YEARS="2000 2001 2002 2003"
QUERY_LANGS="de fr fi so sw en"
QUERY_YEARS="2000 2001 2002 2003"
PASSAGE_LENGTH=180
OVERLAP=90
TOKENIZER="bert-base-multilingual-uncased"
NUM_PROCESSES=4

python -m src.preprocessing.preprocess_clef \
        --doc-langs $DOC_LANGS \
        --doc-years $DOC_YEARS \
        --rel-langs $REL_LANGS \
        --rel-years $REL_YEARS \
        --query-langs $QUERY_LANGS \
        --query-years $QUERY_YEARS \
        --output-dir "$DATA_ROOT" \
        --aggregate-queries \
        --split-passages \
        --passage-length "$PASSAGE_LENGTH" \
        --overlap "$OVERLAP" \
        --tokenizer "$TOKENIZER" \
        --num-processes "$NUM_PROCESSES"
