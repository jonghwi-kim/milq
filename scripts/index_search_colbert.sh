#!/bin/bash
# Index and search with ColBERT-X
# Usage: bash scripts/index_search_colbert.sh <query_lang> <doc_lang>

set -e

QUERY_LANG="$1"
DOC_LANG="$2"

[ -z "$QUERY_LANG" ] || [ -z "$DOC_LANG" ] && {
    echo "Usage: bash scripts/index_search_colbert.sh <query_lang> <doc_lang>"
    exit 1
}

DATA_DIR="./data/neuclir"
COLLECTION="${DATA_DIR}/${DOC_LANG}/${DOC_LANG}_passages_xlm-roberta-large_180_90.tsv"
QUERIES="${DATA_DIR}/${DOC_LANG}/${QUERY_LANG}_queries_concat.tsv"
MAPPING="${DATA_DIR}/${DOC_LANG}/${DOC_LANG}_mapping_xlm-roberta-large_180_90.tsv"
QRELS="${DATA_DIR}/${DOC_LANG}/2022-qrels.${DOC_LANG}"
CHECKPOINT="bert-base-multilingual-uncased"
CHECKPOINT_NAME=$(basename "$CHECKPOINT" | sed 's/[\/:]/_/g')
INDEX_NAME="${DOC_LANG}_${CHECKPOINT_NAME}_1bits"
NUM_GPUS=1

# Index
for STEP in prepare encode finalize; do
    python -m colbert.scripts.index \
        --dataset_name "neuclir_${DOC_LANG}" \
        --coll_dir "$COLLECTION" \
        --index_name "$INDEX_NAME" \
        --nbits 1 \
        --doc_maxlen 180 \
        --checkpoint "$CHECKPOINT" \
        --experiment milq \
        --step "$STEP" \
        --n_gpus "$NUM_GPUS"
done

# Search
python -m colbert.scripts.search \
    --index_name "$INDEX_NAME" \
    --passage_mapping "$MAPPING" \
    --query_file "$QUERIES" \
    --metrics nDCG@20 MAP R@100 \
    --qrel "$QRELS" \
    --experiment milq \
    --n_gpus "$NUM_GPUS"
