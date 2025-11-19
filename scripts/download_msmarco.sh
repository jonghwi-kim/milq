#!/bin/bash
# Download training data resources for MSMarco multi-lingual retrieval
# Usage: bash scripts/download_msmarco.sh [data_dir]

set -e

DATA_ROOT="${1:-./data/msmarco}"

mkdir -p "$DATA_ROOT"
echo "Downloading to: $DATA_ROOT"
echo ""

# English MSMarco (TREC DL 2019 Passage Ranking)
echo "[1/4] English MSMarco..."
mkdir -p "$DATA_ROOT/english"

# Collection
[ ! -f "$DATA_ROOT/english/english_collection.tsv" ] && \
    wget -q --show-progress -O "$DATA_ROOT/english/collection.tar.gz" \
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz" && \
    tar -xzf "$DATA_ROOT/english/collection.tar.gz" -C "$DATA_ROOT/english/" && \
    mv "$DATA_ROOT/english/collection.tsv" "$DATA_ROOT/english/english_collection.tsv" && \
    rm "$DATA_ROOT/english/collection.tar.gz"

# Queries
[ ! -f "$DATA_ROOT/english/english_queries.tsv" ] && \
    wget -q --show-progress -O "$DATA_ROOT/english/queries.tar.gz" \
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz" && \
    tar -xzf "$DATA_ROOT/english/queries.tar.gz" -C "$DATA_ROOT/english/" && \
    mv "$DATA_ROOT/english/queries.train.tsv" "$DATA_ROOT/english/english_queries.tsv" && \
    rm "$DATA_ROOT/english/queries.tar.gz"

# mMARCO (French, German)
echo "[2/4] mMARCO (French, German)..."
for lang in french german; do
    mkdir -p "$DATA_ROOT/$lang"
    
    [ ! -f "$DATA_ROOT/$lang/${lang}_collection.tsv" ] && \
        wget -q --show-progress -O "$DATA_ROOT/$lang/${lang}_collection.tsv" \
        "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/${lang}_collection.tsv"
    
    [ ! -f "$DATA_ROOT/$lang/${lang}_queries.tsv" ] && \
        wget -q --show-progress -O "$DATA_ROOT/$lang/${lang}_queries.tsv" \
        "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/${lang}_queries.train.tsv"
done

# NeuMARCO (Farsi, Chinese, Russian)
echo "[3/4] NeuMARCO (Farsi, Chinese, Russian)..."

# Download and extract NeuMARCO archive (temporary extraction)
TEMP_DIR=$(mktemp -d)
if [ ! -f "$DATA_ROOT/farsi/farsi_collection.tsv" ] || [ ! -f "$DATA_ROOT/chinese/chinese_collection.tsv" ] || [ ! -f "$DATA_ROOT/russian/russian_collection.tsv" ]; then
    echo "Downloading NeuMARCO collections..."
    wget -q --show-progress -O "$TEMP_DIR/neumarco.tar.gz" \
        "https://huggingface.co/datasets/neuclir/neumarco/resolve/main/data/neumarco.tar.gz"
    tar -xzf "$TEMP_DIR/neumarco.tar.gz" -C "$TEMP_DIR/"
fi

# Farsi
mkdir -p "$DATA_ROOT/farsi"
[ ! -f "$DATA_ROOT/farsi/farsi_collection.tsv" ] && \
    cp "$TEMP_DIR/eng-fas/msmarco.collection.20210731-scale21-sockeye2-tm1.tsv" "$DATA_ROOT/farsi/farsi_collection.tsv"

[ ! -f "$DATA_ROOT/farsi/farsi_queries.tsv" ] && \
    wget -q --show-progress -O "$DATA_ROOT/farsi/farsi_queries.tsv.gz" \
    "https://huggingface.co/datasets/hltcoe/tdist-msmarco-scores/resolve/main/msmarco.train.query.fas.tsv.gz" && \
    gunzip -f "$DATA_ROOT/farsi/farsi_queries.tsv.gz"

# Chinese
mkdir -p "$DATA_ROOT/chinese"
[ ! -f "$DATA_ROOT/chinese/chinese_collection.tsv" ] && \
    cp "$TEMP_DIR/eng-zho/msmarco.collection.20210731-scale21-sockeye2-tm1.tsv" "$DATA_ROOT/chinese/chinese_collection.tsv"

[ ! -f "$DATA_ROOT/chinese/chinese_queries.tsv" ] && \
    wget -q --show-progress -O "$DATA_ROOT/chinese/chinese_queries.tsv" \
    "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/chinese_queries.train.tsv"

# Russian
mkdir -p "$DATA_ROOT/russian"
[ ! -f "$DATA_ROOT/russian/russian_collection.tsv" ] && \
    cp "$TEMP_DIR/eng-rus/msmarco.collection.20210731-scale21-sockeye2-tm1.tsv" "$DATA_ROOT/russian/russian_collection.tsv"

[ ! -f "$DATA_ROOT/russian/russian_queries.tsv" ] && \
    wget -q --show-progress -O "$DATA_ROOT/russian/russian_queries.tsv" \
    "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/russian_queries.train.tsv"

# Clean up temporary directory
[ -d "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"

# Teacher Scores
echo "[4/4] Teacher scores..."
mkdir -p "$DATA_ROOT/teacher-scores"

[ ! -f "$DATA_ROOT/teacher-scores/mt5xxl-monot5-mmarco-engeng.jsonl" ] && \
    wget -q --show-progress -O "$DATA_ROOT/teacher-scores/mt5xxl-monot5-mmarco-engeng.jsonl.gz" \
    "https://huggingface.co/datasets/hltcoe/tdist-msmarco-scores/resolve/main/mt5xxl-monot5-mmarco-engeng.jsonl.gz" && \
    gunzip "$DATA_ROOT/teacher-scores/mt5xxl-monot5-mmarco-engeng.jsonl.gz"

echo ""
echo "Done. Data saved to: $DATA_ROOT"
