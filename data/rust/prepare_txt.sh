#!/bin/bash
# Download the cleaned Rust train split as sharded .txt files under /scratch.
#
# Usage:
#   bash data/rust/prepare_txt.sh

set -euo pipefail

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
mkdir -p "$HF_HOME"

OUTPUT_DIR=${RUST_TXT_DATA_DIR:-/scratch/$USER/the-stack-rust-clean/train_txt}
ROWS_PER_SHARD=${ROWS_PER_SHARD:-10000}

echo "=== Downloading Rust train split as .txt ==="
echo "Dataset: ammarnasr/the-stack-rust-clean"
echo "HF cache: $HF_HOME"
echo "Output: $OUTPUT_DIR"
echo "Rows per shard: $ROWS_PER_SHARD"
echo ""

python data/rust/prepare_txt.py \
  --output-dir "$OUTPUT_DIR" \
  --rows-per-shard "$ROWS_PER_SHARD"

echo ""
echo "=== Done! ==="
echo "Text shards: $OUTPUT_DIR/train_*.txt"
echo "Metadata: $OUTPUT_DIR/metadata.json"
