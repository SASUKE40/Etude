#!/bin/bash
# Prepare the Rust subset of The Stack Dedup for training.
#
# Usage:
#   bash data/rust/prepare.sh

set -e

# Use scratch for HF cache to avoid home directory quota issues
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
mkdir -p "$HF_HOME"

# Output to scratch to avoid home quota
OUTPUT_DIR=${RUST_DATA_DIR:-/scratch/$USER/etude_data/rust}

echo "=== Preparing Rust dataset from The Stack Dedup ==="
echo "HF cache: $HF_HOME"
echo "Output: $OUTPUT_DIR"
echo ""

python data/rust/prepare.py --output-dir "$OUTPUT_DIR" --num-proc 48

echo ""
echo "=== Done! ==="
echo "Files: $OUTPUT_DIR/train.bin, $OUTPUT_DIR/val.bin"
echo ""
echo "To use in training, symlink or pass the path:"
echo "  ln -sf $OUTPUT_DIR/train.bin data/rust/train.bin"
echo "  ln -sf $OUTPUT_DIR/val.bin data/rust/val.bin"
