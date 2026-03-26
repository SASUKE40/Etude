#!/bin/bash
# Prepare the FineWeb-Edu dataset for training.
#
# Usage:
#   bash data/fineweb-edu/prepare.sh

set -e

# Use scratch for HF cache to avoid home directory quota issues
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
mkdir -p "$HF_HOME"

# Output to scratch to avoid home quota
OUTPUT_DIR=${FINEWEB_DATA_DIR:-/scratch/$USER/etude_data/fineweb-edu}

echo "=== Preparing FineWeb-Edu dataset ==="
echo "HF cache: $HF_HOME"
echo "Output: $OUTPUT_DIR"
echo ""

python data/fineweb-edu/prepare.py --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Done! ==="
echo "Files in: $OUTPUT_DIR"
echo ""
echo "To use in training, symlink:"
echo "  ln -sf $OUTPUT_DIR data/fineweb-edu/parquets"
