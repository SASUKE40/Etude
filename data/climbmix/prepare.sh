#!/bin/bash
# Prepare the Nemotron-ClimbMix dataset for training.
# This downloads, tokenizes, and merges the data into train.bin and val.bin.
#
# Usage:
#   bash data/climbmix/prepare.sh
#
# For a quick test with just one part:
#   python data/climbmix/prepare.py --part 0
#   python data/climbmix/merge.py

set -e

# Use scratch for HF cache to avoid home directory quota issues
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
mkdir -p "$HF_HOME"

# Output to scratch to avoid home quota
OUTPUT_DIR=${CLIMBMIX_DATA_DIR:-/scratch/$USER/etude_data/climbmix}

echo "=== Preparing Nemotron-ClimbMix dataset ==="
echo "HF cache: $HF_HOME"
echo "Output: $OUTPUT_DIR"
echo ""

# Prepare all 10 parts
for i in $(seq 0 9); do
    echo "--- Preparing part $i ---"
    python data/climbmix/prepare.py --part $i --output-dir "$OUTPUT_DIR" --num-proc 48
    echo ""
done

# Merge into train.bin and val.bin
echo "--- Merging parts ---"
python data/climbmix/merge.py --data-dir "$OUTPUT_DIR"

echo ""
echo "=== Done! ==="
echo "Files: $OUTPUT_DIR/train.bin, $OUTPUT_DIR/val.bin"
echo ""
echo "To use in training, symlink:"
echo "  ln -sf $OUTPUT_DIR/train.bin data/climbmix/train.bin"
echo "  ln -sf $OUTPUT_DIR/val.bin data/climbmix/val.bin"
