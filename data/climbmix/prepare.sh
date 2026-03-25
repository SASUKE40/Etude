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

echo "=== Preparing Nemotron-ClimbMix dataset ==="
echo "HF cache: $HF_HOME"
echo ""

# Prepare all 10 parts
for i in $(seq 0 9); do
    echo "--- Preparing part $i ---"
    python data/climbmix/prepare.py --part $i --num-proc 48
    echo ""
done

# Merge into train.bin and val.bin
echo "--- Merging parts ---"
python data/climbmix/merge.py

echo ""
echo "=== Done! ==="
echo "Files: data/climbmix/train.bin, data/climbmix/val.bin"
