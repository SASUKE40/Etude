#!/bin/bash
# Prepare the Rust subset of The Stack Dedup for training.
# This downloads, tokenizes, and writes train.bin and val.bin.
#
# Usage:
#   bash data/rust/prepare.sh

set -e

echo "=== Preparing Rust dataset from The Stack Dedup ==="
echo ""

python data/rust/prepare.py

echo ""
echo "=== Done! ==="
echo "Files: data/rust/train.bin, data/rust/val.bin"
