#!/bin/bash

# Prepare NVIDIA Nemotron Cascade SFT Stage 2 for chat SFT.
#
# Usage:
#   bash data/nemotron-cascade-sft-stage-2/prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
OUTPUT_DIR=${1:-/scratch/$USER/etude/datasets/nemotron-cascade-sft-stage-2}

mkdir -p "$HF_HOME"
mkdir -p "$OUTPUT_DIR"

echo "=== Preparing Nemotron Cascade SFT Stage 2 ==="
echo "HF_HOME: $HF_HOME"
echo "OUTPUT_DIR: $OUTPUT_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    exec uv run python data/nemotron-cascade-sft-stage-2/prepare.py --output-dir "$OUTPUT_DIR"
else
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" data/nemotron-cascade-sft-stage-2/prepare.py --output-dir "$OUTPUT_DIR"
