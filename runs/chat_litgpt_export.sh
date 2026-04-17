#!/bin/bash
# Chat with a self-contained exported LitGPT checkpoint directory.
#
# Example:
#   bash runs/chat_litgpt_export.sh
#   bash runs/chat_litgpt_export.sh /scratch/$USER/exports/qwen3-0.6b-rust

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

CHECKPOINT_DIR="${1:-/scratch/$USER/exports/qwen3-0.6b-rust}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B}"

if [[ -f "$CHECKPOINT_DIR/model.pth" && ! -f "$CHECKPOINT_DIR/lit_model.pth" ]]; then
  cp -n "$CHECKPOINT_DIR/model.pth" "$CHECKPOINT_DIR/lit_model.pth"
fi

python scripts/litgpt_infer_checkpoint.py chat \
  "$CHECKPOINT_DIR" \
  --base-checkpoint-dir "$BASE_CHECKPOINT_DIR" \
  --multiline
