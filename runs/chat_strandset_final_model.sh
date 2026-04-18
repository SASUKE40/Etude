#!/bin/bash
# Chat with the final Strandset-Rust LitGPT checkpoint via a direct model.pth path.
#
# Examples:
#   bash runs/chat_strandset_final_model.sh
#   bash runs/chat_strandset_final_model.sh /scratch/$USER/.../out/final/model.pth

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

CHECKPOINT_PATH="${1:-/scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/model.pth}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B}"

python scripts/litgpt_infer_checkpoint.py chat \
  "$CHECKPOINT_PATH" \
  --base-checkpoint-dir "$BASE_CHECKPOINT_DIR"
