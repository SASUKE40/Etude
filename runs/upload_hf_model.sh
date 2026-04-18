#!/bin/bash
# Upload a local LitGPT checkpoint folder to a Hugging Face model repo.
#
# Examples:
#   bash runs/upload_hf_model.sh
#   HF_TOKEN=... bash runs/upload_hf_model.sh
#   bash runs/upload_hf_model.sh /path/to/final Edward40/qwen3-0.6b-rust-strandset-sft

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

CHECKPOINT_DIR="${1:-/scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final}"
REPO_ID="${2:-Edward40/etude-base}"
REPO_TYPE="${REPO_TYPE:-model}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_DIR/model.pth" ]]; then
  echo "ERROR: missing model.pth in $CHECKPOINT_DIR" >&2
  exit 1
fi

export CHECKPOINT_DIR
export REPO_ID
export REPO_TYPE

python - <<'PY'
import os
from huggingface_hub import login, upload_folder

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    login()

upload_folder(
    folder_path=os.environ["CHECKPOINT_DIR"],
    repo_id=os.environ["REPO_ID"],
    repo_type=os.environ["REPO_TYPE"],
)
PY
