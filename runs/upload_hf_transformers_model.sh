#!/bin/bash
# Upload a converted Hugging Face Transformers model folder to the Hugging Face Hub.
#
# Examples:
#   bash runs/upload_hf_transformers_model.sh
#   bash runs/upload_hf_transformers_model.sh /scratch/$USER/.../out/final/hf Edward40/qwen3-0.6b-rust-strandset
#   HF_TOKEN=... bash runs/upload_hf_transformers_model.sh /path/to/hf_dir Edward40/my-model

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

MODEL_DIR="${1:-/scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/hf}"
REPO_ID="${2:-Edward40/etude-base}"
REPO_TYPE="${REPO_TYPE:-model}"
PRIVATE="${PRIVATE:-false}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload converted Transformers checkpoint}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: missing config.json in $MODEL_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/model.safetensors" && ! -f "$MODEL_DIR/pytorch_model.bin" && ! -f "$MODEL_DIR/model.safetensors.index.json" && ! -f "$MODEL_DIR/pytorch_model.bin.index.json" ]]; then
  echo "ERROR: no Hugging Face weight file found in $MODEL_DIR" >&2
  exit 1
fi

export MODEL_DIR
export REPO_ID
export REPO_TYPE
export PRIVATE
export COMMIT_MESSAGE

python - <<'PY'
import os
from huggingface_hub import HfApi, HfFolder, login, upload_folder

repo_id = os.environ["REPO_ID"]
repo_type = os.environ["REPO_TYPE"]
model_dir = os.environ["MODEL_DIR"]
private = os.environ["PRIVATE"].lower() == "true"
commit_message = os.environ["COMMIT_MESSAGE"]

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
else:
    saved_token = HfFolder.get_token()
    if not saved_token:
        raise SystemExit(
            "HF_TOKEN is not set and no cached Hugging Face login was found.\n"
            "Run `hf auth login` first or re-run with `HF_TOKEN=... bash runs/upload_hf_transformers_model.sh`."
        )

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message=commit_message,
)

print(f"Uploaded {model_dir} to https://huggingface.co/{repo_id}")
PY
