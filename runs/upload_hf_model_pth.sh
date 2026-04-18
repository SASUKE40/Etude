#!/bin/bash
# Upload a single model.pth file to a Hugging Face model repo.
#
# Examples:
#   bash runs/upload_hf_model_pth.sh
#   bash runs/upload_hf_model_pth.sh /scratch/zhu.shili/.../model.pth Edward40/etude
#   HF_TOKEN=... bash runs/upload_hf_model_pth.sh /path/to/model.pth Edward40/etude checkpoints/step-003200/model.pth

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

MODEL_PATH="${1:-/scratch/zhu.shili/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/step-003200/hf/model.pth}"
REPO_ID="${2:-Edward40/etude}"
PATH_IN_REPO="${3:-model.pth}"
REPO_TYPE="${REPO_TYPE:-model}"
PRIVATE="${PRIVATE:-false}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload model.pth from step-003200}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model file does not exist: $MODEL_PATH" >&2
  exit 1
fi

export MODEL_PATH
export REPO_ID
export PATH_IN_REPO
export REPO_TYPE
export PRIVATE
export COMMIT_MESSAGE

python - <<'PY'
import os
from huggingface_hub import HfApi, HfFolder, login, upload_file

model_path = os.environ["MODEL_PATH"]
repo_id = os.environ["REPO_ID"]
path_in_repo = os.environ["PATH_IN_REPO"]
repo_type = os.environ["REPO_TYPE"]
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
            "Run `hf auth login` first or re-run with `HF_TOKEN=... bash runs/upload_hf_model_pth.sh`."
        )

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

url = upload_file(
    path_or_fileobj=model_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message=commit_message,
)

print(f"Uploaded {model_path} to {url}")
PY
