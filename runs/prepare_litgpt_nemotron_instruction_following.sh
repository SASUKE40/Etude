#!/bin/bash
# Prepare the Nemotron instruction-following dataset for LitGPT full finetuning.
#
# Example:
#   bash runs/prepare_litgpt_nemotron_instruction_following.sh
#   FORCE_PREPARE=1 MAX_ROWS=1000 bash runs/prepare_litgpt_nemotron_instruction_following.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

SFT_ROOT="${SFT_ROOT:-/scratch/$USER/litgpt-nemotron-sft}"
RUN_TAG="${RUN_TAG:-qwen3-0.6b-rust-step-00001800-nemotron-instruction-following}"
WORK_ROOT="$SFT_ROOT/$RUN_TAG"
DATA_URL="${DATA_URL:-https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2/blob/main/instruction-following.jsonl}"
RAW_DATA_PATH="${RAW_DATA_PATH:-$WORK_ROOT/data/instruction-following.jsonl}"
PREPARED_JSON_PATH="${PREPARED_JSON_PATH:-$WORK_ROOT/data/instruction-following-litgpt.json}"
MAX_ROWS="${MAX_ROWS:--1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

mkdir -p "$WORK_ROOT/data"

echo "Repo root: $REPO_ROOT"
echo "Working directory: $WORK_ROOT"
echo "Raw dataset path: $RAW_DATA_PATH"
echo "Prepared JSON path: $PREPARED_JSON_PATH"

PREPARE_ARGS=()
if [[ "$FORCE_PREPARE" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
if [[ "$MAX_ROWS" != "-1" ]]; then
  PREPARE_ARGS+=(--max-rows "$MAX_ROWS")
fi

python data/nemotron-cascade-sft-stage-2/prepare_litgpt_instruction_following.py \
  --source-url "$DATA_URL" \
  --download-path "$RAW_DATA_PATH" \
  --output-path "$PREPARED_JSON_PATH" \
  "${PREPARE_ARGS[@]}"
