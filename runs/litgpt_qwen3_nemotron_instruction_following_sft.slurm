#!/bin/bash
# LitGPT chat finetune launcher for Qwen3-0.6B continued-pretraining checkpoints.
#
# Default source checkpoint:
#   /scratch/zhu.shili/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-00001800
#
# Default dataset source:
#   https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2/blob/main/instruction-following.jsonl
#
# Example:
#   sbatch runs/litgpt_qwen3_nemotron_instruction_following_sft.slurm
#   SOURCE_STEP_DIR=/scratch/$USER/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-00002000 \
#   RUN_TAG=qwen3-0.6b-rust-step-2000-nemotron-if \
#   LITGPT_FINETUNE_EXTRA_ARGS="--precision bf16-true" \
#   sbatch runs/litgpt_qwen3_nemotron_instruction_following_sft.slurm

#SBATCH --job-name=litgpt-sft
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/litgpt-sft-%j.log

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-etude-litgpt-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

if ! command -v litgpt >/dev/null 2>&1; then
  echo "ERROR: litgpt is not on PATH." >&2
  echo "Activate the correct environment before submitting, or install LitGPT in .venv." >&2
  exit 1
fi

SOURCE_STEP_DIR="${SOURCE_STEP_DIR:-/scratch/zhu.shili/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-00001800}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B}"
DATA_URL="${DATA_URL:-https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2/blob/main/instruction-following.jsonl}"
SFT_ROOT="${SFT_ROOT:-/scratch/$USER/litgpt-nemotron-sft}"
RUN_TAG="${RUN_TAG:-qwen3-0.6b-rust-step-00001800-nemotron-instruction-following}"
WORK_ROOT="$SFT_ROOT/$RUN_TAG"
RAW_DATA_PATH="${RAW_DATA_PATH:-$WORK_ROOT/data/instruction-following.jsonl}"
PREPARED_JSON_PATH="${PREPARED_JSON_PATH:-$WORK_ROOT/data/instruction-following-litgpt.json}"
STAGED_CHECKPOINT_DIR="${STAGED_CHECKPOINT_DIR:-$WORK_ROOT/checkpoint}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/out}"
VAL_SPLIT_FRACTION="${VAL_SPLIT_FRACTION:-0.005}"
MAX_ROWS="${MAX_ROWS:--1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
LITGPT_FINETUNE_EXTRA_ARGS="${LITGPT_FINETUNE_EXTRA_ARGS:-}"

export SOURCE_STEP_DIR
export BASE_CHECKPOINT_DIR
export STAGED_CHECKPOINT_DIR

mkdir -p "$WORK_ROOT" "$WORK_ROOT/data" "$STAGED_CHECKPOINT_DIR" "$OUT_DIR"

if [[ ! -f "$SOURCE_STEP_DIR/lit_model.pth" ]]; then
  echo "ERROR: source step directory does not contain lit_model.pth: $SOURCE_STEP_DIR" >&2
  exit 1
fi

if [[ ! -d "$BASE_CHECKPOINT_DIR" ]]; then
  echo "ERROR: base checkpoint directory does not exist: $BASE_CHECKPOINT_DIR" >&2
  exit 1
fi

echo "Repo root: $REPO_ROOT"
echo "Source step dir: $SOURCE_STEP_DIR"
echo "Base checkpoint dir: $BASE_CHECKPOINT_DIR"
echo "Working directory: $WORK_ROOT"
echo "Prepared JSON path: $PREPARED_JSON_PATH"
echo "Output dir: $OUT_DIR"

python - <<'PY'
import os
from pathlib import Path

metadata_files = (
    "config.json",
    "generation_config.json",
    "model_config.yaml",
    "prompt_style.yaml",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)

source_step_dir = Path(os.environ["SOURCE_STEP_DIR"]).expanduser().resolve()
base_checkpoint_dir = Path(os.environ["BASE_CHECKPOINT_DIR"]).expanduser().resolve()
staged_checkpoint_dir = Path(os.environ["STAGED_CHECKPOINT_DIR"]).expanduser().resolve()
staged_checkpoint_dir.mkdir(parents=True, exist_ok=True)

for filename in ("lit_model.pth", "hyperparameters.yaml"):
    source = source_step_dir / filename
    if not source.exists():
        continue
    target = staged_checkpoint_dir / filename
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)

for filename in metadata_files:
    source = base_checkpoint_dir / filename
    if not source.exists():
        continue
    target = staged_checkpoint_dir / filename
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)

print(f"Staged LitGPT checkpoint in {staged_checkpoint_dir}")
PY

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

CMD=(
  litgpt finetune "$STAGED_CHECKPOINT_DIR"
  --data JSON
  --data.json_path "$PREPARED_JSON_PATH"
  --data.val_split_fraction "$VAL_SPLIT_FRACTION"
  --out_dir "$OUT_DIR"
)

if [[ -n "$LITGPT_FINETUNE_EXTRA_ARGS" ]]; then
  # Intentional word splitting: this env var is for additional LitGPT CLI args.
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $LITGPT_FINETUNE_EXTRA_ARGS )
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
