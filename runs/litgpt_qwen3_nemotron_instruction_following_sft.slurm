#!/bin/bash
# LitGPT full-parameter chat finetune launcher for Qwen3-0.6B continued-pretraining checkpoints.
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
#   PRECISION=bf16-true TRAIN_EPOCHS=3 \
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
PRECISION="${PRECISION:-bf16-true}"
DEVICES="${DEVICES:-1}"
NUM_NODES="${NUM_NODES:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_MAX_ITERS="${EVAL_MAX_ITERS:-50}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-100}"
LOGGER_NAME="${LOGGER_NAME:-csv}"
SEED="${SEED:-1337}"

export SOURCE_STEP_DIR
export BASE_CHECKPOINT_DIR
export STAGED_CHECKPOINT_DIR
export PREPARED_JSON_PATH
export OUT_DIR
export VAL_SPLIT_FRACTION
export PRECISION
export DEVICES
export NUM_NODES
export TRAIN_EPOCHS
export TRAIN_MAX_STEPS
export GLOBAL_BATCH_SIZE
export MICRO_BATCH_SIZE
export MAX_SEQ_LENGTH
export SAVE_INTERVAL
export LOG_INTERVAL
export EVAL_INTERVAL
export EVAL_MAX_ITERS
export EVAL_MAX_NEW_TOKENS
export LOGGER_NAME
export SEED

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

python - <<'PY'
import os
from pathlib import Path

try:
    from litgpt.finetune.full import setup as full_setup
except ImportError as exc:
    raise SystemExit(
        "Your installed LitGPT does not expose litgpt.finetune.full. "
        "Install a LitGPT build that supports full finetuning."
    ) from exc

try:
    from litgpt.data import JSON
except ImportError:
    from litgpt.data.json_data import JSON

from litgpt.args import EvalArgs, TrainArgs

checkpoint_dir = Path(os.environ["STAGED_CHECKPOINT_DIR"]).expanduser().resolve()
json_path = Path(os.environ["PREPARED_JSON_PATH"]).expanduser().resolve()
out_dir = Path(os.environ["OUT_DIR"]).expanduser().resolve()

train_max_steps_raw = os.environ.get("TRAIN_MAX_STEPS", "").strip()
train_max_steps = int(train_max_steps_raw) if train_max_steps_raw else None

data = JSON(
    json_path=json_path,
    val_split_fraction=float(os.environ["VAL_SPLIT_FRACTION"]),
)

train = TrainArgs(
    save_interval=int(os.environ["SAVE_INTERVAL"]),
    log_interval=int(os.environ["LOG_INTERVAL"]),
    global_batch_size=int(os.environ["GLOBAL_BATCH_SIZE"]),
    micro_batch_size=int(os.environ["MICRO_BATCH_SIZE"]),
    epochs=int(os.environ["TRAIN_EPOCHS"]),
    max_steps=train_max_steps,
    max_seq_length=int(os.environ["MAX_SEQ_LENGTH"]),
)

eval_args = EvalArgs(
    interval=int(os.environ["EVAL_INTERVAL"]),
    max_new_tokens=int(os.environ["EVAL_MAX_NEW_TOKENS"]),
    max_iters=int(os.environ["EVAL_MAX_ITERS"]),
)

print("Launching LitGPT full finetune")
print(f"  checkpoint_dir={checkpoint_dir}")
print(f"  json_path={json_path}")
print(f"  out_dir={out_dir}")
print(f"  precision={os.environ['PRECISION']}")
print(f"  devices={os.environ['DEVICES']}")
print(f"  epochs={os.environ['TRAIN_EPOCHS']}")
print(f"  max_steps={train_max_steps}")

full_setup(
    checkpoint_dir=checkpoint_dir,
    out_dir=out_dir,
    precision=os.environ["PRECISION"],
    devices=os.environ["DEVICES"],
    num_nodes=int(os.environ["NUM_NODES"]),
    data=data,
    train=train,
    eval=eval_args,
    logger_name=os.environ["LOGGER_NAME"],
    seed=int(os.environ["SEED"]),
)
PY
