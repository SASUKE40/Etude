#!/bin/bash
# Continued pretraining for Qwen3-0.6B with LitGPT + LitData + Thunder.
#
# Prerequisites:
#   python -m pip install -U \
#     "litgpt[all]==0.5.12" \
#     "lightning-thunder>=0.2.dev20250119" \
#     "transformers>=4.51.3,<4.57" \
#     "huggingface-hub>=0.30,<1.4"
#
# Usage:
#   bash runs/litgpt_qwen3_rust_pretrain.sh
#   MAX_TOKENS=100000000 MICRO_BATCH_SIZE=2 bash runs/litgpt_qwen3_rust_pretrain.sh

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/litgpt-rust-qwen3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/$USER/litgpt-checkpoints}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/Qwen/Qwen3-0.6B}"
TEXT_DIR="${TEXT_DIR:-$SCRATCH_ROOT/text}"
LITDATA_DIR="${LITDATA_DIR:-$SCRATCH_ROOT/litdata}"
OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/out/qwen3-0.6b-rust}"
COMPILER="${COMPILER:-thunder}"
PRECISION="${PRECISION:-bf16-true}"
DEVICES="${DEVICES:-auto}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-10000}"
CHUNK_BYTES="${CHUNK_BYTES:-200MB}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-500000000}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_MAX_ITERS="${EVAL_MAX_ITERS:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
MAX_NORM="${MAX_NORM:-1.0}"
LOGGER_NAME="${LOGGER_NAME:-tensorboard}"
PROJECT="${PROJECT:-}"
RUN_NAME="${RUN_NAME:-}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$CHECKPOINT_ROOT" "$SCRATCH_ROOT"

python - <<'PY'
from importlib import metadata
from packaging.version import Version
import sys

def version_or_none(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

litgpt_version = version_or_none("litgpt")
transformers_version = version_or_none("transformers")
hub_version = version_or_none("huggingface-hub")

errors = []
if litgpt_version is None:
    errors.append("litgpt is not installed")
if transformers_version is None:
    errors.append("transformers is not installed")
elif Version(transformers_version) >= Version("4.57"):
    errors.append(
        f"transformers=={transformers_version} is too new for LitGPT 0.5.12; use transformers>=4.51.3,<4.57"
    )
if hub_version is None:
    errors.append("huggingface-hub is not installed")
elif Version(hub_version) >= Version("1.4"):
    errors.append(
        f"huggingface-hub=={hub_version} is outside LitGPT's supported range; use huggingface-hub>=0.30,<1.4"
    )

if errors:
    print("Dependency check failed for the LitGPT Qwen3 runner:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(file=sys.stderr)
    print("Install a compatible set with:", file=sys.stderr)
    print(
        '  python -m pip install -U "litgpt[all]==0.5.12" '
        '"lightning-thunder>=0.2.dev20250119" '
        '"transformers>=4.51.3,<4.57" '
        '"huggingface-hub>=0.30,<1.4"',
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo "=== Downloading LitGPT checkpoint ==="
litgpt download "$MODEL_NAME" --checkpoint_dir "$CHECKPOINT_ROOT"

echo ""
echo "=== Preparing LitData dataset ==="
python data/rust/prepare_litgpt_litdata.py \
  --tokenizer-dir "$CHECKPOINT_DIR" \
  --root-dir "$SCRATCH_ROOT" \
  --text-output-dir "$TEXT_DIR" \
  --litdata-output-dir "$LITDATA_DIR" \
  --rows-per-shard "$ROWS_PER_SHARD" \
  --chunk-bytes "$CHUNK_BYTES" \
  --max-seq-length "$MAX_SEQ_LENGTH"

echo ""
echo "=== Starting LitGPT continued pretraining ==="

CMD=(
  python scripts/litgpt_qwen3_rust_pretrain.py
  --model-name "$MODEL_NAME"
  --data-path "$LITDATA_DIR"
  --tokenizer-dir "$CHECKPOINT_DIR"
  --out-dir "$OUT_DIR"
  --precision "$PRECISION"
  --devices "$DEVICES"
  --num-nodes "$NUM_NODES"
  --num-workers "$NUM_WORKERS"
  --compiler "$COMPILER"
  --logger-name "$LOGGER_NAME"
  --seed 42
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --max-tokens "$MAX_TOKENS"
  --save-interval "$SAVE_INTERVAL"
  --eval-interval "$EVAL_INTERVAL"
  --eval-max-iters "$EVAL_MAX_ITERS"
  --learning-rate "$LEARNING_RATE"
  --lr-warmup-steps "$LR_WARMUP_STEPS"
  --max-norm "$MAX_NORM"
)

if [[ -n "$MAX_STEPS" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "$PROJECT" ]]; then
  CMD+=(--project "$PROJECT")
fi

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run-name "$RUN_NAME")
fi

"${CMD[@]}"
