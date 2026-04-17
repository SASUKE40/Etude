#!/bin/bash
# Continued pretraining for Qwen3-0.6B with LitGPT + LitData + Thunder.
#
# Prerequisites:
#   python -m pip install -U \
#     "litgpt[extra,compiler]==0.5.12" \
#     "lightning-thunder>=0.2.dev20250119" \
#     "transformers>=4.51.3,<4.57" \
#     "huggingface-hub>=0.30,<1.4" \
#     "torch==2.9.1" \
#     "torchvision==0.24.1" \
#     "torchaudio==2.9.1"
#
# Usage:
#   bash runs/litgpt_qwen3_rust_pretrain.sh
#   bash runs/litgpt_qwen3_rust_pretrain.sh prepare
#   bash runs/litgpt_qwen3_rust_pretrain.sh train
#   PHASE=prepare bash runs/litgpt_qwen3_rust_pretrain.sh
#   MAX_TOKENS=100000000 MICRO_BATCH_SIZE=2 bash runs/litgpt_qwen3_rust_pretrain.sh train

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
COMPILER="${COMPILER:-torch}"
PRECISION="${PRECISION:-bf16-true}"
DEVICES="${DEVICES:-auto}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-10000}"
CHUNK_BYTES="${CHUNK_BYTES:-200MB}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
FILES_PER_BATCH="${FILES_PER_BATCH:-8}"
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
PHASE="${PHASE:-${1:-all}}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$CHECKPOINT_ROOT" "$SCRATCH_ROOT"

python - <<'PY'
from importlib import metadata
from packaging.version import Version
import sys

def base_version(value):
    if value is None:
        return None
    return value.split("+", 1)[0]

def version_or_none(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

litgpt_version = version_or_none("litgpt")
transformers_version = version_or_none("transformers")
hub_version = version_or_none("huggingface-hub")
torch_version = version_or_none("torch")
torchvision_version = version_or_none("torchvision")
torchaudio_version = version_or_none("torchaudio")

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
if torch_version is None:
    errors.append("torch is not installed")
elif base_version(torch_version) != "2.9.1":
    errors.append(f"torch=={torch_version} does not match the tested install here; use torch==2.9.1")
if torchvision_version is None:
    errors.append("torchvision is not installed")
elif base_version(torchvision_version) != "0.24.1":
    errors.append(
        f"torchvision=={torchvision_version} does not match torch==2.9.1; use torchvision==0.24.1"
    )
if torchaudio_version is None:
    errors.append("torchaudio is not installed")
elif base_version(torchaudio_version) != "2.9.1":
    errors.append(
        f"torchaudio=={torchaudio_version} does not match torch==2.9.1; use torchaudio==2.9.1"
    )

if errors:
    print("Dependency check failed for the LitGPT Qwen3 runner:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(file=sys.stderr)
    print("Install a compatible set with:", file=sys.stderr)
    print(
        '  python -m pip install -U "litgpt[extra,compiler]==0.5.12" '
        '"lightning-thunder>=0.2.dev20250119" '
        '"transformers>=4.51.3,<4.57" '
        '"huggingface-hub>=0.30,<1.4" '
        '"torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1"',
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import torchaudio  # noqa: F401
    import litgpt  # noqa: F401
except Exception as exc:
    print("Import smoke test failed for the LitGPT Qwen3 runner:", file=sys.stderr)
    print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
    print(file=sys.stderr)
    print("Reinstall a compatible set with:", file=sys.stderr)
    print(
        '  python -m pip install -U "litgpt[extra,compiler]==0.5.12" '
        '"lightning-thunder>=0.2.dev20250119" '
        '"transformers>=4.51.3,<4.57" '
        '"huggingface-hub>=0.30,<1.4" '
        '"torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1"',
        file=sys.stderr,
    )
    sys.exit(1)
PY

run_prepare() {
  echo "=== Downloading LitGPT checkpoint ==="
  litgpt download "$MODEL_NAME" --checkpoint_dir "$CHECKPOINT_ROOT"

  echo ""
  echo "=== Preparing LitData dataset ==="
  python data/rust/prepare_litgpt_litdata.py \
    --tokenizer-dir "$CHECKPOINT_DIR" \
    --root-dir "$SCRATCH_ROOT" \
    --text-output-dir "$TEXT_DIR" \
    --litdata-output-dir "$LITDATA_DIR" \
    --num-workers "$NUM_WORKERS" \
    --files-per-batch "$FILES_PER_BATCH" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --chunk-bytes "$CHUNK_BYTES" \
    --max-seq-length "$MAX_SEQ_LENGTH"
}

run_train() {
  echo "=== Starting LitGPT continued pretraining ==="

  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint directory: $CHECKPOINT_DIR" >&2
    echo "Run the prepare phase first: bash runs/litgpt_qwen3_rust_pretrain.sh prepare" >&2
    exit 1
  fi

  if [[ ! -d "$LITDATA_DIR/train" || ! -d "$LITDATA_DIR/val" ]]; then
    echo "Missing LitData dataset under $LITDATA_DIR" >&2
    echo "Run the prepare phase first: bash runs/litgpt_qwen3_rust_pretrain.sh prepare" >&2
    exit 1
  fi

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
}

case "$PHASE" in
  all)
    run_prepare
    echo ""
    run_train
    ;;
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    echo "Expected one of: all, prepare, train" >&2
    exit 1
    ;;
esac
