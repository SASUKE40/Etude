#!/bin/bash
# Stage-2 Rust fine-tune starting from a base checkpoint.
#
# Defaults:
#   source tag: d24-h100
#   source step: 10200
#   output tag: d24-rust-s2
#
# Example:
#   sbatch runs/d24_rust_resume.slurm
#   RESUME_STEP=12000 sbatch runs/d24_rust_resume.slurm
#   SOURCE_MODEL_TAG=d24-h100 MODEL_TAG=d24-rust-s2-v2 sbatch runs/d24_rust_resume.slurm

#SBATCH --job-name=d24-rust
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/d24-rust-%j.log

set -euo pipefail

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="${ETUDE_BASE_DIR:-/scratch/$USER/etude}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-etude}"
export WANDB_ENTITY="${WANDB_ENTITY:-edward40}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME"

source .venv/bin/activate

SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-d24-h100}"
MODEL_TAG="${MODEL_TAG:-d24-rust-s2}"
RUN_NAME="${RUN_NAME:-d24-rust-s2}"
RESUME_STEP="${RESUME_STEP:-10200}"

SAVE_EVERY="${SAVE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-3}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-$ETUDE_BASE_DIR/base_checkpoints/$SOURCE_MODEL_TAG}"
STEP_PADDED=$(printf "%06d" "$RESUME_STEP")

MODEL_PATH="$SOURCE_CHECKPOINT_DIR/model_${STEP_PADDED}.pt"
META_PATH="$SOURCE_CHECKPOINT_DIR/meta_${STEP_PADDED}.json"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Source checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$META_PATH" ]; then
    echo "ERROR: Source checkpoint metadata not found: $META_PATH"
    exit 1
fi

if (( TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) != 0 )); then
    echo "ERROR: TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN"
    exit 1
fi

GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) ))
echo "Starting Rust stage transition from ${SOURCE_MODEL_TAG} step ${RESUME_STEP}"
echo "Source dir: ${SOURCE_CHECKPOINT_DIR}"
echo "Output tag: ${MODEL_TAG}"
echo "Launch config: seq=${MAX_SEQ_LEN} batch=${DEVICE_BATCH_SIZE} total_batch=${TOTAL_BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS}"

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-type=cuda \
    --fp8 \
    --depth=24 \
    --dataset=rust \
    --max-seq-len="${MAX_SEQ_LEN}" \
    --device-batch-size="${DEVICE_BATCH_SIZE}" \
    --total-batch-size="${TOTAL_BATCH_SIZE}" \
    --target-param-data-ratio="${TARGET_PARAM_DATA_RATIO}" \
    --log-every="${LOG_EVERY}" \
    --eval-every="${EVAL_EVERY}" \
    --eval-tokens="${EVAL_TOKENS}" \
    --core-metric-every="${CORE_METRIC_EVERY}" \
    --sample-every="${SAMPLE_EVERY}" \
    --save-every="${SAVE_EVERY}" \
    --run="${RUN_NAME}" \
    --model-tag="${MODEL_TAG}" \
    --resume-from-step="${RESUME_STEP}" \
    --resume-from-dir="${SOURCE_CHECKPOINT_DIR}"
