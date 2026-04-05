#!/bin/bash
# Stage-2 Rust fine-tune starting from a base checkpoint.
#
# Defaults:
#   source checkpoint dir: /scratch/zhu.shili/etude/base_checkpoints/d24-h100-rust-s2-010200
#   source tag: basename of source checkpoint dir
#   source step: latest complete checkpoint in source dir
#   output tag: same as source tag (in-place resume)
#
# Example:
#   sbatch runs/d24_rust_resume.slurm
#   RESUME_STEP=12000 sbatch runs/d24_rust_resume.slurm
#   SOURCE_MODEL_TAG=d24-h100 MODEL_TAG=d24-rust-s2-v2 sbatch runs/d24_rust_resume.slurm
#   SOURCE_MODEL_TAG=d24-h100-rust-s2-010200 MODEL_TAG=d24-h100-rust-s2-010200 sbatch runs/d24_rust_resume.slurm

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

DEFAULT_SOURCE_CHECKPOINT_DIR="/scratch/zhu.shili/etude/base_checkpoints/d24-h100-rust-s2-010200"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-}"
SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-}"
if [ -z "${SOURCE_CHECKPOINT_DIR}" ]; then
    if [ -n "${SOURCE_MODEL_TAG}" ]; then
        SOURCE_CHECKPOINT_DIR="$ETUDE_BASE_DIR/base_checkpoints/$SOURCE_MODEL_TAG"
    else
        SOURCE_CHECKPOINT_DIR="$DEFAULT_SOURCE_CHECKPOINT_DIR"
    fi
fi
SOURCE_MODEL_TAG="${SOURCE_MODEL_TAG:-$(basename "$SOURCE_CHECKPOINT_DIR")}"
MODEL_TAG="${MODEL_TAG:-$SOURCE_MODEL_TAG}"
RUN_NAME="${RUN_NAME:-$MODEL_TAG}"
RESUME_STEP="${RESUME_STEP:-}"

IN_PLACE_RESUME=0
if [ -n "${MODEL_TAG}" ] && [ "${MODEL_TAG}" = "${SOURCE_MODEL_TAG}" ]; then
    IN_PLACE_RESUME=1
fi

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

if [ -z "${RESUME_STEP}" ]; then
    RESUME_STEP=$(
        find "$SOURCE_CHECKPOINT_DIR" -maxdepth 1 -name 'model_*.pt' -print | \
        sed -E 's#^.*/model_([0-9]+)\.pt$#\1#' | \
        sort -n | \
        while read -r step; do
            if [ -f "$SOURCE_CHECKPOINT_DIR/meta_${step}.json" ] && \
               { [ "${IN_PLACE_RESUME}" -eq 0 ] || [ -f "$SOURCE_CHECKPOINT_DIR/optim_${step}_rank0.pt" ]; }; then
                echo "$step"
            fi
        done | \
        tail -1
    )
fi

if [ -z "${RESUME_STEP}" ]; then
    echo "ERROR: No complete source checkpoint found in $SOURCE_CHECKPOINT_DIR"
    exit 1
fi

RESUME_STEP=$((10#$RESUME_STEP))

STEP_PADDED=$(printf "%06d" "$RESUME_STEP")

if [ "${MODEL_TAG}" = "${SOURCE_MODEL_TAG}" ]; then
    IN_PLACE_RESUME=1
fi

MODEL_PATH="$SOURCE_CHECKPOINT_DIR/model_${STEP_PADDED}.pt"
META_PATH="$SOURCE_CHECKPOINT_DIR/meta_${STEP_PADDED}.json"
OPTIM_PATH="$SOURCE_CHECKPOINT_DIR/optim_${STEP_PADDED}_rank0.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Source checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$META_PATH" ]; then
    echo "ERROR: Source checkpoint metadata not found: $META_PATH"
    exit 1
fi

if [ "${IN_PLACE_RESUME}" -eq 1 ] && [ ! -f "$OPTIM_PATH" ]; then
    echo "ERROR: Source optimizer checkpoint not found for in-place resume: $OPTIM_PATH"
    exit 1
fi

if (( TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) != 0 )); then
    echo "ERROR: TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN"
    exit 1
fi

GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) ))
echo "Starting Rust run from ${SOURCE_MODEL_TAG} step ${RESUME_STEP}"
echo "Source dir: ${SOURCE_CHECKPOINT_DIR}"
echo "Output tag: ${MODEL_TAG}"
if [ "${IN_PLACE_RESUME}" -eq 1 ]; then
    echo "Mode: in-place Rust resume (model, optimizer, and dataloader state)"
else
    echo "Mode: Rust stage transition (model weights only, fresh optimizer/dataloader)"
fi
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
