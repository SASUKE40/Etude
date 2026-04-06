#!/bin/bash
# Chat SFT starting from a base checkpoint.
#
# Defaults:
#   source checkpoint: /scratch/zhu.shili/etude/base_checkpoints/d24-h100-rust-s2-010200/model_008600.pt
#   chat dataset: nemotron-cascade-sft-stage-2
#   output tag: d24-h100-rust-s2-010200-sft-008600
#
# Examples:
#   sbatch runs/d24_chat_sft.slurm
#   MODEL_TAG=d24-h100-rust-s2-chat CHAT_DATASET=legacy sbatch runs/d24_chat_sft.slurm
#   SOURCE_MODEL_PATH=/scratch/zhu.shili/etude/base_checkpoints/other/model_009000.pt sbatch runs/d24_chat_sft.slurm

#SBATCH --job-name=d24-sft
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/d24-sft-%j.log

set -euo pipefail

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="${ETUDE_BASE_DIR:-/scratch/$USER/etude}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-etude-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-edward40}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME" "$ETUDE_BASE_DIR/base_checkpoints"

source .venv/bin/activate

DEFAULT_SOURCE_MODEL_PATH="/scratch/zhu.shili/etude/base_checkpoints/d24-h100-rust-s2-010200/model_008600.pt"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-$DEFAULT_SOURCE_MODEL_PATH}"
SOURCE_CHECKPOINT_DIR="$(dirname "$SOURCE_MODEL_PATH")"
SOURCE_MODEL_FILE="$(basename "$SOURCE_MODEL_PATH")"
SOURCE_MODEL_TAG="$(basename "$SOURCE_CHECKPOINT_DIR")"

if [[ "$SOURCE_MODEL_FILE" =~ ^model_([0-9]+)\.pt$ ]]; then
    SOURCE_STEP="${SOURCE_STEP:-${BASH_REMATCH[1]}}"
else
    echo "ERROR: SOURCE_MODEL_PATH must end with model_<step>.pt"
    exit 1
fi

SOURCE_STEP=$((10#$SOURCE_STEP))
STEP_PADDED=$(printf "%06d" "$SOURCE_STEP")

MODEL_TAG="${MODEL_TAG:-${SOURCE_MODEL_TAG}-sft-${STEP_PADDED}}"
RUN_NAME="${RUN_NAME:-$MODEL_TAG}"
CHAT_DATASET="${CHAT_DATASET:-nemotron-cascade-sft-stage-2}"
CHAT_DATA_DIR="${CHAT_DATA_DIR:-$ETUDE_BASE_DIR/datasets/nemotron-cascade-sft-stage-2}"
CHAT_SUBSETS="${CHAT_SUBSETS:-}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-1}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
CHATCORE_EVERY="${CHATCORE_EVERY:--1}"

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

if [ "$LOAD_OPTIMIZER" != "0" ] && [ ! -f "$OPTIM_PATH" ]; then
    echo "ERROR: Source optimizer checkpoint not found: $OPTIM_PATH"
    echo "Set LOAD_OPTIMIZER=0 to start SFT with a fresh optimizer."
    exit 1
fi

if (( TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) != 0 )); then
    echo "ERROR: TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN"
    exit 1
fi

MODEL_ALIAS_DIR="$ETUDE_BASE_DIR/base_checkpoints/$MODEL_TAG"
if [ ! -e "$MODEL_ALIAS_DIR" ]; then
    ln -s "$SOURCE_CHECKPOINT_DIR" "$MODEL_ALIAS_DIR"
elif [ -L "$MODEL_ALIAS_DIR" ]; then
    EXISTING_TARGET="$(readlink "$MODEL_ALIAS_DIR")"
    if [ "$EXISTING_TARGET" != "$SOURCE_CHECKPOINT_DIR" ]; then
        echo "ERROR: $MODEL_ALIAS_DIR already points to $EXISTING_TARGET"
        echo "       Expected: $SOURCE_CHECKPOINT_DIR"
        exit 1
    fi
elif [ "$MODEL_ALIAS_DIR" != "$SOURCE_CHECKPOINT_DIR" ]; then
    echo "ERROR: $MODEL_ALIAS_DIR already exists and is not the requested source dir"
    exit 1
fi

GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) ))
echo "Starting chat SFT from ${SOURCE_MODEL_TAG} step ${SOURCE_STEP}"
echo "Source model path: ${MODEL_PATH}"
echo "Load alias tag: ${MODEL_TAG}"
echo "Output SFT tag: ${MODEL_TAG}"
echo "Chat dataset: ${CHAT_DATASET}"
echo "Chat data dir: ${CHAT_DATA_DIR}"
if [ -n "$CHAT_SUBSETS" ]; then
    echo "Chat subsets: ${CHAT_SUBSETS}"
fi
echo "Load optimizer: ${LOAD_OPTIMIZER}"
echo "Launch config: seq=${MAX_SEQ_LEN} batch=${DEVICE_BATCH_SIZE} total_batch=${TOTAL_BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS}"

CHAT_SUBSETS_ARGS=()
if [ -n "$CHAT_SUBSETS" ]; then
    CHAT_SUBSETS_ARGS+=(--chat-subsets="$CHAT_SUBSETS")
fi

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device-type=cuda \
    --model-tag="${MODEL_TAG}" \
    --model-step="${SOURCE_STEP}" \
    --load-optimizer="${LOAD_OPTIMIZER}" \
    --chat-dataset="${CHAT_DATASET}" \
    --chat-data-dir="${CHAT_DATA_DIR}" \
    "${CHAT_SUBSETS_ARGS[@]}" \
    --max-seq-len="${MAX_SEQ_LEN}" \
    --device-batch-size="${DEVICE_BATCH_SIZE}" \
    --total-batch-size="${TOTAL_BATCH_SIZE}" \
    --eval-every="${EVAL_EVERY}" \
    --eval-tokens="${EVAL_TOKENS}" \
    --chatcore-every="${CHATCORE_EVERY}" \
    --run="${RUN_NAME}"
