#!/bin/bash
# Example:
#   sbatch runs/qwen_base_train.slurm
#   TOTAL_STEPS=10000 BATCH_SIZE=2 sbatch runs/qwen_base_train.slurm
#   MODEL_NAME=Qwen/Qwen3.5-0.8B-Base OUTPUT_DIR=/scratch/$USER/qwen-rust sbatch runs/qwen_base_train.slurm

#SBATCH --job-name=qwen-rust
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/qwen-rust-%j.log

set -euo pipefail

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="${ETUDE_BASE_DIR:-/scratch/$USER/etude}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen-rust-pretrain}"
export WANDB_ENTITY="${WANDB_ENTITY:-edward40}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

source .venv/bin/activate

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ETUDE_BASE_DIR/checkpoints/qwen-rust}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen3.5-0.8b-rust}"
PROCESSED_DATASET_DIR="${PROCESSED_DATASET_DIR:-$ETUDE_BASE_DIR/datasets/qwen-rust-clean-${MAX_LENGTH}}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-auto}"
STRATEGY="${STRATEGY:-ddp}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-5000}"

python -m scripts.qwen_base_train \
    --model-name="${MODEL_NAME}" \
    --max-length="${MAX_LENGTH}" \
    --batch-size="${BATCH_SIZE}" \
    --num-workers="${NUM_WORKERS}" \
    --preprocessing-num-workers="${PREPROCESSING_NUM_WORKERS}" \
    --processed-dataset-dir="${PROCESSED_DATASET_DIR}" \
    --learning-rate="${LEARNING_RATE}" \
    --warmup-steps="${WARMUP_STEPS}" \
    --total-steps="${TOTAL_STEPS}" \
    --checkpoint-dir="${CHECKPOINT_DIR}" \
    --output-dir="${OUTPUT_DIR}" \
    --project="${WANDB_PROJECT}" \
    --accelerator="${ACCELERATOR}" \
    --devices="${DEVICES}" \
    --strategy="${STRATEGY}" \
    --val-check-interval="${VAL_CHECK_INTERVAL}" \
    --accumulate-grad-batches="${ACCUMULATE_GRAD_BATCHES}" \
    --gradient-clip-val="${GRADIENT_CLIP_VAL}" \
    --checkpoint-every-n-train-steps="${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
