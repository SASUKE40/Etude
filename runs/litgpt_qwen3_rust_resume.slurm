#!/bin/bash
# Resumable LitGPT Qwen3-0.6B Rust training on a single H100.
#
# Prerequisite:
#   Run the prepare phase first so /scratch/$USER/litgpt-rust-qwen3/litdata exists.
#
# Examples:
#   sbatch runs/litgpt_qwen3_rust_resume.slurm
#   WANDB_RUN=qwen3-rust-h100 SAVE_INTERVAL=100 sbatch runs/litgpt_qwen3_rust_resume.slurm
#   OUT_DIR=/scratch/$USER/litgpt-rust-qwen3/out/debug MAX_STEPS=100 sbatch runs/litgpt_qwen3_rust_resume.slurm

#SBATCH --job-name=litgpt-rust
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/litgpt-rust-%j.log

set -euo pipefail

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-etude}"
export WANDB_ENTITY="${WANDB_ENTITY:-edward40}"
export WANDB_RUN="${WANDB_RUN:-qwen3-rust-h100}"

source .venv/bin/activate

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/litgpt-rust-qwen3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/$USER/litgpt-checkpoints}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/Qwen/Qwen3-0.6B}"
LITDATA_DIR="${LITDATA_DIR:-$SCRATCH_ROOT/litdata}"
OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/out/qwen3-0.6b-rust}"

if [[ ! -d "$LITDATA_DIR/train" || ! -d "$LITDATA_DIR/val" ]]; then
  echo "Missing prepared LitData dataset under $LITDATA_DIR" >&2
  echo "Run: bash runs/litgpt_qwen3_rust_pretrain.sh prepare" >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$SCRATCH_ROOT"

export PHASE=train
export RESUME="${RESUME:-auto}"
export COMPILER="${COMPILER:-none}"
export PRECISION="${PRECISION:-bf16-true}"
export DEVICES="${DEVICES:-1}"
export NUM_NODES="${NUM_NODES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKENS="${MAX_TOKENS:-500000000}"
export MAX_STEPS="${MAX_STEPS:-}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export EVAL_MAX_ITERS="${EVAL_MAX_ITERS:-50}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
export MAX_NORM="${MAX_NORM:-1.0}"
export LOGGER_NAME="${LOGGER_NAME:-wandb}"
export PROJECT="${PROJECT:-$WANDB_PROJECT}"
export RUN_NAME="${RUN_NAME:-$WANDB_RUN}"

bash runs/litgpt_qwen3_rust_pretrain.sh train
