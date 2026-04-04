#!/bin/bash
#SBATCH --job-name=d24-h100
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/d24-h100-%j.log

set -euo pipefail

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="${ETUDE_BASE_DIR:-/scratch/$USER/etude}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-etude}"
export WANDB_ENTITY="${WANDB_ENTITY:-edward40}"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME"

source .venv/bin/activate

SAVE_EVERY="${SAVE_EVERY:-100}"
RUN_NAME="${RUN_NAME:-full-train-h100}"
MODEL_TAG="${MODEL_TAG:-d24-h100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ETUDE_BASE_DIR/base_checkpoints/$MODEL_TAG}"

if [ -z "${RESUME_STEP:-}" ]; then
    RESUME_STEP=$(
        find "$CHECKPOINT_DIR" -maxdepth 1 -name 'model_*.pt' -print | \
        sed -E 's#^.*/model_([0-9]+)\.pt$#\1#' | \
        sort -n | \
        while read -r step; do
            if [ -f "$CHECKPOINT_DIR/meta_${step}.json" ] && [ -f "$CHECKPOINT_DIR/optim_${step}_rank0.pt" ]; then
                echo "$step"
            fi
        done | \
        tail -1
    )
fi

if [ -z "${RESUME_STEP:-}" ]; then
    echo "ERROR: No complete checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Resuming ${MODEL_TAG} from step ${RESUME_STEP} (dir: ${CHECKPOINT_DIR})"

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-type=cuda \
    --fp8 \
    --depth=24 \
    --max-seq-len=1024 \
    --device-batch-size=2 \
    --total-batch-size=32768 \
    --save-every="${SAVE_EVERY}" \
    --run="${RUN_NAME}" \
    --model-tag="${MODEL_TAG}" \
    --resume-from-step="${RESUME_STEP}"
