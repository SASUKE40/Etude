#!/bin/bash

# Two-stage training pipeline:
#   Stage 1: Pretrain on FineWeb-Edu (general language understanding)
#   Stage 2: Fine-tune on The Stack Rust (Rust code generation)
#
# The tokenizer is trained on combined FineWeb-Edu + Rust data to ensure
# good coverage of both natural language and Rust syntax.
#
# Usage:
#   bash runs/twostage.sh
#
# In a screen session (recommended):
#   screen -L -Logfile runs/twostage.log -S twostage bash runs/twostage.sh

set -e

export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="${ETUDE_BASE_DIR:-/scratch/$USER/etude}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME"

# Number of GPUs (adjust to your allocation)
NGPU=${NGPU:-1}

# Model depth
DEPTH=${DEPTH:-24}

# WandB run name
WANDB_RUN=${WANDB_RUN:-dummy}

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Report initialization
python -m etude.report reset

# -----------------------------------------------------------------------------
# Step 1: Prepare datasets

echo "=== Step 1: Preparing datasets ==="

# Download FineWeb-Edu (10BT sample)
echo "--- Downloading FineWeb-Edu ---"
python data/fineweb-edu/prepare.py --output-dir "$ETUDE_BASE_DIR/fineweb-edu"

# Download Rust data (requires HF login for gated dataset)
echo "--- Downloading Rust data ---"
python data/rust/prepare.py --output-dir "$ETUDE_BASE_DIR/rust"

echo ""

# -----------------------------------------------------------------------------
# Step 2: Train tokenizer on combined FineWeb-Edu + Rust

echo "=== Step 2: Training tokenizer on FineWeb-Edu + Rust ==="
python -m scripts.tok_train --datasets fineweb-edu,rust
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Step 3: Stage 1 — Pretrain on FineWeb-Edu (general language)

echo "=== Step 3: Stage 1 — Pretraining on FineWeb-Edu ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --depth=$DEPTH \
    --dataset=fineweb-edu \
    --device-batch-size=16 \
    --save-every=500 \
    --model-tag="twostage-s1" \
    --run="${WANDB_RUN}-s1"

torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- \
    --model-tag="twostage-s1" \
    --device-batch-size=16

# -----------------------------------------------------------------------------
# Step 4: Stage 2 — Fine-tune on Rust code

echo "=== Step 4: Stage 2 — Fine-tuning on Rust ==="

# Find the last checkpoint step from stage 1
S1_DIR="$ETUDE_BASE_DIR/base_checkpoints/twostage-s1"
LAST_STEP=$(find "$S1_DIR" -maxdepth 1 -name 'model_*.pt' | sed -E 's#^.*/model_([0-9]+)\.pt$#\1#' | sort -n | tail -1)
if [ -z "$LAST_STEP" ]; then
    echo "ERROR: No stage 1 checkpoints found in $S1_DIR"
    exit 1
fi
echo "Loading stage 1 checkpoint from step $LAST_STEP"

torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --depth=$DEPTH \
    --dataset=rust \
    --device-batch-size=16 \
    --target-param-data-ratio=3 \
    --save-every=100 \
    --model-tag="twostage-s2" \
    --resume-from-step=$LAST_STEP \
    --resume-from-dir="$S1_DIR" \
    --run="${WANDB_RUN}-s2"

torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- \
    --model-tag="twostage-s2" \
    --device-batch-size=16

# -----------------------------------------------------------------------------
# Generate report
echo "=== Generating report ==="
python -m etude.report generate

echo ""
echo "=== Two-stage training complete! ==="
echo "Stage 1 checkpoints: $ETUDE_BASE_DIR/base_checkpoints/twostage-s1/"
echo "Stage 2 checkpoints: $ETUDE_BASE_DIR/base_checkpoints/twostage-s2/"
echo ""
echo "Chat with the model:"
echo "  python -m scripts.chat_cli -g twostage-s2"
