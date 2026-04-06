#!/bin/bash
# Prepare Nemotron Cascade SFT Stage 2 on a compute node.
#
# This dataset prep is CPU / memory heavy and should not run on the login node.
#
# Examples:
#   sbatch runs/nemotron_prepare.slurm
#   MAX_ROWS_PER_SUBSET=1000 SUBSETS=general,instruction-following,tool_calling sbatch runs/nemotron_prepare.slurm

#SBATCH --job-name=nemotron-prep
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --mem=141GB
#SBATCH --time=2:00:00
#SBATCH --output=runs/nemotron-prepare-%j.log

set -euo pipefail

module load cuda/12.8 cuDNN/9.10.2

export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="${ETUDE_BASE_DIR:-/scratch/$USER/etude}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME"

source .venv/bin/activate

OUTPUT_DIR="${OUTPUT_DIR:-$ETUDE_BASE_DIR/datasets/nemotron-cascade-sft-stage-2}"
SUBSETS="${SUBSETS:-general,instruction-following,tool_calling,math,science,code,swe_localization,swe_repair,swe_testgen}"
VAL_RATIO="${VAL_RATIO:-0.005}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-50000}"
FLUSH_ROWS="${FLUSH_ROWS:-16}"
COMPRESSION="${COMPRESSION:-snappy}"
MAX_ROWS_PER_SUBSET="${MAX_ROWS_PER_SUBSET:--1}"

echo "Preparing Nemotron chat SFT data on compute node"
echo "HF_HOME: $HF_HOME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SUBSETS: $SUBSETS"
echo "VAL_RATIO: $VAL_RATIO"
echo "ROWS_PER_SHARD: $ROWS_PER_SHARD"
echo "FLUSH_ROWS: $FLUSH_ROWS"
echo "COMPRESSION: $COMPRESSION"
echo "MAX_ROWS_PER_SUBSET: $MAX_ROWS_PER_SUBSET"

python -u data/nemotron-cascade-sft-stage-2/prepare.py \
    --output-dir "$OUTPUT_DIR" \
    --subsets "$SUBSETS" \
    --val-ratio "$VAL_RATIO" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --flush-rows "$FLUSH_ROWS" \
    --compression "$COMPRESSION" \
    --max-rows-per-subset "$MAX_ROWS_PER_SUBSET"
