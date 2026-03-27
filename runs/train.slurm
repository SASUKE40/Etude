#!/bin/bash
#SBATCH --job-name=etude
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --output=runs/etude-%j.log

module load cuda/12.8 cuDNN/9.10.2
export OMP_NUM_THREADS=1
export ETUDE_BASE_DIR="/scratch/$USER/etude"
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$ETUDE_BASE_DIR" "$HF_HOME"

source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 \
    -m scripts.base_train \
    --depth=24 \
    --device-batch-size=16 \
    --save-every=100 \
    --run="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
