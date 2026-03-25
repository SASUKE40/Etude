# Contributing to Etude

Thanks for your interest in contributing to Etude! This document covers how to get started.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 12.8+ (for GPU training) or CPU/MPS for development

### Setup

```bash
git clone https://github.com/SASUKE40/Etude.git
cd Etude

# GPU
uv sync --extra gpu

# CPU / Apple Silicon
uv sync --extra cpu
```

### Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
etude/          Core library (model, training utilities, inference)
scripts/        Training and evaluation entry points
data/           Data preparation scripts
tasks/          Evaluation task definitions
runs/           Shell scripts for full training pipelines
tests/          Unit tests
```

Key files:

- `etude/gpt.py` — Model architecture (hybrid Gated DeltaNet + Gated Attention)
- `etude/deltanet.py` — Gated DeltaNet layer implementation
- `etude/optim.py` — Muon + AdamW optimizer
- `scripts/base_train.py` — Pretraining script

## How to Contribute

### Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (GPU, PyTorch version, OS)

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run tests: `uv run pytest tests/ -v`
5. Submit a pull request

### Pull Request Guidelines

- Keep PRs focused — one logical change per PR
- Include a clear description of what changed and why
- Add tests for new functionality
- Make sure existing tests pass
- Follow the existing code style

## Code Style

- No unnecessary abstractions — keep it simple and hackable
- Minimal comments — only where the logic isn't self-evident
- No bias in Linear layers unless explicitly needed
- Use `RMSNorm` (parameter-free) throughout
- Explicit dtype management via `COMPUTE_DTYPE` instead of autocast

## Architecture Notes

The model uses a hybrid layout: 6 groups of 4 layers each.

```
Group = 3 × (Gated DeltaNet → SwiGLU FFN) + 1 × (Gated Attention → SwiGLU FFN)
```

When making changes:

- **DeltaNet layers** handle long-range dependencies with O(T) complexity
- **Attention layers** (1 per group) provide full quadratic attention for precision
- Changes should work across the full architecture, not just individual layer types
- Test at small scale first (e.g. `--depth=4 --max-seq-len=512`)

## Development Workflow

### Quick iteration loop

```bash
# Train a small model (~5 min)
python -m scripts.base_train --depth=4 --max-seq-len=512 \
    --device-batch-size=1 --total-batch-size=512 \
    --num-iterations=20 --core-metric-every=-1

# Evaluate
python -m scripts.base_eval
```

### Data Preparation

The project uses the [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) dataset (400B tokens). There are two paths:

#### Path 1: Binary Files (Recommended)

Downloads from HuggingFace, tokenizes with GPT-2 BPE (`tiktoken`), and produces `train.bin` / `val.bin` (uint16 memmap) in `data/climbmix/`.

```bash
# Quick: single part (sufficient for testing)
python data/climbmix/prepare.py --part 0
python data/climbmix/merge.py

# Full dataset (all 10 parts)
bash data/climbmix/prepare.sh

# Prepare a specific part with custom workers
python data/climbmix/prepare.py --part 3 --num-proc 32
```

| Script | Purpose |
|---|---|
| `data/climbmix/prepare.py` | Download & tokenize individual parts (0–9) |
| `data/climbmix/merge.py` | Merge part files into `train.bin` / `val.bin` |
| `data/climbmix/prepare.sh` | End-to-end script (all parts + merge) |

#### Path 2: Parquet Streaming

Downloads parquet shards and tokenizes on-the-fly during training.

```bash
# Download ~170 shards (enough for GPT-2 scale training)
python -m etude.dataset -n 170

# Download all shards
python -m etude.dataset -n -1
```

The dataloader uses **BOS-aligned best-fit packing** — 100% utilization with no padding. Data is automatically sharded across ranks for DDP training.

## Khoury Discovery Cluster

### SSH Access

```bash
ssh zhu.shili@login.explorer.northeastern.edu

# If using Ghostty terminal, fix TERM compatibility before starting tmux
TERM=xterm-256color tmux
```

### GPU Resource Monitor

Check available GPU resources on the cluster: [GPU Monitor Dashboard](https://ood.explorer.northeastern.edu/pun/sys/dashboard/apps/show/gpu-monitor)

### Interactive GPU Session

Request an interactive shell with a GPU:

```bash
# H200 (recommended — 96 GPUs across gpu-interactive/gpu-short/gpu partitions, higher availability)
srun --partition=gpu-interactive --nodes=1 --pty --gres=gpu:h200:1 --ntasks=1 --mem=141GB --time=2:00:00 /bin/bash

# H100 (4 GPUs on sharing partition, tighter availability)
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=2:00:00 /bin/bash
```

| Flag | Meaning |
|---|---|
| `--partition=gpu-interactive` | Target the gpu-interactive partition (H200) |
| `--partition=sharing` | Target the sharing partition (H100) |
| `--nodes=1` | Single node |
| `--pty` | Allocate a pseudo-terminal (interactive) |
| `--gres=gpu:h200:1` | Request 1x H200 GPU (change model/count as needed) |
| `--ntasks=1` | One task |
| `--mem=141GB` | Allocated memory (use 80GB for H100) |
| `--time=2:00:00` | Wall-time limit of 2 hours |

#### Available GPU Resources

| GPU | Partitions | Total Nodes | Total GPUs |
|---|---|---|---|
| H100 | `sharing` | 1 | 4 |
| H200 | `gpu-interactive`, `gpu-short`, `gpu` | 12 | 96 |

> **Tip:** Replace `h200` / `h100` with `a100`, `v100-sxm2`, `l40s`, etc. to request a different GPU type.

### Job Management

```bash
# List your running/pending jobs
squeue -u $USER

# Cancel all your jobs
scancel -u $USER

# Release a held job
scontrol release <job_id>

# Extend job time limit
scontrol update jobid=<JOBID> TimeLimit=<NEW_TIME>
```

### Training with Checkpoints

When you only have limited GPU time (e.g. 1 hour per session), use `--save-every` to periodically save checkpoints and `--resume-from-step` to continue across sessions.

#### Full training (depth=24, default)

```bash
# First session — start training
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --save-every=100 --run="full-train" --model-tag="d24"

# Next session — resume from last checkpoint
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --save-every=100 --run="full-train" --model-tag="d24" \
    --resume-from-step=<LAST_STEP>
```

#### Smaller model (depth=12)

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 --save-every=100 --run="d12-single" --model-tag="d12"
```

#### Check saved checkpoints

```bash
ls ~/base_checkpoints/d24/   # full model
ls ~/base_checkpoints/d12/   # smaller model
```

> **Tip:** Tune `--save-every` based on step speed. Save often enough that you don't lose more than ~5 min of work if the job gets killed.

### Chat CLI

```bash
python -m scripts.chat_cli -i base -g d24
python -m scripts.chat_cli -i base -g d12
```

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
