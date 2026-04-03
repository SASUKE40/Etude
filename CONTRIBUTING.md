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

`--device-batch-size` is the per-device micro-batch size: how many sequences one GPU processes in a single forward/backward pass. The effective batch seen by the optimizer is:

`device_batch_size × max_seq_len × num_devices × grad_accum_steps`

If you run out of VRAM, reduce `--device-batch-size` first.

### Data Preparation

The project uses the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset (10BT sample). The training pipeline uses **parquet streaming** — raw text parquet shards are downloaded, a custom BPE tokenizer is trained on them, and the dataloader tokenizes on-the-fly during training.

> **Important:** Home directory has limited quota. Set `HF_HOME` to `/scratch` for the HuggingFace cache.

#### Step 1: Download dataset (CPU-only)

```bash
export HF_HOME=/scratch/$USER/hf_cache

python data/fineweb-edu/prepare.py
```

#### Step 2: Train tokenizer (CPU-only)

Trains a custom 32K vocab BPE tokenizer (GPT-4 style) on the downloaded parquets. This creates `tokenizer.pkl` at `~/.cache/etude/tokenizer/`.

> **Important:** You must complete this step before training. Without it you'll get `FileNotFoundError: tokenizer.pkl`.

```bash
python -m scripts.tok_train

# For two-stage training, train on combined data:
python -m scripts.tok_train --datasets fineweb-edu,rust
```

After training, visualize the tokenizer to inspect how it splits text:

```bash
# Show built-in samples (Rust, English, mixed) — opens in browser
python -m scripts.tok_viz

# Tokenize a Rust file
python -m scripts.tok_viz --file src/main.rs

# Tokenize a string
python -m scripts.tok_viz "fn main() { println!(\"Hello\"); }"

# Save HTML (useful on cluster without a browser)
python -m scripts.tok_viz --sample rust -o tokenizer_viz.html
```

The HTML output shows each token as a colored span with hover tooltips displaying the token ID, byte representation, and compression stats.

To download the visualization from the cluster to your local machine:

```bash
# On the cluster: generate the HTML
python -m scripts.tok_viz --sample rust -o /scratch/$USER/etude/tokenizer/tokenizer_viz.html

# On your local machine: download it
scp zhu.shili@login.explorer.northeastern.edu:/scratch/zhu.shili/etude/tokenizer/tokenizer_viz.html .
```

Or serve it directly from the cluster via SSH tunnel:

```bash
# On the cluster: start a simple HTTP server
cd /scratch/$USER/etude/tokenizer
python -m http.server 8080

# On your local machine: open an SSH tunnel
ssh -L 8080:localhost:8080 zhu.shili@login.explorer.northeastern.edu
```

Then open http://localhost:8080/tokenizer_viz.html in your browser.

#### Step 3: Train model (needs GPU)

The dataloader reads parquets and tokenizes on-the-fly with the custom tokenizer.

```bash
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --save-every=100 --run="full-train" --model-tag="d24"
```

#### Alternative: Binary files (nanoGPT-style)

A separate path uses GPT-2 tokenizer to produce pre-tokenized binary files. This is **not** the main training pipeline but useful for nanoGPT-compatible workflows.

```bash
export HF_HOME=/scratch/$USER/hf_cache

# Download 10BT sample (recommended)
python data/fineweb-edu/prepare.py --output-dir /scratch/$USER/etude/fineweb-edu

# Or use the shell script
bash data/fineweb-edu/prepare.sh
```

#### Rust fine-tuning data

Fine-tune on Rust code from [The Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup).

> **Important:** This is a gated dataset. You must:
> 1. Accept the terms at https://huggingface.co/datasets/bigcode/the-stack-dedup
> 2. Login with `huggingface-cli login` (get a token from https://huggingface.co/settings/tokens with "Read" access)

```bash
export HF_HOME=/scratch/$USER/hf_cache

huggingface-cli login

# Full prepare (outputs to /scratch)
bash data/rust/prepare.sh

# Or manually
python data/rust/prepare.py --output-dir /scratch/$USER/etude_data/rust --num-proc 48
```

The dataloader uses **BOS-aligned best-fit packing** — 100% utilization with no padding. Data is automatically sharded across ranks for DDP training.

## Khoury Discovery Cluster

### SSH Access

```bash
ssh zhu.shili@login.explorer.northeastern.edu

# If using Ghostty terminal, fix TERM compatibility before starting tmux
TERM=xterm-256color tmux

# Activate the virtual environment
source .venv/bin/activate
```

### GPU Resource Monitor

Check available GPU resources on the cluster: [GPU Monitor Dashboard](https://ood.explorer.northeastern.edu/pun/sys/dashboard/apps/show/gpu-monitor)

### Interactive GPU Session

Request an interactive shell with a GPU:

```bash
# H200 (recommended — 96 GPUs across gpu-interactive/gpu-short/gpu partitions, higher availability)
srun --partition=gpu-interactive --nodes=1 --pty --gres=gpu:h200:1 --ntasks=1 --mem=141GB --time=2:00:00 /bin/bash

# H100 (4 GPUs on sharing partition, tighter availability)
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
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
| `--time=2:00:00` | Wall-time limit (2h for H200, 1h max for H100) |

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

### Training with SLURM Batch Jobs

Submit training as a batch job using the provided SLURM script:

```bash
sbatch runs/train.slurm
```

This submits to the `sharing` partition with 1×H100, 80GB, 1h time limit. Checkpoints are saved every 100 steps.

To resume after the time limit, edit `runs/train.slurm` to add `--resume-from-step=<LAST_STEP>` and resubmit:

```bash
# Check last saved step
ls $ETUDE_BASE_DIR/base_checkpoints/

# Resubmit
sbatch runs/train.slurm
```

Monitor jobs:

```bash
squeue -u $USER           # job status
tail -f runs/etude-*.log   # live output
scancel <job_id>           # cancel a job
```

### Training with Interactive Sessions

When you only have limited GPU time (e.g. 1 hour per session), use `--save-every` to periodically save checkpoints and `--resume-from-step` to continue across sessions.

> **Important:** Always use tmux inside the GPU session so training survives SSH disconnects.

#### Full training (depth=24, default)

```bash
# Request a GPU node
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash

# Start tmux inside the GPU node (survives SSH disconnects)
TERM=xterm-256color tmux

# Setup environment
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache

# First session — start training
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-batch-size=16 --save-every=100 --run="full-train" --model-tag="d24"
```

`torchrun --nproc_per_node=1` starts one training process on this node, which for this setup means one GPU. On a single H100, `--device-batch-size=16` is a reasonable starting point for the default `d24` model. If you OOM, reduce it to `8` or `4`.

With multiple GPUs on one node, set `--nproc_per_node` to the number of GPUs you want to use. For example, `--nproc_per_node=8` launches eight processes, typically one per GPU.

To resume in a new session:

```bash
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
TERM=xterm-256color tmux
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-batch-size=16 --save-every=100 --run="full-train" --model-tag="d24" \
    --resume-from-step=<LAST_STEP>
```

#### Smaller model (depth=12)

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 --device-batch-size=16 --save-every=100 --run="d12-single" --model-tag="d12"
```

#### Check saved checkpoints

```bash
ls $ETUDE_BASE_DIR/base_checkpoints/d24/   # full model
ls $ETUDE_BASE_DIR/base_checkpoints/d12/   # smaller model
```

> **Tip:** Tune `--save-every` based on step speed. Save often enough that you don't lose more than ~5 min of work if the job gets killed. If you OOM with `--device-batch-size=16`, try `8` or `4`.

#### Debugging torch.compile

The first run triggers `torch.compile`, which can take 5–15 minutes on H100. If training appears stuck, enable verbose compile logs:

```bash
TORCH_LOGS="dynamo,inductor" TORCHDYNAMO_VERBOSE=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-batch-size=16 \
    --save-every=100 \
    --run="full-train" \
    --model-tag="d24"
```

### Two-Stage Training (FineWeb-Edu + Rust)

Train a model that understands general language then specializes in Rust code generation:

```bash
# Full pipeline (prepare data, train tokenizer, stage 1 + stage 2)
bash runs/twostage.sh

# Single H100 example
NGPU=1 DEPTH=24 WANDB_RUN=twostage bash runs/twostage.sh
```

Or run stages manually:

```bash
# Request GPU and setup tmux
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
TERM=xterm-256color tmux
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache

# Stage 1: Pretrain on FineWeb-Edu
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --dataset=fineweb-edu --device-batch-size=16 \
    --model-tag="twostage-s1" --save-every=500 --run=dummy

# Stage 2: Fine-tune on Rust (new GPU session if needed)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --dataset=rust --device-batch-size=16 --target-param-data-ratio=3 \
    --model-tag="twostage-s2" --save-every=100 --run=dummy \
    --resume-from-step=<S1_LAST_STEP> \
    --resume-from-dir="$ETUDE_BASE_DIR/base_checkpoints/twostage-s1"
```

### Chat CLI

```bash
python -m scripts.chat_cli -i base -g d24
python -m scripts.chat_cli -i base -g d12
```

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
