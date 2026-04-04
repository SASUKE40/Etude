# Etude

A compact language model for Rust code generation, trained with a two-stage pipeline: general language pretraining on FineWeb-Edu followed by Rust specialization on The Stack.

## Architecture

Etude is a 0.8B parameter causal language model using a hybrid Gated DeltaNet + Gated Attention architecture.

| Component | Spec |
|---|---|
| Parameters | 0.8B |
| Hidden Dimension | 1024 |
| Layers | 24 |
| Layout | 6 × (3 × Gated DeltaNet → FFN + 1 × Gated Attention → FFN) |
| Token Embedding | 248,320 (tied with LM head) |
| Context Length | 262,144 tokens |

### Gated DeltaNet

Linear attention with delta rule recurrence — O(T) complexity.

- 16 heads for Q/K and V, head dimension 128
- Short convolution for local context
- Input (beta) and output gating

### Gated Attention

Standard multi-head attention with grouped-query attention (GQA).

- 8 Q heads, 2 KV heads, head dimension 256
- Rotary position embedding (dimension 64)
- Output gating

### Feed-Forward Network

SwiGLU activation with intermediate dimension 3584.

### Multi-Token Prediction

Trained with auxiliary MTP heads for predicting multiple future tokens.

## Training Pipeline

Etude uses a two-stage training approach with a combined tokenizer:

### Overview

```
1. Prepare datasets (FineWeb-Edu + Rust)
2. Train tokenizer on combined data → 32K vocab BPE
3. Stage 1: Pretrain on FineWeb-Edu → general language understanding
4. Stage 2: Fine-tune on Rust code → Rust code generation
```

### Datasets

| Dataset | Source | Tokens | Disk Size | Purpose |
|---|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | Educational web text | 10B | ~27 GB | Stage 1: general language |
| [The Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) (Rust) | Rust source code | ~5-8B | ~15 GB | Stage 2: Rust specialization |

### Token Budget (d24 model, ~124M scaling params)

| Stage | Dataset | data:param ratio | Tokens Trained |
|---|---|---|---|
| Stage 1 | FineWeb-Edu | 12 (default) | ~1.5B |
| Stage 2 | Rust | 3 | ~0.4B |
| **Total** | | | **~1.9B** |

The `--target-param-data-ratio` controls how many tokens to train on relative to model size. Higher = more tokens, longer training, potentially better results. The default 12 is slightly undertrained for speed (Chinchilla-optimal is ~20).

## Quick Start

### Install

```bash
# GPU (CUDA 12.8)
uv sync --extra gpu

# CPU / Apple Silicon
uv sync --extra cpu
```

### Two-Stage Training (Full Pipeline)

```bash
# End-to-end: prepare data, train tokenizer, stage 1 + stage 2
bash runs/twostage.sh
```

Or run each step manually:

#### 1. Prepare Datasets

```bash
export HF_HOME=/scratch/$USER/hf_cache

# FineWeb-Edu (10BT sample, ~27 GB)
python data/fineweb-edu/prepare.py --output-dir /scratch/$USER/etude/fineweb-edu

# Rust from The Stack (gated — requires `huggingface-cli login`)
python data/rust/prepare.py --output-dir /scratch/$USER/etude/rust
```

#### 2. Train Tokenizer (on combined data)

```bash
python -m scripts.tok_train --datasets fineweb-edu,rust
python -m scripts.tok_eval
```

#### 3. Stage 1 — Pretrain on FineWeb-Edu

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --dataset=fineweb-edu --model-tag="twostage-s1" --save-every=500
```

#### 4. Stage 2 — Fine-tune on Rust

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --dataset=rust --model-tag="twostage-s2" --target-param-data-ratio=3 \
    --resume-from-step=<S1_LAST_STEP> \
    --resume-from-dir="$ETUDE_BASE_DIR/base_checkpoints/twostage-s1"
```

#### 5. Chat with the Model

```bash
python -m scripts.chat_cli -g twostage-s2
python -m scripts.chat_web
```

### Single-Stage Training (FineWeb-Edu)

The single-dataset training path is also supported:

```bash
# Prepare FineWeb-Edu data
python data/fineweb-edu/prepare.py

# Train tokenizer
python -m scripts.tok_train

# Pretrain
torchrun --standalone --nproc_per_node=8 -m scripts.base_train --depth=24

# CPU / MacBook demo (tiny model)
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
```

For a 1×H100 Qwen 3.5 smoke run with per-step W&B logging:

```bash
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache
export WANDB_PROJECT=etude
export WANDB_ENTITY=edward40

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --device-type=cuda \
    --depth=4 \
    --n-embd=256 \
    --max-seq-len=256 \
    --device-batch-size=2 \
    --total-batch-size=512 \
    --num-iterations=10 \
    --log-every=1 \
    --eval-every=1 \
    --eval-tokens=1024 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run="tiny-smoke" \
    --model-tag="qwen35-h100-smoke"
```

If you want to exercise the H100-specific FP8 path too, add `--fp8`.

If your W&B account is already configured locally, you can omit `WANDB_ENTITY`. If you do not want W&B logging for the smoke run, set `--run=dummy`.

To load the resulting base checkpoint and chat with it:

```bash
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache

python -m scripts.chat_cli -i base -g qwen35-h100-smoke -s 10 --device-type cuda
```

For a one-shot prompt:

```bash
python -m scripts.chat_cli -i base -g qwen35-h100-smoke -s 10 --device-type cuda -p "hello"
```

This is only a checkpoint-loading smoke test. A base model trained for 10 steps is not instruction-tuned, so the output will be poor.

For a larger single-H100 checkpoint, for example step 700 of `d24-h100`:

```bash
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache

python -m scripts.chat_cli -i base -g d24-h100 -s 700 --device-type cuda
```

For a one-shot prompt:

```bash
python -m scripts.chat_cli -i base -g d24-h100 -s 700 --device-type cuda -p "hello"
```

For a single-H100 full-depth training run with a safer sequence length and micro-batch size:

```bash
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/$USER/etude
export HF_HOME=/scratch/$USER/hf_cache
export WANDB_PROJECT=etude
export WANDB_ENTITY=edward40

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --device-type=cuda \
  --fp8 \
  --depth=24 \
  --max-seq-len=1024 \
  --device-batch-size=2 \
  --total-batch-size=32768 \
  --save-every=100 \
  --run="full-train-h100" \
  --model-tag="d24-h100"
```

Resume from a saved checkpoint with:

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --device-type=cuda \
  --fp8 \
  --depth=24 \
  --max-seq-len=1024 \
  --device-batch-size=2 \
  --total-batch-size=32768 \
  --save-every=100 \
  --run="full-train-h100" \
  --model-tag="d24-h100" \
  --resume-from-step=5
```

Run the same resume flow as a Slurm batch job with:

```bash
sbatch runs/d24_h100_resume.slurm
```

To automatically resubmit when Slurm kills the job for hitting the time limit, run a watcher against the batch log:

```bash
bash runs/watch_slurm_time_limit.sh runs/d24-h100-<job_id>.log
```

For known resume logs, the watcher infers the right `sbatch` file. You can also pass one explicitly:

```bash
bash runs/watch_slurm_time_limit.sh runs/d24-h100-<job_id>.log runs/d24_h100_resume.slurm
```

Follow Slurm logs with:

```bash
tail -f runs/d24-h100-*.log
```

To follow only the newest log:

```bash
tail -f "$(ls -t runs/d24-h100-*.log | head -1)"
```

Check job status with:

```bash
squeue -u $USER
sacct -j <job_id>
```

`sbatch` jobs are non-interactive, so there is no shell session to reattach to later. Use `tail -f runs/d24-h100-<job_id>.log` to watch progress. If you need a reconnectable shell on the GPU node, start an interactive allocation with `srun --pty` and run `tmux` inside it.

Training logs are printed in a compact one-line format, for example:

```text
step 00499/182414 (0.27%) | loss: 6.397844 | lrm: 1.00 | dt: 3830.24ms | tok/sec: 8,555 | bf16_mfu: 4.03 | epoch: 1 pq: 0 rg: 33 | total time: 63.01m | eta: 23441.6m
```

Field meanings:

- `step 00499/182414`: current optimizer step and total planned steps.
- `(0.27%)`: percent of planned optimization steps completed.
- `loss`: EMA-smoothed training loss for logging.
- `lrm`: learning-rate multiplier from the scheduler.
- `dt`: wall-clock time for one optimizer step, including gradient accumulation.
- `tok/sec`: effective optimizer-step throughput in tokens per second.
- `bf16_mfu`: model FLOPs utilization, as a percent of theoretical BF16 peak FLOPs.
- `epoch`: current pass through the dataset.
- `pq`: current parquet shard index.
- `rg`: current parquet row-group index within the shard.
- `total time`: accumulated training time tracked by the script.
- `eta`: estimated minutes remaining based on average measured step time so far.

See `runs/speedrun.sh` for a full end-to-end example on 8×H100 GPUs.

### Evaluate

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

## Project Structure

```
etude/              Core library
  gpt.py              Model architecture
  deltanet.py          Gated DeltaNet layer
  flash_attention.py   FA4/SDPA unified interface
  optim.py             Muon + AdamW optimizer
  engine.py            Inference engine with KV cache
  tokenizer.py         BPE tokenizer
  dataloader.py        Distributed data loading
  dataset.py           Dataset management (multi-dataset support)
  checkpoint_manager.py  Checkpoint save/load
  common.py            Shared utilities
  fp8.py               FP8 training support
  report.py            Training report generation
scripts/            Training and evaluation
  base_train.py        Pretraining (--dataset flag for data source)
  base_eval.py         Evaluation (CORE metric, BPB)
  chat_sft.py          Supervised fine-tuning
  chat_rl.py           Reinforcement learning
  chat_cli.py          Interactive CLI chat
  chat_web.py          Web chat interface
  tok_train.py         Tokenizer training (--datasets for combined training)
  tok_eval.py          Tokenizer evaluation
data/               Data preparation
  fineweb-edu/         FineWeb-Edu dataset (educational web text)
  rust/                Rust code from The Stack Dedup
tasks/              Evaluation tasks
  mmlu.py, arc.py, gsm8k.py, humaneval.py, ...
runs/               Shell scripts for training pipelines
  twostage.sh          Two-stage: FineWeb-Edu pretrain + Rust fine-tune
  speedrun.sh          Single-stage on 8×H100
tests/              Unit tests
```

## Key Features

- **Two-stage training**: General language pretraining → domain specialization
- **Multi-dataset support**: FineWeb-Edu, The Stack (Rust), with HuggingFace streaming fallback
- **Hybrid architecture**: Gated DeltaNet (linear attention) + Gated Attention for efficiency
- **Flash Attention 4**: Automatic FA4 on Ampere+ GPUs (Hopper, Blackwell), SDPA fallback elsewhere
- **FP8 training**: Optional FP8 for faster training on H100+
- **Muon optimizer**: Combined Muon (for matrices) + AdamW (for embeddings) with distributed support
- **Multi-token prediction**: Auxiliary MTP loss for improved training
- **Full inference stack**: KV cache, tool use (calculator), streaming generation
- **Scaling laws**: Automatic batch size, learning rate, and weight decay scaling based on model size

## License

See [LICENSE](LICENSE).
