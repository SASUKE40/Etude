# Etude

A compact language model for Rust code generation.

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

## Project Structure

```
etude/              Core library
  gpt.py              Model architecture
  deltanet.py          Gated DeltaNet layer
  flash_attention.py   FA3/SDPA unified interface
  optim.py             Muon + AdamW optimizer
  engine.py            Inference engine with KV cache
  tokenizer.py         BPE tokenizer
  dataloader.py        Distributed data loading
  dataset.py           Dataset utilities
  checkpoint_manager.py  Checkpoint save/load
  common.py            Shared utilities
  fp8.py               FP8 training support
  report.py            Training report generation
scripts/            Training and evaluation
  base_train.py        Pretraining
  base_eval.py         Evaluation (CORE metric, BPB)
  chat_sft.py          Supervised fine-tuning
  chat_rl.py           Reinforcement learning
  chat_cli.py          Interactive CLI chat
  chat_web.py          Web chat interface
  tok_train.py         Tokenizer training
  tok_eval.py          Tokenizer evaluation
data/               Data preparation
  climbmix/
    prepare.py         Download & tokenize Nemotron-ClimbMix
    merge.py           Merge tokenized parts into train.bin/val.bin
    prepare.sh         End-to-end data preparation script
tasks/              Evaluation tasks
  mmlu.py, arc.py, gsm8k.py, humaneval.py, ...
runs/               Shell scripts for training pipelines
tests/              Unit tests
```

## Quick Start

### Install

```bash
# GPU (CUDA 12.8)
uv sync --extra gpu

# CPU / Apple Silicon
uv sync --extra cpu
```

### Prepare Data

The training uses the [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) dataset (400B tokens).

**Option 1: Binary files (recommended, nanoGPT-style)**

```bash
# Prepare all 10 parts (downloads from HuggingFace, tokenizes with GPT-2 BPE)
bash data/climbmix/prepare.sh

# Or prepare a single part for quick testing
python data/climbmix/prepare.py --part 0
python data/climbmix/merge.py
```

**Option 2: Parquet files (streaming, tokenized on-the-fly)**

```bash
# Download parquet shards (~170 shards is enough for GPT-2 scale)
python -m etude.dataset -n 170
```

### Train

```bash
# Pretrain on 8 GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24

# Pretrain on single GPU
python -m scripts.base_train --depth=24

# CPU / MacBook demo (tiny model)
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
```

### Evaluate

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

### Chat

```bash
# CLI
python -m scripts.chat_cli

# Web UI
python -m scripts.chat_web
```

## Training Pipeline

1. **Tokenizer training** — `scripts/tok_train.py`
2. **Pretraining** — `scripts/base_train.py` (distributed, gradient accumulation, FP8 optional)
3. **Supervised fine-tuning** — `scripts/chat_sft.py`
4. **Reinforcement learning** — `scripts/chat_rl.py`
5. **Evaluation** — `scripts/base_eval.py`, `scripts/chat_eval.py`

See `runs/speedrun.sh` for a full end-to-end example on 8×H100 GPUs.

## Key Features

- **Hybrid architecture**: Gated DeltaNet (linear attention) + Gated Attention for efficiency
- **Flash Attention 3**: Automatic FA3 on Hopper GPUs, SDPA fallback elsewhere
- **FP8 training**: Optional FP8 for faster training on H100+
- **Muon optimizer**: Combined Muon (for matrices) + AdamW (for embeddings) with distributed support
- **Multi-token prediction**: Auxiliary MTP loss for improved training
- **Full inference stack**: KV cache, tool use (calculator), streaming generation

## License

See [LICENSE](LICENSE).
