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
| [Nemotron-Cascade-SFT-Stage-2](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2) | Multi-domain chat SFT data | ~7.8M conversations | Large, streamable | Optional chat SFT |

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

# Rust train split as sharded .txt files in /scratch
bash data/rust/prepare_txt.sh
```

The text export downloads `ammarnasr/the-stack-rust-clean` and writes sharded
`train_*.txt` files plus `metadata.json` to
`/scratch/$USER/the-stack-rust-clean/train_txt` by default. To override the
destination:

```bash
python data/rust/prepare_txt.py --output-dir /scratch/$USER/the-stack-rust-clean/train_txt
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

For a single-H100 Slurm stage transition from an existing base checkpoint to the Rust dataset, use:

```bash
sbatch runs/d24_rust_resume.slurm
```

That launcher defaults to:

- source checkpoint dir: `/scratch/zhu.shili/etude/base_checkpoints/d24-h100-rust-s2-010200`
- source tag: basename of the source checkpoint directory
- source step: latest complete checkpoint in the source checkpoint directory
- output tag: same as the source tag, so it resumes that Rust run in place by default

Override these at submit time if needed, for example:

```bash
RESUME_STEP=12000 MODEL_TAG=d24-rust-s2-v2 sbatch runs/d24_rust_resume.slurm
```

To auto-resubmit the Rust Slurm chain when jobs hit the time limit, use:

```bash
bash runs/watch_slurm_time_limit.sh d24-rust
```

#### 5. Optional: Prepare Chat SFT Data

```bash
export HF_HOME=/scratch/$USER/hf_cache

python data/nemotron-cascade-sft-stage-2/prepare.py \
    --output-dir /scratch/$USER/etude/datasets/nemotron-cascade-sft-stage-2
```

The prepare step streams the source dataset and writes deterministic `train/` and `val/`
parquet shards. By default it only downloads the `instruction-following` subset.
To override that subset list explicitly:

```bash
python data/nemotron-cascade-sft-stage-2/prepare.py \
    --subsets instruction-following \
    --max-rows-per-subset 50000
```

#### 6. Optional: Chat SFT

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --model-tag="twostage-s2" \
    --output-model-tag="twostage-s2-chat" \
    --chat-dataset=nemotron-cascade-sft-stage-2 \
    --chat-data-dir /scratch/$USER/etude/datasets/nemotron-cascade-sft-stage-2
```

`scripts/chat_sft.py` defaults to `--chat-dataset=auto`: it uses prepared Nemotron data if
present and falls back to the old legacy in-memory mixture for quick local smoke runs.

For the single-GPU Slurm launcher, `runs/d24_chat_sft.slurm` now separates the base
checkpoint source from the SFT output tag, and supports true SFT resume:

```bash
MODEL_TAG=d24-h100-rust-s2-010200-sft-008800 RESUME_FROM_STEP=500 sbatch runs/d24_chat_sft.slurm
```

To auto-resubmit the chat SFT Slurm chain when jobs hit the time limit, use:

```bash
bash runs/watch_slurm_time_limit.sh d24-sft
```

#### 7. Chat with the Model

```bash
python -m scripts.chat_cli -g twostage-s2-chat
python -m scripts.chat_web
```

If you want to inspect the Rust-specialized base checkpoint before chat SFT, use:

```bash
python -m scripts.chat_cli -i base -g twostage-s2
```

For the D24 H100 Rust stage-2 base checkpoint:

```bash
python -m scripts.chat_cli -i base -g d24-h100-rust-s2-010200 --device-type cuda --max-tokens 2048
```

To chat with a specific chat-SFT checkpoint saved under `chatsft_checkpoints`, point
`ETUDE_BASE_DIR` at the directory root, then pass the checkpoint directory name as the
model tag and the checkpoint suffix as the step. For example, for:

```text
/scratch/zhu.shili/etude/chatsft_checkpoints/d24-h100-rust-s2-010200-sft-008800/model_000500.pt
```

use:

```bash
cd ~/Etude && source .venv/bin/activate
export ETUDE_BASE_DIR=/scratch/zhu.shili/etude
export HF_HOME=/scratch/zhu.shili/hf_cache

python -m scripts.chat_cli \
    -g d24-h100-rust-s2-010200-sft-008800 \
    -s 500 \
    --device-type cuda
```

For a one-shot prompt:

```bash
python -m scripts.chat_cli \
    -g d24-h100-rust-s2-010200-sft-008800 \
    -s 500 \
    --device-type cuda \
    -p "Hello"
```

For the web UI:

```bash
python -m scripts.chat_web \
    -g d24-h100-rust-s2-010200-sft-008800 \
    -s 500 \
    --device-type cuda
```

To use the latest saved checkpoint from a chat-SFT run, look for the highest
`model_*.pt` suffix in that checkpoint directory. For example, if:

```text
/scratch/zhu.shili/etude/chatsft_checkpoints/d24-h100-rust-s2-010200-sft-008800/
  model_001500.pt
  meta_001500.json
```

then the latest checkpoint is step `1500`, and you can chat with it directly:

```bash
python -m scripts.chat_cli \
    -g d24-h100-rust-s2-010200-sft-008800 \
    -s 1500 \
    --device-type cuda
```

You can also omit `-s` and let Etude load the latest available checkpoint
automatically for that model tag:

```bash
python -m scripts.chat_cli \
    -g d24-h100-rust-s2-010200-sft-008800 \
    --device-type cuda
```

To back up only the latest base checkpoint from each folder to your home directory:

```bash
bash runs/backup_base_checkpoints.sh /scratch/zhu.shili/etude/base_checkpoints ~/base_checkpoints
```

Without arguments, the script defaults to
`${ETUDE_BASE_DIR:-/scratch/$USER/etude}/base_checkpoints` as the source and
`$HOME/base_checkpoints` as the destination.

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

For the Lightning-based Qwen Rust pretraining entrypoint, run:

```bash
python -m scripts.qwen_base_train \
    --model-name Qwen/Qwen3.5-0.8B-Base \
    --max-length 2048 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --warmup-steps 500 \
    --total-steps 50000 \
    --checkpoint-dir checkpoints/qwen-rust \
    --output-dir qwen3.5-0.8b-rust
```

There is also a matching Slurm launcher for a 1×H100 run:

```bash
sbatch runs/qwen_base_train.slurm
```

Override the Slurm defaults at submit time with environment variables, for example:

```bash
TOTAL_STEPS=10000 BATCH_SIZE=2 OUTPUT_DIR=/scratch/$USER/qwen-rust sbatch runs/qwen_base_train.slurm
```

For a LitGPT-based continued pretraining path using Qwen3-0.6B, LitData, and
Lightning Thunder on the cleaned Rust dataset (`ammarnasr/the-stack-rust-clean`),
install the extra dependencies:

```bash
uv pip install -U \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.9.1" \
    "torchvision==0.24.1" \
    "torchaudio==2.9.1"

uv pip install -U \
    "litgpt[extra,compiler]==0.5.12" \
    "lightning-thunder>=0.2.dev20250119" \
    "transformers>=4.51.3,<4.57" \
    "huggingface-hub>=0.30,<1.4"
```

`transformers>=4.57` is too new for the current LitGPT stack here and can fail
during `litgpt download` with a `huggingface_hub` import error.

If you see a `torchvision` import crash during `litgpt download`, that usually
means your `torch` and `torchvision` builds do not match. The pinned PyTorch
triplet above matches the official PyTorch 2.9.1 compatibility table.

If `litgpt` crashes inside `bitsandbytes` with a CUDA driver error on older
nodes, remove `bitsandbytes` from the environment. It is not required for this
workflow:

```bash
uv pip uninstall bitsandbytes
```

If you see `warnings.warn("transformer_engine module not found!")` from Thunder,
that is only an optional-acceleration warning. The workflow can continue
without Transformer Engine installed. If you want to enable that backend
explicitly, install:

```bash
uv pip install "transformer_engine[pytorch]"
```

Then run:

```bash
bash runs/litgpt_qwen3_rust_pretrain.sh
```

For Slurm or other time-limited environments, it is better to split the
workflow into separate prepare and train phases:

```bash
# CPU / I/O-heavy one-time data preparation
NUM_WORKERS=8 FILES_PER_BATCH=8 bash runs/litgpt_qwen3_rust_pretrain.sh prepare

# GPU training after the LitData dataset already exists
MAX_TOKENS=100000000 MICRO_BATCH_SIZE=2 bash runs/litgpt_qwen3_rust_pretrain.sh train
```

The default phase is `all`, which runs prepare and train in sequence. You can
also select the phase with `PHASE=prepare` or `PHASE=train`.

For a single-H100 Slurm workflow with 1-hour jobs, use the dedicated resumable
training launcher after the prepare phase is complete:

```bash
sbatch runs/litgpt_qwen3_rust_resume.slurm
```

Useful overrides:

```bash
WANDB_RUN=qwen3-rust-h100 SAVE_INTERVAL=100 sbatch runs/litgpt_qwen3_rust_resume.slurm
OUT_DIR=/scratch/$USER/litgpt-rust-qwen3/out/debug MAX_STEPS=100 sbatch runs/litgpt_qwen3_rust_resume.slurm
```

To find the newest LitGPT Slurm logs:

```bash
ls -t runs/litgpt-rust-*.log | head
```

To follow the live LitGPT Slurm logs:

```bash
tail -f runs/litgpt-rust-*.log
```

That Slurm launcher defaults to:

- `PHASE=train`
- `RESUME=auto`
- `COMPILER=none`
- `SAVE_INTERVAL=100`

So rerunning the same `sbatch` command will continue from the latest checkpoint
in the output directory when one exists.

To chat with Qwen3 directly through LitGPT using the Hugging Face model ID:

```bash
litgpt chat Qwen/Qwen3-0.6B
```

If you want the raw base model instead of the chat/post-trained checkpoint:

```bash
litgpt chat Qwen/Qwen3-0.6B-Base
```

To test a saved LitGPT checkpoint on this cluster, use the wrapper script
instead of the raw `litgpt chat` CLI. It links the missing tokenizer/config
files from the base Qwen checkpoint into the step directory and disables cuDNN
SDPA by default to avoid `CUDNN_STATUS_NOT_INITIALIZED` during inference:

```bash
python scripts/litgpt_infer_checkpoint.py chat \
  /scratch/$USER/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-00000100
```

Example on the current cluster:

```bash
python scripts/litgpt_infer_checkpoint.py chat \
  /scratch/zhu.shili/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-00000100
```

For plain completion-style prompting:

```bash
python scripts/litgpt_infer_checkpoint.py generate \
  /scratch/$USER/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-00000100 \
  --prompt "Write a Rust function that parses a TOML file."
```

To grab the newest checkpoint automatically:

```bash
LATEST="$(ls -td /scratch/$USER/litgpt-rust-qwen3/out/qwen3-0.6b-rust/step-* | head -1)"
python scripts/litgpt_infer_checkpoint.py chat "$LATEST"
```

If your base LitGPT checkpoint lives somewhere else, override it explicitly:

```bash
python scripts/litgpt_infer_checkpoint.py chat "$LATEST" \
  --base-checkpoint-dir /scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B
```

To chat with a full-SFT checkpoint produced by
`runs/litgpt_qwen3_nemotron_instruction_following_sft.slurm`, point the wrapper
at the saved `out/step-*` directory and provide the base Qwen checkpoint for
tokenizer/config metadata:

```bash
python scripts/litgpt_infer_checkpoint.py chat \
  /scratch/$USER/litgpt-nemotron-sft/qwen3-0.6b-rust-step-00001800-nemotron-instruction-following/out/step-000700 \
  --base-checkpoint-dir /scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B
```

If that step directory already contains the tokenizer/config files (for example
after exporting or copying a self-contained checkpoint folder), you can also
chat with it directly through LitGPT:

```bash
litgpt chat /scratch/$USER/litgpt-nemotron-sft/qwen3-0.6b-rust-step-00001800-nemotron-instruction-following/out/step-000700 \
  --max_new_tokens 512
```

If you exported a checkpoint into a self-contained folder, you can also use the
helper script:

```bash
bash runs/chat_litgpt_export.sh /scratch/zhu.shili/exports/qwen3-0.6b-rust
```

To watch the LitGPT full-SFT Slurm job and auto-resubmit if it hits the time
limit, use:

```bash
bash runs/watch_slurm_time_limit.sh litgpt-sft
```

To just follow the newest SFT log live:

```bash
tail -f "$(ls -t runs/litgpt-sft-*.log | head -1)"
```

For a Rust-focused LitGPT full-SFT run using
`Fortytwo-Network/Strandset-Rust-v1`, prepare the dataset and submit the
dedicated Slurm launcher:

```bash
bash runs/prepare_litgpt_strandset_rust.sh
sbatch runs/litgpt_qwen3_strandset_rust_sft.slurm
```

That workflow writes under:

```bash
/scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1
```

To chat directly with a saved Strandset-Rust SFT checkpoint through LitGPT:

```bash
srun --partition=sharing --gres=gpu:h100:1 --pty bash
litgpt chat /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/step-000100 \
  --max_new_tokens 10000
```

After training completes, the self-contained final checkpoint can be loaded
directly as well:

```bash
srun --partition=sharing --gres=gpu:h100:1 --pty bash
litgpt chat /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final \
  --max_new_tokens 10000
```

If you want to point at the final `model.pth` file directly instead of the
checkpoint directory, use the wrapper:

```bash
bash runs/chat_strandset_final_model.sh
```

Or call the inference helper directly with a `model.pth` path:

```bash
python scripts/litgpt_infer_checkpoint.py chat \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/model.pth
```

To chat with the same checkpoint through Hugging Face Transformers, use:

```bash
python scripts/chat_transformers_from_litgpt.py \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/model.pth \
  --hf-dir /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/hf
```

That script does not convert the LitGPT checkpoint. It expects an already
converted Hugging Face checkpoint directory and loads it with
`AutoTokenizer` and `AutoModelForCausalLM`:

```bash
python scripts/chat_transformers_from_litgpt.py \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/model.pth \
  --hf-dir /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/hf
```

To convert a LitGPT checkpoint into a Hugging Face checkpoint directory:

```bash
bash runs/convert_litgpt_to_hf.sh \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/hf
```

To upload the raw LitGPT final checkpoint folder to the Hub:

```bash
hf auth login
bash runs/upload_hf_model.sh \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final \
  Edward40/etude
```

To upload the converted Hugging Face checkpoint directory instead:

```bash
hf auth login
bash runs/upload_hf_transformers_model.sh \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final/hf \
  Edward40/etude
```

To upload a single `model.pth` file, for example a converted step checkpoint:

```bash
hf auth login
bash runs/upload_hf_model_pth.sh \
  /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/step-003200/hf/model.pth \
  Edward40/etude \
  checkpoints/step-003200/model.pth
```

The Slurm log for:

```bash
sbatch runs/litgpt_qwen3_strandset_rust_sft.slurm
```

will be written to:

```bash
runs/litgpt-strand-sft-<jobid>.log
```

To follow the newest Strandset SFT log live:

```bash
tail -f "$(ls -t runs/litgpt-strand-sft-*.log | head -1)"
```

To auto-resubmit that job family when it hits the time limit:

```bash
bash runs/watch_slurm_time_limit.sh litgpt-strand-sft
```

That launcher will:

- download the LitGPT-compatible `Qwen/Qwen3-0.6B` checkpoint
- export the Rust `train` and `valid` splits to sharded text under `/scratch`
- convert those shards into LitData token chunks
- start continued pretraining with LitGPT using Thunder as the compile backend by default

Useful overrides:

```bash
MAX_TOKENS=100000000 MICRO_BATCH_SIZE=2 bash runs/litgpt_qwen3_rust_pretrain.sh
COMPILER=torch OUT_DIR=/scratch/$USER/litgpt-rust-qwen3/out/smoke bash runs/litgpt_qwen3_rust_pretrain.sh
```

For manual control, the dataset preparation and training entrypoints are:

```bash
python data/rust/prepare_litgpt_litdata.py \
    --tokenizer-dir /scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B

python scripts/litgpt_qwen3_rust_pretrain.py \
    --model-name Qwen/Qwen3-0.6B \
    --data-path /scratch/$USER/litgpt-rust-qwen3/litdata \
    --tokenizer-dir /scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B \
    --out-dir /scratch/$USER/litgpt-rust-qwen3/out/qwen3-0.6b-rust
```

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

The Slurm resume launcher is tuned a bit more aggressively than the simple one-off `torchrun` example above. Its current defaults are:

- `DEVICE_BATCH_SIZE=4` with `MAX_SEQ_LEN=1024` and `TOTAL_BATCH_SIZE=32768` (`grad_accum=8`)
- `LOG_EVERY=10`
- `SAVE_EVERY=200`
- `EVAL_EVERY=1000`
- `EVAL_TOKENS=4194304`
- `CORE_METRIC_EVERY=-1`
- `SAMPLE_EVERY=-1`

All of these can be overridden at submit time with environment variables, for example:

```bash
DEVICE_BATCH_SIZE=8 EVAL_EVERY=-1 sbatch runs/d24_h100_resume.slurm
```

To automatically resubmit when Slurm kills the job for hitting the time limit, run a watcher against the batch log:

```bash
bash runs/watch_slurm_time_limit.sh d24-h100
```

The known prefixes are `d24-h100`, `d24-h200`, `d24-rust`, `d24-sft`, `litgpt-rust`, and `litgpt-sft`. The watcher auto-detects the newest matching `runs/<prefix>-<job_id>.log` file by modification time, infers the matching `sbatch` script, and keeps following each newly resumed Slurm log until you stop it. If no matching log exists yet, it waits for the first one to appear. You can still pass an explicit log file or `sbatch` file:

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
