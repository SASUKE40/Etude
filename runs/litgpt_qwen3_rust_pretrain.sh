#!/bin/bash
# Continued pretraining for Qwen3-0.6B with LitGPT + LitData + Thunder.
#
# Prerequisites:
#   python -m pip install -U \
#     "litgpt[extra,compiler]==0.5.12" \
#     "lightning-thunder>=0.2.dev20250119" \
#     "transformers>=4.51.3,<4.57" \
#     "huggingface-hub>=0.30,<1.4" \
#     "torch==2.9.1" \
#     "torchvision==0.24.1" \
#     "torchaudio==2.9.1"
#
# Usage:
#   bash runs/litgpt_qwen3_rust_pretrain.sh
#   bash runs/litgpt_qwen3_rust_pretrain.sh prepare
#   bash runs/litgpt_qwen3_rust_pretrain.sh train
#   PHASE=prepare bash runs/litgpt_qwen3_rust_pretrain.sh
#   MAX_TOKENS=100000000 MICRO_BATCH_SIZE=2 bash runs/litgpt_qwen3_rust_pretrain.sh train

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-etude}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN="${WANDB_RUN:-}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/litgpt-rust-qwen3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/$USER/litgpt-checkpoints}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/Qwen/Qwen3-0.6B}"
TEXT_DIR="${TEXT_DIR:-$SCRATCH_ROOT/text}"
LITDATA_DIR="${LITDATA_DIR:-$SCRATCH_ROOT/litdata}"
OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/out/qwen3-0.6b-rust}"
COMPILER="${COMPILER:-none}"
PRECISION="${PRECISION:-bf16-true}"
DEVICES="${DEVICES:-auto}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-10000}"
CHUNK_BYTES="${CHUNK_BYTES:-200MB}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
FILES_PER_BATCH="${FILES_PER_BATCH:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-500000000}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_MAX_ITERS="${EVAL_MAX_ITERS:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
MAX_NORM="${MAX_NORM:-1.0}"
LOGGER_NAME="${LOGGER_NAME:-wandb}"
PROJECT="${PROJECT:-$WANDB_PROJECT}"
RUN_NAME="${RUN_NAME:-$WANDB_RUN}"
RESUME="${RESUME:-auto}"
PHASE="${PHASE:-${1:-all}}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$CHECKPOINT_ROOT" "$SCRATCH_ROOT"

python - <<'PY'
from importlib import metadata
from packaging.version import Version
import sys

def base_version(value):
    if value is None:
        return None
    return value.split("+", 1)[0]

def version_or_none(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

litgpt_version = version_or_none("litgpt")
transformers_version = version_or_none("transformers")
hub_version = version_or_none("huggingface-hub")
torch_version = version_or_none("torch")
torchvision_version = version_or_none("torchvision")
torchaudio_version = version_or_none("torchaudio")

errors = []
if litgpt_version is None:
    errors.append("litgpt is not installed")
if transformers_version is None:
    errors.append("transformers is not installed")
elif Version(transformers_version) >= Version("4.57"):
    errors.append(
        f"transformers=={transformers_version} is too new for LitGPT 0.5.12; use transformers>=4.51.3,<4.57"
    )
if hub_version is None:
    errors.append("huggingface-hub is not installed")
elif Version(hub_version) >= Version("1.4"):
    errors.append(
        f"huggingface-hub=={hub_version} is outside LitGPT's supported range; use huggingface-hub>=0.30,<1.4"
    )
if torch_version is None:
    errors.append("torch is not installed")
elif base_version(torch_version) != "2.9.1":
    errors.append(f"torch=={torch_version} does not match the tested install here; use torch==2.9.1")
if torchvision_version is None:
    errors.append("torchvision is not installed")
elif base_version(torchvision_version) != "0.24.1":
    errors.append(
        f"torchvision=={torchvision_version} does not match torch==2.9.1; use torchvision==0.24.1"
    )
if torchaudio_version is None:
    errors.append("torchaudio is not installed")
elif base_version(torchaudio_version) != "2.9.1":
    errors.append(
        f"torchaudio=={torchaudio_version} does not match torch==2.9.1; use torchaudio==2.9.1"
    )

if errors:
    print("Dependency check failed for the LitGPT Qwen3 runner:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(file=sys.stderr)
    print("Install a compatible set with:", file=sys.stderr)
    print(
        '  python -m pip install -U "litgpt[extra,compiler]==0.5.12" '
        '"lightning-thunder>=0.2.dev20250119" '
        '"transformers>=4.51.3,<4.57" '
        '"huggingface-hub>=0.30,<1.4" '
        '"torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1"',
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import torchaudio  # noqa: F401
    import litgpt  # noqa: F401
except Exception as exc:
    print("Import smoke test failed for the LitGPT Qwen3 runner:", file=sys.stderr)
    print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
    print(file=sys.stderr)
    print("Reinstall a compatible set with:", file=sys.stderr)
    print(
        '  python -m pip install -U "litgpt[extra,compiler]==0.5.12" '
        '"lightning-thunder>=0.2.dev20250119" '
        '"transformers>=4.51.3,<4.57" '
        '"huggingface-hub>=0.30,<1.4" '
        '"torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1"',
        file=sys.stderr,
    )
    sys.exit(1)
PY

run_prepare() {
  echo "=== Downloading LitGPT checkpoint ==="
  litgpt download "$MODEL_NAME" --checkpoint_dir "$CHECKPOINT_ROOT"

  echo ""
  echo "=== Preparing LitData dataset ==="
  python "$REPO_ROOT/data/rust/prepare_litgpt_litdata.py" \
    --tokenizer-dir "$CHECKPOINT_DIR" \
    --root-dir "$SCRATCH_ROOT" \
    --text-output-dir "$TEXT_DIR" \
    --litdata-output-dir "$LITDATA_DIR" \
    --num-workers "$NUM_WORKERS" \
    --files-per-batch "$FILES_PER_BATCH" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --chunk-bytes "$CHUNK_BYTES" \
    --max-seq-length "$MAX_SEQ_LENGTH"
}

run_train() {
  echo "=== Starting LitGPT continued pretraining ==="

  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint directory: $CHECKPOINT_DIR" >&2
    echo "Run the prepare phase first: bash runs/litgpt_qwen3_rust_pretrain.sh prepare" >&2
    exit 1
  fi

  if [[ ! -d "$LITDATA_DIR/train" || ! -d "$LITDATA_DIR/val" ]]; then
    echo "Missing LitData dataset under $LITDATA_DIR" >&2
    echo "Run the prepare phase first: bash runs/litgpt_qwen3_rust_pretrain.sh prepare" >&2
    exit 1
  fi

  echo "Repo root: $REPO_ROOT"
  echo "Using inline Python launcher to avoid CLI/parser drift"

  export ETUDE_REPO_ROOT="$REPO_ROOT"
  export ETUDE_MODEL_NAME="$MODEL_NAME"
  export ETUDE_LITDATA_DIR="$LITDATA_DIR"
  export ETUDE_CHECKPOINT_DIR="$CHECKPOINT_DIR"
  export ETUDE_OUT_DIR="$OUT_DIR"
  export ETUDE_COMPILER="$COMPILER"
  export ETUDE_PRECISION="$PRECISION"
  export ETUDE_DEVICES="$DEVICES"
  export ETUDE_NUM_NODES="$NUM_NODES"
  export ETUDE_NUM_WORKERS="$NUM_WORKERS"
  export ETUDE_LOGGER_NAME="$LOGGER_NAME"
  export ETUDE_PROJECT="$PROJECT"
  export ETUDE_RUN_NAME="$RUN_NAME"
  export ETUDE_RESUME="$RESUME"
  export ETUDE_GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
  export ETUDE_MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE"
  export ETUDE_MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH"
  export ETUDE_MAX_TOKENS="$MAX_TOKENS"
  export ETUDE_MAX_STEPS="$MAX_STEPS"
  export ETUDE_SAVE_INTERVAL="$SAVE_INTERVAL"
  export ETUDE_LOG_INTERVAL="${LOG_INTERVAL:-10}"
  export ETUDE_EVAL_INTERVAL="$EVAL_INTERVAL"
  export ETUDE_EVAL_MAX_ITERS="$EVAL_MAX_ITERS"
  export ETUDE_LEARNING_RATE="$LEARNING_RATE"
  export ETUDE_LR_WARMUP_STEPS="$LR_WARMUP_STEPS"
  export ETUDE_MAX_NORM="$MAX_NORM"
  export ETUDE_ALLOW_CUDNN_SDP="${ALLOW_CUDNN_SDP:-0}"

  python - <<'PY'
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

repo_root = Path(os.environ["ETUDE_REPO_ROOT"]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.data import LitData
from litgpt.pretrain import setup as litgpt_pretrain


@contextmanager
def compile_backend(name: str):
    import torch

    original_compile = torch.compile
    if name == "none":
        torch.compile = lambda model, *args, **kwargs: model
        try:
            yield
        finally:
            torch.compile = original_compile
        return
    if name == "torch":
        yield
        return
    if name != "thunder":
        raise ValueError(f"Unsupported compiler backend: {name}")

    import thunder

    torch.compile = thunder.compile
    try:
        yield
    finally:
        torch.compile = original_compile


def coerce_resume(value: str):
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return Path(value)


def has_existing_checkpoint(out_dir: Path) -> bool:
    return out_dir.exists() and any(out_dir.rglob("lit_model.pth"))


def normalize_devices(value: str):
    if value == "auto":
        return value
    if value.isdigit():
        return int(value)
    return value


def install_metric_aliases(max_tokens: int) -> None:
    from lightning.fabric.utilities.throughput import ThroughputMonitor
    import lightning as L

    original_compute = ThroughputMonitor.compute
    if not getattr(ThroughputMonitor.compute, "_etude_mfu_alias", False):
        def compute_with_alias(self):
            metrics = original_compute(self)
            if "device/mfu" in metrics and "mfu" not in metrics:
                metrics["mfu"] = metrics["device/mfu"]
            alias_map = {
                "throughput/device/batches_per_sec": "batches_per_sec",
                "throughput/device/samples_per_sec": "samples_per_sec",
                "throughput/device/items_per_sec": "items_per_sec",
                "throughput/device/flops_per_sec": "flops_per_sec",
                "device/batches_per_sec": "batches_per_sec",
                "device/samples_per_sec": "samples_per_sec",
                "device/items_per_sec": "items_per_sec",
                "device/flops_per_sec": "flops_per_sec",
                "device/tokens_per_sec": "tokens_per_sec",
                "throughput/device/tokens_per_sec": "tokens_per_sec",
            }
            for source_key, alias_key in alias_map.items():
                if source_key in metrics and alias_key not in metrics:
                    metrics[alias_key] = metrics[source_key]
            return metrics

        compute_with_alias._etude_mfu_alias = True
        ThroughputMonitor.compute = compute_with_alias

    original_log_dict = L.Fabric.log_dict
    if not getattr(L.Fabric.log_dict, "_etude_metric_aliases", False):
        def log_dict_with_aliases(self, metrics, *args, **kwargs):
            aliased = dict(metrics)
            if "total_tokens" in aliased and "percent_of_token_budget" not in aliased and max_tokens > 0:
                aliased["percent_of_token_budget"] = 100.0 * aliased["total_tokens"] / max_tokens
            if "iter_time" in aliased and "tokens" in aliased and "tokens_per_sec" not in aliased:
                iter_time = aliased["iter_time"]
                tokens = aliased["tokens"]
                if isinstance(iter_time, (int, float)) and iter_time > 0 and isinstance(tokens, (int, float)):
                    aliased["tokens_per_sec"] = tokens / iter_time
            return original_log_dict(self, aliased, *args, **kwargs)

        log_dict_with_aliases._etude_metric_aliases = True
        L.Fabric.log_dict = log_dict_with_aliases


def install_checkpoint_hparam_saver(
    *,
    model_name: str,
    data_path: Path,
    tokenizer_dir: Path,
    out_dir: Path,
    precision: str,
    resume,
    num_workers: int,
    seed: int,
    save_interval: int,
    log_interval: int,
    global_batch_size: int,
    micro_batch_size: int,
    max_tokens: int,
    max_steps,
    max_seq_length: int,
    max_norm: float,
    learning_rate: float,
    lr_warmup_steps: int,
    eval_interval: int,
    eval_max_iters: int,
    project: str,
    run_name,
    devices,
    num_nodes: int,
    logger_name: str,
) -> None:
    import litgpt.pretrain as litgpt_pretrain_module

    def save_hyperparameters(_function, checkpoint_dir: Path, known_commands=None) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": model_name,
            "out_dir": str(out_dir),
            "precision": precision,
            "initial_checkpoint_dir": str(tokenizer_dir) if resume is False else None,
            "resume": str(resume),
            "data": {
                "class_path": "litgpt.data.LitData",
                "init_args": {
                    "data_path": str(data_path),
                    "split_names": ["train", "val"],
                    "num_workers": num_workers,
                    "seed": seed,
                },
            },
            "train": {
                "save_interval": save_interval,
                "log_interval": log_interval,
                "global_batch_size": global_batch_size,
                "micro_batch_size": micro_batch_size,
                "max_tokens": max_tokens,
                "max_steps": max_steps,
                "max_seq_length": max_seq_length,
                "max_norm": max_norm,
                "min_lr": learning_rate,
                "lr_warmup_steps": lr_warmup_steps,
            },
            "eval": {
                "interval": eval_interval,
                "max_iters": eval_max_iters,
                "initial_validation": False,
                "final_validation": True,
            },
            "log": {
                "project": project,
                "run": run_name,
            },
            "devices": devices,
            "num_nodes": num_nodes,
            "tokenizer_dir": str(tokenizer_dir),
            "logger_name": logger_name,
            "seed": seed,
        }
        (checkpoint_dir / "hyperparameters.yaml").write_text(json.dumps(payload, indent=2) + "\n")

    litgpt_pretrain_module.save_hyperparameters = save_hyperparameters


model_name = os.environ["ETUDE_MODEL_NAME"]
data_path = Path(os.environ["ETUDE_LITDATA_DIR"])
tokenizer_dir = Path(os.environ["ETUDE_CHECKPOINT_DIR"])
out_dir = Path(os.environ["ETUDE_OUT_DIR"])
compiler = os.environ["ETUDE_COMPILER"]
precision = os.environ["ETUDE_PRECISION"]
devices = normalize_devices(os.environ["ETUDE_DEVICES"])
num_nodes = int(os.environ["ETUDE_NUM_NODES"])
num_workers = int(os.environ["ETUDE_NUM_WORKERS"])
logger_name = os.environ["ETUDE_LOGGER_NAME"]
project = os.environ["ETUDE_PROJECT"]
run_name = os.environ.get("ETUDE_RUN_NAME") or None
resume = coerce_resume(os.environ["ETUDE_RESUME"])
global_batch_size = int(os.environ["ETUDE_GLOBAL_BATCH_SIZE"])
micro_batch_size = int(os.environ["ETUDE_MICRO_BATCH_SIZE"])
max_seq_length = int(os.environ["ETUDE_MAX_SEQ_LENGTH"])
max_tokens = int(os.environ["ETUDE_MAX_TOKENS"])
max_steps_raw = os.environ.get("ETUDE_MAX_STEPS", "")
max_steps = int(max_steps_raw) if max_steps_raw else None
save_interval = int(os.environ["ETUDE_SAVE_INTERVAL"])
log_interval = int(os.environ["ETUDE_LOG_INTERVAL"])
eval_interval = int(os.environ["ETUDE_EVAL_INTERVAL"])
eval_max_iters = int(os.environ["ETUDE_EVAL_MAX_ITERS"])
learning_rate = float(os.environ["ETUDE_LEARNING_RATE"])
lr_warmup_steps = int(os.environ["ETUDE_LR_WARMUP_STEPS"])
max_norm = float(os.environ["ETUDE_MAX_NORM"])

import torch
if os.environ.get("ETUDE_ALLOW_CUDNN_SDP", "0") != "1" and torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    print("Disabled cuDNN SDPA for stability; using alternate SDPA backends.")

install_metric_aliases(max_tokens)
install_checkpoint_hparam_saver(
    model_name=model_name,
    data_path=data_path,
    tokenizer_dir=tokenizer_dir,
    out_dir=out_dir,
    precision=precision,
    resume=resume,
    num_workers=num_workers,
    seed=42,
    save_interval=save_interval,
    log_interval=log_interval,
    global_batch_size=global_batch_size,
    micro_batch_size=micro_batch_size,
    max_tokens=max_tokens,
    max_steps=max_steps,
    max_seq_length=max_seq_length,
    max_norm=max_norm,
    learning_rate=learning_rate,
    lr_warmup_steps=lr_warmup_steps,
    eval_interval=eval_interval,
    eval_max_iters=eval_max_iters,
    project=project,
    run_name=run_name,
    devices=devices,
    num_nodes=num_nodes,
    logger_name=logger_name,
)

train_args = TrainArgs(
    save_interval=save_interval,
    log_interval=log_interval,
    global_batch_size=global_batch_size,
    micro_batch_size=micro_batch_size,
    max_tokens=max_tokens,
    max_steps=max_steps,
    max_seq_length=max_seq_length,
    max_norm=max_norm,
    min_lr=learning_rate,
    lr_warmup_steps=lr_warmup_steps,
)
eval_args = EvalArgs(
    interval=eval_interval,
    max_iters=eval_max_iters,
    initial_validation=False,
    final_validation=True,
)
log_args = LogArgs(project=project, run=run_name)
data = LitData(data_path=data_path, split_names=("train", "val"), num_workers=num_workers, seed=42)

initial_checkpoint_dir = None
if resume == "auto":
    if has_existing_checkpoint(out_dir):
        initial_checkpoint_dir = None
    else:
        resume = False
        initial_checkpoint_dir = tokenizer_dir
elif resume is False:
    initial_checkpoint_dir = tokenizer_dir

with compile_backend(compiler):
    litgpt_pretrain(
        model_name=model_name,
        out_dir=out_dir,
        precision=precision,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        data=data,
        train=train_args,
        eval=eval_args,
        tokenizer_dir=tokenizer_dir,
        devices=devices,
        num_nodes=num_nodes,
        logger_name=logger_name,
        log=log_args,
        seed=42,
    )
PY
}

case "$PHASE" in
  all)
    run_prepare
    echo ""
    run_train
    ;;
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    echo "Expected one of: all, prepare, train" >&2
    exit 1
    ;;
esac
