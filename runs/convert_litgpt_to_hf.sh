#!/bin/bash
# Convert a LitGPT checkpoint directory into a Hugging Face Transformers checkpoint.
#
# Examples:
#   bash runs/convert_litgpt_to_hf.sh
#   bash runs/convert_litgpt_to_hf.sh /scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final
#   bash runs/convert_litgpt_to_hf.sh /path/to/checkpoint /path/to/output

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

CHECKPOINT_DIR="${1:-/scratch/$USER/litgpt-strandset-rust-sft/qwen3-0.6b-rust-step-00001800-strandset-rust-v1/out/final}"
OUTPUT_DIR="${2:-${CHECKPOINT_DIR%/}/hf}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 1
fi

export CHECKPOINT_DIR
export OUTPUT_DIR
export BASE_CHECKPOINT_DIR

python - <<'PY'
import importlib.util
import os
from pathlib import Path

metadata_files = (
    "config.json",
    "generation_config.json",
    "model_config.yaml",
    "prompt_style.yaml",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]).expanduser().resolve()
output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser().resolve()
base_checkpoint_dir = Path(os.environ["BASE_CHECKPOINT_DIR"]).expanduser().resolve()
lit_weights_path = checkpoint_dir / "lit_model.pth"

if not lit_weights_path.is_file():
    raise SystemExit(f"ERROR: missing LitGPT weights at {lit_weights_path}")

if not base_checkpoint_dir.is_dir():
    raise SystemExit(f"ERROR: base checkpoint directory does not exist: {base_checkpoint_dir}")

if importlib.util.find_spec("litgpt") is None:
    raise SystemExit(
        "ERROR: Python package `litgpt` is not installed in the active environment.\n"
        "Activate the correct environment or install the repo dependencies first, e.g.:\n"
        '  python -m pip install -U "litgpt[extra,compiler]==0.5.12" '
        '"transformers>=4.51.3,<4.57" "huggingface-hub>=0.30,<1.4"'
    )

linked = []
for filename in metadata_files:
    source = base_checkpoint_dir / filename
    target = checkpoint_dir / filename
    if target.exists() or not source.exists():
        continue
    target.symlink_to(source)
    linked.append(filename)

output_dir.parent.mkdir(parents=True, exist_ok=True)

print(f"Checkpoint: {checkpoint_dir}")
print(f"Output: {output_dir}")
if linked:
    print(f"Linked metadata from {base_checkpoint_dir}: {', '.join(linked)}")
PY

python -m litgpt convert from_litgpt \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR"

echo "Converted Hugging Face checkpoint written to: $OUTPUT_DIR"
