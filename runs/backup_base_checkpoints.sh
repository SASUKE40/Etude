#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash runs/backup_base_checkpoints.sh [source_dir] [dest_dir]

Back up only the latest model checkpoint from each base checkpoint folder.
The matching meta_*.json file is copied too when present, so the model remains
loadable by the Etude checkpoint loader.

Defaults:
  source_dir: ${ETUDE_BASE_DIR:-/scratch/$USER/etude}/base_checkpoints
  dest_dir:   $HOME/base_checkpoints

Examples:
  bash runs/backup_base_checkpoints.sh
  bash runs/backup_base_checkpoints.sh /scratch/zhu.shili/etude/base_checkpoints ~/base_checkpoints
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SOURCE_DIR="${1:-${ETUDE_BASE_DIR:-/scratch/$USER/etude}/base_checkpoints}"
DEST_DIR="${2:-$HOME/base_checkpoints}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: source directory does not exist: $SOURCE_DIR" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required but was not found in PATH" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

echo "Backing up base checkpoints"
echo "  from: $SOURCE_DIR/"
echo "  to:   $DEST_DIR/"
echo

copy_latest_checkpoint() {
    local checkpoint_dir="$1"
    local checkpoint_name
    local latest_model
    local step
    local meta_file
    local output_dir

    checkpoint_name="$(basename "$checkpoint_dir")"
    latest_model="$(
        find "$checkpoint_dir" -maxdepth 1 -type f -name 'model_*.pt' -print | awk '
            {
                step = $0
                sub(/^.*\/model_/, "", step)
                sub(/\.pt$/, "", step)
                if (!seen || step + 0 > max_step) {
                    max_step = step + 0
                    latest = $0
                    seen = 1
                }
            }
            END {
                if (seen) {
                    print latest
                }
            }
        '
    )"

    if [ -z "$latest_model" ]; then
        echo "Skipping $checkpoint_name: no model_*.pt files found"
        return 0
    fi

    step="$(basename "$latest_model")"
    step="${step#model_}"
    step="${step%.pt}"
    meta_file="$checkpoint_dir/meta_${step}.json"
    output_dir="$DEST_DIR/$checkpoint_name"

    mkdir -p "$output_dir"

    echo "Backing up $checkpoint_name step $step"
    rsync -aP "$latest_model" "$output_dir/"

    if [ -f "$meta_file" ]; then
        rsync -aP "$meta_file" "$output_dir/"
    else
        echo "  WARNING: matching metadata not found: $meta_file" >&2
    fi
    echo
}

while IFS= read -r checkpoint_dir; do
    copy_latest_checkpoint "$checkpoint_dir"
done < <(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -print | sort)

echo
echo "Backup complete."
du -sh "$DEST_DIR" 2>/dev/null || true
