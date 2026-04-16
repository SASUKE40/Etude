"""
Download the Rust train split as sharded plain-text files under /scratch.

Usage:
    python data/rust/prepare_txt.py
    python data/rust/prepare_txt.py --output-dir /scratch/$USER/the-stack-rust-clean
    python data/rust/prepare_txt.py --rows-per-shard 5000

Dataset: https://huggingface.co/datasets/ammarnasr/the-stack-rust-clean
Split: train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

HF_DATASET = "ammarnasr/the-stack-rust-clean"
DEFAULT_ROWS_PER_SHARD = 10_000
DEFAULT_SEPARATOR = "\n\n<|endoffile|>\n\n"


def _default_output_dir() -> Path:
    scratch_root = Path("/scratch") / os.environ.get("USER", "unknown")
    return scratch_root / "the-stack-rust-clean" / "train_txt"


def prepare(
    output_dir: Path,
    rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
    separator: str = DEFAULT_SEPARATOR,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("train_*.txt"))
    if existing and not overwrite:
        print(f"Found {len(existing)} existing shard(s) in {output_dir}")
        print("Use --overwrite to replace them.")
        return

    if overwrite:
        for path in existing:
            path.unlink()

    print(f"Streaming {HF_DATASET} train split...")
    print(f"Writing sharded text files to {output_dir}")

    dataset = load_dataset(HF_DATASET, split="train", streaming=True)

    shard_idx = 0
    shard_rows = 0
    total_rows = 0
    total_chars = 0
    shard_path = output_dir / f"train_{shard_idx:05d}.txt"
    handle = shard_path.open("w", encoding="utf-8")

    try:
        for example in dataset:
            content = example["content"]
            if not isinstance(content, str) or not content:
                continue

            if shard_rows > 0:
                handle.write(separator)
                total_chars += len(separator)

            handle.write(content)
            shard_rows += 1
            total_rows += 1
            total_chars += len(content)

            if shard_rows >= rows_per_shard:
                handle.close()
                print(
                    f"Wrote {shard_path.name} with {shard_rows:,} rows "
                    f"(total rows: {total_rows:,})"
                )
                shard_idx += 1
                shard_rows = 0
                shard_path = output_dir / f"train_{shard_idx:05d}.txt"
                handle = shard_path.open("w", encoding="utf-8")
    finally:
        handle.close()

    if shard_rows == 0 and shard_path.exists():
        shard_path.unlink()
    else:
        print(
            f"Wrote {shard_path.name} with {shard_rows:,} rows "
            f"(total rows: {total_rows:,})"
        )

    metadata = {
        "dataset": HF_DATASET,
        "split": "train",
        "format": "txt",
        "rows_per_shard": rows_per_shard,
        "separator": separator,
        "num_rows": total_rows,
        "num_chars": total_chars,
        "num_shards": shard_idx + (1 if shard_rows > 0 else 0),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Done!")
    print(f"Rows written: {total_rows:,}")
    print(f"Shards written: {metadata['num_shards']:,}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the Rust train split as sharded plain-text files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Output directory for .txt shards under /scratch",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=DEFAULT_ROWS_PER_SHARD,
        help=f"Number of rows per .txt shard (default: {DEFAULT_ROWS_PER_SHARD})",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=DEFAULT_SEPARATOR,
        help="Separator inserted between files inside each shard",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing train_*.txt shards before writing",
    )
    args = parser.parse_args()

    if args.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be greater than 0")

    prepare(
        output_dir=args.output_dir,
        rows_per_shard=args.rows_per_shard,
        separator=args.separator,
        overwrite=args.overwrite,
    )
