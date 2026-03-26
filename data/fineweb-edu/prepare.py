"""
Download FineWeb-Edu dataset as parquet files for training.

FineWeb-Edu is a high-quality educational subset of FineWeb, filtered using
a classifier trained on LLM-annotated educational content.

Uses the 10BT sample split by default, which is a good size for pretraining.

Usage:
    # Download 10BT sample (recommended, ~10B tokens)
    python data/fineweb-edu/prepare.py

    # Custom output directory (e.g. scratch to avoid home quota)
    python data/fineweb-edu/prepare.py --output-dir /scratch/$USER/etude_data/fineweb-edu

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import os
import glob
import argparse

DEFAULT_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HF_DATASET = "HuggingFaceFW/fineweb-edu"


def prepare(output_dir, name="sample-10BT"):
    """Download FineWeb-Edu parquet files using HuggingFace datasets."""
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    existing = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
    if existing:
        print(f"Found {len(existing)} existing parquet files in {output_dir}")
        print("Delete them to re-download, or use as-is.")
        return

    print(f"Downloading {HF_DATASET} ({name}) ...")
    print(f"Output directory: {output_dir}")

    # load_dataset will download and cache the parquet files
    # We then save them to our output directory
    dataset = load_dataset(HF_DATASET, name=name, split="train")

    # Save as parquet shards
    num_shards = max(1, len(dataset) // 500_000)  # ~500K rows per shard
    print(f"Saving {len(dataset):,} rows as {num_shards} parquet shards...")

    for shard_idx in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        filepath = os.path.join(output_dir, f"part_{shard_idx:04d}.parquet")
        shard.to_parquet(filepath)
        print(f"  Saved shard {shard_idx}/{num_shards} -> {filepath} ({len(shard):,} rows)")

    print(f"\nDone! {num_shards} parquet files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--name", type=str, default="sample-10BT",
        help="Dataset config name (default: sample-10BT). Use 'sample-100BT' for 100B tokens.",
    )
    args = parser.parse_args()

    prepare(output_dir=args.output_dir, name=args.name)
