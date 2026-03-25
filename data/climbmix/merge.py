"""
Merge multiple binary token files into single train.bin and val.bin files.

Usage:
    python data/climbmix/merge.py

Run this after prepare.py has produced part_*_train.bin and part_*_val.bin files.
"""

import os
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def merge_bin_files(input_files, output_file):
    """Merge multiple binary token files into one file."""
    # Filter to existing files
    existing = [f for f in input_files if os.path.exists(f)]
    if not existing:
        print(f"No input files found for {output_file}, skipping.")
        return

    # Calculate total length
    total_length = 0
    for file_path in existing:
        arr = np.memmap(file_path, dtype=np.uint16, mode="r")
        total_length += len(arr)
        print(f"  {os.path.basename(file_path)}: {len(arr):,} tokens")

    print(f"  Total: {total_length:,} tokens")

    # Create output
    merged = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(total_length,))

    current_idx = 0
    batch_size = 10 * 1024 * 1024  # ~20MB per batch
    for file_path in tqdm(existing, desc="Merging"):
        arr = np.memmap(file_path, dtype=np.uint16, mode="r")
        file_len = len(arr)
        for i in range(0, file_len, batch_size):
            end = min(i + batch_size, file_len)
            batch = arr[i:end]
            merged[current_idx : current_idx + len(batch)] = batch
            current_idx += len(batch)

    merged.flush()
    print(f"  Written to {output_file}")


if __name__ == "__main__":
    print("Merging training files...")
    train_files = [os.path.join(DATA_DIR, f"part_{i}_train.bin") for i in range(10)]
    merge_bin_files(train_files, os.path.join(DATA_DIR, "train.bin"))

    print("\nMerging validation files...")
    val_files = [os.path.join(DATA_DIR, f"part_{i}_val.bin") for i in range(10)]
    merge_bin_files(val_files, os.path.join(DATA_DIR, "val.bin"))

    print("\nDone!")
