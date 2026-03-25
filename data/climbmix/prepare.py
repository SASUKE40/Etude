"""
Download and tokenize the Nemotron-ClimbMix dataset into binary files for training.

Usage:
    # Prepare a single part (0-9)
    python data/climbmix/prepare.py --part 0

    # Prepare all parts
    python data/climbmix/prepare.py --all

    # Prepare with custom num_proc
    python data/climbmix/prepare.py --part 0 --num-proc 32

The output is a set of binary files (uint16 memmap) that can be used directly
for training. After preparing all parts, run merge.py to combine them.

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix
"""

import os
import argparse

import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# GPT-2 BPE tokenizer (max token value 50256 < 2^16, so uint16 is safe)
enc = tiktoken.get_encoding("gpt2")

DEFAULT_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HF_DATASET = "nvidia/Nemotron-ClimbMix"


def process(example):
    """Tokenize a single document and append EOT token."""
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)  # 50256 for GPT-2
    return {"ids": ids, "len": len(ids)}


def prepare_part(part_idx, output_dir=DEFAULT_DATA_DIR, num_proc=32, val_ratio=0.0005, total_parts=10):
    """Download and tokenize one part of the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading part {part_idx} from {HF_DATASET}...")
    print(f"Output directory: {output_dir}")

    # The dataset has 100 data files. Load a subset (10 files per part)
    # to avoid OOM from loading the entire 400B token dataset at once.
    files_per_part = 100 // total_parts
    start = part_idx * files_per_part
    end = start + files_per_part
    data_files = [f"data/train-{i:05d}-of-00100.parquet" for i in range(start, end)]
    dataset = load_dataset(
        HF_DATASET,
        data_files=data_files,
        split="train",
        num_proc=num_proc,
    )

    # If the dataset has multiple configs/splits, handle accordingly
    # For a single-split dataset, create train/val split
    split_dataset = dataset.train_test_split(
        test_size=val_ratio, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    print(f"Dataset split: {split_dataset}")

    # Tokenize
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc=f"Tokenizing part {part_idx}",
        num_proc=num_proc,
    )

    # Write to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(output_dir, f"part_{part_idx}_{split}.bin")
        print(f"Writing {filename} ({arr_len:,} tokens)...")

        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {split}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print(f"Part {part_idx} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Nemotron-ClimbMix data for training"
    )
    parser.add_argument(
        "--part", type=int, default=0, help="Part index to prepare (default: 0)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Prepare all 10 parts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory for binary files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=32,
        help="Number of workers for tokenization (default: 32)",
    )
    args = parser.parse_args()

    if args.all:
        for i in range(10):
            prepare_part(i, output_dir=args.output_dir, num_proc=args.num_proc)
    else:
        prepare_part(args.part, output_dir=args.output_dir, num_proc=args.num_proc)
