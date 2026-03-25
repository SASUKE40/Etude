"""
Download and tokenize the Rust subset of The Stack Dedup into binary files for training.

Usage:
    # Prepare with default settings
    python data/rust/prepare.py

    # Custom output directory (e.g. scratch to avoid home quota)
    python data/rust/prepare.py --output-dir /scratch/$USER/etude_data/rust

    # Custom number of workers
    python data/rust/prepare.py --num-proc 48

Dataset: https://huggingface.co/datasets/bigcode/the-stack-dedup
Language: Rust
"""

import os
import argparse

import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# GPT-2 BPE tokenizer (max token value 50256 < 2^16, so uint16 is safe)
enc = tiktoken.get_encoding("gpt2")

HF_DATASET = "bigcode/the-stack-dedup"


def process(example):
    """Tokenize a single document and append EOT token."""
    ids = enc.encode_ordinary(example["content"])
    ids.append(enc.eot_token)  # 50256 for GPT-2
    return {"ids": ids, "len": len(ids)}


def prepare(output_dir, num_proc=32, val_ratio=0.0005):
    """Download and tokenize the Rust subset."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading Rust subset from {HF_DATASET}...")
    print(f"Output directory: {output_dir}")

    dataset = load_dataset(
        HF_DATASET,
        data_dir="data/rust",
        split="train",
        num_proc=num_proc,
    )

    # Create train/val split
    split_dataset = dataset.train_test_split(
        test_size=val_ratio, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    print(f"Dataset split: {split_dataset}")

    # Tokenize
    tokenized = split_dataset.map(
        process,
        remove_columns=dataset.column_names,
        desc="Tokenizing Rust",
        num_proc=num_proc,
    )

    # Write to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(output_dir, f"{split}.bin")
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

    print("Done!")


if __name__ == "__main__":
    default_output = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Prepare Rust data from The Stack Dedup for training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help=f"Output directory for binary files (default: {default_output})",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=32,
        help="Number of workers for tokenization (default: 32)",
    )
    args = parser.parse_args()

    prepare(output_dir=args.output_dir, num_proc=args.num_proc)
