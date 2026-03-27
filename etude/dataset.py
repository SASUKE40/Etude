"""
Dataset management for Etude training.

Supports multiple datasets:
- FineWeb-Edu: High-quality educational web text (10BT sample)
- Rust: Rust code from The Stack Dedup

Data formats:
1. Binary (.bin) files — pre-tokenized with GPT-2 tokenizer, loaded via numpy memmap.
2. Parquet files — raw text, tokenized on-the-fly by the dataloader.

Datasets:
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Rust: https://huggingface.co/datasets/bigcode/the-stack-dedup
"""

import os
import numpy as np
import pyarrow.parquet as pq

from etude.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration

base_dir = get_base_dir()

# Project root for local data directories
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Binary data directories
BIN_DATA_DIR_RUST = os.path.join(_PROJECT_ROOT, "data", "rust")

BIN_DATA_DIRS = {
    "rust": BIN_DATA_DIR_RUST,
}

# Parquet data directories — checked in order, first existing dir with parquets wins
PARQUET_DATA_DIRS = {
    "fineweb-edu": [os.path.join(base_dir, "fineweb-edu"), os.path.join(_PROJECT_ROOT, "data", "fineweb-edu")],
    "rust": [os.path.join(base_dir, "rust"), os.path.join(_PROJECT_ROOT, "data", "rust")],
}

# HuggingFace dataset IDs for streaming fallback
HF_DATASETS = {
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "rust": "bigcode/the-stack-dedup",
}

# Text column name varies by dataset
TEXT_COLUMNS = {
    "fineweb-edu": "text",
    "rust": "content",
}

# -----------------------------------------------------------------------------
# Binary data support (nanoGPT-style memmap)

def get_bin_data_path(split, dataset="fineweb-edu"):
    """Return the path to the binary data file for a given split and dataset."""
    data_dir = BIN_DATA_DIRS.get(dataset)
    if data_dir is None:
        raise ValueError(f"No binary data directory configured for dataset '{dataset}'")

    if split == "train":
        return os.path.join(data_dir, "train.bin")
    elif split == "val":
        return os.path.join(data_dir, "val.bin")
    raise ValueError(f"Unknown split: {split}")


def has_bin_data(dataset="fineweb-edu"):
    """Check if binary data files exist."""
    return os.path.exists(get_bin_data_path("train", dataset=dataset))


def load_bin_data(split, dataset="fineweb-edu"):
    """Load a binary data file as a numpy memmap."""
    path = get_bin_data_path(split, dataset=dataset)
    if not os.path.exists(path):
        prepare_cmd = {
            "rust": "python data/rust/prepare.py",
        }.get(dataset, f"Prepare data in {path}")
        raise FileNotFoundError(
            f"Binary data not found at {path}. Run: {prepare_cmd}"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_bin_batch(split, batch_size, block_size, device="cuda"):
    """
    Get a random batch from binary data (nanoGPT-style).

    Returns (x, y) tensors of shape (batch_size, block_size).
    """
    import torch

    data = load_bin_data(split)
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Parquet data support (streaming from HuggingFace)

def list_parquet_files(data_dir=None, dataset="fineweb-edu", warn_on_legacy=False):
    """Looks into data dirs and returns full paths to all parquet files.

    Searches multiple candidate directories for the given dataset.
    """
    if data_dir is not None:
        candidate_dirs = [data_dir]
    else:
        candidate_dirs = PARQUET_DATA_DIRS.get(dataset, [])
        if isinstance(candidate_dirs, str):
            candidate_dirs = [candidate_dirs]

    for d in candidate_dirs:
        if not os.path.exists(d):
            continue
        parquet_files = sorted([
            f for f in os.listdir(d)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ])
        if parquet_files:
            return [os.path.join(d, f) for f in parquet_files]

    if warn_on_legacy:
        print()
        print("=" * 80)
        print(f"  WARNING: DATASET NOT FOUND ({dataset})")
        print("=" * 80)
        print()
        print(f"  Searched: {candidate_dirs}")
        print()
        if dataset == "fineweb-edu":
            print("  To download FineWeb-Edu, run:")
            print("    python data/fineweb-edu/prepare.py")
        elif dataset == "rust":
            print("  To download Rust data, run:")
            print("    python data/rust/prepare.py")
        print()
        print("=" * 80)
        print()

    return []


def parquets_iter_batched(split, start=0, step=1, batch_size=10000, dataset="fineweb-edu"):
    """
    Iterate through the dataset, in batches of text strings.

    Auto-selects local parquet files if available, otherwise streams from HuggingFace.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - dataset: which dataset to iterate ("fineweb-edu", "rust")
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    text_col = TEXT_COLUMNS.get(dataset, "text")

    try:
        parquet_paths = list_parquet_files(dataset=dataset)
        if parquet_paths:
            parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
            for filepath in parquet_paths:
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(start, pf.num_row_groups, step):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column(text_col).to_pylist()
                    yield texts
            return
    except (FileNotFoundError, OSError):
        pass

    # HuggingFace streaming fallback
    from datasets import load_dataset

    hf_id = HF_DATASETS.get(dataset)

    if dataset == "fineweb-edu":
        ds = load_dataset(hf_id, name="sample-10BT", split="train", streaming=True)
    elif dataset == "rust":
        ds = load_dataset(hf_id, data_dir="data/rust", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    batch = []
    row_idx = 0
    for example in ds:
        if row_idx % step != start:
            row_idx += 1
            continue
        row_idx += 1
        text = example.get(text_col)
        if text is None:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
