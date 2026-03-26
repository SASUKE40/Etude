"""
Dataset management for Etude training.

Supports multiple datasets:
- ClimbMix: NVIDIA Nemotron-ClimbMix (400B tokens, general text)
- FineWeb-Edu: High-quality educational web text (10BT sample)
- Rust: Rust code from The Stack Dedup

Data formats:
1. Binary (.bin) files — pre-tokenized with GPT-2 tokenizer, loaded via numpy memmap.
2. Parquet files — raw text, tokenized on-the-fly by the dataloader.

Datasets:
- ClimbMix: https://huggingface.co/datasets/OptimalScale/ClimbMix
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Rust: https://huggingface.co/datasets/bigcode/the-stack-dedup
"""

import os
import argparse
import numpy as np
import pyarrow.parquet as pq

from etude.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration

HF_DATASET = "OptimalScale/ClimbMix"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# Project root for local data directories
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Binary data directories
BIN_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "climbmix")
BIN_DATA_DIR_RUST = os.path.join(_PROJECT_ROOT, "data", "rust")

BIN_DATA_DIRS = {
    "climbmix": BIN_DATA_DIR,
    "rust": BIN_DATA_DIR_RUST,
}

# Parquet data directories — checked in order, first existing dir with parquets wins
PARQUET_DATA_DIRS = {
    "climbmix": [DATA_DIR, os.path.join(_PROJECT_ROOT, "data", "climbmix")],
    "fineweb-edu": [os.path.join(base_dir, "fineweb-edu"), os.path.join(_PROJECT_ROOT, "data", "fineweb-edu")],
    "rust": [os.path.join(base_dir, "rust"), os.path.join(_PROJECT_ROOT, "data", "rust")],
}

# HuggingFace dataset IDs for streaming fallback
HF_DATASETS = {
    "climbmix": "OptimalScale/ClimbMix",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "rust": "bigcode/the-stack-dedup",
}

# Text column name varies by dataset
TEXT_COLUMNS = {
    "climbmix": "text",
    "fineweb-edu": "text",
    "rust": "content",
}

# Total number of JSONL files in the ClimbMix dataset (part_0.jsonl to part_99.jsonl)
TOTAL_PARTS = 100

# -----------------------------------------------------------------------------
# Binary data support (nanoGPT-style memmap)

def get_bin_data_path(split, dataset="climbmix"):
    """Return the path to the binary data file for a given split and dataset."""
    data_dir = BIN_DATA_DIRS.get(dataset, BIN_DATA_DIR)
    if split == "train":
        return os.path.join(data_dir, "train.bin")
    elif split == "val":
        return os.path.join(data_dir, "val.bin")
    raise ValueError(f"Unknown split: {split}")


def has_bin_data(dataset="climbmix"):
    """Check if binary data files exist."""
    return os.path.exists(get_bin_data_path("train", dataset=dataset))


def load_bin_data(split, dataset="climbmix"):
    """Load a binary data file as a numpy memmap."""
    path = get_bin_data_path(split, dataset=dataset)
    if not os.path.exists(path):
        prepare_cmd = {
            "climbmix": "python data/climbmix/prepare.py && python data/climbmix/merge.py",
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

def list_parquet_files(data_dir=None, dataset="climbmix", warn_on_legacy=False):
    """Looks into data dirs and returns full paths to all parquet files.

    Searches multiple candidate directories for the given dataset.
    """
    if data_dir is not None:
        candidate_dirs = [data_dir]
    else:
        candidate_dirs = PARQUET_DATA_DIRS.get(dataset, [DATA_DIR])
        if isinstance(candidate_dirs, str):
            candidate_dirs = [candidate_dirs]
        if dataset == "climbmix":
            candidate_dirs.append(os.path.join(base_dir, "base_data"))  # legacy fallback

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
        if dataset == "climbmix":
            print("  To download the ClimbMix dataset, run:")
            print("    python -m etude.dataset -n 10")
        elif dataset == "fineweb-edu":
            print("  To download FineWeb-Edu, run:")
            print("    python data/fineweb-edu/prepare.py")
        elif dataset == "rust":
            print("  To download Rust data, run:")
            print("    python data/rust/prepare.py")
        print()
        print("=" * 80)
        print()

    return []


def parquets_iter_batched(split, start=0, step=1, batch_size=10000, dataset="climbmix"):
    """
    Iterate through the dataset, in batches of text strings.

    Auto-selects local parquet files if available, otherwise streams from HuggingFace.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - dataset: which dataset to iterate ("climbmix", "fineweb-edu", "rust")
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

    hf_id = HF_DATASETS.get(dataset, HF_DATASET)

    if dataset == "climbmix":
        data_files = ["part_99.jsonl"] if split == "val" else [f"part_{i}.jsonl" for i in range(99)]
        ds = load_dataset(hf_id, data_files=data_files, split="train", streaming=True)
    elif dataset == "fineweb-edu":
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


# -----------------------------------------------------------------------------
# Download utility — uses HuggingFace datasets to download and convert to parquet

def download_part(part_idx, data_dir, num_proc=8):
    """Download one JSONL part from HuggingFace and convert to parquet.

    Uses huggingface_hub for fast file download, then converts JSONL to parquet locally.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow as pa
    import pyarrow.parquet as pq_write
    import json

    filename = f"part_{part_idx}.parquet"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping part {part_idx} (already exists at {filepath})")
        return True

    print(f"Downloading part {part_idx} from {HF_DATASET}...")
    try:
        # Fast download via huggingface_hub (uses hf_transfer if available)
        jsonl_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=f"part_{part_idx}.jsonl",
            repo_type="dataset",
        )

        # Convert JSONL to parquet in batches to limit memory usage
        temp_path = filepath + ".tmp"
        writer = None
        num_rows = 0
        batch_size = 50000
        batch = []

        with open(jsonl_path, "r") as f:
            for line in f:
                text = json.loads(line).get("text")
                if text is None:
                    continue
                batch.append(text)
                if len(batch) >= batch_size:
                    table = pa.table({"text": batch})
                    if writer is None:
                        writer = pq_write.ParquetWriter(temp_path, table.schema)
                    writer.write_table(table)
                    num_rows += len(batch)
                    batch = []

        # Write remaining
        if batch:
            table = pa.table({"text": batch})
            if writer is None:
                writer = pq_write.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)
            num_rows += len(batch)

        if writer is not None:
            writer.close()

        os.rename(temp_path, filepath)
        print(f"  Saved part {part_idx} -> {filepath} ({num_rows:,} rows)")
        return True
    except Exception as e:
        print(f"  ERROR downloading part {part_idx}: {e}")
        for path in [filepath + ".tmp", filepath]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ClimbMix dataset and convert to parquet")
    parser.add_argument(
        "-n", "--num-parts", type=int, default=-1,
        help="Number of parts to download (0-99). -1 = all 100 parts (default: -1)",
    )
    parser.add_argument(
        "-w", "--num-workers", type=int, default=8,
        help="Number of workers for dataset processing (default: 8)",
    )
    parser.add_argument(
        "--val-part", type=int, default=99,
        help="Which part to use as validation (default: 99, the last part)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    num_parts = TOTAL_PARTS if args.num_parts == -1 else min(args.num_parts, TOTAL_PARTS)

    # Build list of parts to download
    parts_to_download = list(range(num_parts))
    if args.val_part not in parts_to_download:
        parts_to_download.append(args.val_part)  # always include validation part

    print(f"Downloading {len(parts_to_download)} parts from {HF_DATASET}")
    print(f"Target directory: {DATA_DIR}")
    print(f"Validation part: {args.val_part}")
    print()

    from tqdm import tqdm

    successful = 0
    for part_idx in tqdm(parts_to_download, desc="Overall", unit="part"):
        if download_part(part_idx, DATA_DIR, num_proc=args.num_workers):
            successful += 1

    print(f"\nDone! Downloaded: {successful}/{len(parts_to_download)} parts to {DATA_DIR}")
