"""
The base/pretraining dataset: NVIDIA Nemotron-ClimbMix.

Supports two data formats:
1. Binary (.bin) files — pre-tokenized with GPT-2 tokenizer, loaded via numpy memmap.
   Prepare with: python data/climbmix/prepare.py && python data/climbmix/merge.py
2. Parquet files — raw text, tokenized on-the-fly by the dataloader.
   Download with: python -m etude.dataset -n 170

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix
"""

import os
import argparse
import time
import requests
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

from etude.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration

HF_DATASET = "nvidia/Nemotron-ClimbMix"
BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
MAX_SHARD = 6542
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# Binary data directories
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "climbmix")
BIN_DATA_DIR_RUST = os.path.join(_PROJECT_ROOT, "data", "rust")

# -----------------------------------------------------------------------------
# Binary data support (nanoGPT-style memmap)

BIN_DATA_DIRS = {
    "climbmix": BIN_DATA_DIR,
    "rust": BIN_DATA_DIR_RUST,
}


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

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir

    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET NOT FOUND")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  To download the Nemotron-ClimbMix dataset, run:")
            print()
            print("    python -m etude.dataset -n 170     # download ~170 shards")
            print()
            print("  Or prepare binary data for nanoGPT-style training:")
            print()
            print("    python data/climbmix/prepare.py --all")
            print("    python data/climbmix/merge.py")
            print()
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


# -----------------------------------------------------------------------------
# Download utility

def download_single_file(index):
    """Downloads a single file index, with retries and backoff."""
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nemotron-ClimbMix dataset shards")
    parser.add_argument(
        "-n", "--num-files", type=int, default=-1,
        help="Number of train shards to download (default: -1 = all)",
    )
    parser.add_argument(
        "-w", "--num-workers", type=int, default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download the validation shard

    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Dataset: {HF_DATASET}")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
