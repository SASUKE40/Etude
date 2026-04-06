"""
Prepare NVIDIA Nemotron Cascade SFT Stage 2 as local parquet shards for chat SFT.

The source dataset ships only with a train split, so this script creates a
deterministic train/val split by hashing the subset name plus conversation.

Usage:
    python data/nemotron-cascade-sft-stage-2/prepare.py
    python data/nemotron-cascade-sft-stage-2/prepare.py --output-dir /scratch/$USER/etude/datasets/nemotron-cascade-sft-stage-2
    python data/nemotron-cascade-sft-stage-2/prepare.py --subsets instruction-following,code
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required to write local parquet shards for Nemotron chat SFT.\n"
        "Install project dependencies first, for example:\n"
        "  source .venv/bin/activate && uv sync --extra gpu\n"
        "or:\n"
        "  uv sync --extra gpu"
    ) from exc
from datasets import load_dataset
from tqdm import tqdm

from tasks.nemotron_cascade_sft_stage2 import (
    DEFAULT_SUBSETS,
    HF_DATASET,
    get_default_data_dir,
    has_prepared_split,
    has_prepared_data,
    normalize_messages,
    parse_subset_names,
    pick_split_for_messages,
    prepared_split_dir,
)

MESSAGE_STRUCT = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.string()),
    ]
)
PARQUET_SCHEMA = pa.schema(
    [
        ("subset", pa.string()),
        ("category", pa.string()),
        ("source", pa.string()),
        ("generator", pa.string()),
        ("thinking", pa.bool_()),
        ("messages", pa.list_(MESSAGE_STRUCT)),
    ]
)


class RollingParquetWriter:
    def __init__(self, output_dir, split, rows_per_shard, flush_rows, compression):
        self.split = split
        self.rows_per_shard = rows_per_shard
        self.flush_rows = flush_rows
        self.compression = compression
        self.split_dir = prepared_split_dir(output_dir, split)
        os.makedirs(self.split_dir, exist_ok=True)
        self.shard_idx = 0
        self.rows_in_shard = 0
        self.total_rows = 0
        self.writer = None
        self.buffer = []
        assert self.flush_rows >= 1, f"flush_rows must be >= 1, got {self.flush_rows}"

    def _open_writer(self):
        path = os.path.join(self.split_dir, f"part_{self.shard_idx:05d}.parquet")
        # For long, high-cardinality reasoning traces, dictionary/statistics buffers
        # can become expensive. Keep parquet writes simple and low-memory.
        self.writer = pq.ParquetWriter(
            path,
            PARQUET_SCHEMA,
            compression=self.compression,
            use_dictionary=False,
            write_statistics=False,
        )

    def _rotate(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.shard_idx += 1
        self.rows_in_shard = 0

    def _write_chunk(self, rows):
        if self.writer is None:
            self._open_writer()
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        self.writer.write_table(table)
        self.rows_in_shard += len(rows)
        self.total_rows += len(rows)
        if self.rows_in_shard >= self.rows_per_shard:
            self._rotate()

    def write(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def flush(self):
        while self.buffer:
            capacity = self.rows_per_shard - self.rows_in_shard
            if capacity == 0:
                self._rotate()
                capacity = self.rows_per_shard
            chunk = self.buffer[:capacity]
            self.buffer = self.buffer[capacity:]
            self._write_chunk(chunk)

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def prepare(
    output_dir,
    subsets=None,
    val_ratio=0.005,
    rows_per_shard=50_000,
    flush_rows=16,
    compression="snappy",
    max_rows_per_subset=-1,
):
    subsets = parse_subset_names(subsets) or DEFAULT_SUBSETS
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if has_prepared_data(output_dir):
        print(f"Found prepared train/val parquet shards in {output_dir}")
        print("Delete them to re-prepare, or use as-is.")
        return
    if has_prepared_split(output_dir, "train") or has_prepared_split(output_dir, "val"):
        raise RuntimeError(
            f"Found a partial prepared dataset in {output_dir}. "
            "Delete the existing parquet shards before re-running prepare."
        )

    print(f"Preparing {HF_DATASET}")
    print(f"Subsets: {', '.join(subsets)}")
    print(f"Output directory: {output_dir}")
    print(f"Validation ratio: {val_ratio:.4f}")
    print(f"Rows per shard: {rows_per_shard:,}")
    print(f"Flush rows: {flush_rows}")
    print(f"Parquet compression: {compression}")

    writers = {
        "train": RollingParquetWriter(output_dir, "train", rows_per_shard, flush_rows, compression),
        "val": RollingParquetWriter(output_dir, "val", rows_per_shard, flush_rows, compression),
    }
    skipped = 0
    counts = {
        "train": {subset: 0 for subset in subsets},
        "val": {subset: 0 for subset in subsets},
    }

    try:
        for subset in subsets:
            print(f"Starting subset: {subset}", flush=True)
            ds = load_dataset(HF_DATASET, subset, split="train", streaming=True)
            progress = tqdm(desc=f"Preparing {subset}", unit="rows")
            seen = 0
            for row in ds:
                if 0 <= max_rows_per_subset <= seen:
                    break
                seen += 1
                progress.update(1)
                try:
                    messages = normalize_messages(row["messages"])
                except (KeyError, ValueError):
                    skipped += 1
                    continue

                split = pick_split_for_messages(messages, subset=subset, val_ratio=val_ratio)
                writers[split].write(
                    {
                        "subset": subset,
                        "category": row.get("category"),
                        "source": row.get("source"),
                        "generator": row.get("generator"),
                        "thinking": bool(row.get("thinking", False)),
                        "messages": messages,
                    }
                )
                counts[split][subset] += 1
            progress.close()
    finally:
        for writer in writers.values():
            writer.close()

    metadata = {
        "hf_dataset": HF_DATASET,
        "subsets": list(subsets),
        "val_ratio": val_ratio,
        "rows_per_shard": rows_per_shard,
        "flush_rows": flush_rows,
        "compression": compression,
        "max_rows_per_subset": max_rows_per_subset,
        "counts": counts,
        "skipped": skipped,
    }
    metadata_path = os.path.join(output_dir, "meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    train_total = sum(counts["train"].values())
    val_total = sum(counts["val"].values())
    print(f"Done. Train rows: {train_total:,} | Val rows: {val_total:,} | Skipped: {skipped:,}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Nemotron Cascade SFT Stage 2 for chat SFT")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=get_default_data_dir(),
        help="Directory to write train/val parquet shards",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=",".join(DEFAULT_SUBSETS),
        help="Comma-separated subset names to include",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.005,
        help="Deterministic validation split ratio",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=50_000,
        help="Maximum rows per parquet shard",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=16,
        help="Rows to accumulate before each parquet write (lower = less RAM, slower)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "zstd", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--max-rows-per-subset",
        type=int,
        default=-1,
        help="Optional cap per subset for smoke runs (-1 = no cap)",
    )
    args = parser.parse_args()

    prepare(
        output_dir=args.output_dir,
        subsets=args.subsets,
        val_ratio=args.val_ratio,
        rows_per_shard=args.rows_per_shard,
        flush_rows=args.flush_rows,
        compression=None if args.compression == "none" else args.compression,
        max_rows_per_subset=args.max_rows_per_subset,
    )
