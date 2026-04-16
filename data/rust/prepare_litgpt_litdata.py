"""
Prepare the cleaned Rust dataset for LitGPT continued pretraining.

This script:
1. Downloads the Hugging Face dataset splits as sharded plain-text files.
2. Converts those text shards into LitData token chunks using a LitGPT tokenizer.

Usage:
    python data/rust/prepare_litgpt_litdata.py \
        --tokenizer-dir /scratch/$USER/litgpt-checkpoints/Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from functools import partial
from pathlib import Path

from datasets import load_dataset

HF_DATASET = "ammarnasr/the-stack-rust-clean"
DEFAULT_ROWS_PER_SHARD = 10_000
DEFAULT_SEPARATOR = "\n\n<|endoftext|>\n\n"


def _default_root() -> Path:
    return Path("/scratch") / os.environ.get("USER", "unknown") / "litgpt-rust-qwen3"


def _write_split_text(
    dataset_name: str,
    split_name: str,
    output_dir: Path,
    rows_per_shard: int,
    separator: str,
) -> dict[str, int]:
    existing_shards = sorted(output_dir.glob("*.txt"))
    if existing_shards:
        print(f"Found existing text shards in {output_dir}, skipping download")
        return {
            "num_rows": -1,
            "num_chars": -1,
            "num_shards": len(existing_shards),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split_name, streaming=True)

    shard_idx = 0
    shard_rows = 0
    total_rows = 0
    total_chars = 0
    shard_path = output_dir / f"{split_name}_{shard_idx:05d}.txt"
    handle = shard_path.open("w", encoding="utf-8")

    try:
        for example in dataset:
            content = example.get("content")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
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
                shard_idx += 1
                shard_rows = 0
                shard_path = output_dir / f"{split_name}_{shard_idx:05d}.txt"
                handle = shard_path.open("w", encoding="utf-8")
    finally:
        handle.close()

    if shard_rows == 0 and shard_path.exists():
        shard_path.unlink()

    return {
        "num_rows": total_rows,
        "num_chars": total_chars,
        "num_shards": shard_idx + (1 if shard_rows > 0 else 0),
    }


def _tokenize_file(filename: str, tokenizer) -> object:
    with open(filename, encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return
    yield tokenizer.encode(text, bos=True, eos=False)


def _optimize_split(
    input_dir: Path,
    output_dir: Path,
    tokenizer_dir: Path,
    max_seq_length: int,
    chunk_bytes: str,
    num_workers: int,
    overwrite: bool,
) -> dict[str, int]:
    from litdata import optimize
    from litdata.streaming import TokensLoader
    from litgpt import Tokenizer

    text_files = sorted(str(path) for path in input_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    if output_dir.exists():
        if not overwrite:
            index_files = list(output_dir.glob("index.json"))
            if index_files:
                return {"num_input_files": len(text_files), "skipped": 1}
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_dir)
    optimize(
        fn=partial(_tokenize_file, tokenizer=tokenizer),
        inputs=text_files,
        output_dir=str(output_dir),
        num_workers=min(num_workers, len(text_files)),
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(block_size=max_seq_length + 1),
    )
    return {"num_input_files": len(text_files), "skipped": 0}


def prepare(args: argparse.Namespace) -> None:
    text_root = args.text_output_dir
    litdata_root = args.litdata_output_dir

    if args.overwrite:
        if text_root.exists():
            shutil.rmtree(text_root)
        if litdata_root.exists():
            shutil.rmtree(litdata_root)

    text_train_dir = text_root / "train"
    text_val_dir = text_root / "val"
    litdata_train_dir = litdata_root / "train"
    litdata_val_dir = litdata_root / "val"

    print(f"Preparing text shards from {args.dataset_name}")
    train_meta = _write_split_text(
        dataset_name=args.dataset_name,
        split_name="train",
        output_dir=text_train_dir,
        rows_per_shard=args.rows_per_shard,
        separator=args.separator,
    )
    val_meta = _write_split_text(
        dataset_name=args.dataset_name,
        split_name=args.val_split,
        output_dir=text_val_dir,
        rows_per_shard=args.rows_per_shard,
        separator=args.separator,
    )

    print("Optimizing train split with LitData")
    train_opt_meta = _optimize_split(
        input_dir=text_train_dir,
        output_dir=litdata_train_dir,
        tokenizer_dir=args.tokenizer_dir,
        max_seq_length=args.max_seq_length,
        chunk_bytes=args.chunk_bytes,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )

    print("Optimizing validation split with LitData")
    val_opt_meta = _optimize_split(
        input_dir=text_val_dir,
        output_dir=litdata_val_dir,
        tokenizer_dir=args.tokenizer_dir,
        max_seq_length=args.max_seq_length,
        chunk_bytes=args.chunk_bytes,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )

    metadata = {
        "dataset": args.dataset_name,
        "train_split": "train",
        "val_split": args.val_split,
        "tokenizer_dir": str(args.tokenizer_dir),
        "text_output_dir": str(text_root),
        "litdata_output_dir": str(litdata_root),
        "rows_per_shard": args.rows_per_shard,
        "chunk_bytes": args.chunk_bytes,
        "max_seq_length": args.max_seq_length,
        "splits": {
            "train": {**train_meta, **train_opt_meta},
            "val": {**val_meta, **val_opt_meta},
        },
    }
    metadata_path = args.root_dir / "prepare_litgpt_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Done!")
    print(f"Text shards: {text_root}")
    print(f"LitData chunks: {litdata_root}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    default_root = _default_root()
    parser = argparse.ArgumentParser(description="Prepare the Rust dataset for LitGPT + LitData")
    parser.add_argument("--dataset-name", type=str, default=HF_DATASET)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, default=default_root)
    parser.add_argument("--text-output-dir", type=Path, default=default_root / "text")
    parser.add_argument("--litdata-output-dir", type=Path, default=default_root / "litdata")
    parser.add_argument("--val-split", type=str, default="valid")
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument("--separator", type=str, default=DEFAULT_SEPARATOR)
    parser.add_argument("--chunk-bytes", type=str, default="200MB")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()

    if parsed.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be greater than 0")
    if parsed.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be greater than 0")

    prepare(parsed)
