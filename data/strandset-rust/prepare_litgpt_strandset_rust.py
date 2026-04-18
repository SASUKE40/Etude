#!/usr/bin/env python3
"""
Prepare Fortytwo-Network/Strandset-Rust-v1 for LitGPT JSON finetuning.

The installed LitGPT stack here expects Alpaca-style JSON records:

    {"instruction": "...", "input": "...", "output": "..."}

This converter also keeps a `messages` copy for compatibility with newer
chat-oriented loaders.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset


HF_DATASET = "Fortytwo-Network/Strandset-Rust-v1"
TARGET_TASK_CATEGORY = "code_generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Strandset-Rust-v1 for LitGPT JSON finetuning"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to write the normalized LitGPT JSON dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Optional cap for smoke tests; negative means no limit.",
    )
    return parser.parse_args()


def output_has_required_schema(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"Existing prepared dataset is unreadable, rebuilding {output_path}: {exc}")
        return False

    if not isinstance(payload, list) or not payload:
        print(f"Existing prepared dataset is empty or not a JSON list, rebuilding {output_path}")
        return False

    first = payload[0]
    required_keys = {"instruction", "input", "output"}
    if not isinstance(first, dict) or not required_keys.issubset(first):
        print(
            f"Existing prepared dataset is missing keys {sorted(required_keys)}, rebuilding {output_path}"
        )
        return False
    return True


def parse_structured_field(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def build_instruction(row: dict) -> str:
    task_category = str(row.get("task_category") or "").strip()
    if task_category != TARGET_TASK_CATEGORY:
        raise ValueError(f"unsupported task_category: {task_category!r}")

    input_data = parse_structured_field(row.get("input_data"))
    title = str(input_data.get("title") or "").strip()
    if not title:
        raise ValueError("row is missing input_data.title")
    return title


def build_input(row: dict) -> str:
    return ""


def build_output(row: dict) -> str:
    output_data = parse_structured_field(row.get("output_data"))
    code = str(output_data.get("code") or "").strip()
    if not code:
        raise ValueError("row is missing output_data.code")
    return code


def row_to_record(row: dict) -> dict[str, object]:
    instruction = build_instruction(row)
    input_text = build_input(row)
    output = build_output(row)
    messages = [{"role": "user", "content": instruction}]
    if input_text:
        messages[0]["content"] += "\n\n" + input_text
    messages.append({"role": "assistant", "content": output})
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "messages": messages,
        "task_category": row.get("task_category"),
        "crate_name": row.get("crate_name"),
    }


def convert_to_json(output_path: Path, force: bool, max_rows: int) -> None:
    if output_path.exists() and not force and output_has_required_schema(output_path):
        print(f"Using existing prepared dataset: {output_path}")
        return

    print(f"Loading {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split="train")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    kept = 0
    skipped = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for idx, row in enumerate(ds):
            if max_rows >= 0 and kept >= max_rows:
                break
            try:
                payload = row_to_record(row)
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"Skipping row {idx}: {type(exc).__name__}: {exc}")
                continue
            if not first:
                out.write(",\n")
            json.dump(payload, out, ensure_ascii=False)
            first = False
            kept += 1
        out.write("\n]\n")

    tmp_path.replace(output_path)
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "hf_dataset": HF_DATASET,
                "task_category": TARGET_TASK_CATEGORY,
                "output_path": str(output_path),
                "kept": kept,
                "skipped": skipped,
                "max_rows": max_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {kept:,} LitGPT samples to {output_path}")
    print(f"Skipped {skipped:,} malformed rows")


def main() -> None:
    args = parse_args()
    convert_to_json(args.output_path, force=args.force, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
