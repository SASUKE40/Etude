#!/usr/bin/env python3
"""
Download and normalize NVIDIA Nemotron instruction-following data for LitGPT.

Different LitGPT releases expect different JSON schemas. The install on this
cluster uses the older Alpaca-style path that reads:

    {"instruction": "...", "input": "...", "output": "..."}

This script therefore writes those keys, and also keeps a `messages` copy for
newer chat-oriented loaders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.nemotron_cascade_sft_stage2 import normalize_messages


DEFAULT_URL = (
    "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2/"
    "resolve/main/instruction-following.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Nemotron instruction-following data for LitGPT JSON finetuning"
    )
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument(
        "--download-path",
        type=Path,
        required=True,
        help="Where to store the raw downloaded JSONL.",
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
        help="Re-download and rebuild outputs even if they already exist.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Optional cap for smoke tests; negative means no limit.",
    )
    return parser.parse_args()


def resolve_download_url(raw_url: str) -> str:
    parsed = urllib.parse.urlparse(raw_url)
    if parsed.netloc != "huggingface.co":
        return raw_url
    return raw_url.replace("/blob/", "/resolve/")


def download_file(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"Using existing raw dataset: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(resolve_download_url(url))

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp_path.replace(dest)
    print(f"Downloaded dataset to {dest}")


def coerce_messages(row: dict) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        return normalize_messages(row["messages"])

    if isinstance(row.get("conversations"), list):
        converted = []
        for message in row["conversations"]:
            if not isinstance(message, dict):
                raise ValueError("conversation entry is not a dict")
            role = message.get("role") or message.get("from")
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            content = message.get("content") or message.get("value")
            converted.append({"role": role, "content": content})
        return normalize_messages(converted)

    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    response = row.get("output") or row.get("response") or row.get("answer")
    if isinstance(instruction, str) and isinstance(response, str):
        messages = [{"role": "user", "content": instruction}]
        system = row.get("system") or row.get("system_prompt")
        if isinstance(system, str) and system.strip():
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "assistant", "content": response})
        return normalize_messages(messages)

    raise ValueError("row does not expose a supported chat schema")


def messages_to_instruction_record(messages: list[dict[str, str]]) -> dict[str, object]:
    system_parts = []
    dialogue_parts = []

    for idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            system_parts.append(content)
            continue
        if idx == len(messages) - 1:
            if role != "assistant":
                raise ValueError("conversation must end with assistant for instruction conversion")
            output = content
            break
        speaker = "User" if role == "user" else "Assistant"
        dialogue_parts.append(f"{speaker}: {content}")
    else:
        raise ValueError("missing final assistant turn")

    first_user = next((m["content"].strip() for m in messages if m["role"] == "user"), None)
    if first_user is None:
        raise ValueError("conversation is missing a user turn")

    instruction = first_user
    if system_parts:
        instruction = "\n\n".join(system_parts + [instruction])

    remaining_parts = []
    seen_first_user = False
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role == "user" and not seen_first_user:
            seen_first_user = True
            continue
        if role == "assistant" and message["content"].strip() == output and message is messages[-1]:
            continue
        speaker = "User" if role == "user" else "Assistant"
        remaining_parts.append(f"{speaker}: {message['content'].strip()}")

    return {
        "instruction": instruction,
        "input": "\n\n".join(remaining_parts),
        "output": output,
        "messages": messages,
    }


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


def convert_jsonl_to_json(source_path: Path, output_path: Path, force: bool, max_rows: int) -> None:
    if output_path.exists() and not force and output_has_required_schema(output_path):
        print(f"Using existing prepared dataset: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    kept = 0
    skipped = 0
    with open(source_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for line_number, raw_line in enumerate(src, start=1):
            if max_rows >= 0 and kept >= max_rows:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                messages = coerce_messages(row)
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"Skipping line {line_number}: {type(exc).__name__}: {exc}")
                continue

            payload = messages_to_instruction_record(messages)
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
                "source_path": str(source_path),
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
    download_file(args.source_url, args.download_path, force=args.force)
    convert_jsonl_to_json(
        args.download_path,
        args.output_path,
        force=args.force,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
