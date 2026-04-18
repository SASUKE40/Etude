#!/usr/bin/env python3
"""
Chat with a LitGPT checkpoint through Hugging Face Transformers.

This script does not load a raw LitGPT `model.pth` directly into Transformers.
Instead, it resolves the checkpoint directory, converts it to a Hugging Face
checkpoint if needed, and then loads that converted checkpoint with
`AutoModelForCausalLM` and `AutoTokenizer`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    scratch_root = Path("/scratch") / Path.home().name
    default_checkpoint = (
        scratch_root
        / "litgpt-strandset-rust-sft"
        / "qwen3-0.6b-rust-step-00001800-strandset-rust-v1"
        / "out"
        / "final"
        / "model.pth"
    )
    default_base = scratch_root / "litgpt-checkpoints" / "Qwen" / "Qwen3-0.6B"

    parser = argparse.ArgumentParser(description="Chat with a LitGPT checkpoint via Hugging Face Transformers")
    parser.add_argument("checkpoint_path", type=Path, nargs="?", default=default_checkpoint)
    parser.add_argument("--hf-dir", type=Path, default=None, help="Existing or target converted HF checkpoint dir.")
    parser.add_argument(
        "--base-checkpoint-dir",
        type=Path,
        default=default_base,
        help="Base Qwen checkpoint dir that provides tokenizer/config metadata for conversion.",
    )
    parser.add_argument(
        "--conversion-python",
        type=str,
        default=sys.executable,
        help="Python executable used for LitGPT conversion when HF files are missing.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    return parser.parse_args()


def resolve_checkpoint_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.is_dir():
        return checkpoint_path
    if checkpoint_path.is_file() and checkpoint_path.name in {"model.pth", "lit_model.pth"}:
        return checkpoint_path.parent
    raise FileNotFoundError(
        f"Checkpoint path must be a directory or a model.pth/lit_model.pth file: {checkpoint_path}"
    )


def ensure_litgpt_checkpoint_layout(checkpoint_dir: Path, base_checkpoint_dir: Path) -> None:
    metadata_files = (
        "config.json",
        "generation_config.json",
        "model_config.yaml",
        "prompt_style.yaml",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    )

    lit_model = checkpoint_dir / "lit_model.pth"
    model_pth = checkpoint_dir / "model.pth"
    if not lit_model.exists() and model_pth.exists():
        lit_model.symlink_to(model_pth.name)

    if not lit_model.is_file():
        raise FileNotFoundError(f"Missing LitGPT weights at {lit_model}")
    if not base_checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Base checkpoint directory does not exist: {base_checkpoint_dir}")

    for filename in metadata_files:
        source = base_checkpoint_dir / filename
        target = checkpoint_dir / filename
        if target.exists() or not source.exists():
            continue
        target.symlink_to(source)


def has_hf_checkpoint(hf_dir: Path) -> bool:
    if not hf_dir.is_dir():
        return False
    if not (hf_dir / "config.json").is_file():
        return False
    return any(
        (hf_dir / filename).is_file()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "model.pth",
        )
    )


def run_conversion(conversion_python: str, checkpoint_dir: Path, hf_dir: Path) -> None:
    help_proc = subprocess.run(
        [conversion_python, "-m", "litgpt", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    help_text = help_proc.stdout

    if "convert_from_litgpt" in help_text:
        cmd = [conversion_python, "-m", "litgpt", "convert_from_litgpt", str(checkpoint_dir), str(hf_dir)]
    else:
        cmd = [
            conversion_python,
            "-m",
            "litgpt",
            "convert",
            "from_litgpt",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--output_dir",
            str(hf_dir),
        ]

    subprocess.run(cmd, check=True)


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_path)
    hf_dir = (args.hf_dir.expanduser().resolve() if args.hf_dir else checkpoint_dir / "hf")
    base_checkpoint_dir = args.base_checkpoint_dir.expanduser().resolve()

    if not has_hf_checkpoint(hf_dir):
        ensure_litgpt_checkpoint_layout(checkpoint_dir, base_checkpoint_dir)
        print(f"Converting LitGPT checkpoint to Hugging Face format in {hf_dir}", file=sys.stderr)
        try:
            run_conversion(args.conversion_python, checkpoint_dir, hf_dir)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "Failed to convert the LitGPT checkpoint to Hugging Face format. "
                "Your active LitGPT install may be too old for this checkpoint layout.\n"
                f"Conversion command exited with status {exc.returncode}."
            ) from exc

    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    print(f"Loaded Transformers checkpoint from {hf_dir}", file=sys.stderr)
    print("Type /exit to quit.", file=sys.stderr)

    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        messages = build_messages(args.system, prompt)
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(text.strip())


if __name__ == "__main__":
    main()
