#!/usr/bin/env python3
"""
Chat with a LitGPT checkpoint through Hugging Face Transformers.

This script does not load a raw LitGPT `model.pth` directly into Transformers.
It expects an already-converted Hugging Face checkpoint directory and loads it
with `AutoModelForCausalLM` and `AutoTokenizer`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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

    parser = argparse.ArgumentParser(description="Chat with an already-converted HF checkpoint via Hugging Face Transformers")
    parser.add_argument("checkpoint_path", type=Path, nargs="?", default=default_checkpoint)
    parser.add_argument(
        "--hf-dir",
        type=Path,
        required=True,
        help="Converted Hugging Face checkpoint dir containing config/tokenizer/model files.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    return parser.parse_args()


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


def load_model(hf_dir: Path, dtype_name: str, device: str):
    model_pth = hf_dir / "model.pth"
    dtype = resolve_dtype(dtype_name)

    if model_pth.is_file():
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            dtype=dtype,
        )
        state_dict = torch.load(model_pth, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise SystemExit(
                f"Missing weights when loading {model_pth}: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
        if unexpected:
            raise SystemExit(
                f"Unexpected weights when loading {model_pth}: {unexpected[:10]}"
                + (" ..." if len(unexpected) > 10 else "")
            )
        model.to(device)
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model

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
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    hf_dir = args.hf_dir.expanduser().resolve()

    if not has_hf_checkpoint(hf_dir):
        raise SystemExit(
            f"Hugging Face checkpoint directory is missing or incomplete: {hf_dir}\n"
            "Provide an already-converted HF directory with --hf-dir."
        )

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint path does not exist: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    model = load_model(hf_dir, args.dtype, args.device)

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
