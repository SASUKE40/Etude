#!/usr/bin/env python3
"""
Inference wrapper for LitGPT checkpoints produced by continued pretraining.

This script makes LitGPT checkpoint directories easier to use on clusters where:
- step checkpoints do not include tokenizer/config metadata files
- cuDNN SDPA can crash with CUDNN_STATUS_NOT_INITIALIZED during inference
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path


METADATA_FILES = (
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


def parse_args() -> argparse.Namespace:
    scratch_root = Path("/scratch") / Path.home().name
    default_base = scratch_root / "litgpt-checkpoints" / "Qwen" / "Qwen3-0.6B"
    parser = argparse.ArgumentParser(description="Chat with or generate from a LitGPT training checkpoint")
    parser.add_argument("mode", choices=("chat", "generate"))
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Checkpoint directory or direct path to model.pth/lit_model.pth.",
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        type=Path,
        default=default_base,
        help="Base LitGPT checkpoint directory that provides tokenizer/config metadata.",
    )
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--quantize", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--context-length",
        type=int,
        default=20000,
        help=(
            "Override LitGPT's `block_size` in model_config.yaml for inference "
            "(default: 20000). This controls the maximum prompt + generated tokens."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per response (default: 512).",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--allow-cudnn-sdp",
        action="store_true",
        help="Keep cuDNN SDPA enabled. Disabled by default to avoid CUDNN_STATUS_NOT_INITIALIZED on some clusters.",
    )
    parser.add_argument("--multiline", action="store_true", help="Enable multiline prompts in chat mode.")
    parser.add_argument("--access-token", type=str, default=None, help="Optional Hugging Face access token for chat mode.")
    parser.add_argument("--prompt", type=str, default="Write a Rust function that parses a TOML file.")
    parser.add_argument("--sys-prompt", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    return parser.parse_args()


def _override_context_length(checkpoint_dir: Path, context_length: int) -> None:
    if context_length <= 0:
        raise ValueError(f"context length must be positive, got {context_length}")

    model_config_path = checkpoint_dir / "model_config.yaml"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Missing LitGPT model config at {model_config_path}")

    original_text = model_config_path.read_text()
    updated_text, replacements = re.subn(
        r"^(\s*block_size:\s*)\d+(\s*)$",
        rf"\g<1>{context_length}\g<2>",
        original_text,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError(f"Could not find `block_size` in {model_config_path}")
    if updated_text == original_text:
        return

    # If the config is symlinked from the base checkpoint, replace it locally so the
    # override only applies to this inference checkpoint directory.
    if model_config_path.is_symlink():
        model_config_path.unlink()
    model_config_path.write_text(updated_text)
    print(f"Overrode LitGPT context length to {context_length} tokens in {model_config_path}")


def _resolve_checkpoint_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.is_dir():
        return checkpoint_path
    if checkpoint_path.is_file() and checkpoint_path.name in {"lit_model.pth", "model.pth"}:
        return checkpoint_path.parent
    raise FileNotFoundError(
        f"Checkpoint path must be a directory or a model.pth/lit_model.pth file: {checkpoint_path}"
    )


def _prepare_checkpoint_dir(checkpoint_dir: Path, base_checkpoint_dir: Path, context_length: int | None = None) -> None:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    base_checkpoint_dir = base_checkpoint_dir.expanduser().resolve()
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    model_path = checkpoint_dir / "model.pth"
    if not checkpoint_path.is_file() and model_path.is_file():
        checkpoint_path.symlink_to(model_path.name)
        print(f"Linked LitGPT weights into {checkpoint_dir}: lit_model.pth -> {model_path.name}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing LitGPT weights at {checkpoint_path}")
    if not base_checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Base checkpoint directory does not exist: {base_checkpoint_dir}")

    linked = []
    for filename in METADATA_FILES:
        source = base_checkpoint_dir / filename
        target = checkpoint_dir / filename
        if target.exists() or not source.exists():
            continue
        target.symlink_to(source)
        linked.append(filename)
    if linked:
        print(f"Linked checkpoint metadata into {checkpoint_dir}: {', '.join(linked)}")
    if context_length is not None:
        _override_context_length(checkpoint_dir, context_length)


def _configure_runtime(allow_cudnn_sdp: bool) -> None:
    warnings.filterwarnings(
        "ignore",
        message="transformer_engine module not found!",
        category=UserWarning,
    )
    import torch

    if not allow_cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("Disabled cuDNN SDPA for inference stability; using alternate SDPA backends.")


def main() -> None:
    args = parse_args()
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_path)
    _prepare_checkpoint_dir(checkpoint_dir, args.base_checkpoint_dir, context_length=args.context_length)
    _configure_runtime(args.allow_cudnn_sdp)

    if args.mode == "chat":
        from litgpt.chat.base import main as litgpt_chat

        litgpt_chat(
            checkpoint_dir=checkpoint_dir,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            quantize=args.quantize,
            precision=args.precision,
            compile=args.compile,
            multiline=args.multiline,
            access_token=args.access_token,
        )
        return

    from litgpt.generate.base import main as litgpt_generate

    litgpt_generate(
        checkpoint_dir=checkpoint_dir,
        prompt=args.prompt,
        sys_prompt=args.sys_prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        quantize=args.quantize,
        precision=args.precision,
        compile=args.compile,
    )


if __name__ == "__main__":
    main()
