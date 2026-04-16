"""
Continued pretraining launcher for Qwen3-0.6B with LitGPT, LitData, and Thunder.

This script uses LitGPT's public pretrain API with the LitData streaming dataset.
LitGPT's current pretrain path hard-codes `torch.compile(model)`, so the Thunder
integration swaps that single call by temporarily monkeypatching `torch.compile`
to `thunder.compile`.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.data import LitData
from litgpt.pretrain import setup as litgpt_pretrain


@contextmanager
def _compile_backend(name: str):
    import torch

    original_compile = torch.compile
    if name == "torch":
        yield
        return
    if name != "thunder":
        raise ValueError(f"Unsupported compiler backend: {name}")

    import thunder

    torch.compile = thunder.compile
    try:
        yield
    finally:
        torch.compile = original_compile


def parse_args() -> argparse.Namespace:
    scratch_root = Path("/scratch") / Path.home().name
    parser = argparse.ArgumentParser(description="Continued pretraining for Qwen3-0.6B with LitGPT")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", type=Path, default=scratch_root / "litgpt-rust-qwen3" / "litdata")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=scratch_root / "litgpt-checkpoints" / "Qwen" / "Qwen3-0.6B",
    )
    parser.add_argument("--initial-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=scratch_root / "litgpt-rust-qwen3" / "out" / "qwen3-0.6b-rust")
    parser.add_argument("--precision", type=str, default="bf16-true")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--compiler", choices=("thunder", "torch"), default="thunder")
    parser.add_argument("--logger-name", type=str, default="tensorboard")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=500_000_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-max-iters", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--train-from-scratch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_args = TrainArgs(
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        max_norm=args.max_norm,
        min_lr=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
    )
    eval_args = EvalArgs(
        interval=args.eval_interval,
        max_iters=args.eval_max_iters,
        initial_validation=False,
        final_validation=True,
    )
    log_args = LogArgs(project=args.project, run=args.run_name)
    data = LitData(
        data_path=args.data_path,
        split_names=("train", "val"),
        num_workers=args.num_workers,
        seed=args.seed,
    )

    initial_checkpoint_dir = None
    if not args.train_from_scratch:
        initial_checkpoint_dir = args.initial_checkpoint_dir or args.tokenizer_dir

    with _compile_backend(args.compiler):
        litgpt_pretrain(
            model_name=args.model_name,
            out_dir=args.out_dir,
            precision=args.precision,
            initial_checkpoint_dir=initial_checkpoint_dir,
            data=data,
            train=train_args,
            eval=eval_args,
            tokenizer_dir=args.tokenizer_dir,
            devices=args.devices,
            num_nodes=args.num_nodes,
            logger_name=args.logger_name,
            log=log_args,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
