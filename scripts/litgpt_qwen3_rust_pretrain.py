"""
Continued pretraining launcher for Qwen3-0.6B with LitGPT, LitData, and Thunder.

This script uses LitGPT's public pretrain API with the LitData streaming dataset.
LitGPT's current pretrain path hard-codes `torch.compile(model)`, so the Thunder
integration swaps that single call by temporarily monkeypatching `torch.compile`
to `thunder.compile`.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def _compile_backend(name: str):
    import torch

    original_compile = torch.compile
    if name == "none":
        torch.compile = lambda model, *args, **kwargs: model
        try:
            yield
        finally:
            torch.compile = original_compile
        return
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
    parser.add_argument("model_name_positional", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-name", dest="model_name", type=str, default=None)
    parser.add_argument("--data-path", dest="data_path", type=Path, default=None)
    parser.add_argument("--data", dest="data_path_legacy", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--tokenizer-dir",
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--initial-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Resume mode: auto, true, false, or an explicit checkpoint path.",
    )
    parser.add_argument("--precision", type=str, default="bf16-true")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--num-nodes", "--num_nodes", dest="num_nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--compiler", choices=("none", "thunder", "torch"), default="none")
    parser.add_argument("--logger-name", "--logger_name", dest="logger_name", type=str, default="wandb")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--log.project", dest="log_project", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log.run", dest="log_run", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--train.global_batch_size", dest="train_global_batch_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--train.micro_batch_size", dest="train_micro_batch_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--train.max_seq_length", dest="train_max_seq_length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--train.max_tokens", dest="train_max_tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train.max_steps", dest="train_max_steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--train.save_interval", dest="train_save_interval", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--train.log_interval", dest="train_log_interval", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval.interval", dest="eval_interval_legacy", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-max-iters", type=int, default=None)
    parser.add_argument("--eval.max_iters", dest="eval_max_iters_legacy", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--train.min_lr", dest="train_min_lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--train.lr_warmup_steps", dest="train_lr_warmup_steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-norm", type=float, default=None)
    parser.add_argument("--train.max_norm", dest="train_max_norm", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-cudnn-sdp",
        action="store_true",
        help="Keep cuDNN SDPA enabled. Disabled by default to avoid CUDNN_STATUS_NOT_INITIALIZED on some clusters.",
    )
    parser.add_argument("--train-from-scratch", action="store_true")
    args = parser.parse_args()

    args.model_name = args.model_name or args.model_name_positional or "Qwen/Qwen3-0.6B"
    args.data_path = args.data_path or args.data_path_legacy or (scratch_root / "litgpt-rust-qwen3" / "litdata")
    args.tokenizer_dir = args.tokenizer_dir or (scratch_root / "litgpt-checkpoints" / "Qwen" / "Qwen3-0.6B")
    args.out_dir = args.out_dir or (scratch_root / "litgpt-rust-qwen3" / "out" / "qwen3-0.6b-rust")
    args.project = args.project or args.log_project or os.environ.get("WANDB_PROJECT", "etude")
    args.run_name = args.run_name or args.log_run or os.environ.get("WANDB_RUN")
    args.global_batch_size = args.global_batch_size or args.train_global_batch_size or 128
    args.micro_batch_size = args.micro_batch_size or args.train_micro_batch_size or 1
    args.max_seq_length = args.max_seq_length or args.train_max_seq_length or 2048
    args.max_tokens = args.max_tokens or args.train_max_tokens or 500_000_000
    args.max_steps = args.max_steps or args.train_max_steps
    args.save_interval = args.save_interval or args.train_save_interval or 1000
    args.log_interval = args.log_interval or args.train_log_interval or 10
    args.eval_interval = args.eval_interval or args.eval_interval_legacy or 1000
    args.eval_max_iters = args.eval_max_iters or args.eval_max_iters_legacy or 50
    args.learning_rate = args.learning_rate or args.train_min_lr or 2e-5
    args.lr_warmup_steps = args.lr_warmup_steps or args.train_lr_warmup_steps or 200
    args.max_norm = args.max_norm or args.train_max_norm or 1.0
    return args


def _coerce_resume(value: str | None) -> bool | str | Path:
    if value is None:
        return "auto"
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return Path(value)


def _has_existing_checkpoint(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    return any(out_dir.rglob("lit_model.pth"))


def _normalize_devices(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    if value == "auto":
        return value
    if value.isdigit():
        return int(value)
    return value


def _install_mfu_metric_alias() -> None:
    from lightning.fabric.utilities.throughput import ThroughputMonitor

    original_compute = ThroughputMonitor.compute
    if getattr(ThroughputMonitor.compute, "_etude_mfu_alias", False):
        return

    def compute_with_alias(self):
        metrics = original_compute(self)
        if "device/mfu" in metrics and "mfu" not in metrics:
            metrics["mfu"] = metrics["device/mfu"]
        alias_map = {
            "throughput/device/batches_per_sec": "batches_per_sec",
            "throughput/device/samples_per_sec": "samples_per_sec",
            "throughput/device/items_per_sec": "items_per_sec",
            "throughput/device/flops_per_sec": "flops_per_sec",
            "device/batches_per_sec": "batches_per_sec",
            "device/samples_per_sec": "samples_per_sec",
            "device/items_per_sec": "items_per_sec",
            "device/flops_per_sec": "flops_per_sec",
            "device/tokens_per_sec": "tokens_per_sec",
            "throughput/device/tokens_per_sec": "tokens_per_sec",
        }
        for source_key, alias_key in alias_map.items():
            if source_key in metrics and alias_key not in metrics:
                metrics[alias_key] = metrics[source_key]
        return metrics

    compute_with_alias._etude_mfu_alias = True
    ThroughputMonitor.compute = compute_with_alias


def _install_log_dict_metric_aliases() -> None:
    import lightning as L

    original_log_dict = L.Fabric.log_dict
    if getattr(L.Fabric.log_dict, "_etude_metric_aliases", False):
        return

    def log_dict_with_aliases(self, metrics, *args, **kwargs):
        aliased = dict(metrics)
        if "total_tokens" in aliased and "percent_of_token_budget" not in aliased:
            max_tokens = getattr(self, "_etude_max_tokens", None)
            if isinstance(max_tokens, int) and max_tokens > 0:
                aliased["percent_of_token_budget"] = 100.0 * aliased["total_tokens"] / max_tokens
        if "iter_time" in aliased and "tokens" in aliased and "tokens_per_sec" not in aliased:
            iter_time = aliased["iter_time"]
            tokens = aliased["tokens"]
            if isinstance(iter_time, (int, float)) and iter_time > 0 and isinstance(tokens, (int, float)):
                aliased["tokens_per_sec"] = tokens / iter_time
        return original_log_dict(self, aliased, *args, **kwargs)

    log_dict_with_aliases._etude_metric_aliases = True
    L.Fabric.log_dict = log_dict_with_aliases


def _install_checkpoint_hparam_saver(args: argparse.Namespace) -> None:
    import litgpt.pretrain as litgpt_pretrain_module

    def save_hyperparameters(_function, checkpoint_dir: Path, known_commands=None) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": args.model_name,
            "out_dir": str(args.out_dir),
            "precision": args.precision,
            "initial_checkpoint_dir": (
                str(args.initial_checkpoint_dir) if args.initial_checkpoint_dir is not None else None
            ),
            "resume": str(args.resume),
            "data": {
                "class_path": "litgpt.data.LitData",
                "init_args": {
                    "data_path": str(args.data_path),
                    "split_names": ["train", "val"],
                    "num_workers": args.num_workers,
                    "seed": args.seed,
                },
            },
            "train": {
                "save_interval": args.save_interval,
                "log_interval": args.log_interval,
                "global_batch_size": args.global_batch_size,
                "micro_batch_size": args.micro_batch_size,
                "max_tokens": args.max_tokens,
                "max_steps": args.max_steps,
                "max_seq_length": args.max_seq_length,
                "max_norm": args.max_norm,
                "min_lr": args.learning_rate,
                "lr_warmup_steps": args.lr_warmup_steps,
            },
            "eval": {
                "interval": args.eval_interval,
                "max_iters": args.eval_max_iters,
                "initial_validation": False,
                "final_validation": True,
            },
            "log": {
                "project": args.project,
                "run": args.run_name,
            },
            "devices": _normalize_devices(args.devices),
            "num_nodes": args.num_nodes,
            "tokenizer_dir": str(args.tokenizer_dir),
            "logger_name": args.logger_name,
            "seed": args.seed,
        }
        (checkpoint_dir / "hyperparameters.yaml").write_text(json.dumps(payload, indent=2) + "\n")

    litgpt_pretrain_module.save_hyperparameters = save_hyperparameters


def main() -> None:
    args = parse_args()
    from litgpt.args import EvalArgs, LogArgs, TrainArgs
    from litgpt.data import LitData
    from litgpt.pretrain import setup as litgpt_pretrain

    import torch

    if not args.allow_cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("Disabled cuDNN SDPA for stability; using alternate SDPA backends.")

    _install_mfu_metric_alias()
    _install_log_dict_metric_aliases()
    _install_checkpoint_hparam_saver(args)

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

    resume = _coerce_resume(args.resume)
    initial_checkpoint_dir = None
    if args.train_from_scratch:
        resume = False
    elif resume == "auto":
        if _has_existing_checkpoint(args.out_dir):
            initial_checkpoint_dir = None
        else:
            resume = False
            initial_checkpoint_dir = args.initial_checkpoint_dir or args.tokenizer_dir
    elif resume is False:
        initial_checkpoint_dir = args.initial_checkpoint_dir or args.tokenizer_dir

    with _compile_backend(args.compiler):
        import lightning as L

        L.Fabric._etude_max_tokens = args.max_tokens
        litgpt_pretrain(
            model_name=args.model_name,
            out_dir=args.out_dir,
            precision=args.precision,
            initial_checkpoint_dir=initial_checkpoint_dir,
            resume=resume,
            data=data,
            train=train_args,
            eval=eval_args,
            tokenizer_dir=args.tokenizer_dir,
            devices=_normalize_devices(args.devices),
            num_nodes=args.num_nodes,
            logger_name=args.logger_name,
            log=log_args,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
