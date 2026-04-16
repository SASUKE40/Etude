import argparse

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import torch

from etude.common import COMPUTE_DTYPE
from etude.qwen_data_module import RustCodeDataModule
from etude.qwen_trainer import QwenRustPretrainer


if COMPUTE_DTYPE == torch.bfloat16:
    amp_precision = "bf16-mixed"
elif COMPUTE_DTYPE == torch.float16:
    amp_precision = "16-mixed"
else:
    amp_precision = "32-true"


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the Qwen Rust pretraining setup.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device micro batch size.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preprocessing-num-workers", type=int, default=None)
    parser.add_argument("--processed-dataset-dir", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/qwen-rust")
    parser.add_argument("--output-dir", type=str, default="qwen3.5-0.8b-rust")
    parser.add_argument("--project", type=str, default="qwen-rust-pretrain")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--val-check-interval", type=int, default=2000)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-n-train-steps", type=int, default=5000)
    return parser


def main():
    args = _build_arg_parser().parse_args()
    dm = RustCodeDataModule(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocessing_num_workers=args.preprocessing_num_workers,
        processed_dataset_dir=args.processed_dataset_dir,
    )

    model = QwenRustPretrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            every_n_train_steps=args.checkpoint_every_n_train_steps,
            save_top_k=3,
            monitor="val/loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=amp_precision,
        max_steps=args.total_steps,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=WandbLogger(project=args.project),
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    # Save final HF-compatible model
    if trainer.is_global_zero:
        model.model.save_pretrained(args.output_dir)
        dm.tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
