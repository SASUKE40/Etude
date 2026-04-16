import os

import lightning as L
from datasets import load_dataset, load_from_disk
from filelock import FileLock
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from etude.common import get_base_dir


class RustCodeDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B-Base",
        max_length: int = 2048,
        batch_size: int = 4,
        num_workers: int = 4,
        preprocessing_num_workers: int | None = None,
        processed_dataset_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _processed_dataset_dir(self) -> str:
        if self.hparams.processed_dataset_dir:
            return self.hparams.processed_dataset_dir
        return os.path.join(
            get_base_dir(),
            "datasets",
            f"qwen-rust-clean-{self.hparams.max_length}",
        )

    def _build_processed_dataset(self):
        # Option A: Official The Stack, Rust subset
        # ds = load_dataset("bigcode/the-stack", data_dir="data/rust", split="train")

        # Option B: Pre-cleaned Rust subset (smaller, easier to start)
        ds = load_dataset("ammarnasr/the-stack-rust-clean")

        # Tokenize: concatenate & chunk into fixed-length sequences
        def tokenize(examples):
            return self.tokenizer(
                examples["content"],
                padding=False,
            )

        tokenized = ds.map(
            tokenize,
            batched=True,
            remove_columns=ds["train"].column_names,
            num_proc=self.hparams.preprocessing_num_workers,
            desc="Tokenizing Rust dataset",
        )

        # Group texts into chunks of max_length for efficient training
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = (
                len(concatenated["input_ids"]) // self.hparams.max_length
            ) * self.hparams.max_length
            result = {
                k: [
                    t[i : i + self.hparams.max_length]
                    for i in range(0, total_length, self.hparams.max_length)
                ]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        grouped = tokenized.map(
            group_texts,
            batched=True,
            num_proc=self.hparams.preprocessing_num_workers,
            desc="Packing Rust token sequences",
        )
        return grouped

    def prepare_data(self):
        processed_dataset_dir = self._processed_dataset_dir()
        lock_path = processed_dataset_dir + ".lock"

        with FileLock(lock_path):
            if os.path.exists(processed_dataset_dir):
                return

            os.makedirs(os.path.dirname(processed_dataset_dir), exist_ok=True)
            grouped = self._build_processed_dataset()
            grouped.save_to_disk(processed_dataset_dir)

    def setup(self, stage=None):
        grouped = load_from_disk(self._processed_dataset_dir())
        grouped.set_format("torch")

        self.train_ds = grouped["train"]
        if "validation" in grouped:
            self.val_ds = grouped["validation"]
        elif "test" in grouped:
            self.val_ds = grouped["test"]
        else:
            self.val_ds = self.train_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
