import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class RustCodeDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B-Base",
        max_length: int = 2048,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        # Option A: Official The Stack, Rust subset
        # ds = load_dataset("bigcode/the-stack", data_dir="data/rust", split="train")

        # Option B: Pre-cleaned Rust subset (smaller, easier to start)
        ds = load_dataset("ammarnasr/the-stack-rust-clean")

        # Tokenize: concatenate & chunk into fixed-length sequences
        def tokenize(examples):
            return self.tokenizer(
                examples["content"],
                truncation=True,
                max_length=self.hparams.max_length,
                padding=False,
            )

        tokenized = ds.map(
            tokenize, batched=True, remove_columns=ds["train"].column_names
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

        grouped = tokenized.map(group_texts, batched=True)
        grouped.set_format("torch")

        self.train_ds = grouped["train"]
        self.val_ds = grouped.get("validation", grouped["test"])

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
