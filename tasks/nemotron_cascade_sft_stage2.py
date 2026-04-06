"""
Local task loader and shared helpers for NVIDIA Nemotron Cascade SFT Stage 2.
"""

import glob
import hashlib
import json
import os

from datasets import load_dataset

from etude.common import get_base_dir
from tasks.common import Task

HF_DATASET = "nvidia/Nemotron-Cascade-SFT-Stage-2"
SUPPORTED_SUBSETS = (
    "general",
    "instruction-following",
    "tool_calling",
    "math",
    "science",
    "code",
    "swe_localization",
    "swe_repair",
    "swe_testgen",
)
DEFAULT_SUBSETS = (
    "instruction-following",
    "code",
)
SUPPORTED_ROLES = {"system", "user", "assistant"}


def get_default_data_dir():
    return os.path.join(get_base_dir(), "datasets", "nemotron-cascade-sft-stage-2")


def parse_subset_names(raw_subsets):
    if raw_subsets is None:
        return None
    if isinstance(raw_subsets, str):
        subsets = [part.strip() for part in raw_subsets.split(",")]
    else:
        subsets = [str(part).strip() for part in raw_subsets]
    subsets = [subset for subset in subsets if subset]
    if not subsets:
        return None
    unknown = sorted(set(subsets) - set(SUPPORTED_SUBSETS))
    if unknown:
        raise ValueError(f"Unknown Nemotron subsets: {unknown}")
    return tuple(subsets)


def normalize_messages(messages):
    """
    Normalize and validate conversations so they fit the tokenizer's chat format.
    Returns a cleaned list of {role, content} dicts.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Conversation must contain at least 2 messages")

    cleaned = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {idx} must be a dict")
        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Message {idx} has unsupported role: {role}")
        if not isinstance(content, str):
            raise ValueError(f"Message {idx} content must be a string")
        if not content.strip():
            raise ValueError(f"Message {idx} content is empty")
        cleaned.append({"role": role, "content": content})

    if cleaned[0]["role"] == "system":
        if len(cleaned) < 3:
            raise ValueError("System conversations must include user and assistant turns")
        alternating = cleaned[1:]
    else:
        alternating = cleaned

    if alternating[0]["role"] != "user":
        raise ValueError("Conversation must start with a user message after any system prompt")
    if alternating[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with an assistant message")

    for idx, message in enumerate(alternating):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError(
                f"Message {idx} has role {message['role']} but should be {expected_role}"
            )

    return cleaned


def pick_split_for_messages(messages, subset, val_ratio):
    payload = json.dumps(
        {"subset": subset, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    score = int.from_bytes(digest, byteorder="big") / 2**64
    return "val" if score < val_ratio else "train"


def prepared_split_dir(data_dir, split):
    return os.path.join(data_dir, split)


def list_prepared_files(data_dir=None, split="train"):
    data_dir = data_dir or get_default_data_dir()
    pattern = os.path.join(prepared_split_dir(data_dir, split), "*.parquet")
    return sorted(glob.glob(pattern))


def has_prepared_split(data_dir=None, split="train"):
    return len(list_prepared_files(data_dir=data_dir, split=split)) > 0


def has_prepared_data(data_dir=None):
    return has_prepared_split(data_dir=data_dir, split="train") and has_prepared_split(
        data_dir=data_dir, split="val"
    )


class NemotronCascadeSFTStage2(Task):
    """
    Load locally prepared Nemotron Stage 2 parquet shards for chat SFT.
    """

    def __init__(self, split, data_dir=None, subsets=None, shuffle_seed=42, **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "val"}, "split must be train|val"
        self.split = split
        self.data_dir = data_dir or get_default_data_dir()
        self.subsets = parse_subset_names(subsets)

        parquet_files = list_prepared_files(data_dir=self.data_dir, split=split)
        if not parquet_files:
            prepare_cmd = (
                "python data/nemotron-cascade-sft-stage-2/prepare.py "
                f"--output-dir {self.data_dir}"
            )
            raise FileNotFoundError(
                f"No prepared Nemotron data found for split '{split}' in {self.data_dir}. "
                f"Run: {prepare_cmd}"
            )

        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        if self.subsets is not None:
            allowed = set(self.subsets)
            ds = ds.filter(
                lambda row: row["subset"] in allowed,
                desc=f"Filtering Nemotron subsets for {split}",
            )
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed)

        self.ds = ds
        self.length = len(self.ds)
        if self.length == 0:
            raise ValueError(
                f"No rows left in Nemotron split '{split}' after filtering subsets {self.subsets}"
            )

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = normalize_messages(row["messages"])
        return {
            "messages": messages,
            "subset": row.get("subset"),
            "category": row.get("category"),
            "source": row.get("source"),
            "thinking": bool(row.get("thinking", False)),
        }
