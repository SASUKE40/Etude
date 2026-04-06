from pathlib import Path

import pytest

from tasks.nemotron_cascade_sft_stage2 import (
    has_prepared_data,
    has_prepared_split,
    normalize_messages,
    parse_subset_names,
    pick_split_for_messages,
)


def test_normalize_messages_accepts_system_prefix_and_preserves_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "  hi  "},
        {"role": "assistant", "content": "  hello  "},
    ]

    normalized = normalize_messages(messages)

    assert normalized == messages


def test_normalize_messages_rejects_bad_role_order():
    messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "hi"},
    ]

    with pytest.raises(ValueError, match="must start with a user message"):
        normalize_messages(messages)


def test_parse_subset_names_rejects_unknown_subset():
    with pytest.raises(ValueError, match="Unknown Nemotron subsets"):
        parse_subset_names("general,not-a-real-subset")


def test_pick_split_is_deterministic():
    messages = [
        {"role": "user", "content": "Write a haiku about Rust."},
        {"role": "assistant", "content": "Borrowed autumn wind\nThreads of iron sing softly\nSafe embers endure"},
    ]

    split_a = pick_split_for_messages(messages, subset="general", val_ratio=0.005)
    split_b = pick_split_for_messages(messages, subset="general", val_ratio=0.005)

    assert split_a == split_b
    assert split_a in {"train", "val"}


def test_prepared_split_detection(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    assert not has_prepared_split(tmp_path, "train")
    assert not has_prepared_data(tmp_path)

    (train_dir / "part_00000.parquet").touch()
    assert has_prepared_split(tmp_path, "train")
    assert not has_prepared_data(tmp_path)

    (val_dir / "part_00000.parquet").touch()
    assert has_prepared_data(tmp_path)
