import torch

from etude.sft_dataloader import PackedConversationDataLoader


class DummyDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class DummyTokenizer:
    def get_bos_token_id(self):
        return 1

    def render_conversation(self, conversation, max_tokens):
        token_ids = conversation["tokens"][:max_tokens]
        loss_mask = conversation.get("mask", [1] * len(token_ids))
        return token_ids, loss_mask


def build_loader(resume_state_dict=None):
    dataset = DummyDataset(
        [
            {"tokens": [1, 11, 12, 13, 14], "mask": [0, 1, 1, 1, 1]},
            {"tokens": [1, 21, 22], "mask": [0, 1, 1]},
            {"tokens": [1, 31, 32, 33], "mask": [0, 1, 1, 1]},
            {"tokens": [1, 41, 42, 43, 44, 45], "mask": [0, 1, 1, 1, 1, 1]},
            {"tokens": [1, 51, 52, 53], "mask": [0, 1, 1, 1]},
            {"tokens": [1, 61, 62], "mask": [0, 1, 1]},
        ]
    )
    tokenizer = DummyTokenizer()
    return PackedConversationDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        device_batch_size=2,
        max_seq_len=7,
        split="train",
        device="cpu",
        device_type="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        buffer_size=4,
        resume_state_dict=resume_state_dict,
    )


def test_packed_conversation_loader_resume_replays_same_next_batch():
    loader = build_loader()

    _, _, _ = next(loader)
    expected_inputs, expected_targets, resume_state = next(loader)

    resumed_loader = build_loader(resume_state_dict=resume_state)
    resumed_inputs, resumed_targets, _ = next(resumed_loader)

    assert torch.equal(resumed_inputs, expected_inputs)
    assert torch.equal(resumed_targets, expected_targets)

    expected_next_inputs, expected_next_targets, expected_next_state = next(loader)
    resumed_next_inputs, resumed_next_targets, resumed_next_state = next(resumed_loader)

    assert torch.equal(resumed_next_inputs, expected_next_inputs)
    assert torch.equal(resumed_next_targets, expected_next_targets)
    assert resumed_next_state == expected_next_state


def test_packed_conversation_loader_resume_preserves_progress():
    loader = PackedConversationDataLoader(
        dataset=DummyDataset(
            [
                {"tokens": [1, 11, 12], "mask": [0, 1, 1]},
                {"tokens": [1, 21, 22], "mask": [0, 1, 1]},
                {"tokens": [1, 31, 32], "mask": [0, 1, 1]},
                {"tokens": [1, 41, 42], "mask": [0, 1, 1]},
            ]
        ),
        tokenizer=DummyTokenizer(),
        device_batch_size=1,
        max_seq_len=3,
        split="train",
        device="cpu",
        device_type="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        num_iterations=4,
        buffer_size=2,
    )

    _, _, resume_state = next(loader)
    assert loader.iteration == 1
    assert loader.approx_progress == 0.25

    resumed_loader = PackedConversationDataLoader(
        dataset=loader.dataset,
        tokenizer=loader.tokenizer,
        device_batch_size=1,
        max_seq_len=3,
        split="train",
        device="cpu",
        device_type="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        num_iterations=4,
        buffer_size=2,
        resume_state_dict=resume_state,
    )

    next(resumed_loader)

    assert resumed_loader.iteration == 1
    assert resumed_loader.approx_progress == 0.25
