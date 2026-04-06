"""
Stateful dataloader helpers for chat SFT.
"""

import torch


class PackedConversationDataLoader:
    """
    Deterministic BOS-aligned SFT packing with resumable loader state.

    The saved state captures the buffered conversation indices in addition to the
    cursor, so resuming can continue from the same packed stream instead of
    restarting the epoch from scratch.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        device_batch_size,
        max_seq_len,
        split,
        device,
        device_type,
        ddp_rank,
        ddp_world_size,
        num_iterations=-1,
        buffer_size=100,
        resume_state_dict=None,
    ):
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device_batch_size = device_batch_size
        self.max_seq_len = max_seq_len
        self.split = split
        self.device = device
        self.device_type = device_type
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.num_iterations = num_iterations
        self.buffer_size = buffer_size

        self.dataset_size = len(dataset)
        assert self.dataset_size > 0

        self.row_capacity = max_seq_len + 1
        self.render_max_tokens = self.row_capacity
        self.bos_token = tokenizer.get_bos_token_id()

        self.last_step = False
        self.approx_progress = 0.0
        self.current_epoch = 1

        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.iteration = 0
        self.conv_buffer = []

        if resume_state_dict is not None:
            self._load_state_dict(resume_state_dict)

    def _encode_conversation(self, dataset_idx):
        conversation = self.dataset[dataset_idx]
        token_ids, loss_mask = self.tokenizer.render_conversation(
            conversation,
            max_tokens=self.render_max_tokens,
        )
        return dataset_idx, token_ids, loss_mask

    def _load_state_dict(self, state_dict):
        self.cursor = int(state_dict.get("cursor", self.ddp_rank))
        self.consumed = int(state_dict.get("consumed", self.ddp_rank))
        self.epoch = int(state_dict.get("epoch", 1))
        self.iteration = int(state_dict.get("iteration", 0))
        self.last_step = bool(state_dict.get("last_step", False))
        self.approx_progress = float(state_dict.get("approx_progress", 0.0))
        self.current_epoch = int(state_dict.get("current_epoch", self.epoch))

        buffer_indices = state_dict.get("buffer_indices", [])
        self.conv_buffer = [self._encode_conversation(int(idx)) for idx in buffer_indices]

    def _state_dict(self):
        return {
            "cursor": self.cursor,
            "consumed": self.consumed,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "buffer_indices": [dataset_idx for dataset_idx, _, _ in self.conv_buffer],
            "last_step": self.last_step,
            "approx_progress": self.approx_progress,
            "current_epoch": self.current_epoch,
        }

    def _refill_buffer(self):
        while len(self.conv_buffer) < self.buffer_size:
            self.conv_buffer.append(self._encode_conversation(self.cursor))
            self.cursor += self.ddp_world_size
            if self.cursor >= self.dataset_size:
                self.cursor %= self.dataset_size
                self.epoch += 1

    def __iter__(self):
        return self

    def __next__(self):
        resume_state = self._state_dict()
        rows = []
        mask_rows = []
        row_lengths = []

        for _ in range(self.device_batch_size):
            row = []
            mask_row = []
            padded = False
            content_len = 0

            while len(row) < self.row_capacity:
                while len(self.conv_buffer) < self.buffer_size:
                    self._refill_buffer()

                remaining = self.row_capacity - len(row)

                best_idx = -1
                best_len = 0
                for i, (_, conv, _) in enumerate(self.conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    _, conv, conv_mask = self.conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    self.consumed += self.ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([self.bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            row_lengths.append(content_len if padded else self.row_capacity)
            rows.append(row[:self.row_capacity])
            mask_rows.append(mask_row[:self.row_capacity])

        self.iteration += 1
        if self.split == "train":
            self.current_epoch = self.epoch
            if self.num_iterations > 0:
                self.approx_progress = self.iteration / self.num_iterations
                if self.iteration >= self.num_iterations:
                    self.last_step = True
            else:
                self.approx_progress = self.consumed / self.dataset_size
            if self.consumed >= self.dataset_size:
                self.last_step = True

        use_cuda = self.device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=use_cuda,
        ).contiguous()
        targets = batch_tensor[:, 1:].to(
            device=self.device,
            dtype=torch.int64,
            non_blocking=use_cuda,
        ).contiguous()

        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=self.device)
        targets[mask_targets == 0] = -1

        for row_idx, content_len in enumerate(row_lengths):
            if content_len < self.row_capacity:
                targets[row_idx, content_len - 1 :] = -1

        return inputs, targets, resume_state
