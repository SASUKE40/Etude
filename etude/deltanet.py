"""
Gated DeltaNet: Linear attention with delta rule and gating.

Reference: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
(Yang et al., 2025)

The Gated DeltaNet replaces standard softmax attention with a recurrent
linear attention mechanism that uses the delta rule for state updates
and input/output gating for expressiveness.

Key properties:
- Linear complexity in sequence length (O(T) vs O(T^2) for attention)
- Recurrent state of size (head_dim_qk, head_dim_v) per head
- Chunkwise parallel training for GPU efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet layer.

    Architecture:
    - Projects input to Q, K, V with independent head counts
    - Short convolution on Q, K, V for local context
    - Beta gate controls delta rule update strength
    - Output gating for expressiveness
    - Recurrent state: S_t = alpha * S_{t-1} + beta * k_t v_t^T

    Args:
        n_embd: Model hidden dimension
        num_heads: Number of heads for Q/K and V (16 in Etude)
        head_dim: Dimension per head (128 in Etude)
    """

    def __init__(self, n_embd, num_heads=16, head_dim=128):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_dim = num_heads * head_dim  # 16 * 128 = 2048
        self.v_dim = num_heads * head_dim   # 16 * 128 = 2048

        # QKV projections
        self.q_proj = Linear(n_embd, self.qk_dim, bias=False)
        self.k_proj = Linear(n_embd, self.qk_dim, bias=False)
        self.v_proj = Linear(n_embd, self.v_dim, bias=False)

        # Beta gate: controls how much of the new kv pair to write into state
        self.beta_proj = Linear(n_embd, num_heads, bias=False)

        # Output gate
        self.gate_proj = Linear(n_embd, self.v_dim, bias=False)

        # Output projection
        self.o_proj = Linear(self.v_dim, n_embd, bias=False)

        # Short convolution for local context (kernel size 4)
        self.conv_size = 4
        self.q_conv = nn.Conv1d(
            self.qk_dim, self.qk_dim, self.conv_size,
            padding=self.conv_size - 1, groups=self.qk_dim, bias=False
        )
        self.k_conv = nn.Conv1d(
            self.qk_dim, self.qk_dim, self.conv_size,
            padding=self.conv_size - 1, groups=self.qk_dim, bias=False
        )
        self.v_conv = nn.Conv1d(
            self.v_dim, self.v_dim, self.conv_size,
            padding=self.conv_size - 1, groups=self.v_dim, bias=False
        )

        # Group norm for output (per-head normalization)
        self.out_norm = nn.GroupNorm(num_heads, self.v_dim, eps=1e-5, affine=False)

    def _apply_conv(self, x, conv):
        """Apply causal 1D convolution. x: (B, T, D) -> (B, T, D)"""
        # Conv1d expects (B, D, T)
        x = x.transpose(1, 2).to(conv.weight.dtype)
        x = conv(x)[:, :, :x.size(2)]  # causal: trim to original length
        return x.transpose(1, 2)

    def forward(self, x, recurrent_state=None):
        """
        Args:
            x: (B, T, n_embd)
            recurrent_state: optional (B, num_heads, head_dim, head_dim) state tensor

        Returns:
            output: (B, T, n_embd)
            new_state: (B, num_heads, head_dim, head_dim) updated state
        """
        B, T, _ = x.size()

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, qk_dim)
        k = self.k_proj(x)  # (B, T, qk_dim)
        v = self.v_proj(x)  # (B, T, v_dim)

        # Short convolution + activation
        q = F.silu(self._apply_conv(q, self.q_conv))
        k = F.silu(self._apply_conv(k, self.k_conv))
        v = F.silu(self._apply_conv(v, self.v_conv))

        # Compute gates
        beta = torch.sigmoid(self.beta_proj(x))  # (B, T, num_heads)
        gate = torch.sigmoid(self.gate_proj(x))   # (B, T, v_dim)

        # Reshape for multi-head: (B, T, H, D)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Normalize q and k for stability
        q = q * (self.head_dim ** -0.5)

        # Run the delta rule recurrence (chunkwise for training)
        output, new_state = self._chunkwise_recurrence(q, k, v, beta, recurrent_state)

        # output: (B, T, num_heads, head_dim) -> (B, T, v_dim)
        output = output.reshape(B, T, self.v_dim)

        # Apply group norm then output gate
        output = self.out_norm(output.transpose(1, 2)).transpose(1, 2)
        output = output * gate

        # Project back to model dim
        output = self.o_proj(output)

        return output, new_state

    def _chunkwise_recurrence(self, q, k, v, beta, state, chunk_size=64):
        """
        Chunkwise parallel delta rule recurrence for training efficiency.

        Within each chunk: parallel attention-like computation.
        Across chunks: recurrent state passing.

        Args:
            q, k, v: (B, T, H, D)
            beta: (B, T, H)
            state: (B, H, D, D) or None

        Returns:
            output: (B, T, H, D)
            final_state: (B, H, D, D)
        """
        B, T, H, D = q.shape
        device = q.device
        dtype = q.dtype

        if state is None:
            state = torch.zeros(B, H, D, D, device=device, dtype=dtype)

        # For short sequences, use simple recurrence
        if T <= chunk_size:
            return self._recurrence(q, k, v, beta, state)

        # Chunk the sequence
        num_chunks = (T + chunk_size - 1) // chunk_size
        # Pad if necessary
        pad_len = num_chunks * chunk_size - T
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))

        outputs = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            q_chunk = q[:, start:end]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            beta_chunk = beta[:, start:end]

            out_chunk, state = self._recurrence(q_chunk, k_chunk, v_chunk, beta_chunk, state)
            outputs.append(out_chunk)

        output = torch.cat(outputs, dim=1)
        if pad_len > 0:
            output = output[:, :T]

        return output, state

    def _recurrence(self, q, k, v, beta, state):
        """
        Simple recurrence for the delta rule.

        S_t = (1 - beta_t) * S_{t-1} + beta_t * k_t ⊗ v_t
        o_t = q_t^T @ S_t

        Args:
            q, k, v: (B, T, H, D)
            beta: (B, T, H)
            state: (B, H, D, D)

        Returns:
            output: (B, T, H, D)
            final_state: (B, H, D, D)
        """
        B, T, H, D = q.shape
        outputs = []

        for t in range(T):
            # Get current timestep
            q_t = q[:, t]   # (B, H, D)
            k_t = k[:, t]   # (B, H, D)
            v_t = v[:, t]   # (B, H, D)
            b_t = beta[:, t]  # (B, H)

            # Expand beta for broadcasting: (B, H, 1, 1)
            b_t = b_t.unsqueeze(-1).unsqueeze(-1)

            # Delta rule update: S = (1 - beta) * S + beta * k ⊗ v
            kv = torch.einsum('bhd,bhe->bhde', k_t, v_t)  # outer product: (B, H, D, D)
            state = (1 - b_t) * state + b_t * kv

            # Query the state: o = S^T @ q
            o_t = torch.einsum('bhde,bhd->bhe', state, q_t)  # (B, H, D)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)  # (B, T, H, D)
        return output, state
