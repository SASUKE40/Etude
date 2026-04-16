import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from etude.flash_attention import flash_attn


def flash_attention_4_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    # Fall back to Transformers' SDPA path for features our local FA4 wrapper
    # does not implement, such as arbitrary masks or attention outputs.
    if attention_mask is not None or kwargs.get("output_attentions", False) or softcap is not None:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            is_causal=is_causal,
            **kwargs,
        )

    if any(dim == 0 for dim in query.shape):
        raise ValueError("Flash Attention 4 does not support zero-sized attention inputs.")

    # HF attention backends receive (B, H, T, D); our wrapper expects (B, T, H, D).
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    window_size = (-1, -1) if sliding_window is None else (sliding_window, 0)
    output = flash_attn.flash_attn_func(
        query,
        key,
        value,
        causal=module.is_causal if is_causal is None else is_causal,
        window_size=window_size,
    )
    return output.transpose(1, 2), None


AttentionInterface.register("flash_attention_4", flash_attention_4_forward)
