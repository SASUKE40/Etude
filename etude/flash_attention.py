"""
Unified Flash Attention interface with automatic FA4/SDPA switching.

Exports `flash_attn` module that matches the FA4 API, but falls back
to PyTorch SDPA on unsupported GPUs, MPS, and CPU.

FA4 (CuTeDSL) supports Ampere (sm80+), Hopper (sm90), and Blackwell (sm100+).
It supports both bf16 and fp16 dtypes for training. KV cache inference always
uses the SDPA fallback since FA4 does not provide flash_attn_with_kvcache.

Usage:
    from etude.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache) — always uses SDPA fallback
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA4 on Ampere+ GPUs
# =============================================================================
def _load_flash_attention_4():
    """Try to load Flash Attention 4 (requires Ampere+ GPU, sm80+)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA4 CuTeDSL kernels support SM80 (Ampere), SM90 (Hopper), SM100/SM110/SM120 (Blackwell)
        if major < 8:
            return None
        from flash_attn.cute import flash_attn_func as fa4_func
        return fa4_func
    except Exception:
        return None


_fa4_func = _load_flash_attention_4()
HAS_FA4 = _fa4_func is not None

# Backward compat aliases
HAS_FA3 = HAS_FA4

# Override for testing: set to 'fa4', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa4():
    """Decide once whether to use FA4, based on availability, override, and dtype."""
    if _override_impl == 'fa4':
        assert HAS_FA4, "Cannot override to FA4: not available on this hardware"
        return True
    # Legacy alias
    if _override_impl == 'fa3':
        assert HAS_FA4, "Cannot override to FA4: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA4:
        # FA4 supports bf16 and fp16; fp32 must use SDPA fallback
        from etude.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE in (torch.bfloat16, torch.float16):
            return True
        return False
    return False

USE_FA4 = _resolve_use_fa4()

# Backward compat aliases
USE_FA3 = USE_FA4


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Window size conversion: our API uses -1 for unlimited, FA4 uses None
# =============================================================================
def _to_fa4_window_size(window_size):
    """Convert (-1, -1) style window_size to FA4's (None, None) convention."""
    left, right = window_size
    return (None if left == -1 else left, None if right == -1 else right)


# =============================================================================
# Public API: Same interface as FA4
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA4:
        # FA4 only supports fp16/bf16 — cast float32 inputs to COMPUTE_DTYPE
        if q.dtype == torch.float32:
            from etude.common import COMPUTE_DTYPE
            cast_dtype = COMPUTE_DTYPE if COMPUTE_DTYPE in (torch.bfloat16, torch.float16) else torch.bfloat16
            q, k, v = q.to(cast_dtype), k.to(cast_dtype), v.to(cast_dtype)
        return _fa4_func(q, k, v, causal=causal, window_size=_to_fa4_window_size(window_size))

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA4 does not provide a flash_attn_with_kvcache function, so this always
    uses the SDPA fallback which manages the KV cache manually (in-place updates).

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
