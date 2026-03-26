"""
Etude GPT model — hybrid Gated DeltaNet + Gated Attention architecture.

Architecture (0.8B parameters):
- Hidden dimension: 1024
- Token embedding: 248320 (padded)
- 24 layers in layout: 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- Gated DeltaNet: 16 heads for V and QK, head dim 128
- Gated Attention: 8 Q heads, 2 KV heads, head dim 256, RoPE dim 64
- FFN: SwiGLU with intermediate dim 3584
- LM output: tied to token embedding
- MTP: multi-token prediction
- Context length: 262,144 natively
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from etude.common import get_dist_info, print0, COMPUTE_DTYPE
from etude.optim import MuonAdamW, DistMuonAdamW
from etude.flash_attention import flash_attn
from etude.deltanet import GatedDeltaNet


@dataclass
class GPTConfig:
    sequence_len: int = 262144          # 262K native context
    vocab_size: int = 248320            # padded token embedding size
    n_layer: int = 24                   # total layers
    n_embd: int = 1024                  # hidden dimension
    # Gated DeltaNet config
    deltanet_heads: int = 16            # heads for QK and V
    deltanet_head_dim: int = 128        # head dimension
    # Gated Attention config
    attn_q_heads: int = 8               # Q heads
    attn_kv_heads: int = 2              # KV heads (GQA)
    attn_head_dim: int = 256            # head dimension
    rope_dim: int = 64                  # RoPE dimension (applied to first rope_dim dims of each head)
    # FFN config
    ffn_intermediate: int = 3584        # SwiGLU intermediate dimension
    # MTP config
    mtp_steps: int = 1                  # number of additional future tokens to predict (0 = standard next-token only)
    # Layout config
    group_size: int = 4                 # layers per group (3 DeltaNet + 1 Attention)
    deltanet_per_group: int = 3         # DeltaNet layers per group


def norm(x):
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin, rope_dim):
    """Apply RoPE to the first rope_dim dimensions of each head, pass through the rest."""
    d = rope_dim // 2
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x1, x2 = x_rope[..., :d], x_rope[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2, x_pass], dim=-1)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    FFN(x) = W_down(SiLU(W_gate(x)) * W_up(x))
    """
    def __init__(self, n_embd, intermediate_dim):
        super().__init__()
        self.w_gate = Linear(n_embd, intermediate_dim, bias=False)
        self.w_up = Linear(n_embd, intermediate_dim, bias=False)
        self.w_down = Linear(intermediate_dim, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GatedAttention(nn.Module):
    """
    Gated multi-head attention with GQA and partial RoPE.

    - 8 Q heads, 2 KV heads, head dim 256
    - RoPE applied to first 64 dims of each head
    - Output gating for consistency with DeltaNet blocks
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.n_q_heads = config.attn_q_heads
        self.n_kv_heads = config.attn_kv_heads
        self.head_dim = config.attn_head_dim
        self.rope_dim = config.rope_dim

        q_dim = self.n_q_heads * self.head_dim    # 8 * 256 = 2048
        kv_dim = self.n_kv_heads * self.head_dim  # 2 * 256 = 512

        self.c_q = Linear(self.n_embd, q_dim, bias=False)
        self.c_k = Linear(self.n_embd, kv_dim, bias=False)
        self.c_v = Linear(self.n_embd, kv_dim, bias=False)
        self.c_proj = Linear(q_dim, self.n_embd, bias=False)

        # Output gating
        self.gate_proj = Linear(self.n_embd, q_dim, bias=False)

    def forward(self, x, cos_sin, kv_cache=None):
        B, T, _ = x.size()

        q = self.c_q(x).view(B, T, self.n_q_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply partial RoPE (first rope_dim dimensions)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin, self.rope_dim)
        k = apply_rotary_emb(k, cos, sin, self.rope_dim)

        # QK norm for stability
        q, k = norm(q), norm(k)

        # Flash Attention
        if kv_cache is None:
            # Training: full causal attention
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
        else:
            # Inference with KV cache
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=(-1, 0),
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Output gating
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, q_dim)
        y = y.contiguous().view(B, T, -1)
        y = y * gate

        y = self.c_proj(y)
        return y


class DeltaNetBlock(nn.Module):
    """Block with Gated DeltaNet attention + SwiGLU FFN."""
    def __init__(self, config):
        super().__init__()
        self.attn = GatedDeltaNet(
            n_embd=config.n_embd,
            num_heads=config.deltanet_heads,
            head_dim=config.deltanet_head_dim,
        )
        self.ffn = SwiGLUFFN(config.n_embd, config.ffn_intermediate)

    def forward(self, x, recurrent_state=None, **kwargs):
        # Pre-norm DeltaNet + residual
        attn_out, new_state = self.attn(norm(x), recurrent_state)
        x = x + attn_out
        # Pre-norm FFN + residual
        x = x + self.ffn(norm(x))
        return x, new_state


class AttentionBlock(nn.Module):
    """Block with Gated Attention + SwiGLU FFN."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = GatedAttention(config, layer_idx)
        self.ffn = SwiGLUFFN(config.n_embd, config.ffn_intermediate)

    def forward(self, x, cos_sin, kv_cache=None, **kwargs):
        # Pre-norm Attention + residual
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        # Pre-norm FFN + residual
        x = x + self.ffn(norm(x))
        return x


class MTPHead(nn.Module):
    """Multi-Token Prediction head.

    Predicts a future token by combining the current hidden state with
    the embedding of the next ground-truth token (teacher forcing).
    """
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.proj = Linear(n_embd * 2, n_embd, bias=False)
        # lm_head weight will be shared with the main embedding

    def forward(self, hidden, next_token_emb):
        """
        Args:
            hidden: (B, T, n_embd) - current hidden states
            next_token_emb: (B, T, n_embd) - embeddings of next ground-truth tokens
        Returns:
            projected: (B, T, n_embd) - projected states for logit computation
        """
        combined = torch.cat([hidden, next_token_emb], dim=-1)
        return self.proj(combined)


class GPT(nn.Module):
    def __init__(self, config):
        """
        NOTE: this __init__ runs in meta device context.
        All actual data initialization happens in init_weights().
        """
        super().__init__()
        self.config = config

        assert config.n_layer == (config.n_layer // config.group_size) * config.group_size, \
            f"n_layer ({config.n_layer}) must be divisible by group_size ({config.group_size})"
        num_groups = config.n_layer // config.group_size

        # Token embedding (tied with lm_head)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(),
        })

        # Build layers: 6 × (3 × DeltaNet + 1 × Attention)
        attn_layer_idx = 0  # separate index for attention layers (for KV cache)
        self._attn_layer_indices = []  # track which global indices are attention layers
        for g in range(num_groups):
            # 3 DeltaNet blocks
            for _ in range(config.deltanet_per_group):
                self.transformer["h"].append(DeltaNetBlock(config))
            # 1 Attention block
            self._attn_layer_indices.append(g * config.group_size + config.deltanet_per_group)
            self.transformer["h"].append(AttentionBlock(config, attn_layer_idx))
            attn_layer_idx += 1

        self.n_attn_layers = attn_layer_idx

        # LM head (tied to wte)
        # We use a Linear but will tie weights in init_weights
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # MTP heads for multi-token prediction
        self.mtp_heads = nn.ModuleList()
        if config.mtp_steps > 0:
            for _ in range(config.mtp_steps):
                self.mtp_heads.append(MTPHead(config.n_embd, config.vocab_size))

        # Rotary embeddings for attention layers (precomputed, not learned)
        # Only need rope_dim/2 frequencies since we only apply RoPE to first rope_dim dims
        self.rotary_seq_len = min(config.sequence_len, 262144)  # precompute up to context length
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, config.rope_dim
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize all model parameters."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)

        # Tie lm_head weights to embedding
        self.lm_head.weight = self.transformer.wte.weight

        for block in self.transformer.h:
            if isinstance(block, DeltaNetBlock):
                # DeltaNet projections
                dn = block.attn
                torch.nn.init.uniform_(dn.q_proj.weight, -s, s)
                torch.nn.init.uniform_(dn.k_proj.weight, -s, s)
                torch.nn.init.uniform_(dn.v_proj.weight, -s, s)
                torch.nn.init.zeros_(dn.o_proj.weight)
                torch.nn.init.uniform_(dn.beta_proj.weight, -0.01, 0.01)
                torch.nn.init.uniform_(dn.gate_proj.weight, -s, s)
                # Conv weights
                torch.nn.init.normal_(dn.q_conv.weight, std=0.02)
                torch.nn.init.normal_(dn.k_conv.weight, std=0.02)
                torch.nn.init.normal_(dn.v_conv.weight, std=0.02)
                # FFN
                torch.nn.init.uniform_(block.ffn.w_gate.weight, -s * 0.4, s * 0.4)
                torch.nn.init.uniform_(block.ffn.w_up.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.ffn.w_down.weight)

            elif isinstance(block, AttentionBlock):
                # Attention projections
                attn = block.attn
                torch.nn.init.uniform_(attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(attn.c_proj.weight)
                torch.nn.init.uniform_(attn.gate_proj.weight, -s, s)
                # FFN
                torch.nn.init.uniform_(block.ffn.w_gate.weight, -s * 0.4, s * 0.4)
                torch.nn.init.uniform_(block.ffn.w_up.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.ffn.w_down.weight)

        # MTP heads
        for mtp_head in self.mtp_heads:
            torch.nn.init.uniform_(mtp_head.proj.weight, -s, s)

        # Rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.config.rope_dim
        )
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, rope_dim, base=500000, device=None):
        """Precompute RoPE cos/sin for the given sequence length.

        Only computes for rope_dim/2 frequency pairs since RoPE is applied
        to the first rope_dim dimensions of each head.
        """
        if device is None:
            device = self.transformer.wte.weight.device
        half_dim = rope_dim // 2
        channel_range = torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / rope_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (seq_len, half_dim)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos = cos[None, :, None, :]  # (1, seq_len, 1, half_dim)
        sin = sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return estimated FLOPs per token (forward + backward)."""
        config = self.config
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude embeddings (lookup, not matmul)
        nparams_exclude = self.transformer.wte.weight.numel()
        # Attention FLOPs for attention layers only
        num_groups = config.n_layer // config.group_size
        h = config.attn_q_heads
        d = config.attn_head_dim
        t = config.sequence_len
        attn_flops = num_groups * 12 * h * d * t  # 1 attention layer per group
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return parameter counts for scaling law analysis."""
        wte = self.transformer.wte.weight.numel()
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        mtp_params = sum(p.numel() for p in self.mtp_heads.parameters())
        # lm_head is tied to wte, so don't count separately
        total = wte + transformer_matrices + mtp_params
        return {
            'wte': wte,
            'lm_head': 0,  # tied to wte
            'transformer_matrices': transformer_matrices,
            'mtp_params': mtp_params,
            'total': total,
        }

    def setup_optimizer(self, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        mtp_params = list(self.mtp_heads.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        # lm_head is tied, don't add separately

        # Scale LR for AdamW params by 1/sqrt(dmodel) (tuned for 768-dim reference)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        ]

        # MTP params go into AdamW
        if mtp_params:
            param_groups.append(
                dict(kind='adamw', params=mtp_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            )

        # Muon groups (matrix params, grouped by shape)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Rotary embeddings for attention layers
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Forward through hybrid layers
        for i, block in enumerate(self.transformer.h):
            if isinstance(block, DeltaNetBlock):
                x, _ = block(x)  # DeltaNet doesn't use cos_sin or kv_cache during training
            elif isinstance(block, AttentionBlock):
                x = block(x, cos_sin, kv_cache)

        x = norm(x)

        # LM head (tied weights)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            # Main next-token prediction loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )

            # Multi-token prediction losses
            # MTP produces different-length outputs due to shifts, so only use
            # mean reduction (skip MTP when loss_reduction='none' for BPB eval)
            if self.config.mtp_steps > 0 and len(self.mtp_heads) > 0 and loss_reduction == 'mean':
                mtp_weight = 0.1  # weight for MTP auxiliary losses
                for step_i, mtp_head in enumerate(self.mtp_heads):
                    # For step i+1, predict token at position t+i+2 given hidden state at t
                    shift = step_i + 1
                    if T > shift + 1:
                        # Get embeddings of intermediate ground-truth tokens
                        next_emb = self.transformer.wte(targets[:, shift-1:-1])
                        next_emb = next_emb.to(COMPUTE_DTYPE)
                        hidden = x[:, :-shift-1]  # hidden states for prediction positions
                        next_emb = next_emb[:, :hidden.size(1)]  # align lengths

                        mtp_proj = mtp_head(norm(hidden), norm(next_emb))
                        mtp_logits = self.lm_head(mtp_proj)
                        mtp_logits = mtp_logits.float()
                        mtp_logits = softcap * torch.tanh(mtp_logits / softcap)

                        mtp_targets = targets[:, shift+1:shift+1+hidden.size(1)]
                        mtp_loss = F.cross_entropy(
                            mtp_logits.reshape(-1, mtp_logits.size(-1)),
                            mtp_targets.reshape(-1),
                            ignore_index=-1,
                            reduction=loss_reduction,
                        )
                        loss = loss + mtp_weight * mtp_loss

            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive autoregressive streaming inference."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
