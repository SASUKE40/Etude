from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from etude.common import COMPUTE_DTYPE, get_dist_info, print0
from etude.optim import DistMuonAdamW, MuonAdamW
from etude.qwen3_5_transformers import Qwen3_5GatedDeltaNet


class Linear(nn.Linear):
    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


@dataclass
class Qwen3_5Config:
    sequence_len: int = 262144
    vocab_size: int = 248320
    n_layer: int = 24
    n_embd: int = 1024
    n_heads: int = 8
    n_kv_groups: int = 2
    head_dim: int = 256
    hidden_dim: int = 3584
    qk_norm: bool = True
    rope_base: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    mtp_steps: int = 0
    layer_types: list[str] = field(default_factory=lambda: [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ])


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.n_embd, config.hidden_dim, bias=False)
        self.fc2 = Linear(config.n_embd, config.hidden_dim, bias=False)
        self.fc3 = Linear(config.hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float() / rotary_dim
        )
    )
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    _, _, seq_len, head_dim = x.shape
    rot_dim = cos.shape[-1]
    if rot_dim > head_dim:
        raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}.")
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat([(x_rot * cos) + (rotated * sin), x_pass], dim=-1).to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_heads % config.n_kv_groups == 0
        self.num_heads = config.n_heads
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_heads // config.n_kv_groups
        self.head_dim = config.head_dim
        self.d_out = config.n_heads * config.head_dim

        self.W_query = Linear(config.n_embd, self.d_out * 2, bias=False)
        self.W_key = Linear(config.n_embd, config.n_kv_groups * config.head_dim, bias=False)
        self.W_value = Linear(config.n_embd, config.n_kv_groups * config.head_dim, bias=False)
        self.out_proj = Linear(self.d_out, config.n_embd, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        q_and_gate = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores * (self.head_dim ** -0.5), dim=-1, dtype=torch.float32).to(queries.dtype)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        context = context * torch.sigmoid(gate)
        return self.out_proj(context)


class _Qwen3_5ConfigAdapter:
    def __init__(self, config):
        self.hidden_size = config.n_embd
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_num_key_heads = config.linear_num_key_heads
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim
        self.linear_conv_kernel_dim = config.linear_conv_kernel_dim
        self.hidden_act = "silu"
        self.rms_norm_eps = config.rms_norm_eps
        self.dtype = None


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_type, layer_idx):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(config)
        elif layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(_Qwen3_5ConfigAdapter(config), layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        self.ff = FeedForward(config)
        self.norm1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        if self.layer_type == "full_attention":
            x = self.token_mixer(x, mask, cos, sin)
        else:
            x = self.token_mixer(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + shortcut


class Qwen3_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if len(config.layer_types) != config.n_layer:
            raise ValueError("len(layer_types) must equal n_layer")

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(config, layer_type, idx) for idx, layer_type in enumerate(config.layer_types)]
        )
        self.final_norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.out_head = Linear(config.n_embd, config.vocab_size, bias=False)

        cos, sin = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_base,
            context_length=config.sequence_len,
            partial_rotary_factor=config.partial_rotary_factor,
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.n_attn_layers = sum(layer_type == "full_attention" for layer_type in config.layer_types)

    @torch.no_grad()
    def init_weights(self):
        s = (3 ** 0.5) * (self.config.n_embd ** -0.5)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.8)
        self.out_head.weight = self.tok_emb.weight

        for module in self.modules():
            if isinstance(module, Linear) and module is not self.out_head:
                nn.init.uniform_(module.weight, -s, s)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, std=0.02)

        for block in self.trf_blocks:
            nn.init.zeros_(block.ff.fc3.weight)
            if hasattr(block.token_mixer, "out_proj"):
                nn.init.zeros_(block.token_mixer.out_proj.weight)

        self.tok_emb.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.tok_emb.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.tok_emb.weight.numel()
        attn_flops = self.n_attn_layers * 12 * self.config.n_heads * self.config.head_dim * self.config.sequence_len
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = self.tok_emb.weight.numel()
        transformer_matrices = sum(p.numel() for p in self.trf_blocks.parameters()) + sum(p.numel() for p in self.final_norm.parameters())
        total = wte + transformer_matrices
        return {
            "wte": wte,
            "lm_head": 0,
            "transformer_matrices": transformer_matrices,
            "mtp_params": 0,
            "total": total,
        }

    def setup_optimizer(self, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        ddp, rank, local_rank, world_size = get_dist_info()
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        embedding_params = list(self.tok_emb.parameters())
        final_norm_params = list(self.final_norm.parameters())
        final_norm_param_ids = {id(param) for param in final_norm_params}
        adamw_params = list(final_norm_params)
        muon_params = []
        for name, param in self.named_parameters():
            if param is self.tok_emb.weight or id(param) in final_norm_param_ids:
                continue
            if name == "out_head.weight":
                continue
            if param.ndim == 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        ]
        if adamw_params:
            param_groups.append(
                dict(kind="adamw", params=adamw_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            )

        for shape in sorted({tuple(p.shape) for p in muon_params}):
            group_params = [p for p in muon_params if tuple(p.shape) == shape]
            param_groups.append(
                dict(kind="muon", params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay),
            )

        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        if kv_cache is not None:
            raise NotImplementedError("Qwen3.5 base training path does not support KV-cache generation")

        x = self.tok_emb(idx).to(COMPUTE_DTYPE)
        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x).float()

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        device = self.get_device()
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        for _ in range(max_tokens):
            if ids.size(1) > self.config.sequence_len:
                ids = ids[:, -self.config.sequence_len :]
            logits = self.forward(ids)[:, -1, :]
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
