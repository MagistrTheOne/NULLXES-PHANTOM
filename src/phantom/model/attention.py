from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.model.config import ModelConfig
from phantom.model.norm import RMSNorm
from phantom.model.rotary import apply_rotary_pos_emb, build_rope_cache


class MultiHeadLatentAttention(nn.Module):
    """
    Latent-compressed Q/K/V paths with RoPE on Q/K and optional QK-Norm (PHANTOM spec).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.nh = config.num_attention_heads
        self.hd = config.head_dim
        qd = self.nh * self.hd

        self.q_a = nn.Linear(d, config.q_latent_dim, bias=False)
        self.q_b = nn.Linear(config.q_latent_dim, qd, bias=False)
        self.kv_a = nn.Linear(d, config.kv_latent_dim, bias=False)
        self.k_b = nn.Linear(config.kv_latent_dim, qd, bias=False)
        self.v_b = nn.Linear(config.kv_latent_dim, qd, bias=False)
        self.o_proj = nn.Linear(qd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, eps=config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.hd, eps=config.rms_norm_eps) if config.qk_norm else None
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)
        self._rope_len = 0

    def _ensure_rope(self, t: int, device: torch.device, dtype: torch.dtype) -> None:
        if t <= self._rope_len and self._rope_cos.device == device:
            return
        max_len = max(t, self.config.max_position_embeddings)
        cos, sin = build_rope_cache(
            max_len, self.hd, self.config.rope_theta, device, dtype
        )
        self._rope_cos = cos
        self._rope_sin = sin
        self._rope_len = max_len

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        self._ensure_rope(t, x.device, x.dtype)

        q = self.q_b(self.q_a(x)).view(b, t, self.nh, self.hd).transpose(1, 2)
        kv = self.kv_a(x)
        k = self.k_b(kv).view(b, t, self.nh, self.hd).transpose(1, 2)
        v = self.v_b(kv).view(b, t, self.nh, self.hd).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        cos = self._rope_cos[:t].to(dtype=x.dtype)
        sin = self._rope_sin[:t].to(dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        sd = 1.0 / math.sqrt(self.hd)
        attn = torch.matmul(q, k.transpose(-2, -1)) * sd
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.nh * self.hd)
        return self.o_proj(out)
