from __future__ import annotations

import torch
import torch.nn as nn

from phantom.model.attention import MultiHeadLatentAttention
from phantom.model.config import ModelConfig
from phantom.model.norm import RMSNorm
from phantom.moe.experts import SwiGLUExpert
from phantom.moe.layer import MoELayer


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MultiHeadLatentAttention(config)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if layer_idx < config.num_dense_layers:
            self.ffn = SwiGLUExpert(config.hidden_size, config.ffn_intermediate)
            self.moe = None
        else:
            self.ffn = None
            self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_norm(x), attn_mask=attn_mask)
        if self.ffn is not None:
            x = x + self.ffn(self.post_attn_norm(x))
        else:
            assert self.moe is not None
            x = x + self.moe(self.post_attn_norm(x))
        return x
