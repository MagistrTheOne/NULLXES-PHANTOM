from __future__ import annotations

import torch
import torch.nn as nn

from phantom.model.block import DecoderBlock
from phantom.model.config import ModelConfig
from phantom.model.norm import RMSNorm


def causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask shape [1, 1, T, T] with float min above diagonal."""
    t = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
    t = torch.triu(t, diagonal=1)
    return t.view(1, 1, seq_len, seq_len).to(dtype)


class PhantomCausalLM(nn.Module):
    """Decoder-only MLA + MoE (+ MTP depth 1)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderBlock(config, i) for i in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp_depth = config.mtp_depth
        if config.mtp_depth > 0:
            self.mtp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mtp_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed(input_ids)
        t = x.shape[1]
        if attn_mask is None:
            attn_mask = causal_mask(t, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        mtp_logits: torch.Tensor | None = None
        if self.mtp_depth > 0:
            h = self.mtp_proj(self.mtp_norm(x))
            mtp_logits = self.mtp_head(h)
        return logits, mtp_logits
