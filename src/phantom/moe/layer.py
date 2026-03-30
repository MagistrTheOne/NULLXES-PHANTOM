from __future__ import annotations

import torch
import torch.nn as nn

from phantom.moe.experts import SwiGLUExpert
from phantom.moe.router import MoERouter
from phantom.model.config import ModelConfig


class MoELayer(nn.Module):
    """
    Routed MoE FFN + optional shared expert(s).

    Naive token dispatch for correctness; replace with EP + grouped GEMM at scale
    (see `phantom.scale_notes`).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        e = config.num_routed_experts
        k = config.num_experts_per_tok
        h = config.hidden_size
        inter = config.ffn_intermediate
        self.router = MoERouter(h, e, k)
        self.experts = nn.ModuleList(SwiGLUExpert(h, inter) for _ in range(e))
        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(h, inter) for _ in range(config.num_shared_experts)
        )
        self._last_top_indices: torch.Tensor | None = None
        self._last_full_scores: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        flat = x.reshape(b * t, h)
        w, idx, full_scores = self.router.forward_route(flat)
        out = torch.zeros_like(flat)
        k = self.config.num_experts_per_tok
        for j in range(k):
            e_ids = idx[:, j]
            wt = w[:, j]
            for e in range(self.config.num_routed_experts):
                m = e_ids == e
                if not m.any():
                    continue
                y = self.experts[e](flat[m])
                out[m] = out[m] + (wt[m, None] * y)
        for se in self.shared_experts:
            out = out + se(flat)
        out = out.view(b, t, h)
        self._last_top_indices = idx.detach()
        self._last_full_scores = full_scores.detach()
        return out
