from __future__ import annotations

import torch
from torch import nn

from .experts import SwiGLUExpert
from .router import TopKRouter


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model=d_model, hidden_size=expert_hidden_size, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        flat_x = x.reshape(batch * seq_len, d_model)
        _, probs, topk_indices, topk_weights = self.router(flat_x)

        output = torch.zeros_like(flat_x)
        token_count = flat_x.size(0)

        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(flat_x)
            selection_mask = topk_indices == expert_idx
            if not selection_mask.any():
                continue
            combine_weights = (topk_weights * selection_mask.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
            output = output + expert_out * combine_weights

        expert_usage = torch.zeros(self.num_experts, device=x.device, dtype=flat_x.dtype)
        for expert_idx in range(self.num_experts):
            expert_usage[expert_idx] = (topk_indices == expert_idx).any(dim=-1).to(flat_x.dtype).mean()
        expert_prob_mass = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_usage * expert_prob_mass)

        return output.view(batch, seq_len, d_model), aux_loss
