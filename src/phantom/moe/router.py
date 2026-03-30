from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def expert_load_metrics(
    top_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, float]:
    """Histogram counts and max/min load ratio proxy (coefficient of variation)."""
    flat = top_indices.reshape(-1)
    counts = torch.bincount(flat, minlength=num_experts).to(torch.float32)
    mean = counts.mean().clamp_min(1e-6)
    cv = (counts.std() / mean).item()
    return counts, cv


class MoERouter(nn.Module):
    """
    Sigmoid affinity + top-k normalized routing (PHANTOM spec).

    Auxiliary-loss-free load handling: `expert_bias` is updated outside autograd
    (see `maybe_update_load_balance_bias`) so router logits stay the trainable signal.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward_scores(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.expert_bias)

    def forward_route(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, H]
        Returns:
          weights: [N, top_k] normalized
          indices: [N, top_k] int64
          full_scores: [N, E] (for metrics)
        """
        scores = self.forward_scores(x)
        top_w, top_idx = torch.topk(scores, self.top_k, dim=-1)
        denom = top_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_w = top_w / denom
        return top_w, top_idx, scores

    @torch.no_grad()
    def maybe_update_load_balance_bias(
        self,
        top_indices: torch.Tensor,
        scores: torch.Tensor,
        *,
        lr: float,
        epsilon: float = 1e-6,
    ) -> None:
        """
        DeepSeek-style bias update: nudge experts that receive less than uniform share.

        top_indices: [N, top_k]; scores unused (kept for API / future temperature).
        """
        del scores
        e = self.num_experts
        k = self.top_k
        flat = top_indices.reshape(-1)
        counts = torch.bincount(flat, minlength=e).to(self.expert_bias.dtype)
        total = counts.sum().clamp_min(1.0)
        target = (total * k) / e
        delta = (target - counts) / (target.clamp_min(epsilon))
        self.expert_bias.add_(lr * delta)
