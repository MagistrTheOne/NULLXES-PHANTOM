from __future__ import annotations

import torch
from torch import nn


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_size)
        self.w3 = nn.Linear(d_model, hidden_size)
        self.w2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(gated))
