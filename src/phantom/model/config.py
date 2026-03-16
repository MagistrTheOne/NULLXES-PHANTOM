from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: float
    dropout: float


def load_model_config(path: str | Path) -> ModelConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelConfig(**data)
