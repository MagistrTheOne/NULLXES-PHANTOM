from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Frozen hyperparameters for PHANTOM decoder (spec-aligned fields)."""

    vocab_size: int = 160_000
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_dense_layers: int = 4
    num_attention_heads: int = 128
    head_dim: int = 128
    q_latent_dim: int = 2048
    kv_latent_dim: int = 768
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    ffn_intermediate: int = 18_944
    num_routed_experts: int = 192
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    max_position_embeddings: int = 32_768
    rms_norm_eps: float = 1e-6
    mtp_depth: int = 1
    tie_word_embeddings: bool = False
    qk_norm: bool = True
    load_balance_bias_lr: float = 1e-3
    expert_score_bias_epsilon: float = 1e-6
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_moe_layers(self) -> int:
        return self.num_hidden_layers - self.num_dense_layers

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if k != "extra"} | self.extra

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls) if f.name != "extra"}
        core = {k: data[k] for k in data if k in known and k != "extra"}
        extra = {k: data[k] for k in data if k not in known}
        return cls(**core, extra=extra)

    @classmethod
    def from_json_file(cls, path: str | Path) -> ModelConfig:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError("model config JSON must be an object")
        return cls.from_dict(data)


def resolve_model_config_path(train_config_path: str | Path, relative: str | Path) -> Path:
    base = Path(train_config_path).resolve().parent
    return (base / relative).resolve()
