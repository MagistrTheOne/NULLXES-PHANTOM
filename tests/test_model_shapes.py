from __future__ import annotations

import json
from pathlib import Path

import torch

from phantom.model.causal_lm import PhantomCausalLM
from phantom.model.config import ModelConfig


def test_phantom_smoke_config_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "configs" / "model" / "phantom_smoke.json"
    cfg = ModelConfig.from_json_file(p)
    out = tmp_path / "roundtrip.json"
    out.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    cfg2 = ModelConfig.from_json_file(out)
    assert cfg.hidden_size == cfg2.hidden_size
    assert cfg.num_routed_experts == cfg2.num_routed_experts


def test_forward_shapes_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ModelConfig.from_json_file(root / "configs" / "model" / "phantom_smoke.json")
    m = PhantomCausalLM(cfg)
    b, t = 2, 32
    x = torch.randint(0, cfg.vocab_size, (b, t))
    logits, mtp = m(x)
    assert logits.shape == (b, t, cfg.vocab_size)
    assert mtp is not None and mtp.shape == logits.shape
