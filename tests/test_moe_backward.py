from __future__ import annotations

from pathlib import Path

import torch

from phantom.model.causal_lm import PhantomCausalLM
from phantom.model.config import ModelConfig


def test_moe_backward_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ModelConfig.from_json_file(root / "configs" / "model" / "phantom_smoke.json")
    m = PhantomCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, _ = m(x)
    loss = logits.float().mean()
    loss.backward()
    assert m.layers[-1].moe is not None
    assert m.layers[-1].moe.router.linear.weight.grad is not None
