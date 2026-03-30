from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    model_config: Path
    seq_len: int = 2048
    micro_batch_size: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 10
    log_every: int = 10
    checkpoint_dir: Path = Path("checkpoints/run")
    seed: int = 42
    manifest_path: Path | None = None
    tokenizer_path: Path | None = None
    use_synthetic: bool = False

    @classmethod
    def from_json_file(cls, path: str | Path) -> TrainConfig:
        cfg_path = Path(path).resolve()
        base = cfg_path.parent
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        def rp(key: str) -> Path | None:
            if key not in data or data[key] is None:
                return None
            p = Path(data[key])
            if p.is_absolute():
                return p
            if key in {"checkpoint_dir"}:
                return (Path.cwd() / p).resolve()
            candidates = [(Path.cwd() / p).resolve(), (base / p).resolve()]
            for c in candidates:
                if c.exists():
                    return c
            return candidates[1]

        mc = rp("model_config")
        if mc is None or not mc.exists():
            raise ValueError("model_config is required and must resolve to an existing file")
        return cls(
            model_config=mc,
            seq_len=int(data.get("seq_len", 2048)),
            micro_batch_size=int(data.get("micro_batch_size", 1)),
            lr=float(data.get("lr", 3e-4)),
            weight_decay=float(data.get("weight_decay", 0.1)),
            max_steps=int(data.get("max_steps", 1000)),
            warmup_steps=int(data.get("warmup_steps", 10)),
            log_every=int(data.get("log_every", 10)),
            checkpoint_dir=rp("checkpoint_dir") or Path("checkpoints/run"),
            seed=int(data.get("seed", 42)),
            manifest_path=rp("manifest_path"),
            tokenizer_path=rp("tokenizer_path"),
            use_synthetic=bool(data.get("use_synthetic", False)),
        )


def save_checkpoint(
    path: Path,
    *,
    model_state: dict,
    optimizer_state: dict,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optimizer_state,
        },
        path,
    )
