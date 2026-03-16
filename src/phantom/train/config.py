from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    run_name: str
    model_config: str
    tokenizer_dir: str
    train_data_glob: str
    output_dir: str
    seq_len: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    steps: int
    log_every: int
    checkpoint_every: int
    seed: int
    device: str


def load_train_config(path: str | Path) -> TrainConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(**data)
