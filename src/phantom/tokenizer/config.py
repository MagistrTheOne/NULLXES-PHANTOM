from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenizerTrainConfig:
    vocab_size: int = 160_000
    num_reserved_special_tokens: int = 512
    input_paths: tuple[str, ...] = ()
    pretokenizer: str = "gpt2_regex"

    @property
    def max_learned_tokens(self) -> int:
        return self.vocab_size - self.num_reserved_special_tokens

    @classmethod
    def from_json_file(cls, path: str | Path) -> TokenizerTrainConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        input_paths = tuple(data.get("input_paths") or ())
        known = {
            "vocab_size",
            "num_reserved_special_tokens",
            "pretokenizer",
        }
        core = {k: data[k] for k in data if k in known}
        return cls(input_paths=input_paths, **core)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_paths"] = list(self.input_paths)
        return d
