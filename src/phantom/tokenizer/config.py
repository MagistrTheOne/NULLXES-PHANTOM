from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class TokenizerConfig:
    name: str
    vocab_size: int
    reserved_special_ids: int
    learned_token_limit: int
    merges_limit: int
    min_pair_count: int
    pretokenize_regex: str
    normalize_newlines: bool
    lowercase: bool
    sample_bytes: int
    special_tokens: list[str]
    corpus_globs: list[str]
    output_dir: str

    @property
    def byte_vocab_size(self) -> int:
        return 256


def load_config(path: str | Path) -> TokenizerConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TokenizerConfig(**data)
