from __future__ import annotations

import json
from pathlib import Path
import random

import torch

from phantom.tokenizer.runtime import PhantomTokenizer


def load_token_sequences(data_glob: str, tokenizer: PhantomTokenizer) -> list[int]:
    tokens: list[int] = []
    for path in sorted(Path().glob(data_glob)):
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = str(record.get("text", ""))
                if not text:
                    continue
                tokens.extend(tokenizer.encode(text))
                tokens.append(tokenizer.vocab.get("<|eos|>", 0))
    return tokens


class PackedTokenDataset:
    def __init__(self, tokens: list[int], seq_len: int) -> None:
        if len(tokens) <= seq_len:
            raise ValueError("Not enough tokens to build a training dataset.")
        self.tokens = tokens
        self.seq_len = seq_len

    def sample_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        starts = [random.randint(0, len(self.tokens) - self.seq_len - 2) for _ in range(batch_size)]
        inputs = [self.tokens[start : start + self.seq_len] for start in starts]
        labels = [self.tokens[start + 1 : start + self.seq_len + 1] for start in starts]
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(labels, dtype=torch.long, device=device)
        return x, y
