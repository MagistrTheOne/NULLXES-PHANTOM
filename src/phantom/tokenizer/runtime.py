from __future__ import annotations

import json
from pathlib import Path


class PhantomTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.byte_tokens = {idx: token_id for idx, token_id in self._extract_byte_tokens(vocab).items()}
        self.non_byte_tokens = sorted(
            ((token, token_id) for token, token_id in vocab.items() if not token.startswith("<|byte:")),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    @staticmethod
    def _extract_byte_tokens(vocab: dict[str, int]) -> dict[int, int]:
        result: dict[int, int] = {}
        for token, token_id in vocab.items():
            if token.startswith("<|byte:") and token.endswith("|>"):
                index = int(token[len("<|byte:") : -2])
                result[index] = token_id
        return result

    @classmethod
    def from_dir(cls, path: str | Path) -> "PhantomTokenizer":
        vocab_path = Path(path) / "vocab.json"
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        return cls(vocab=vocab)

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        index = 0
        while index < len(text):
            matched = False
            for token, token_id in self.non_byte_tokens:
                if token.startswith("<|") and token.endswith("|>"):
                    continue
                if text.startswith(token, index):
                    encoded.append(token_id)
                    index += len(token)
                    matched = True
                    break
            if matched:
                continue
            for byte in text[index].encode("utf-8"):
                encoded.append(self.byte_tokens[byte])
            index += 1
        return encoded
