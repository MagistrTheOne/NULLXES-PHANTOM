from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from phantom.tokenizer.trainer import _pretokenize_line, build_id_to_bytes


class PhantomBBPE:
    """Runtime BBPE with mandatory byte fallback (emit raw byte ids)."""

    def __init__(self, payload: dict) -> None:
        self.vocab_size = int(payload["vocab_size"])
        self.num_reserved = int(payload.get("num_reserved_special_tokens", 0))
        self.pretokenizer = str(payload.get("pretokenizer", "gpt2_regex"))
        merges_raw = payload["merges"]
        self._merges: list[tuple[int, int]] = [
            (int(a), int(b)) for a, b in merges_raw
        ]
        self._id_to_bytes = build_id_to_bytes(self._merges)
        self._merge_rank: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self._merges)
        }

    @classmethod
    def from_json_file(cls, path: str | Path) -> PhantomBBPE:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload)

    def encode_word(self, word_bytes: tuple[int, ...]) -> list[int]:
        if len(word_bytes) == 0:
            return []
        word = list(word_bytes)
        if len(word) == 1:
            return list(word)

        def get_pairs(ws: list[int]) -> set[tuple[int, int]]:
            return {(ws[i], ws[i + 1]) for i in range(len(ws) - 1)}

        while True:
            pairs = get_pairs(word)
            if not pairs:
                break
            bigram = min(
                pairs,
                key=lambda p: self._merge_rank.get(p, float("inf")),
            )
            if bigram not in self._merge_rank:
                break
            new_id = 256 + self._merge_rank[bigram]
            i = 0
            nxt: list[int] = []
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == bigram[0]
                    and word[i + 1] == bigram[1]
                ):
                    nxt.append(new_id)
                    i += 2
                else:
                    nxt.append(word[i])
                    i += 1
            word = nxt
            if len(word) == 1:
                break
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _pretokenize_line(text, self.pretokenizer):
            if not piece:
                continue
            b = tuple(bt for bt in piece.encode("utf-8"))
            if not b:
                continue
            ids.extend(self.encode_word(b))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out = bytearray()
        for i in ids:
            if i < 0 or i >= self.vocab_size:
                continue
            if i < 256:
                out.append(i)
            elif i in self._id_to_bytes:
                out.extend(self._id_to_bytes[i])
            else:
                out.extend(bytes([i & 0xFF]))
        return out.decode("utf-8", errors="replace")
