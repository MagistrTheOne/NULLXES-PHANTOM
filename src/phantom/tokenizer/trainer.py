from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from phantom.tokenizer.config import TokenizerTrainConfig

try:
    import regex

    _HAS_REGEX = True
except ImportError:
    _HAS_REGEX = False

import re as std_re

_GPT2_SPLIT = std_re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def _pretokenize_line(line: str, mode: str) -> list[str]:
    line = line.rstrip("\n")
    if mode == "gpt2_regex":
        if _HAS_REGEX:
            return regex.findall(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                line,
            )
        return _GPT2_SPLIT.findall(line)
    if mode == "whitespace":
        return line.split()
    raise ValueError(f"unknown pretokenizer: {mode}")


def _pair_counts(words: dict[tuple[int, ...], int]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for word, freq in words.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _merge_words(
    words: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, ...], int]:
    first, second = pair
    new_words: dict[tuple[int, ...], int] = defaultdict(int)
    for word, freq in words.items():
        if len(word) < 2:
            new_words[word] += freq
            continue
        i = 0
        merged: list[int] = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                merged.append(new_id)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        new_words[tuple(merged)] += freq
    return dict(new_words)


def build_id_to_bytes(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    id_to: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        new_id = 256 + i
        id_to[new_id] = id_to[a] + id_to[b]
    return id_to


def train_bbpe(
    text_iter: Iterable[str],
    train_cfg: TokenizerTrainConfig,
) -> dict:
    """
    Byte-level BPE: ids 0..255 are raw bytes. Each merge introduces id 256.. until
    `max_learned_tokens - 1` (last `num_reserved_special_tokens` ids stay unused for specials).
    """
    max_learned = train_cfg.max_learned_tokens
    if max_learned < 256:
        raise ValueError("vocab_size - reserved must be >= 256 for byte foundation")

    words: dict[tuple[int, ...], int] = defaultdict(int)
    for chunk in text_iter:
        for piece in _pretokenize_line(chunk, train_cfg.pretokenizer):
            if not piece:
                continue
            b = tuple(bt for bt in piece.encode("utf-8"))
            if not b:
                continue
            words[b] += 1

    merges: list[tuple[int, int]] = []
    next_id = 256
    while next_id < max_learned:
        counts = _pair_counts(words)
        if not counts:
            break
        pair = counts.most_common(1)[0][0]
        merges.append(pair)
        words = _merge_words(words, pair, next_id)
        next_id += 1

    id_to_bytes = {str(i): list(b) for i, b in build_id_to_bytes(merges).items()}
    return {
        "version": 1,
        "vocab_size": train_cfg.vocab_size,
        "num_reserved_special_tokens": train_cfg.num_reserved_special_tokens,
        "merges": [[int(a), int(b)] for a, b in merges],
        "id_to_token_bytes": id_to_bytes,
        "pretokenizer": train_cfg.pretokenizer,
    }


def save_tokenizer_json(payload: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def iter_files(paths: tuple[str, ...]) -> Iterable[str]:
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            with p.open("r", encoding="utf-8", errors="replace") as f:
                yield from f
        elif p.is_dir():
            for fp in sorted(p.rglob("*.txt")):
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    yield from f
        else:
            raise FileNotFoundError(raw)
