from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path

from .config import TokenizerConfig


def _iter_corpus_files(config: TokenizerConfig) -> list[Path]:
    files: list[Path] = []
    for pattern in config.corpus_globs:
        files.extend(Path().glob(pattern))
    return sorted(path for path in files if path.is_file())


def _load_sample_text(config: TokenizerConfig, files: list[Path]) -> str:
    chunks: list[str] = []
    consumed = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if config.normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if config.lowercase:
            text = text.lower()
        if consumed + len(text.encode("utf-8")) > config.sample_bytes:
            remaining = max(config.sample_bytes - consumed, 0)
            chunks.append(text.encode("utf-8")[:remaining].decode("utf-8", errors="ignore"))
            break
        chunks.append(text)
        consumed += len(text.encode("utf-8"))
        if consumed >= config.sample_bytes:
            break
    return "\n".join(chunks)


def _pretokenize(text: str, pattern: str) -> list[bytes]:
    tokens: list[bytes] = []
    regex = re.compile(pattern)
    for match in regex.finditer(text):
        piece = match.group(0)
        if piece:
            tokens.append(piece.encode("utf-8"))
    return tokens


def _train_bpe_merges(token_stream: list[bytes], config: TokenizerConfig) -> tuple[list[tuple[int, int]], dict[bytes, int]]:
    words = [list(token) for token in token_stream if token]
    vocab_counter: Counter[bytes] = Counter(token_stream)
    merges: list[tuple[int, int]] = []
    token_to_id: dict[bytes, int] = {bytes([idx]): idx for idx in range(256)}
    id_to_token: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    next_id = 256
    target_vocab = min(config.vocab_size - len(config.special_tokens), 256 + config.learned_token_limit)

    while next_id < target_vocab and len(merges) < config.merges_limit:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word in words:
            for left, right in zip(word, word[1:]):
                pair_counts[(left, right)] += 1
        if not pair_counts:
            break
        (best_left, best_right), count = pair_counts.most_common(1)[0]
        if count < config.min_pair_count:
            break

        merged_bytes = id_to_token[best_left] + id_to_token[best_right]
        token_to_id[merged_bytes] = next_id
        id_to_token[next_id] = merged_bytes
        merges.append((best_left, best_right))

        replacement_id = next_id
        next_id += 1
        updated_words: list[list[int]] = []
        for word in words:
            merged_word: list[int] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and word[index] == best_left and word[index + 1] == best_right:
                    merged_word.append(replacement_id)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1
            updated_words.append(merged_word)
        words = updated_words

    learned_tokens = {token: idx for token, idx in token_to_id.items() if idx >= 256}
    if vocab_counter:
        _ = vocab_counter.most_common(1)[0]
    return merges, learned_tokens


def _materialize_vocab(config: TokenizerConfig, learned_tokens: dict[bytes, int]) -> dict[str, int]:
    vocab = {f"<|byte:{idx}|>": idx for idx in range(256)}
    base_index = 256
    for token_bytes, idx in sorted(learned_tokens.items(), key=lambda item: item[1]):
        vocab[token_bytes.decode("utf-8", errors="backslashreplace")] = idx
    next_id = max(vocab.values(), default=255) + 1
    for token in config.special_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    return vocab


def train_tokenizer(config: TokenizerConfig) -> dict[str, object]:
    files = _iter_corpus_files(config)
    if not files:
        raise FileNotFoundError("No corpus files matched the configured globs.")

    text = _load_sample_text(config, files)
    token_stream = _pretokenize(text, config.pretokenize_regex)
    merges, learned_tokens = _train_bpe_merges(token_stream, config)
    vocab = _materialize_vocab(config, learned_tokens)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": asdict(config),
        "matched_files": [str(path) for path in files],
        "sample_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pretoken_count": len(token_stream),
        "merge_count": len(merges),
        "vocab_size": len(vocab),
    }

    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(metadata["config"], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / "merges.txt").write_text(
        "\n".join(f"{left} {right}" for left, right in merges),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return metadata
