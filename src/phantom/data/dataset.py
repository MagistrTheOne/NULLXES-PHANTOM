from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from phantom.data.config import DataManifestConfig, ManifestEntry


class SyntheticTokenDataset(IterableDataset):
    """Deterministic random-token stream for integration smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        wid = 0 if worker is None else worker.id
        rng = random.Random(self.seed + wid)
        while True:
            tokens = torch.tensor(
                [rng.randrange(self.vocab_size) for _ in range(self.seq_len + 1)],
                dtype=torch.long,
            )
            yield {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }


class ManifestTextDataset(IterableDataset):
    """
    Streams fixed-length windows from manifest-listed text files.
    Expects `tokenizer.json` from PHANTOM BBPE train for encoding.
    """

    def __init__(
        self,
        manifest: DataManifestConfig,
        tokenizer_path: str | Path,
        seq_len: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        from phantom.tokenizer.runtime import PhantomBBPE

        self._tokenizer = PhantomBBPE.from_json_file(tokenizer_path)
        self.entries = manifest.entries
        self.seq_len = seq_len
        self.seed = seed
        self._paths = [e.path for e in manifest.entries]

    def _read_corpus(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        weights = [max(e.weight, 0.0) for e in self.entries]
        s = sum(weights)
        if s <= 0:
            raise ValueError("manifest weights must sum > 0")
        probs = [w / s for w in weights]
        while True:
            e = rng.choices(self.entries, weights=probs, k=1)[0]
            path = Path(e.path)
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            ids = self._tokenizer.encode(text)
            for t in ids:
                yield t

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf: list[int] = []
        need = self.seq_len + 1
        for t in self._read_corpus():
            buf.append(t)
            if len(buf) >= need:
                chunk = torch.tensor(buf[:need], dtype=torch.long)
                buf = buf[need:]
                yield {
                    "input_ids": chunk[:-1].clone(),
                    "labels": chunk[1:].clone(),
                }


def load_simple_manifest(path: str | Path) -> DataManifestConfig:
    if str(path).lower().endswith(".json"):
        return DataManifestConfig.from_json_file(path)
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    entries = [ManifestEntry(path=ln.strip()) for ln in lines if ln.strip()]
    return DataManifestConfig(entries=tuple(entries))


def manifest_json_template() -> str:
    return json.dumps(
        [{"path": "manifests/corpora/bootstrap/sample_corpus.txt", "weight": 1.0}],
        indent=2,
    )
