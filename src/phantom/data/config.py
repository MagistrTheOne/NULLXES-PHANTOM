from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    path: str
    weight: float = 1.0


@dataclass
class DataManifestConfig:
    entries: tuple[ManifestEntry, ...]

    @classmethod
    def from_json_file(cls, path: str | Path) -> DataManifestConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise TypeError("manifest must be a JSON array")
        entries: list[ManifestEntry] = []
        for item in raw:
            if isinstance(item, str):
                entries.append(ManifestEntry(path=item))
            elif isinstance(item, dict):
                entries.append(
                    ManifestEntry(
                        path=str(item["path"]),
                        weight=float(item.get("weight", 1.0)),
                    )
                )
            else:
                raise TypeError("manifest entries must be str or object")
        return cls(entries=tuple(entries))

    def to_dict(self) -> list[dict[str, Any]]:
        return [{"path": e.path, "weight": e.weight} for e in self.entries]
