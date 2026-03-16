from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class DataSourceConfig:
    source_id: str
    family: str
    dataset: str
    split: str
    text_field: str
    streaming: bool
    max_documents: int
    weight: float
    license_id: str
    config: str | None = None


@dataclass(slots=True)
class DataBootstrapConfig:
    name: str
    output_root: str
    default_streaming: bool
    default_format: str
    sources: list[DataSourceConfig]


def load_data_config(path: str | Path) -> DataBootstrapConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = [DataSourceConfig(**item) for item in data["sources"]]
    return DataBootstrapConfig(
        name=data["name"],
        output_root=data["output_root"],
        default_streaming=data["default_streaming"],
        default_format=data["default_format"],
        sources=sources,
    )
