from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
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


def override_sources(
    config: DataBootstrapConfig,
    source_ids: list[str] | None = None,
    max_documents: int | None = None,
) -> DataBootstrapConfig:
    selected = config.sources
    if source_ids:
        allowed = set(source_ids)
        selected = [source for source in selected if source.source_id in allowed]
    if max_documents is not None:
        selected = [replace(source, max_documents=max_documents) for source in selected]
    return DataBootstrapConfig(
        name=config.name,
        output_root=config.output_root,
        default_streaming=config.default_streaming,
        default_format=config.default_format,
        sources=selected,
    )
