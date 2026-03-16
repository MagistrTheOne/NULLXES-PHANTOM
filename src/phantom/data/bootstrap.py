from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import DataBootstrapConfig, DataSourceConfig


def _load_dataset(source: DataSourceConfig):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to download corpus sources.") from exc

    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": source.streaming,
    }
    if source.config:
        kwargs["name"] = source.config
    return load_dataset(**kwargs)


def _pick_text(record: dict[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _record_to_doc(source: DataSourceConfig, index: int, record: dict[str, Any]) -> dict[str, Any] | None:
    text = _pick_text(record, source.text_field)
    if not text or not text.strip():
        return None
    return {
        "source_id": source.source_id,
        "document_id": f"{source.source_id}-{index}",
        "license_id": source.license_id,
        "source_family": source.family,
        "text": text,
        "metadata": {key: value for key, value in record.items() if key != source.text_field},
    }


def bootstrap_sources(config: DataBootstrapConfig) -> dict[str, Any]:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "name": config.name,
        "output_root": config.output_root,
        "sources": [],
    }

    for source in config.sources:
        dataset = _load_dataset(source)
        output_path = output_root / f"{source.source_id}.jsonl"
        written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(dataset):
                if written >= source.max_documents:
                    break
                doc = _record_to_doc(source, index, record)
                if doc is None:
                    continue
                handle.write(json.dumps(doc, ensure_ascii=True) + "\n")
                written += 1

        manifest["sources"].append(
            {
                "source": asdict(source),
                "output_path": str(output_path),
                "documents_written": written,
            }
        )

    manifest_path = output_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest
