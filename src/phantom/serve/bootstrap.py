from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_runtime_manifest(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "service": "phantom-inference",
        "tokenizer_dir": "artifacts/tokenizer/phantom_bbpe_160k",
        "model_dir": "artifacts/model",
        "max_context_tokens": 262144,
        "max_generation_tokens": 16384,
        "healthcheck_path": "/healthz"
    }
    manifest_path = path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap PHANTOM serving runtime artifacts.")
    parser.add_argument("--output-dir", default="artifacts/runtime", help="Directory for runtime manifests.")
    args = parser.parse_args()
    path = write_runtime_manifest(args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
