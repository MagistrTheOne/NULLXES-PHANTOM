from __future__ import annotations

import argparse
import json

from .bootstrap import bootstrap_sources
from .config import load_data_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and materialize the PHANTOM first-run corpus mix.")
    parser.add_argument(
        "--config",
        default="configs/data/first_run_text_code.json",
        help="Path to the data bootstrap config.",
    )
    args = parser.parse_args()

    config = load_data_config(args.config)
    manifest = bootstrap_sources(config)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
