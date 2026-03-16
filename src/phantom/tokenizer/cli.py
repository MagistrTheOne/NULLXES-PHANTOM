from __future__ import annotations

import argparse
import json

from .config import load_config
from .trainer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PHANTOM byte-level BPE tokenizer.")
    parser.add_argument(
        "--config",
        default="configs/tokenizer/phantom_bbpe_160k.json",
        help="Path to tokenizer config JSON.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = train_tokenizer(config)
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
