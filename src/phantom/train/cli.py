from __future__ import annotations

import argparse
import json

from .config import load_train_config
from .loop import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PHANTOM bootstrap LM training.")
    parser.add_argument("--config", default="configs/train/bootstrap_300_steps.json", help="Path to train config.")
    args = parser.parse_args()

    config = load_train_config(args.config)
    summary = run_training(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
