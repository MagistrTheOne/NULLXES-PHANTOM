from __future__ import annotations

import argparse
from pathlib import Path

from phantom.train.config import TrainConfig
from phantom.train.loop import run_smoke_training


def main() -> None:
    p = argparse.ArgumentParser(description="PHANTOM pretrain loop (smoke / bootstrap).")
    p.add_argument("--config", type=str, required=True, help="Train JSON (see configs/train/).")
    args = p.parse_args()
    cfg = TrainConfig.from_json_file(args.config)
    run_smoke_training(cfg)


if __name__ == "__main__":
    main()
