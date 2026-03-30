from __future__ import annotations

import argparse
import json
from pathlib import Path

from phantom.tokenizer.config import TokenizerTrainConfig
from phantom.tokenizer.trainer import iter_files, save_tokenizer_json, train_bbpe


def main() -> None:
    p = argparse.ArgumentParser(description="Train PHANTOM byte-level BPE tokenizer.")
    p.add_argument("--config", type=str, help="Tokenizer JSON config path.")
    p.add_argument("--input", nargs="*", help="Input .txt files or dirs (overrides config).")
    p.add_argument("--output", type=str, required=True, help="Output tokenizer.json path.")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--reserved", type=int, default=None)
    args = p.parse_args()

    if args.config:
        tcfg = TokenizerTrainConfig.from_json_file(args.config)
    else:
        tcfg = TokenizerTrainConfig()

    if args.vocab_size is not None:
        tcfg.vocab_size = args.vocab_size
    if args.reserved is not None:
        tcfg.num_reserved_special_tokens = args.reserved

    paths = tuple(args.input) if args.input else tcfg.input_paths
    if not paths:
        raise SystemExit("No inputs: pass --input or set input_paths in --config")

    tcfg = TokenizerTrainConfig(
        vocab_size=tcfg.vocab_size,
        num_reserved_special_tokens=tcfg.num_reserved_special_tokens,
        input_paths=paths,
        pretokenizer=tcfg.pretokenizer,
    )

    payload = train_bbpe(iter_files(paths), tcfg)
    save_tokenizer_json(payload, args.output)
    meta = Path(args.output).with_suffix(".meta.json")
    meta.write_text(
        json.dumps({"output": args.output, "num_merges": len(payload["merges"])}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
