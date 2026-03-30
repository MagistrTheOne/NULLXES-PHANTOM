from __future__ import annotations

from pathlib import Path

from phantom.tokenizer.config import TokenizerTrainConfig
from phantom.tokenizer.runtime import PhantomBBPE
from phantom.tokenizer.trainer import train_bbpe


def test_bbpe_train_decode_roundtrip(tmp_path: Path) -> None:
    corpus = ["hello world\n", "PHANTOM MoE MLA\n"]
    tcfg = TokenizerTrainConfig(
        vocab_size=512,
        num_reserved_special_tokens=16,
        input_paths=(),
        pretokenizer="whitespace",
    )
    payload = train_bbpe(iter(corpus), tcfg)
    path = tmp_path / "tok.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    tok = PhantomBBPE.from_json_file(path)
    # Whitespace pretokenizer does not preserve spaces between pieces on decode;
    # single-span roundtrip validates encode/decode invariants.
    text = "PHANTOM"
    ids = tok.encode(text)
    back = tok.decode(ids)
    assert back == text
