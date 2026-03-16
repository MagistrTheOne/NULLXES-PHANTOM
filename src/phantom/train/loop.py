from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from phantom.model import PhantomCausalLM, load_model_config
from phantom.tokenizer.runtime import PhantomTokenizer

from .config import TrainConfig
from .dataset import PackedTokenDataset, load_token_sequences


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(config: TrainConfig) -> dict[str, object]:
    _set_seed(config.seed)

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PhantomTokenizer.from_dir(config.tokenizer_dir)
    tokens = load_token_sequences(config.train_data_glob, tokenizer)
    dataset = PackedTokenDataset(tokens=tokens, seq_len=config.seq_len)

    model_config = load_model_config(config.model_config)
    if model_config.vocab_size != len(tokenizer.vocab):
        raise ValueError(
            f"Model vocab_size ({model_config.vocab_size}) does not match tokenizer vocab ({len(tokenizer.vocab)})."
        )
    model = PhantomCausalLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs: list[dict[str, float | int]] = []
    started = time.time()

    for step in range(1, config.steps + 1):
        model.train()
        x, y = dataset.sample_batch(batch_size=config.batch_size, device=device)
        _, lm_loss, aux_loss = model(x, labels=y)
        assert lm_loss is not None
        loss = lm_loss + aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == 1 or step == config.steps:
            entry = {
                "step": step,
                "loss": float(loss.item()),
                "lm_loss": float(lm_loss.item()),
                "aux_loss": float(aux_loss.item()),
            }
            logs.append(entry)
            print(json.dumps(entry, ensure_ascii=True))

        if step % config.checkpoint_every == 0 or step == config.steps:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": asdict(config),
            }
            torch.save(checkpoint, output_dir / f"checkpoint_step_{step}.pt")

    summary = {
        "run_name": config.run_name,
        "device": device,
        "steps": config.steps,
        "elapsed_sec": round(time.time() - started, 3),
        "train_tokens": len(tokens),
        "final_loss": logs[-1]["loss"] if logs else None,
        "checkpoints": sorted(str(path.name) for path in output_dir.glob("checkpoint_step_*.pt")),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (output_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=True) for item in logs) + ("\n" if logs else ""),
        encoding="utf-8",
    )
    return summary
