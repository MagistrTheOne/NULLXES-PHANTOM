from __future__ import annotations

from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phantom.data.config import DataManifestConfig
from phantom.data.dataset import ManifestTextDataset, SyntheticTokenDataset
from phantom.model.causal_lm import PhantomCausalLM
from phantom.model.config import ModelConfig
from phantom.moe.router import expert_load_metrics
from phantom.train.config import TrainConfig, save_checkpoint


def _maybe_distributed(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if dist.is_available() and not dist.is_initialized():
        return


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _lr_at_step(step: int, warmup: int, base_lr: float, max_steps: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    # cosine decay to 0.1 * base
    t = (step - warmup) / max(1, max_steps - warmup)
    t = min(1.0, max(0.0, t))
    import math

    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * t))


def update_moe_router_biases(model: PhantomCausalLM, cfg: ModelConfig) -> None:
    for block in model.layers:
        if block.moe is None:
            continue
        m = block.moe
        if m._last_top_indices is None or m._last_full_scores is None:
            continue
        m.router.maybe_update_load_balance_bias(
            m._last_top_indices,
            m._last_full_scores,
            lr=cfg.load_balance_bias_lr,
            epsilon=cfg.expert_score_bias_epsilon,
        )


def pretrain_one_step(
    model: PhantomCausalLM,
    batch: dict[str, torch.Tensor],
    *,
    cfg: ModelConfig,
    mtp_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    logits, mtp_logits = model(input_ids)
    loss_main = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        labels.reshape(-1),
    )
    loss = loss_main
    metrics: dict[str, float] = {"loss_main": float(loss_main.detach())}
    if mtp_logits is not None:
        loss_mtp = F.cross_entropy(
            mtp_logits.reshape(-1, cfg.vocab_size),
            labels.reshape(-1),
        )
        loss = loss + mtp_weight * loss_mtp
        metrics["loss_mtp"] = float(loss_mtp.detach())
    metrics["loss_total"] = float(loss.detach())

    # Expert load telemetry (first MoE layer only for brevity)
    for block in model.layers:
        if block.moe is not None and block.moe._last_top_indices is not None:
            _, cv = expert_load_metrics(
                block.moe._last_top_indices,
                cfg.num_routed_experts,
            )
            metrics["expert_load_cv"] = float(cv)
            break
    return loss, metrics


def run_smoke_training(train_cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _maybe_distributed(device)
    _set_seed(train_cfg.seed)

    mcfg = ModelConfig.from_json_file(train_cfg.model_config)
    model = PhantomCausalLM(mcfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    if train_cfg.use_synthetic or train_cfg.manifest_path is None:
        ds: torch.utils.data.IterableDataset = SyntheticTokenDataset(
            mcfg.vocab_size,
            train_cfg.seq_len,
            seed=train_cfg.seed,
        )
    else:
        assert train_cfg.tokenizer_path is not None
        manifest = DataManifestConfig.from_json_file(train_cfg.manifest_path)
        ds = ManifestTextDataset(
            manifest,
            train_cfg.tokenizer_path,
            seq_len=train_cfg.seq_len,
            seed=train_cfg.seed,
        )

    loader = DataLoader(
        ds,
        batch_size=train_cfg.micro_batch_size,
        num_workers=0,
    )
    it: Iterator[dict[str, torch.Tensor]] = iter(loader)

    for step in range(train_cfg.max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        lr = _lr_at_step(step, train_cfg.warmup_steps, train_cfg.lr, train_cfg.max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss, metrics = pretrain_one_step(model, batch, cfg=mcfg)
        loss.backward()
        opt.step()
        update_moe_router_biases(model, mcfg)

        if step % train_cfg.log_every == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"step={step} lr={lr:.2e} {msg}")

        if step == train_cfg.max_steps - 1:
            ckpt = train_cfg.checkpoint_dir / "last.pt"
            save_checkpoint(
                ckpt,
                model_state=model.state_dict(),
                optimizer_state=opt.state_dict(),
                step=step + 1,
            )


def ddp_placeholder_world() -> None:
    """Call torchrun from CLI for multi-GPU; wire process group here when enabling DDP."""
    return
