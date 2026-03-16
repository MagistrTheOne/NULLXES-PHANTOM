# PHANTOM Bootstrap Training

## Scope

This is the first real training stack for PHANTOM bootstrap runs. It trains a small causal LM with sparse MoE feed-forward layers on your own downloaded corpus and your own tokenizer artifacts.

It is not the flagship architecture. It exists to validate:

- tokenizer artifact loading
- corpus tokenization
- packed next-token batches
- forward/backward/optimizer flow
- router, experts, dispatch, and load-balancing loss
- checkpoint writing

## Prerequisites

- `artifacts/data/first_run_smoke_v1/*.jsonl` exists
- `artifacts/tokenizer/phantom_bbpe_160k/` exists
- PyTorch is installed

## Run

```bash
PYTHONPATH=src python scripts/train/run_bootstrap.py
```

## Outputs

- `artifacts/train/bootstrap_300_steps/checkpoint_step_100.pt`
- `artifacts/train/bootstrap_300_steps/checkpoint_step_200.pt`
- `artifacts/train/bootstrap_300_steps/checkpoint_step_300.pt`
- `artifacts/train/bootstrap_300_steps/train_log.jsonl`
- `artifacts/train/bootstrap_300_steps/train_summary.json`

## Notes

- This run uses a small causal LM with MoE FFN blocks in `configs/model/bootstrap_small.json`.
- The objective is standard next-token prediction on your own data.
- The weights are fully your own bootstrap weights.
