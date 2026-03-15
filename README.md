# NULLXES PHANTOM

`PHANTOM` is a from-scratch foundation model program for `NULLXES LC`.

Current repository state is intentionally architecture-first:

- [PHANTOM_ARCHITECTURE.md](./PHANTOM_ARCHITECTURE.md): locked model direction for the `500B MoE` flagship
- [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md): repo layout and ownership boundaries
- [docs/DATASET_PLAN.md](./docs/DATASET_PLAN.md): full-pretrain data plan for the `34T` token target

## Working Principles

- no borrowed base model weights
- no training plan built around fine-tuning an external checkpoint
- tokenizer may borrow the Qwen family approach, but vocabulary and merges are trained in-house
- data provenance, deduplication, and licensing are first-class artifacts, not afterthoughts

## Repository Layout

Top-level directories:

- `docs/`: architecture, data, training, eval, and governance specs
- `configs/`: model, data, train, eval, and serve configuration trees
- `manifests/`: source registries, corpus manifests, licensing manifests, and run manifests
- `src/phantom/`: implementation packages for tokenizer, data, model, train, eval, and serving
- `scripts/`: one-shot operational entrypoints for data prep, training, eval, and deployment
- `tests/`: smoke and integration coverage
- `infra/`: storage, orchestration, and serving infrastructure definitions

No training code is committed yet by design. The next stage is to turn the architecture and data specs into executable configs, manifests, and implementation modules.
