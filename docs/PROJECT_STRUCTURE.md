# Project structure

| Path | Role |
|------|------|
| [PHANTOM_ARCHITECTURE.md](../PHANTOM_ARCHITECTURE.md) | Locked flagship spec |
| `configs/` | Model, train, tokenizer JSON |
| `manifests/` | Corpus paths, licensing registries |
| `src/phantom/` | Tokenizer, data, model (MLA/MoE), train |
| `scripts/` | Operational entrypoints |
| `tests/` | Smoke and integration |
| `infra/` | Orchestration and runtime ops |

Governance: no external base weights in-repo; proprietary license.
