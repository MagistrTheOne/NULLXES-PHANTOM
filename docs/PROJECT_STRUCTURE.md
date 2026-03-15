# PHANTOM Project Structure

## Goal

The repository should separate six concerns cleanly:

1. source acquisition and governance
2. corpus normalization, deduplication, scoring, and packing
3. tokenizer training and validation
4. model definition and distributed training
5. evaluation and regression tracking
6. inference and deployment

## Recommended Tree

```text
NULLXES PHANTOM/
├── PHANTOM_ARCHITECTURE.md
├── README.md
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   └── DATASET_PLAN.md
├── configs/
│   ├── model/
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── serve/
├── manifests/
│   ├── sources/
│   ├── corpora/
│   ├── licenses/
│   └── runs/
├── src/
│   └── phantom/
│       ├── tokenizer/
│       ├── data/
│       ├── model/
│       ├── attention/
│       ├── moe/
│       ├── train/
│       ├── eval/
│       ├── serve/
│       └── utils/
├── scripts/
│   ├── data/
│   ├── tokenizer/
│   ├── train/
│   ├── eval/
│   └── serve/
├── tests/
│   ├── smoke/
│   └── integration/
└── infra/
    ├── storage/
    ├── orchestration/
    └── serving/
```

## Directory Ownership

### `docs/`

Human-readable specs. This is where architecture, data policy, eval policy, and rollout decisions live. If a major system decision exists only in code, the repo will drift.

### `configs/`

Machine-readable configuration only. Keep each axis isolated:

- `model/`: hidden size, layers, experts, attention, norms, MTP
- `data/`: source weights, filters, dedup thresholds, packing rules
- `train/`: optimizer, LR schedule, parallelism, precision, checkpointing
- `eval/`: benchmark suites, prompt templates, decoding settings
- `serve/`: runtime kernels, quantization, batching, context limits

### `manifests/`

This is the control plane for data and runs.

- `sources/`: one record per upstream source or crawl family
- `corpora/`: frozen compositions after cleaning and dedup
- `licenses/`: source-level and shard-level rights metadata
- `runs/`: checkpoint lineage, exact config hashes, and eval lineage

This directory matters as much as model code. Without it, you cannot reproduce what actually trained the model.

### `src/phantom/`

The implementation boundary.

- `tokenizer/`: BBPE trainer, validation, analysis
- `data/`: parsing, normalization, quality scoring, dedup, packing
- `model/`: transformer blocks, embeddings, output heads
- `attention/`: MLA kernels and related abstractions
- `moe/`: router, expert blocks, balancing logic, dispatch paths
- `train/`: distributed train loop, checkpoint IO, metrics, resumption
- `eval/`: benchmark runners, contamination checks, error slicing
- `serve/`: prefill/decode runtime, cache management, batching
- `utils/`: shared helpers only

### `scripts/`

Operational wrappers. These should call reusable modules from `src/phantom/`, not contain business logic that bypasses the main stack.

### `tests/`

- `smoke/`: import tests, tiny-forward tests, tokenizer sanity, config loading
- `integration/`: miniature end-to-end flows for data -> tokenize -> pack -> train step -> eval

### `infra/`

Infrastructure definitions and operational documents for:

- storage layout
- distributed job orchestration
- serving deployment topology

Keep environment-specific details here instead of leaking them into model code.

## First Build Order

Do not implement the stack in model-first order. The correct order is:

1. `manifests/` and data contracts
2. `src/phantom/data/` pipeline
3. `src/phantom/tokenizer/`
4. `configs/data/` and frozen corpus snapshots
5. `src/phantom/model/`, `attention/`, and `moe/`
6. `src/phantom/train/`
7. `src/phantom/eval/`
8. `src/phantom/serve/`

## Required Artifacts Before First Large Run

- source manifest with URL, license, capture date, language, and source family
- document-level dedup identifiers and cluster ids
- quality score tables and filter thresholds
- packed token shard manifest
- eval contamination blacklist
- checkpoint/run lineage manifest

If these artifacts are missing, the first large run will not be auditable.
