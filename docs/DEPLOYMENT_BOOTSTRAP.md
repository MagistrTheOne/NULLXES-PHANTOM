# PHANTOM Deployment Bootstrap

## Goal

Bring the repository from documentation-only state to a runnable tokenizer and serving bootstrap.

## What Exists Now

- `pyproject.toml` packages `src/phantom/`
- `scripts/tokenizer/train.py` trains the bootstrap tokenizer
- `scripts/serve/bootstrap_runtime.py` writes a serving runtime manifest
- `configs/tokenizer/phantom_bbpe_160k.json` locks the tokenizer training contract
- `configs/serve/runtime.json` locks the serving contract
- `infra/serving/docker-compose.yml` provides a one-command bootstrap path

## Local Bootstrap

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/tokenizer/train.py
python scripts/serve/bootstrap_runtime.py
```

Expected artifacts:

- `artifacts/tokenizer/phantom_bbpe_160k/vocab.json`
- `artifacts/tokenizer/phantom_bbpe_160k/merges.txt`
- `artifacts/tokenizer/phantom_bbpe_160k/metadata.json`
- `artifacts/runtime/runtime_manifest.json`

## Container Bootstrap

```bash
docker compose -f infra/serving/docker-compose.yml up --build
```

## Scope Limit

This is a bootstrap tokenizer trainer and runtime manifest generator, not the final production tokenizer stack.
The next step is to replace the in-repo sample corpus with frozen manifest-backed shard inputs and add evaluation gates.
