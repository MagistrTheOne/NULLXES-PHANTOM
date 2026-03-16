# PHANTOM Phase-2 Dataset

## Scope

This is the next launch-ready corpus bootstrap after the smoke path. It keeps the source set intentionally narrow:

- `HuggingFaceFW/fineweb`
- `HuggingFaceFW/fineweb-2`

The goal is to prepare a clean English + multilingual base without introducing problematic or unverified sources.

## Config

The exact contract is locked in [configs/data/phase2_fineweb_dual.json](/d:/NULLXES/NULLXES%20PHANTOM/configs/data/phase2_fineweb_dual.json).

Default document counts:

- `fineweb_english`: `10000`
- `fineweb2_multilingual`: `4000`

## Pod Run

```bash
mkdir -p /workspace/hf_home /workspace/hf_cache /workspace/hf_tmp
export HF_HOME=/workspace/hf_home
export HF_DATASETS_CACHE=/workspace/hf_cache
export TMPDIR=/workspace/hf_tmp
export HF_HUB_DISABLE_TELEMETRY=1
export HF_TOKEN=YOUR_HF_TOKEN

PYTHONPATH=src python scripts/data/bootstrap_first_run.py --config configs/data/phase2_fineweb_dual.json
```

If you need a quicker probe first:

```bash
PYTHONPATH=src python scripts/data/bootstrap_first_run.py --config configs/data/phase2_fineweb_dual.json --max-documents 1000
```

## Output

- `artifacts/data/phase2_fineweb_dual_v1/fineweb_english.jsonl`
- `artifacts/data/phase2_fineweb_dual_v1/fineweb2_multilingual.jsonl`
- `artifacts/data/phase2_fineweb_dual_v1/download_manifest.json`
