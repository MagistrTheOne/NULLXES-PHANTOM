# PHANTOM First-Run Dataset

## Scope

This bootstrap mix is the first practical download set for `PHANTOM` before the full manifest-backed corpus pipeline exists.

It is not the final production corpus. It is a runnable subset for:

- tokenizer training
- cleaning and dedup experiments
- small pretrain smoke runs
- data pipeline validation

## Mix

- `HuggingFaceFW/fineweb`
- `allenai/dolma`
- `HuggingFaceFW/fineweb-2`
- `HuggingFaceFW/finepdfs`
- `open-web-math/open-web-math`
- `bigcode/the-stack-v2`

The exact first-run contract is locked in [configs/data/first_run_text_code.json](/d:/NULLXES/NULLXES%20PHANTOM/configs/data/first_run_text_code.json).

## Run

```bash
pip install -e .
python scripts/data/bootstrap_first_run.py
```

Or without editable install:

```bash
PYTHONPATH=src python scripts/data/bootstrap_first_run.py
```

For the safest first remote run, use the smoke config:

```bash
PYTHONPATH=src python scripts/data/bootstrap_first_run.py --config configs/data/first_run_smoke.json
```

You can also restrict sources and document counts without editing JSON:

```bash
PYTHONPATH=src python scripts/data/bootstrap_first_run.py --sources fineweb_english --max-documents 5000
```

## Output

Artifacts are written to:

- `artifacts/data/first_run_text_code_v1/*.jsonl`
- `artifacts/data/first_run_text_code_v1/download_manifest.json`

Each JSONL record carries:

- `source_id`
- `document_id`
- `license_id`
- `source_family`
- `text`
- `metadata`

## Notes

- This bootstrap downloader assumes Hugging Face dataset schemas remain compatible.
- Large sources should stay in streaming mode for the first pass.
- `The Stack v2` often needs tighter filtering later by language, license, and repository family.
- `Dolma` may require a different ingestion path than `datasets>=4` because some Hub layouts still rely on legacy dataset scripts.
- Final production ingestion still needs cross-source dedup, quality scoring, and contamination checks.
