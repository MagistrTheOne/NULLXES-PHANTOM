---
language:
  - en
license: other
library_name: pytorch
pipeline_tag: text-generation
tags:
  - nullxes
  - phantom
  - moe
  - causal-lm
  - pretraining
  - bootstrap
model_type: custom
datasets:
  - HuggingFaceFW/fineweb
  - HuggingFaceFW/fineweb-2
base_model: "none"
---

# NULLXES-PHANTOM-500B-MoE

`NULLXES-PHANTOM-500B-MoE` is the Hugging Face repository for the `PHANTOM` foundation model program by `NULLXES LC`.

This repository currently holds early bootstrap artifacts for the PHANTOM training stack:

- tokenizer bootstrap artifacts
- dataset manifests
- bootstrap sparse MoE checkpoints
- bootstrap train and model configs

## Current Status

This is not the final `PHANTOM-500B` release.

Current uploaded artifacts represent:

- a working `data -> tokenizer -> train -> checkpoint` pipeline
- a bootstrap causal language model with sparse `MoE` feed-forward layers
- router, experts, top-k gating, and auxiliary load-balancing loss
- checkpoints trained on owned pipeline artifacts

The flagship target architecture remains the larger `PHANTOM-500B MoE` program described in the project repository.

## Training Summary

Current bootstrap run characteristics:

- tokenizer: `PHANTOM-BBPE-160K` bootstrap variant
- corpus bootstrap: `FineWeb` smoke slice, with phase-2 expansion prepared for `FineWeb + FineWeb-2`
- model class: small causal LM with sparse `MoE` layers
- objective: next-token prediction
- training stack: custom PHANTOM bootstrap implementation

## Included Artifacts

- `checkpoint_bootstrap_300/`
- `tokenizer/`
- `data/download_manifest.json`
- `configs/model/bootstrap_small.json`
- `configs/train/bootstrap_300_steps.json`

## Intended Use

This repository is intended for:

- internal checkpoint tracking
- reproducibility of PHANTOM bootstrap runs
- continued initialization of future PHANTOM training stages

It is not intended to be presented as the final flagship model checkpoint.

## Limitations

- bootstrap-scale model, not `500B`
- bootstrap corpus, not full production corpus
- bootstrap tokenizer training budget, not final tokenizer training budget
- current checkpoint is for pipeline validation and initialization, not final deployment

## Provenance

All weights in the bootstrap checkpoints are trained through the PHANTOM bootstrap pipeline using owned training runs and owned repository code.

## Repository

Project repository:

- `https://github.com/MagistrTheOne/NULLXES-PHANTOM`
