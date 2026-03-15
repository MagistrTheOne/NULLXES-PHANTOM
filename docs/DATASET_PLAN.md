# PHANTOM Dataset Plan

As of March 15, 2026, this is the recommended data plan for a full `PHANTOM-500B` pretraining run.

## Core Assumption

The `34T` number in [PHANTOM_ARCHITECTURE.md](../PHANTOM_ARCHITECTURE.md) should be treated as consumed training tokens, not unique raw text.

For a run of this size, target three layers of data:

| Layer | Target | Purpose |
|---|---|---|
| Raw acquisition lake | `60T` to `100T+` extracted tokens | broad coverage before filtering |
| Canonical clean corpus | `12T` to `16T` unique tokens | deduplicated, licensed, decontaminated base pool |
| Consumed train tokens | `34T` | stage-weighted repeated sampling from the canonical pool |

Do not add public dataset sizes together and assume you have that many useful tokens. Open corpora overlap heavily. Cross-deduplication is mandatory.

## Public Anchor Sources

These are the highest-value open anchors to build around. They should be treated as seeds and reference pools, not blindly concatenated.

| Source family | Official anchor | Approximate public scale | Role in PHANTOM |
|---|---|---|---|
| English filtered web | Hugging Face `FineWeb` | `~15T` tokens | main English web seed |
| Multilingual filtered web | Hugging Face `FineWeb2` | `1000+` languages and `3T+` non-English words | main multilingual web seed |
| Common Crawl benchmark pool | `DCLM-Pool` / `DCLM-Baseline` | `240T` pool, `4T` baseline subset | methodology reference and seed subset |
| Curated mixed corpus | AllenAI `Dolma` | `~3T` tokens | high-quality mixed seed corpus |
| Code repositories | BigCode `The Stack v2` | `~900B` tokens, `658` languages | primary open code source |
| PDF-derived long documents | Hugging Face `FinePDFs` | `~3T` tokens, `475M` docs, `1733` languages | long-doc and technical PDF seed |
| Math-rich web | `OpenWebMath` | `~14.7B` tokens from `~130k` domains | math-heavy web subset |
| Massive multilingual corpus | `HPLT v2` | `~7.6T` tokens across `193` languages | multilingual expansion |
| Biomedical OA full text | PMC Open Access Subset | millions of reusable articles | scientific long-form text |
| Open scientific papers | arXiv bulk data access | large open paper and source archive | math, CS, physics, technical writing |

## Recommended Final Mixture

This is the mixture I would target for `34T` consumed tokens.

| Bucket | Share | Consumed tokens | Canonical unique target | Main source families |
|---|---|---|---|---|
| High-quality English web and docs | `24%` | `8.2T` | `3.0T` to `4.0T` | FineWeb, Dolma, internal crawl |
| Multilingual web | `12%` | `4.1T` | `1.8T` to `2.6T` | FineWeb2, HPLT, internal regional crawl |
| Code repos, READMEs, build files, issues | `18%` | `6.1T` | `1.5T` to `2.2T` | The Stack v2, permissive mirrors, package ecosystems |
| Technical docs, API docs, standards, manuals | `11%` | `3.7T` | `1.2T` to `1.8T` | official docs crawls, RFCs, standards, manuals |
| PDFs, papers, textbooks, long documents | `11%` | `3.7T` | `1.5T` to `2.2T` | FinePDFs, arXiv, PMC OA, public books |
| Math, theorem, STEM problem text | `6%` | `2.0T` | `0.3T` to `0.6T` | OpenWebMath, FineMath, Proof-Pile-style corpora |
| Books, encyclopedic, reference text | `7%` | `2.4T` | `0.6T` to `1.0T` | Wikipedia, Wikibooks, public-domain books |
| Structured text and schemas | `5%` | `1.7T` | `0.4T` to `0.7T` | JSON, SQL, YAML, config, logs, schemas |
| Conversational and QA-style natural text | `3%` | `1.0T` | `0.2T` to `0.4T` | public forum-style text with clear rights, internal support corpora |
| Internal synthetic data | `3%` max | `1.0T` max | low | only after internal proxy models become useful |

## What You Will Need Beyond Public Data

Public corpora alone are not enough for a best-in-class `500B` run. You need internal acquisition in at least four areas:

### 1. Licensed web and docs crawl

You need your own crawl layer for:

- official documentation sites
- developer portals
- product manuals
- standards and RFC mirrors
- long-form blogs with clear reuse terms

This is the easiest way to improve beyond commodity open mixtures.

### 2. Code ecosystem expansion

Open code corpora are useful but not sufficient. You should ingest:

- repository READMEs
- issue and discussion text where rights allow it
- package manager docs and changelogs
- CI configs, infra files, and schema files

The model should see code as a software ecosystem, not only as source files.

### 3. Long-context document packs

For a `256K` shipping target, you need training units that are longer than normal web pages:

- full repositories packed as dependency-aware bundles
- multi-file manuals
- books and textbook chapters
- paper + appendix + bibliography bundles
- long technical standards

This should be a dedicated corpus product, not a late packing trick.

### 4. Commercial-use scientific subset

PMC and other research sources have split licenses. For a commercial model, keep a clean commercial-use branch and a non-commercial quarantine branch from day one.

## Data Pipeline Requirements

Every document should carry these fields through the pipeline:

- `source_id`
- `document_id`
- `license_id`
- `capture_date`
- `language`
- `domain`
- `source_family`
- `quality_score`
- `dedup_cluster_id`
- `pack_group_id`

Without that metadata you cannot do reliable exclusion, ablation, or takedown.

## Stage-by-Stage Consumption Plan

| Stage | Tokens | Main emphasis |
|---|---|---|
| S1 Foundation | `22T` | high-quality web, code, multilingual base, general knowledge |
| S2 Hard-signal | `6T` | code, math, formal STEM, technical docs, structured text |
| S3 Native long-context | `4T` | repo packs, manuals, papers, transcripts, textbooks |
| S4 Long-context extension | `1T` | `128K`-oriented long-doc continuation |
| S5 Shipping consolidation | `1T` | `256K` stabilization, contamination cleanup, late balancing |

## Exclusion Rules

- Do not train directly on benchmark test sets or their contaminated mirrors.
- Do not mix commercial and non-commercial rights in the same frozen production corpus.
- Do not keep OCR-noisy PDFs unless they pass a document-quality floor.
- Do not allow the same document family to appear in both short-form and long-pack corpora without cross-pack dedup.
- Do not let synthetic data exceed `10%` of consumed tokens before your own smaller PHANTOM models are demonstrably strong.

## Practical Source Notes

- `FineWeb`, `Dolma`, `DCLM`, and `Zyda`-style mixtures overlap. Use them as candidate pools for rescoring, not as additive corpora.
- `The Stack v2` is the cleanest open code anchor, but you still need license filtering and repository-family dedup.
- `FinePDFs`, `PMC OA`, and arXiv are essential for long technical reasoning, but they require aggressive text extraction cleanup.
- `HPLT` and `FineWeb2` should be the backbone for multilingual coverage instead of random low-quality multilingual crawl spillover.

## Minimum Viable Public Stack

If you wanted a strong open-data-only bootstrap before the internal crawl is ready, I would start with:

- FineWeb
- FineWeb2
- Dolma
- DCLM-Baseline as a rescore seed
- The Stack v2
- FinePDFs
- OpenWebMath
- HPLT
- PMC OA commercial-use branch
- arXiv bulk text extraction

That is enough to build the first serious canonical pool. It is not enough by itself to win the class at `500B`.

## Source Anchors

- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- DCLM-Baseline: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- DataComp-LM paper: https://arxiv.org/abs/2406.11794
- Dolma: https://huggingface.co/datasets/allenai/dolma
- The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
- FinePDFs: https://huggingface.co/datasets/HuggingFaceFW/finepdfs
- OpenWebMath: https://huggingface.co/datasets/open-web-math/open-web-math
- HPLT v2 download hub: https://data.hplt-project.org/two/
- HPLT paper: https://arxiv.org/abs/2403.14009
- PMC Open Access Subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist
- arXiv bulk data access: https://info.arxiv.org/help/bulk_data_s3.html
