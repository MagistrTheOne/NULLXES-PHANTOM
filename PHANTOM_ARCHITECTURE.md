# PHANTOM 500B MoE Architecture Spec

## Identity

- Company: NULLXES LC
- Contact: ceo@nullxes.com
- Model codename: PHANTOM
- Scope: from-scratch foundation model, no borrowed base weights, no external teacher distillation, no fine-tuning-first shortcut

## Design Goal

Build a frontier-class text-first foundation model that is:

- strong in code, math, technical reasoning, and long-form instruction following
- efficient enough to serve at scale despite 500B total parameters
- stable to train from scratch on your own stack
- future-proof for a long-context branch without compromising the base model

The safest high-upside design for this target is not a dense 500B transformer. It is a sparse decoder-only MoE model with MLA-style attention, 8 active routed experts, a Qwen-style BBPE tokenizer trained from your own corpus, and a staged context expansion path.

## Recommended Final Config

### PHANTOM-500B-Base

| Item | Recommendation |
|---|---|
| Architecture | Decoder-only Transformer + MoE |
| Total parameters | ~500B |
| Active parameters per token | ~36B to 39B |
| Layers | 61 |
| Dense layers | First 4 layers dense FFN |
| MoE layers | Remaining 57 layers |
| Hidden size | 7168 |
| Attention | Multi-Head Latent Attention |
| Attention heads | 128 |
| Head dim | 128 |
| Query latent dim | 2048 |
| KV latent dim | 768 |
| Positional encoding | RoPE with high base frequency + YaRN extension path |
| Norms | RMSNorm pre-norm + QK-Norm |
| FFN activation | SwiGLU |
| Routed experts per MoE layer | 192 |
| Shared experts per MoE layer | 1 |
| Activated routed experts per token | 8 |
| Routing score | Sigmoid affinity + top-k normalized routing |
| Load balancing | Auxiliary-loss-free bias balancing, with only a minimal anti-collapse floor |
| MTP | Yes, depth 1 |
| Native training context | 32K by the end of pretraining |
| Shipping context target | 256K |
| Optional long-context branch | 1M, as a separate PHANTOM-Long continuation, not the base flagship |

## Why This Shape

### 1. Use MLA, not standard GQA, in the flagship

At 500B total parameters, long-context serving becomes a KV-cache problem before it becomes a pure FLOP problem. MLA is the right choice because it reduces KV memory pressure and keeps long-context decoding practical. This is the main reason to prefer a DeepSeek/Kimi-style attention core over a conventional dense-attention stack.

### 2. Keep 8 active experts

Top-8 routing is the current sweet spot for large frontier MoE systems:

- enough specialization to make 500B worthwhile
- much easier to stabilize than very high active-expert counts
- better serving economics than a denser MoE path

Going to 16 active experts in v1 would inflate active parameters too hard. Going below 8 leaves too much quality on the table for a 500B-class model.

### 3. Use 192 routed experts, not 128 and not 384

192 routed experts is the pragmatic middle:

- 128 is proven, but too conservative for a 500B flagship
- 384 is attractive for specialization, but it increases routing and systems complexity for the first major run
- 192 keeps the architecture aggressive without making the training stack the main research problem

### 4. Keep one shared expert

For a from-scratch run, one shared expert is a useful stability valve:

- it preserves a dense path for universal patterns
- it reduces pressure on the router in early training
- it avoids over-fragmenting common linguistic and formatting behaviors

For PHANTOM v1, this is the lower-risk choice over a no-shared-expert design.

### 5. Keep MTP depth at 1

Depth-1 multi-token prediction is worth it for two reasons:

- it improves training signal density
- it opens the door to speculative decoding later

Deeper MTP is not the right risk profile for the first 500B run.

## Context Strategy

Do not build the flagship around a native 1M context claim. That is the wrong optimization target for v1.

Recommended path:

| Stage | Context | Purpose |
|---|---|---|
| S1 | 4K | language, code, syntax, core world model |
| S2 | 4K | harder STEM, code, reasoning-heavy continuation |
| S3 | 32K | native long-document competence |
| S4 | 128K | continuation stage for long-context robustness |
| S5 | 256K | final shipping window consolidation |

### Shipping recommendation

- Default production context: 256K
- Default max generation: 16K
- Hard generation cap: 32K

### Separate long-context branch

If you want a headline window beyond that, do it as:

- `PHANTOM-500B-Long`
- same base weights
- dedicated continuation on long-context corpora
- dedicated serving kernels and chunked prefill path

That branch can target 1M. The base flagship should not be optimized around that constraint from day one.

## Tokenizer Strategy

### Recommended tokenizer: PHANTOM-BBPE-160K

Borrow only the tokenizer family idea from Qwen:

- byte-level BPE
- tiktoken-style pretokenization
- byte fallback so there is no unknown-token failure mode

But train the vocabulary, merges, and specials fully on your own corpus.

### Tokenizer spec

| Item | Recommendation |
|---|---|
| Tokenizer family | Byte-level BPE |
| Vocab size | 160,000 total ids |
| Reserved special ids | 512 |
| Learned ids | 159,488 |
| Unknown token | None; byte fallback only |
| Unicode normalization | Minimal; preserve code and whitespace behavior |
| Whitespace policy | Explicit newline and indentation-friendly behavior |
| Numeric policy | Ensure single digits remain cheap; let corpus learn common numeric chunks |
| Structured text | Keep JSON, XML, Markdown, URLs, file paths, stack traces, SQL patterns well represented |

### Why 160K

160K is the right point for this project:

- large enough for multilingual text, code, markup, and technical corpora
- small enough to avoid wasting softmax capacity
- gives room for reserved control and future expansion tokens

Do not simply clone Qwen's exact vocabulary. Use the architecture style, not the exact token inventory.

### Tokenizer training corpus

Build the tokenizer on a weighted sample of the final pretraining mixture, not on raw-frequency web text alone. Weight the sample so English web prose does not dominate the merges. The tokenizer should be intentionally biased toward:

- code and repositories
- technical documentation
- structured text
- multilingual coverage
- long-form documents

## Training Token Budget

If the goal is "best in class", the token budget cannot be conservative.

### Recommended total

- Target: 34T training tokens
- Acceptable floor: 28T
- Stretch target: 40T if data quality remains high

### Recommended stage split

| Stage | Tokens | Context | Notes |
|---|---|---|---|
| S1 Foundation | 22T | 4K | broad text, code, books, technical web, multilingual |
| S2 Hard-signal | 6T | 4K | code, math, STEM, formal reasoning, structured data |
| S3 Native long-context | 4T | 32K | long documents, repo packs, manuals, transcripts |
| S4 Long-context extension | 1T | 128K | robustness and retrieval continuation |
| S5 Final context consolidation | 1T | 256K | shipping-window stabilization |

### Data mix guardrails

- Keep at least 20% of the full budget code + technical artifacts.
- Keep at least 10% genuinely multilingual, not machine-translated filler.
- Keep synthetic data below 10% until your own internal smaller PHANTOM proxy models are good enough to generate useful self-augmentation.
- Do not use external frontier model traces as the backbone of the corpus.

## What Not To Do

- Do not build a dense 500B flagship. The serving economics are worse for no good reason.
- Do not ship the first major version as a 1M-native model. Make 256K excellent first.
- Do not copy another model's exact tokenizer vocabulary.
- Do not make the router research problem bigger than the language-model problem. Avoid 384+ experts in v1.
- Do not over-index on synthetic data early. From-scratch models still live or die by real corpus quality.
- Do not fuse "reasoning mode" into a separate architecture. Keep one strong base model and solve reasoning in later training stages.

## Final Call

If I had to lock PHANTOM today, I would lock this:

- 61-layer decoder-only MoE
- first 4 layers dense, remaining 57 MoE
- hidden size 7168
- MLA with 128 heads, 2048 query latent, 768 KV latent
- 192 routed experts + 1 shared expert per MoE layer
- top-8 routed activation
- auxiliary-loss-free routing balance
- MTP depth 1
- PHANTOM-BBPE-160K tokenizer
- 34T-token training plan
- 256K shipping context
- 1M only as a dedicated continuation branch

That is the strongest low-regret starting point for a 500B from-scratch model on your stated constraints.

## Source Anchors

These sources were used as architecture anchors. The final PHANTOM configuration above is an inference and recommendation, not a copy of any one model.

- DeepSeek-V3 Technical Report: https://ar5iv.labs.arxiv.org/html/2412.19437
- Qwen3 Technical Report: https://ar5iv.labs.arxiv.org/html/2505.09388
- Qwen2.5-1M Technical Report: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf
- Qwen official repo tokenizer note: https://github.com/QwenLM/Qwen
- Kimi K2 paper page: https://arxiv.org/abs/2507.20534
