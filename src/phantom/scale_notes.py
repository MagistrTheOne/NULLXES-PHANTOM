"""
Megatron-Core / large-scale parallel porting boundaries for PHANTOM.

Design intent: keep tensor semantics here in `phantom.model` / `phantom.moe`
unchanged; swap execution strategy only.

Tensor-parallel (TP)
    Shard attention projections and row/column parallel MLP/MoE matrices along
    the hidden or intermediate dimensions. `MultiHeadLatentAttention` should use
    explicit `d_model`, `q_latent_dim`, `kv_latent_dim` splits compatible with
    `tensor_parallel.ColumnParallelLinear` / `RowParallelLinear` patterns.

Pipeline-parallel (PP)
    Cut `PhantomCausalLM.layers` into pipeline stages; keep per-layer
    residuals local to a stage. MTP modules must live on the last stage with
    the main LM head unless replicated with careful broadcast rules.

Expert-parallel (EP)
    `MoELayer` currently performs naive per-expert dispatch. At scale replace
    the inner loop with:
    - permute tokens to expert-local groups (all-to-all),
    - grouped GEMM / fused MoE kernels,
    - reverse permutation.
    Keep `MoERouter` logits shape `[N, num_routed_experts]`; EP partitions
    experts across ranks, not the router vocabulary.

Load-balancing bias
    `MoERouter.maybe_update_load_balance_bias` is intentionally optimizer-free.
    Mirror DeepSeek auxiliary-loss-free practice in distributed settings: run
    the same bias update on all ranks with identical `top_indices` histograms,
    or all-reduce counts before updating bias to stay in lockstep.

Checkpoints
    Save `ModelConfig` JSON alongside shards; provide a merge script for
    `state_dict` keys → consolidated HF-style or Megatron checkpoints (future).

Fused kernels
    SwiGLU, MLA attention, and MoE dispatch are correctness-first PyTorch. Swap
    in Triton/Cutlass/Megatron fused modules behind identical module I/O.
"""
