# Architecture Extensibility Assessment

How hard is it to add non-standard model architectures? What needs to change to support MoE, Mamba/SSM, DeepSeek MLA, and other diverse architectures?

## Executive Summary

With the Python build / C++ runtime split, adding new model families is now a **Python-only task**. The Python `trtf_build/` package provides a plugin system for family-specific checkpoint mappers and graph builders, while the C++ runtime handles only bundle loading and inference.

For non-standard architectures, the effort is focused on **Python graph builders** (using the TensorRT Python API) and, in some cases, **C++ state management** changes for the autoregressive loop.

| Architecture Class | Current Support | Effort to Add | Where Changes Needed |
|---|---|---|---|
| Dense decoder (LLaMA, Yi, Gemma, Phi-3) | **Works today** | ~30 LOC Python | Python plugin only |
| Dense decoder + different MLP/norm (GPT-2, OPT) | **Needs custom graph builder** | ~150 LOC Python | Python graph builder |
| MoE decoder (Mixtral, Qwen-MoE) | **Needs custom graph builder** | ~300 LOC Python | Python graph builder + checkpoint mapper |
| Multi-Latent Attention -- MLA (DeepSeek-V2/V3) | **Needs custom graph + C++ cache** | ~400 LOC Python + C++ | Python graph builder + C++ KV cache shape |
| SSM / Mamba | **Needs custom graph + C++ state** | ~600 LOC Python + C++ | Python graph builder + C++ `IStepState` impl |
| Hybrid SSM+Attention (Jamba) | **Needs both above** | ~800 LOC Python + C++ | All of the above |

---

## What Is Easy (Python-Only Changes)

### Adding a standard dense decoder family

Create a Python plugin in `trtf_build/families/<family>/` with a checkpoint mapper. Uses the standard graph builder. ~30 LOC.

**Examples**: Yi, InternLM, Baichuan, DeepSeek-dense.

### Adding a custom graph builder

Write a Python graph builder using the TensorRT Python API and shared `graph_ops.py` ops. Register it in the family plugin. No C++ changes.

**Examples**: GPT-2/OPT (LayerNorm + GELU), Phi (parallel attention), Mixtral MoE (expert routing).

---

## What Requires C++ Changes

### Different state management (Mamba/SSM)

The C++ autoregressive loop assumes KV-cache state (`KvCacheStepState`). Models with fundamentally different state (recurrent hidden state, SSM state) need:
1. A new `IStepState` implementation in C++
2. Modifications to the generate loop to handle the new state type

The `IStepState` interface is already abstract (`virtual ~IStepState() = default`), so new implementations can be added without modifying existing code.

### Different KV cache shapes (DeepSeek MLA)

Compressed KV caches (e.g., `[cache_len, kv_lora_rank]` instead of `[cache_len, attention_size]`) need changes to `KvCacheStepState` in C++ to support variable cache sizes.

---

## Architecture-Specific Deep Dives

### MoE (Mixtral, Qwen-MoE, DeepSeek-MoE)

**Python changes**:
- Checkpoint mapper: Map expert weights (`model.layers.N.block_sparse_moe.experts.E.*`)
- Graph builder: Expert routing logic (router -> top_k -> dispatch -> per-expert SwiGLU -> combine)
- New graph ops: `add_topk_routing()`, `add_expert_dispatch()`

**C++ changes**: None. Standard KV-cache autoregressive loop works unchanged.

**Estimated work**: ~300 LOC Python.

### Mamba / SSM

**Python changes**:
- Checkpoint mapper: Map SSM weights (`conv1d`, `x_proj`, `dt_proj`, `A_log`, `B`, `C`, `D`)
- Graph builder: Selective scan, 1D convolution, discrete state-space update
- Engine I/O: No attention mask/position. Input: token + per-layer hidden state. Output: logits + new hidden state.

**C++ changes**:
- New `IStepState` implementation: `RecurrentStepState` (per-layer hidden state, no KV cache)
- Generate loop modifications for different state update pattern

**Estimated work**: ~400 LOC Python + ~200 LOC C++.

### DeepSeek MLA (Multi-Latent Attention)

**Python changes**:
- Checkpoint mapper: Compressed Q/KV projections (`w_dq`, `w_uk`, `w_uv`, `kv_lora_rank`)
- Graph builder: Decomposed attention with low-rank KV

**C++ changes**:
- `KvCacheStepState` generalization for different cache shapes per layer

**Estimated work**: ~300 LOC Python + ~100 LOC C++.

### Hybrid SSM+Attention (Jamba, Zamba)

**Python + C++ changes**: Combination of Mamba and attention approaches.
- Per-layer type: some layers attention (KV cache), some Mamba (recurrent state)
- `HybridStepState` that manages both state types

**Estimated work**: ~500 LOC Python + ~300 LOC C++.

---

## Recommended Approach for New Families

### Tier 1: Python-only (start immediately)
Standard dense decoders using the existing graph builder:
- Yi, InternLM, Baichuan, DeepSeek-dense, CodeLlama, Vicuna
- ~30 LOC each, independent work

### Tier 2: Python custom graph builder
Non-standard graph topologies:
- GPT-2/OPT (GELU MLP), Phi (parallel attention), GPT-NeoX, Falcon
- ~150-200 LOC each, independent work

### Tier 3: Python + C++ state changes
Fundamentally different architectures:
- Mixtral MoE (Python only, ~300 LOC)
- DeepSeek MLA (Python + C++ cache, ~400 LOC)
- Mamba (Python + C++ state, ~600 LOC)
- Jamba hybrid (Python + C++ state, ~800 LOC)

Each tier can be worked on independently. Tier 1 and 2 families are fully parallelizable since they only touch Python plugins with no shared file edits.
