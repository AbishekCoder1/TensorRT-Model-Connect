# Architecture Extensibility Assessment

How hard is it to add non-standard model architectures? What needs to change to support MoE, Mamba/SSM, DeepSeek MLA, and other diverse architectures?

## Executive Summary

The current architecture has **5 hard-coded assumptions** that work perfectly for dense Pre-RMSNorm+GQA+RoPE+SwiGLU decoders but break for non-standard architectures. The good news: each assumption is isolated in a specific file and can be generalized without rewriting the entire system.

| Architecture Class | Current Support | Effort to Add | Blocking Assumptions |
|---|---|---|---|
| Dense decoder (LLaMA, Yi, Gemma, Phi-3) | **Works today** | ~50 LOC | None |
| Dense decoder + different MLP/norm (GPT-2, OPT, Phi-1/2) | **Needs new graph builder** | ~200 LOC | #4 (SwiGLU-only graph builder) |
| MoE decoder (Mixtral, Qwen-MoE, DeepSeek-MoE) | **Needs new layer checkpoint + graph builder** | ~400 LOC + **shared changes** | #1, #4 |
| Multi-Latent Attention — MLA (DeepSeek-V2/V3) | **Needs new checkpoint + graph builder** | ~500 LOC + **shared changes** | #1, #3, #4 |
| SSM / Mamba (Mamba, Mamba2) | **Needs new state model + backend** | ~800 LOC + **shared changes** | #1, #2, #3, #5 |
| Hybrid SSM+Attention (Jamba, Zamba) | **Needs both above** | ~1000 LOC + **shared changes** | #1, #2, #3, #4, #5 |
| Encoder-only (BERT, RoBERTa) | **Needs new task type** | ~600 LOC + **shared changes** | #2, #5, new task |
| Encoder-decoder (T5, BART) | **Needs dual-stack** | ~1200 LOC + **shared changes** | #1, #2, #3, #5, new task |

---

## The 5 Hard-Coded Assumptions

### Assumption #1: Fixed-Schema Layer Checkpoint

**Where**: `DecoderLayerCheckpoint` in `include/trtf/model.h` and `TrtDecoderLayerDefinition` in `src/model/trt_model_definition.h`

**Problem**: Every layer is assumed to have exactly: `{input_norm, q_norm, k_norm, w_q, w_k, w_v, w_o, post_attn_norm, w_gate, w_up, w_down}`. This is correct for standard dense decoders but breaks for:

| Architecture | Missing/Extra Fields |
|---|---|
| MoE (Mixtral) | Needs `num_experts`, `top_k`, per-expert `w_gate[E]`, `w_up[E]`, `w_down[E]`, plus router weight `w_router` |
| DeepSeek MLA | Needs `w_dq` (down-projection for Q), `w_uk` (up-projection for K), `kv_lora_rank`, compressed KV weights instead of full `w_k`/`w_v` |
| Mamba/SSM | No attention weights at all. Needs `A`, `B`, `C`, `D` state-space matrices, `dt_proj`, `x_proj`, `conv1d` weights |
| Hybrid (Jamba) | Some layers are attention, some are Mamba — different weight schemas per layer |

**Fix**: Generalize `DecoderLayerCheckpoint` to hold arbitrary named tensors:

```cpp
struct LayerCheckpoint {
    std::string layer_type;  // "attention", "moe", "mamba", "hybrid"
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, int32_t> int_params;
    std::unordered_map<std::string, float> float_params;
};
```

Or, keep the existing struct as-is for backward compatibility and add an `extra_tensors` escape hatch:

```cpp
struct DecoderLayerCheckpoint {
    // ... existing fields (backward compatible) ...

    // Extension: arbitrary named tensors for non-standard layers
    std::unordered_map<std::string, std::vector<float>> extra_tensors;
};
```

**Impact**: Moderate. Touches `model.h` (public header), checkpoint mappers, populators, and graph builders. But existing code doesn't break if we use the additive approach.

---

### Assumption #2: KV-Cache-Based Autoregressive Loop

**Where**: `TrtBackendShared::generate()` in `src/runtime/trt/trt_backend_shared.cpp`

**Problem**: The generate loop assumes every model uses KV-cache attention:
- Allocates per-layer `cache_k` and `cache_v` vectors
- Calls `append_cache_state()` after each step
- Passes `attention_mask` to the engine
- Uses `position_id` for RoPE

This breaks for:

| Architecture | Issue |
|---|---|
| Mamba/SSM | No KV cache. Has a recurrent hidden state that gets updated each step. No attention mask. |
| Hybrid (Jamba) | Some layers use KV cache (attention), some use recurrent state (Mamba). Need both. |
| Linear attention (RWKV) | Different state update mechanism. |
| Encoder-only (BERT) | No autoregressive generation at all. Single forward pass. |

**Fix**: Abstract the generate loop's state management:

```cpp
// New interface: family-owned state manager
class IStepState {
public:
    virtual ~IStepState() = default;
    virtual void allocate(int32_t num_layers, int32_t max_cache_length) = 0;
    virtual void bind_inputs(IExecutionContext& ctx, int32_t step) = 0;
    virtual void bind_outputs(IExecutionContext& ctx) = 0;
    virtual void update_after_step(int32_t step) = 0;
};

// KV cache state (current behavior, for attention models)
class KvCacheState : public IStepState { ... };

// Recurrent state (for Mamba/SSM)
class RecurrentState : public IStepState { ... };

// Hybrid state (for Jamba — per-layer KV or recurrent)
class HybridState : public IStepState { ... };
```

**Impact**: Significant. Requires refactoring `TrtBackendShared::generate()` to use the state interface instead of hardcoded KV cache management. But the existing KV cache logic becomes `KvCacheState` with no behavioral change.

---

### Assumption #3: Fixed Engine I/O Schema

**Where**: `DecoderStepEngine` in `src/runtime/trt/trt_engine_lifecycle.h` and `run_decoder_step()` in `src/runtime/trt/trt_decode_runtime.cpp`

**Problem**: The engine expects exactly:
- **Inputs**: `token_id`, `position_id`, `attention_mask`, per-layer `cache_k_{i}`, `cache_v_{i}`
- **Outputs**: `logits`, per-layer `present_k_{i}`, `present_v_{i}`

Non-standard architectures need different I/O:

| Architecture | Different I/O |
|---|---|
| Mamba/SSM | Input: `token_id` + per-layer `hidden_state_{i}`. Output: `logits` + per-layer `new_hidden_state_{i}`. No position/mask. |
| DeepSeek MLA | Compressed KV cache with different shapes (`[cache_len, kv_lora_rank]` instead of `[cache_len, attention_size]`). |
| Encoder-decoder | Cross-attention cache from encoder, separate decoder KV cache, encoder hidden states as input. |

**Fix**: Generalize `DecoderStepEngine` to hold arbitrary named tensor bindings:

```cpp
struct StepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    // Generic named I/O (replaces hardcoded cache_k/v names)
    std::vector<TensorBinding> inputs;
    std::vector<TensorBinding> outputs;

    // Metadata
    int32_t num_layers{1};
    int32_t vocab_size{0};
};

struct TensorBinding {
    std::string name;
    nvinfer1::DataType dtype;
    std::vector<int32_t> shape;
    bool per_layer{false};  // if true, name has {layer} placeholder
};
```

**Impact**: Moderate. `finalize_decoder_step_engine()` and `run_decoder_step()` need generalization, but the current KV-cache schema can be expressed as a special case of the generic bindings.

---

### Assumption #4: SwiGLU-Only Graph Builder

**Where**: `StandardDecoderGraphBuilder::add_standard_decoder_layer_block()` in `src/model/standard_decoder_graph_builder.cpp`

**Problem**: The graph builder hardcodes the SwiGLU MLP pattern:
```cpp
gate = matmul(norm2, W_gate)
up = matmul(norm2, W_up)
swish = gate * sigmoid(gate)   // SiLU
gated = swish * up
down = matmul(gated, W_down)
```

Different models use different MLP patterns:

| Model | MLP Pattern |
|---|---|
| GPT-2, OPT | `fc1 → GELU → fc2` (two-layer MLP, no gating) |
| Phi-1/2 | Parallel attention + MLP with GELU |
| Mixtral MoE | Router → top-k expert selection → per-expert SwiGLU |
| Mamba | No MLP block — uses SSM convolution + selective scan |

**Fix**: This is already extensible. Custom `ITrtGraphBuilder` implementations can build any graph topology. The fix is just writing new graph builders for each pattern:
- `GeLUDecoderGraphBuilder` for GPT-2/OPT
- `ParallelAttentionGraphBuilder` for GPT-J/Phi
- `MoEDecoderGraphBuilder` for Mixtral

These can reuse existing `trt_graph_ops.h` primitives and add new ones (e.g., `add_gelu_mlp()`, `add_moe_routing()`).

**Impact**: Low per-family (new file only). May want to add new shared ops to `trt_graph_ops.h`.

---

### Assumption #5: Decoder-Only Model

**Where**: `Pipeline`, `RuntimeAssembly`, `ResolvedModelSpec` — the entire pipeline assumes text-generation with autoregressive decoding.

**Problem**: Non-decoder models need different pipeline tasks:
- **Encoder-only (BERT)**: `fill-mask`, `text-classification`, `token-classification` — single forward pass, no generation loop
- **Encoder-decoder (T5)**: Separate encoder pass, then decoder with cross-attention
- **Vision (ViT, CLIP)**: Image input, different preprocessing, different output heads

**Fix**: Extend the pipeline task system. Currently `Pipeline::CreateTextGeneration()` is the only factory. Add:
```cpp
Pipeline::CreateFillMask(model_id);           // Encoder-only
Pipeline::CreateTextToText(model_id);         // Encoder-decoder
Pipeline::CreateImageClassification(model_id); // Vision
```

Each task type would have its own runtime assembly path and backend interface.

**Impact**: Large architectural change. But it doesn't block decoder-only family expansion.

---

## Architecture-Specific Deep Dives

### MoE (Mixtral, Qwen-MoE, DeepSeek-MoE)

**What changes**:
1. **Checkpoint**: Per-layer has `num_experts` × `{w_gate, w_up, w_down}` + `w_router` (Assumption #1)
2. **TRT graph builder**: Expert routing logic — `router(hidden) → top_k → dispatch → per-expert SwiGLU → combine` (Assumption #4)
3. **TRT graph ops**: New `add_moe_routing()`, `add_expert_dispatch()` ops

**What stays the same**:
- Attention path (RMSNorm + GQA + RoPE) is identical to dense
- KV cache and autoregressive loop work unchanged
- Checkpoint mapper pattern (read different HF key names) works
- Engine caching, tokenizer bridge all work

**Estimated work**: ~400 LOC new + `extra_tensors` addition to `DecoderLayerCheckpoint`

```
src/models/mixtral/
  registration.cpp       # Register family, use MoEDecoderGraphBuilder
  checkpoint_mapper.cpp  # Map expert weights: model.layers.N.block_sparse_moe.experts.E.*
src/runtime/trt/
  moe_decoder_graph_builder.cpp  # New graph builder with expert routing
  trt_graph_ops.cpp              # Add: add_topk_routing(), add_expert_ffn()
```

### Mamba / SSM

**What changes**:
1. **Checkpoint**: Completely different weights — `conv1d`, `x_proj`, `dt_proj`, `A_log`, `B`, `C`, `D` per layer (Assumption #1)
2. **State management**: No KV cache. Recurrent hidden state `[hidden_size]` per layer instead (Assumption #2)
3. **Engine I/O**: No attention mask, no position ID. Input: token + per-layer hidden state. Output: logits + new hidden state (Assumption #3)
4. **TRT graph builder**: Selective scan, 1D convolution, discrete state-space update (Assumption #4)

**What stays the same**:
- Pipeline and resolution stages work
- Tokenizer and embedding
- Greedy sampling after logits
- Engine caching
- Safetensors loading infrastructure

**Estimated work**: ~800 LOC new + shared state abstraction refactor

```
src/models/mamba/
  registration.cpp
  checkpoint_mapper.cpp  # Map SSM weights
src/runtime/trt/
  mamba_graph_builder.cpp    # SSM graph: conv1d → x/dt/B/C proj → selective scan
  trt_graph_ops.cpp          # Add: add_selective_scan(), add_conv1d()
src/runtime/trt/
  trt_backend_shared.cpp     # Refactor: extract state interface (one-time cost)
```

### DeepSeek MLA (Multi-Latent Attention)

**What changes**:
1. **Checkpoint**: Compressed Q/KV projections — `w_dq` (low-rank Q decomposition), `w_uk`/`w_uv` (KV up-projections), `kv_lora_rank` parameter (Assumption #1)
2. **KV cache shape**: `[cache_len, kv_lora_rank]` instead of `[cache_len, attention_size]` — much smaller (Assumption #3)
3. **TRT graph builder**: Decomposed attention: `Q = RoPE(X * W_dq) * W_uq`, `KV_compressed = X * W_kv`, then up-project at attention time (Assumption #4)

**What stays the same**:
- Still autoregressive with KV cache (just different shape)
- MLP is standard SwiGLU
- RoPE still applies (to a subset of heads)
- Most of the generate loop works (cache shape parametrized)

**Estimated work**: ~500 LOC new + cache shape generalization

### Hybrid SSM+Attention (Jamba, Zamba)

**What changes**: Everything above for both Mamba and attention, plus:
- Per-layer type flag: `layer_types = ["mamba", "mamba", "attention", "mamba", ...]`
- Different state per layer type (KV cache for attention layers, recurrent state for Mamba layers)
- Different graph ops per layer

**Estimated work**: ~1000 LOC (assumes Mamba and attention builders already exist)

---

## Recommended Refactoring Roadmap

To enable a dozen subagents to work in parallel on different model families, I recommend these changes in priority order:

### Phase A: Generalize Layer Checkpoint (Unblocks: MoE, MLA, Mamba) — COMPLETED

Added `extra_tensors` to `DecoderLayerCheckpoint` and `TrtDecoderLayerDefinition`, plus `extra_int_params`/`extra_float_params`/`extra_string_params` to `DecoderArchitectureConfig`, and matching fields to `TrtDecoderDefinition`. Engine cache hashing updated for all extra fields (version bumped to v4). Model loader now parses `intermediate_size` into `extra_int_params`. Standard populator copies `extra_tensors` through the layer pipeline.

### Phase B: Generalize Engine I/O (Unblocks: Mamba, MLA) — COMPLETED

Added `TensorBinding` struct and `extra_bindings` vector to `DecoderStepEngine`. Added `find_extra_bindings()` for prefix-based lookup. Added `finalize_decoder_step_engine` overload accepting extra bindings. `has_all_required_tensors()` now validates extra bindings. Existing KV-cache I/O is unchanged; extra bindings are additive for non-standard architectures.

### Phase C: Abstract State Management (Unblocks: Mamba, Hybrid) — COMPLETED

Extracted `IStepState` interface (`src/runtime/trt/step_state.h`) from `TrtBackendShared::generate()`. Implemented `KvCacheStepState` (`kv_cache_step_state.h/cpp`) preserving exact current KV-cache behavior. Refactored `generate()` to use `state->prepare_step()`, `state->cache_k/v_by_layer()`, and `state->update_after_step()`. Zero behavioral change — identical token output for all existing models.

### Phase D: New Graph Ops Library (Unblocks: MoE, Mamba, MLA, GELU models)

Add new reusable ops to `trt_graph_ops.h`:

```cpp
// MoE
ITensor* add_topk_routing(network, hidden, router_weights, num_experts, top_k);
ITensor* add_expert_ffn(network, input, expert_weights, expert_id);

// Mamba/SSM
ITensor* add_selective_scan(network, input, A, B, C, D, delta);
ITensor* add_conv1d(network, input, weight, bias, groups);

// Alternative activations
ITensor* add_gelu_mlp(network, input, w_fc1, b_fc1, w_fc2, b_fc2);
```

**Effort**: ~100-200 LOC per op group. Can be parallelized across subagents.

---

## Subagent Parallelization Strategy

After Phases A-C (~400 LOC shared refactor), subagents can work independently:

### Tier 1: Zero shared changes needed (today)
These only need a checkpoint mapper + use existing StandardDecoderGraphBuilder:

| Model | Family match | Notes |
|---|---|---|
| Yi | `starts_with("yi")` | Identical to LLaMA |
| Mistral-dense | `starts_with("mistral")` | Identical to LLaMA |
| Gemma | `starts_with("gemma")` | Standard + different RoPE formula |
| InternLM | `starts_with("internlm")` | Standard pattern |
| Baichuan | `starts_with("baichuan")` | Standard pattern |
| DeepSeek-dense | `starts_with("deepseek")` | Standard + possible ALiBi |

### Tier 2: New graph builder only (after Phase D ops)
Need a custom `ITrtGraphBuilder` but no shared changes:

| Model | What's different |
|---|---|
| GPT-2 / OPT | LayerNorm + GELU MLP (not RMSNorm + SwiGLU) |
| Phi-1/2 | Parallel attention + MLP |
| GPT-NeoX | Rotary + parallel attention |
| Falcon | Multi-query attention + different norm placement |

### Tier 3: Shared changes needed (after Phases A-C)
Need checkpoint generalization + possibly state abstraction:

| Model | Required phases |
|---|---|
| Mixtral | A (extra_tensors for experts) + D (MoE ops) |
| Qwen-MoE | A + D |
| DeepSeek-V2 MLA | A + C (generic cache shapes) + custom graph builder |
| Mamba | A + B (state abstraction) + C (generic I/O) + D (SSM ops) |
| Jamba | A + B + C + D (all phases) |

### Tier 4: New task type (future)

| Model | What's needed |
|---|---|
| BERT / RoBERTa | New pipeline task (no generation loop) |
| T5 / BART | Dual-stack encoder-decoder |
| ViT / CLIP | Image input preprocessing |

---

## Concrete Recommendation

**Immediate next step**: Implement Phases A-C as a single "extensibility foundation" commit (~400 LOC shared refactor). This:
1. Adds `extra_tensors` to checkpoint structs (Phase A)
2. Extracts `IStepState` from the generate loop (Phase B)
3. Generalizes engine I/O bindings (Phase C)

**Then**: Launch subagents in parallel:
- **6 Tier-1 agents** (Yi, Mistral, Gemma, InternLM, Baichuan, DeepSeek-dense) — can start today, ~50 LOC each
- **4 Tier-2 agents** (GPT-2, Phi, GPT-NeoX, Falcon) — need new graph builders, ~200 LOC each
- **3 Tier-3 agents** (Mixtral MoE, DeepSeek-V2 MLA, Mamba) — start after extensibility foundation, ~500-800 LOC each

Each agent needs:
1. `src/models/<family>/registration.cpp` + `checkpoint_mapper.cpp`
2. Optionally a custom graph builder in `src/runtime/trt/`
3. A test in `tests/test_<family>_family.cpp`
4. One-line registration in `hf_family_registry.cpp` + CMakeLists.txt entries
