# TRT Internals

This page explains how the TensorRT engine is built (Python) and how it runs (C++). The build and runtime phases are split across languages.

## Overview

The TRT pipeline has two phases:

1. **Build phase (Python)**: `trtf_build/` reads HF model weights, constructs a TRT `INetworkDefinition` via the TensorRT Python API, compiles it to an `ICudaEngine`, and serializes the engine plan into a `.trtfb` bundle.
2. **Run phase (C++)**: The C++ runtime deserializes the engine plan from the bundle, creates an execution context, and runs the autoregressive generation loop on GPU with KV-cache management.

---

## Decoder Layer Anatomy

The standard decoder graph builder (in Python) implements the dominant modern LLM decoder pattern. Each layer performs:

### 1. Pre-Attention RMSNorm
```
norm1 = RMSNorm(hidden, input_norm_weights, eps)
```
RMS normalization: `x * gamma / sqrt(mean(x^2) + eps)`.

### 2. QKV Projections
```
Q = norm1 * W_q    [1, hidden] * [hidden, attention_size] -> [1, attention_size]
K = norm1 * W_k    [1, hidden] * [hidden, attention_size] -> [1, attention_size]
V = norm1 * W_v    [1, hidden] * [hidden, attention_size] -> [1, attention_size]
```

### 3. Optional Per-Head QK Norm (Qwen3)
```
Q = PerHeadRMSNorm(Q, q_norm_weights)
K = PerHeadRMSNorm(K, k_norm_weights)
```
LLaMA and Mistral skip this step. Qwen3 applies per-head RMS normalization before RoPE.

### 4. Rotary Position Embeddings (RoPE)
```
Q = ApplyRoPE(Q, position_id, cos_table, sin_table)
K = ApplyRoPE(K, position_id, cos_table, sin_table)
```
The cos/sin tables are precomputed for all positions up to `max_cache_length + 1` and stored as constant tensors in the TRT network.

### 5. KV-Cache Concatenation
```
all_K = Concat(cache_K, current_K)
all_V = Concat(cache_V, current_V)
```

### 6. Grouped Query Attention (GQA)
```
scores = Q_heads @ K_heads^T / sqrt(head_dim)
scores = scores + attention_mask
weights = softmax(scores)
context = weights @ V_heads
```
GQA is handled transparently: the checkpoint mapper expands K/V projections to match the number of query heads during loading.

### 7. Output Projection + Residual
```
attn_output = attn_output * W_o
hidden = hidden + attn_output
```

### 8. Post-Attention RMSNorm + SwiGLU MLP
```
norm2 = RMSNorm(hidden, post_attn_norm_weights, eps)
gate = norm2 * W_gate
up   = norm2 * W_up
swish = gate * sigmoid(gate)
gated = swish * up
down  = gated * W_down
hidden = hidden + down
```

### 9. Final Layer: Norm + LM Head
```
hidden = RMSNorm(hidden, final_norm_weights, eps)
logits = hidden * W_lm_head
```

---

## Graph Building (Python)

The Python `trtf_build/` package builds the TRT network graph using the TensorRT Python API. Shared ops in `trtf_build/graph_ops.py` provide reusable building blocks:

| Function | Description |
|----------|-------------|
| `add_constant_tensor()` | Creates a constant weights tensor in the TRT network |
| `add_matmul()` | Matrix multiply with constant weights |
| `add_bias_sum()` | Adds a bias vector to a tensor |
| `add_rms_norm()` | Full-hidden RMS normalization with gamma weights |
| `add_rms_norm_per_head()` | Per-head RMS normalization (for Qwen3 QK norms) |
| `add_rope()` | Apply RoPE to Q or K tensor using precomputed tables |
| `add_attention()` | Scaled dot-product attention with masking |
| `add_swiglu()` | SwiGLU MLP block (gate + up + SiLU + down) |

The standard decoder graph builder composes these ops into a full N-layer network:

```
create_decoder_step_network():
  1. Create IBuilder, INetworkDefinition, IBuilderConfig
  2. Add inputs: token_id[1], position_id[1], attention_mask[1, window],
     per-layer cache_k/cache_v [max_cache, attn_size]
  3. Add embedding gather: token_id -> hidden[1, H]
  4. Precompute RoPE tables as constants
  5. For each decoder layer:
     - add_rms_norm -> QKV proj -> optional QK norm -> add_rope
     - KV cache concat -> add_attention -> output proj -> residual
     - add_rms_norm -> add_swiglu -> residual
  6. Final RMSNorm + LM head matmul -> logits[1, vocab]
  7. Mark outputs: logits + per-layer present_k/present_v
  8. buildEngineWithConfig(network, config) -> ICudaEngine
  9. Serialize engine plan to bytes
```

Custom family graph builders can reuse these shared ops and compose them differently (e.g., MoE routing, parallel attention, vision encoder).

### VL Image Preprocessing (Non-TRT)

Vision-language models require image preprocessing before the vision TRT engine. This is handled by `image_preprocessor.cpp` (C++) and `debug_runner.py` (Python), both implementing the same 4 strategies:

| Strategy | Pipeline | Use Case |
|----------|----------|----------|
| `qwen_merge_group` | Load -> resize -> normalize -> merge-group patch permutation -> temporal duplication | Qwen2.5-VL |
| `simple_chw` | Load -> resize -> normalize | Standard ViT (LLaVA, InternVL, Phi-3-Vision) |
| `center_crop_chw` | Load -> center-crop to square -> resize -> normalize | CLIP, DINOv2-based models |
| `aspect_preserve_chw` | Load -> aspect-preserving resize -> zero-pad to square -> normalize | InternVL v2 |

Interpolation is configurable: `"bicubic"` (default, Catmull-Rom), `"bilinear"` (triangle), `"nearest"` (point sample). The mode is read from `config.json` (set by the engine builder), with fallback to the HF `preprocessor_config.json` `resample` integer.

---

## TRT Engine Lifecycle

### Engine Compilation (Python)

Engine compilation happens during `trtf-build build`:
- TensorRT compiles the network graph into optimized CUDA kernels
- Compilation takes 30-300 seconds depending on model size
- The serialized plan is written into the `.trtfb` bundle

### Engine Deserialization (C++)

Engine deserialization happens during `trtf_create_pipeline()`:
- `ReadBundleFile()` extracts the engine plan bytes from the bundle
- `createInferRuntime(logger)` creates a TRT runtime
- `deserializeCudaEngine(plan_bytes)` recreates the `ICudaEngine` (~5s)
- `createExecutionContext()` creates the execution context

### DecoderStepEngine

The deserialized engine is wrapped in `DecoderStepEngine` (C++):

```cpp
struct DecoderStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    // Tensor binding names (derived from bundle metadata)
    std::string token_id_name;
    std::string position_id_name;
    std::string attention_mask_name;
    std::vector<std::string> cache_k_names;
    std::vector<std::string> cache_v_names;
    std::vector<std::string> present_k_names;
    std::vector<std::string> present_v_names;
    std::string logits_name;

    // Metadata
    int32_t num_layers;
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t cache_state_size;
    int32_t max_cache_length;
};
```

---

## Autoregressive Generation Loop (C++)

`TrtBackendFastPath::generate()` in `trt_backend_shared.cpp`:

```
generate(input_ids, config):
  1. Create DeviceKvCache + DeviceResources:
     - Allocate per-layer cache_k, cache_v on GPU [max_cache_length, attention_size]
     - Pre-allocate per-step I/O buffers (token, position, mask, logits)

  2. Prefill phase:
     For each token in input_ids:
       - Build causal attention mask (0 for visible, -1e9 for masked)
       - run_decoder_step_device(engine, token_id, position, mask, cache)
       - D2D cache update internal to DeviceKvCache
       - Advance position counter

  3. Decode phase:
     For step = 0 to max_new_tokens:
       - Run one decode step with the previously generated token
       - Read logits from GPU
       - Greedy sampling: argmax over logits -> next_token_id
       - If next_token == eos_token: break
       - Append to output, update cache

  4. Return: input_ids + generated_token_ids
```

### KV-Cache Management

The cache uses a fixed-size circular buffer per layer, held in device memory (`DeviceKvCache`):
- Size: `[max_cache_length, attention_size]` per layer, per K and V, resident on GPU
- D2D cache update is internal to `DeviceKvCache` (present K/V written directly to cache slots on device, no host round-trip)
- Only small inputs (token ID, position, mask) are transferred H2D per step via `DeviceResources`
- Attention mask grows by one position each step
- When cache is full, oldest entries are evicted (sliding window)

### CUDA Resource Management

All GPU resources use RAII wrappers from `trt_common.h`:
- `CudaStream` -- RAII `cudaStream_t`
- `CudaBuffer` -- RAII `cudaMalloc`/`cudaFree`
- `TrtUniquePtr<T>` -- Smart pointer for TRT objects
- `TrtLogger` -- Custom `ILogger` implementation with error tracking
