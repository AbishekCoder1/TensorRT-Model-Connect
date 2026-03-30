# FP16 TensorRT Network Construction Guide

## Skill: Making a TRT Network FP16-Compatible

This guide documents the complete process for building FP16 TensorRT networks
using **strongly-typed mode** (`NetworkDefinitionCreationFlag.STRONGLY_TYPED`).
It is cross-validated against TensorRT 10.15+ documentation and verified
experimentally on Qwen3-0.6B (GB300).

> **Key principle**: In strongly-typed mode, every tensor has an explicit
> dtype. There are NO `BuilderFlag.FP16` or `BuilderFlag.INT8` flags.
> Precision is controlled entirely through typed constants and explicit
> `ICastLayer` boundaries.

---

## 1. Network Creation

```python
import tensorrt as trt

flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flags)
```

**Rules of strongly-typed mode:**
- `BuilderFlag.FP16`, `BuilderFlag.INT8`, `BuilderFlag.BF16` are **forbidden**
- `layer.setPrecision()` and `layer.setOutputType()` are **forbidden**
- `tensor.dtype = ...` assignment is **forbidden**
- All precision changes go through `network.add_cast(tensor, target_dtype)`
- Elementwise ops require **all inputs to have the same dtype**
- Constants must be created in the target dtype

---

## 2. Precision Boundary Map

Not all operations are safe in FP16. The following table documents which
operations must run in FP32 and why.

### Operations that MUST run in FP32

| Operation | Why FP32 Required | Failure Mode if FP16 |
|-----------|-------------------|----------------------|
| **RMSNorm** | Reduction (mean of squares) accumulates error; sqrt/reciprocal lose precision | Repeated tokens, degenerate text (experimentally verified) |
| **LayerNorm** | Same as RMSNorm plus variance computation | Audio degradation in TTS models (Magpie TTS documented this) |
| **GroupNorm** | Per-group reduction same accumulation issue | Image artifacts in diffusion models |
| **L2Norm** | sqrt(sum(x^2)) accumulation | Embedding collapse |
| **Softmax** | exp() overflows FP16 range (max 65504); normalization loses precision | Incorrect attention weights, wrong token probabilities |
| **BatchNorm** | Running mean/var + scale/shift computation | Training statistics corrupted |

### Operations safe in FP16

| Operation | Why Safe | Notes |
|-----------|----------|-------|
| **MatMul (linear projections)** | Tensor Cores natively support FP16 input with FP32 accumulation | Main throughput benefit of FP16 |
| **Embedding lookup (Gather)** | Simple table lookup, no arithmetic | FP16 table halves memory |
| **SiLU / GELU / ReLU** | Activation functions operate element-wise in safe FP16 range | No accumulation |
| **RoPE application** | Element-wise multiply with cos/sin tables | Trigonometric values fit in FP16 range |
| **Residual connection (Add)** | Simple addition | Values stay in reasonable range |
| **Bias addition** | Element-wise addition of small vectors | Bias magnitudes typically < 1.0 |

### I/O tensors

| Tensor | Required Dtype | Reason |
|--------|---------------|--------|
| **Attention mask (input)** | FP32 | Contains -1e4 or -1e9 for masked positions; FP16 max is 65504 |
| **KV cache (input/output)** | Work dtype (FP16) | Matches compute precision; halves cache memory |
| **Logits (output)** | FP32 | Accurate argmax/sampling requires precise logit ranking |
| **token_id, position_id** | INT32 | Integer indices, not affected by precision |

---

## 3. The FP32 Precision Boundary Pattern

Every numerically sensitive operation follows this pattern:

```python
def add_rms_norm(network, inp, hidden_size, gamma, eps_tensor,
                 dtype=np.float32):
    need_cast = (dtype != np.float32)

    # ---- ENTRY: cast inputs to FP32 ----
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    # ---- COMPUTE: all arithmetic in FP32 ----
    # (mean, variance, sqrt, reciprocal, scale by gamma)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    # ... norm computation ...

    # ---- EXIT: cast result back to work dtype ----
    if need_cast:
        result = network.add_cast(result, _np_to_trt_dtype(dtype)).get_output(0)
    return result
```

**Critical details:**
1. **Cast ALL FP16 tensors used in the computation** — including `eps_tensor`.
   Omitting the eps cast causes a TRT build error: "ElementWiseOperation SUM
   must have same input types (Float and Half)".
2. **Norm weights (gamma, beta) are ALWAYS FP32** — even when the working
   dtype is FP16. They participate in FP32 elementwise ops inside the boundary.
3. The boundary is **conditional** (`need_cast`) — FP32 mode has zero overhead.

### Softmax precision boundary

```python
# Before softmax: cast scores to FP32
softmax_inp = scaled_attention_scores
if dtype != np.float32:
    softmax_inp = network.add_cast(softmax_inp, trt.float32).get_output(0)

softmax = network.add_softmax(softmax_inp)
softmax.axes = 1 << 2  # attention dim

attn_weights = softmax.get_output(0)

# After softmax: cast back to work dtype
if dtype != np.float32:
    attn_weights = network.add_cast(
        attn_weights, _np_to_trt_dtype(dtype)).get_output(0)
```

This pattern appears in ALL attention functions:
- `add_self_attention_block()`
- `add_self_attention_block_with_rope()`
- `add_windowed_self_attention_with_rope()`
- `add_cross_attention()`

---

## 4. Cast-MatMul-Cast Kernel Fusion

TensorRT automatically fuses `cast → matmul → cast` into a single kernel
that reads FP16, accumulates in FP32, and writes FP16.

```
# This graph:
q_fp32 = cast(q_fp16, FP32)
k_fp32 = cast(k_fp16, FP32)
scores = matmul(q_fp32, k_fp32^T)
scores_fp16 = cast(scores, FP16)

# Becomes a single kernel:
# FP16 input → FP32 accumulate → FP16 output
```

**When to use this pattern:** For attention score computation (QK^T) where
you want FP32 accumulation without paying the memory bandwidth cost of
FP32 tensors.

**When NOT to use:** For softmax and normalization — TRT does NOT fuse
cast+softmax+cast or cast+reduce+cast. These must be explicit boundaries.

**In practice, for standard decoder models:** We do NOT explicitly wrap
matmuls with cast layers. Instead, weights and activations are already FP16,
and TRT's kernel selection automatically uses FP32 accumulation on Tensor
Cores. The cast-matmul-cast pattern is only needed when you need to force
FP32 accumulation (e.g., for final logit computation or when precision is
critical).

---

## 5. Weight Loading Strategy

```python
def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision in ("fp16", "bf16"):
        return np.float16
    return np.float32
```

### Weight dtype rules

| Weight Type | FP16 Mode Dtype | Reason |
|-------------|----------------|--------|
| Embedding table | `np.float16` | Large tensor, benefits from compression |
| Q/K/V/O projections | `np.float16` | Main compute weights |
| MLP gate/up/down | `np.float16` | Main compute weights |
| LM head | `np.float16` | Final projection |
| RMSNorm gamma | `np.float32` | **Always FP32** — used inside FP32 boundary |
| LayerNorm gamma/beta | `np.float32` | **Always FP32** — used inside FP32 boundary |
| Q/K norm gamma (per-head) | `np.float32` | **Always FP32** — norm weights |

**Key rule:** Norm affine parameters (gamma, beta) are ALWAYS loaded and
stored in FP32. All other weights use the target dtype.

---

## 6. Constant Creation

In strongly-typed mode, constants MUST be created in their target dtype.
TRT does NOT auto-cast constants.

```python
def add_constant(network, shape, values, dtype=np.float32):
    """Add a constant tensor in the specified dtype."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)
```

**Threading dtype through the graph:**
Every function that creates constants accepts a `dtype` parameter:

```python
# Weight constants: use work_np_dtype (FP16)
embedding = add_constant(network, (vocab, hidden), weights["embedding"],
                         dtype=work_np_dtype)
w_q = add_matmul_rhs_constant(network, hidden_state, hidden, attn_size,
                               weights["w_q"], dtype=work_np_dtype)

# Scalar constants in activation functions: use work_np_dtype (FP16)
half = add_constant(network, (1,), np.array([0.5]), dtype=work_np_dtype)
one = add_constant(network, (1,), np.array([1.0]), dtype=work_np_dtype)

# Norm constants: ALWAYS FP32 (inside precision boundary)
gamma = add_constant(network, (1, hidden), gamma_weights, dtype=np.float32)
```

---

## 7. Builder Configuration

```python
def build_standard_decoder_engine(config, weights, max_cache_length, *,
                                   precision="fp32", ...):
    # Compute work dtypes
    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16   # storage
        work_trt_dtype = trt.bfloat16  # TRT compute type
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    # Network creation (strongly-typed)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Input tensor dtypes
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, ...)

    # KV cache inputs: work dtype (FP16 for FP16 mode)
    for i in range(num_layers):
        cache_k = network.add_input(f"cache_k_{i}", work_trt_dtype,
                                     (max_cache_length, attn_size))

    # Cast attention mask to work dtype for elementwise compatibility
    if work_trt_dtype != trt.float32:
        attention_mask = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)

    # ... build graph with dtype=work_np_dtype ...

    # Logits: always FP32 output
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
```

---

## 8. C++ Runtime: Auto-Detecting FP16 Cache

The C++ runtime queries the engine for tensor dtypes at construction:

```cpp
// DeviceKvCache constructor
if (!engine.cache_k_input_names.empty()
    && has_io_tensor(*engine.engine, engine.cache_k_input_names[0]))
{
    auto trt_dtype = engine.engine->getTensorDataType(
        engine.cache_k_input_names[0].c_str());
    switch (trt_dtype)
    {
    case nvinfer1::DataType::kHALF:
        mCacheElementSize = 2;   // sizeof(half)
        break;
    case nvinfer1::DataType::kBF16:
        mCacheElementSize = 2;
        break;
    default:
        mCacheElementSize = sizeof(float);  // 4
        break;
    }
}

// All cache allocations use mCacheElementSize
const size_t cache_bytes = max_length * state_size * mCacheElementSize;
```

**Key design**: The runtime makes NO assumptions about precision — it
queries the engine itself. This means FP32 bundles and FP16 bundles are
handled identically with zero code changes at runtime.

---

## 9. Bundle Metadata

The `.trtfb` bundle carries precision in its config.json:

```json
{
    "runtime_strategy": "decoder_kv_cache",
    "precision": "fp16",
    "vocab_size": 151936,
    ...
}
```

The C++ runtime reads this via `BaseConfig::precision` but primarily
relies on engine tensor dtype introspection (which is authoritative).

---

## 10. Common Pitfalls and Solutions

### Pitfall 1: RMSNorm without FP32 boundary → garbage output
**Symptom**: Model generates repeated tokens ("you, you, you!!")
**Root cause**: FP16 reduction in mean(x^2) accumulates error
**Fix**: Cast input to FP32 before norm, cast back after

### Pitfall 2: eps_tensor type mismatch
**Symptom**: TRT build error "ElementWiseOperation SUM must have same
input types (Float and Half)"
**Root cause**: eps_tensor created in FP16 but used inside FP32 norm
computation
**Fix**: Cast eps_tensor to FP32 alongside the input inside the norm
function

### Pitfall 3: Elementwise ops with mismatched dtypes
**Symptom**: TRT build error about mismatched types
**Root cause**: In strongly-typed mode, ALL inputs to elementwise ops
must have identical dtypes
**Fix**: Ensure constants and tensors in the same elementwise op share
the same dtype. Use `add_constant(..., dtype=work_np_dtype)` consistently.

### Pitfall 4: `set_output_type()` or `tensor.dtype =` in strongly-typed
**Symptom**: TRT error "not supported in strongly typed mode"
**Fix**: Replace with `network.add_cast(tensor, target_dtype)`

### Pitfall 5: Attention mask overflow
**Symptom**: Masked positions not properly ignored, model attends to
padding
**Root cause**: Mask value -1e9 underflows FP16 (max magnitude 65504)
**Fix**: Keep attention_mask input as FP32; cast to work dtype after
loading

### Pitfall 6: Logits precision loss → wrong tokens
**Symptom**: Model generates plausible but slightly wrong text compared
to FP32 reference
**Root cause**: FP16 logits lose precision in closely-ranked tokens
**Fix**: Always cast logits to FP32 before marking as output

---

## 11. Experimental Results

Validated on Qwen3-0.6B (28 layers, 1024 hidden, 16 heads) on GB300:

| Metric | FP32 | FP16 |
|--------|------|------|
| Engine size | 3.1 GB | 1.6 GB (**48% reduction**) |
| Weight memory | 3260 MB | 1630 MB (**50% reduction**) |
| KV cache per layer | 64 * 2048 * 4 = 512 KB | 64 * 2048 * 2 = 256 KB (**50% reduction**) |
| Output quality | "Paris. The capital of Italy is Rome." | "Paris. The capital of Italy is Rome." (**identical**) |
| Build time | 184s | 146s (**21% faster**) |

FP32 E2E parity confirmed via unified test harness comparing TRT output
against HuggingFace Transformers reference (logit cosine similarity > 0.99).

---

## 12. Checklist: Adding FP16 to a New Builder

When extending FP16 support to a new model family or builder:

- [ ] Accept `precision: str = "fp32"` parameter
- [ ] Compute `work_np_dtype` and `work_trt_dtype` from precision
- [ ] Set KV cache / state inputs to `work_trt_dtype`
- [ ] Set attention mask input to `trt.float32` (cast to work dtype if needed)
- [ ] Pass `dtype=work_np_dtype` to ALL `graph_ops` and `graph_blocks` calls
- [ ] Pass `precision` to `checkpoint_mapper.load_standard_weights()`
- [ ] Verify norm functions use FP32 precision boundaries
- [ ] Verify softmax uses FP32 precision boundary
- [ ] Cast logits/final output to FP32 before `mark_output`
- [ ] Write `"precision": precision` to bundle config
- [ ] Add E2E test manifest with `"precision": "fp16"`
- [ ] Verify output quality matches FP32 (text should be coherent, not degenerate)

---

## References

- TensorRT 10.15 Accuracy Considerations: Strongly-typed networks, ICastLayer
- TensorRT Operators Reference: Quantize/Dequantize, Cast, Reduce
- Experimental validation: Qwen3-0.6B on GB300, commit 7fecea9
- Failure case: Magpie TTS disabled FP16 due to LayerNorm overflow
  (`families/magpie_tts.py:638-640`)
