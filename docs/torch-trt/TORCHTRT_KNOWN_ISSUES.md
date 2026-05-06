# Torch-TRT Known Issues & Workarounds

Living registry of issues encountered during model transforms. **Read this
before starting a new transform. Add new entries when you discover issues.**

---

## Quick reference table

| # | Symptom | Root Cause | Severity |
|---|---------|-----------|----------|
| 1 | NaN output from attention layers | SDPA + attention mask in TRT | Critical |
| 2 | `torch.baddbmm` compilation error | TRT can't compile baddbmm | Critical |
| 3 | Image ignores text prompt | Multiplicative masking before softmax | Critical |
| 4 | Engine ignores mask input at runtime | Uniform trace inputs constant-folded | Critical |
| 5 | 2D mask misinterpreted after additive conversion | Model's internal 2D→3D conversion | Medium |
| 6 | `run_decompositions()` AssertionError | Stateful modules (cache buffer mutations) | Medium |
| 7 | Missing EOS token in T5 output | `tokenizer_add_special_tokens=0` | Medium |
| 8 | CFG produces identical cond/uncond predictions | Wrong null-text encoding | Medium |
| 9 | `StaticLayer.update` breaks TRT graph | In-place `index_copy_` not supported | Medium |
| 10 | All-zero logits from decoder engine | Missing `use_explicit_typing=True` | Critical |

---

## Detailed entries

### 1. SDPA with attention masks produces NaN in TRT

**Symptom:** DiT/transformer engine outputs all NaN. Happens even with
all-zeros mask (which should be a no-op).

**Root cause:** `F.scaled_dot_product_attention`'s fused CUDA kernel cannot
handle the `attn_mask` parameter after torch_tensorrt compilation. This is
structural — not a mask value issue.

**Affected models:** Any model using SDPA with attention masks (diffusion
cross-attention, encoder self-attention with padding).

**Fix:** Replace SDPA with explicit basic ops:
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) * (head_dim ** -0.5)
scores = scores + additive_mask  # {0, -10000}
probs = scores.softmax(dim=-1)
out = torch.matmul(probs, V)
```

For diffusers models, set a custom attention processor:
```python
model.set_attn_processor(TrtSafeAttnProcessor())
```

**Reference:** `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/diffusion.py` —
`_TrtSafeAttnProcessor` class.

**Discovered:** 2026-03-24, PixArt-Sigma transform.

---

### 2. `torch.baddbmm` compilation failure

**Symptom:** `UnboundLocalError: cannot access local variable
'interpreter_result'` during torch_tensorrt compilation.

**Root cause:** `torch.baddbmm` (batched add + batched matrix multiply) is
used by diffusers' classic `AttnProcessor`. torch_tensorrt cannot compile it.

**Fix:** Same as issue #1 — use explicit matmul + add instead.

**Discovered:** 2026-03-24, PixArt-Sigma transform.

---

### 3. Multiplicative masking causes prompt non-adherence

**Symptom:** Diffusion model produces high-quality images that completely
ignore the text prompt. Two different prompts produce visually similar images.
`|conditioned_pred - unconditioned_pred|` is very small (0.001-0.006).

**Root cause:** Multiplicative masking (`embeddings * mask`) zeros padding
embeddings, but `softmax(Q @ K^T)` still distributes weight to those positions
because `exp(0) = 1`. With 110 padding tokens and 10 real tokens, ~74% of
attention weight goes to padding, drowning out the text signal.

**Fix:** Use additive masking. Convert binary mask {0, 1} to additive bias
{-10000, 0} and add to attention logits before softmax:
```python
additive_mask = (1.0 - binary_mask) * (-10000.0)
attn_scores = attn_scores + additive_mask
```

**Diagnostic:** Track `|conditioned_pred - unconditioned_pred|` per denoising
step. Should be 0.002-0.013+. If < 0.001, masking is likely wrong.

**Reference:** `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/diffusion.py` —
`PixArtDiTWrapper.forward()`.

**Discovered:** 2026-03-24, PixArt-Sigma transform.

---

### 4. Uniform trace inputs get constant-folded

**Symptom:** Engine ignores a particular input at runtime — changing its value
has no effect on output.

**Root cause:** `torch.export` traces the computation graph using the provided
example inputs. If an input is uniform (all-ones, all-zeros), the export may
constant-fold operations on it, hardcoding the result into the graph.

**Fix:** Use non-trivial patterns for any input that varies at runtime:
```python
# Attention mask: half real, half padding
mask = torch.ones(1, seq_len, dtype=dtype)
mask[0, seq_len // 2:] = 0.0

# Random data for latents (not all-zeros)
latent = torch.randn(1, channels, h, w, dtype=dtype)
```

**Discovered:** 2026-03-22, PixArt-Sigma transform.

---

### 5. 3D vs 2D mask shape confusion

**Symptom:** Additive mask values are re-interpreted as binary by the model's
internal mask conversion logic, producing wrong attention patterns.

**Root cause:** Many HuggingFace models check `if mask.ndim == 2` and convert
2D masks to 3D/4D with their own logic. If you pass a 2D additive mask
(values like -10000), the model treats them as binary flags and re-converts.

**Fix:** Pre-expand the mask to 3D `[B, 1, seq_len]` to bypass the model's
2D→3D conversion:
```python
additive_mask = additive_mask.unsqueeze(1)  # [B, 1, seq_len]
```

**Discovered:** 2026-03-24, PixArt-Sigma transform.

---

### 6. `run_decompositions()` assertion on stateful modules

**Symptom:** `AssertionError` during `torch_tensorrt.dynamo.compile()` when
the exported program contains cache buffer mutations.

**Root cause:** `exported_program.run_decompositions()` (called internally by
torch_tensorrt) fails on stateful modules where buffers are mutated in-place
(e.g., StaticCache key/value updates).

**Fix:** Patch `run_decompositions()` in `compiler.py` to catch this specific
assertion and return the program unchanged. The non-decomposed graph compiles
correctly via TRT.

**Reference:** `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py` —
`_patch_run_decompositions()`.

**Discovered:** 2026-03-06, Qwen3-0.6B transform.

---

### 7. Missing EOS token with `tokenizer_add_special_tokens=0`

**Symptom:** T5 text encoder produces degraded embeddings. Downstream model
output quality is lower than HuggingFace reference.

**Root cause:** Bundle's `tokenizer_add_special_tokens=0` causes
`HfPythonTokenizer` to call `encode(add_special_tokens=False)`, omitting the
EOS token. T5 was trained to expect EOS (token_id=1) at the end of every input.

**Fix:** Append EOS explicitly in the pipeline if missing:
```cpp
if (input_ids.empty() || input_ids.back() != 1) {
    input_ids.push_back(1);  // T5 EOS token
}
```

**Discovered:** 2026-03-23, PixArt-Sigma transform.

---

### 8. CFG produces identical conditioned/unconditioned predictions

**Symptom:** Classifier-free guidance has no effect. `|cond - uncond|` ≈ 0.
Image quality is okay but prompt is ignored.

**Root cause:** Null-text (unconditional) embedding was encoded incorrectly —
e.g., encoding an empty string with all-zeros mask, or using a different
sequence length.

**Fix:** Match HuggingFace's unconditional encoding exactly. For T5-based
models: encode just the EOS token `[1]` with an all-ones attention mask:
```python
null_ids = [1]  # Just EOS
run_t5_encoder(null_ids, null_text, zero_padding=False)  # all-ones mask
```

The `zero_padding=False` flag causes all positions to get mask=1, matching
HF's behavior for empty/unconditional text.

**Discovered:** 2026-03-23, PixArt-Sigma transform.

---

### 9. `StaticLayer.update` in-place mutation breaks TRT

**Symptom:** `torch.export` or TRT compilation fails with errors about
in-place operations or graph mutations.

**Root cause:** HuggingFace's `StaticCache` uses `index_copy_` (in-place
scatter) in `StaticLayer.update()`. TRT requires functional (out-of-place)
operations.

**Fix:** Patch `StaticLayer.update` to use functional `torch.scatter`:
```python
def patch_static_cache_scatter():
    with _patch_lock:
        if getattr(StaticLayer.update, '_scatter_patched', False):
            return
        original_update = StaticLayer.update
        def patched_update(self, key_states, value_states, cache_position, ...):
            # Use torch.scatter instead of index_copy_
            ...
        StaticLayer.update = patched_update
        patched_update._scatter_patched = True
```

**Reference:** `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/decoder.py` —
`patch_static_cache_scatter()`.

**Discovered:** 2026-03-07, Qwen3-0.6B transform.

---

### 10. All-zero logits without `use_explicit_typing=True`

**Symptom:** Decoder engine produces all-zero or near-zero logits. Top-1 token
is always the same regardless of input.

**Root cause:** `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine()`
without `use_explicit_typing=True` applies its own precision conversion that
destroys fp16 model weights.

**Fix:** Always pass `use_explicit_typing=True`:
```python
engine_bytes = convert_exported_program_to_serialized_trt_engine(
    exported,
    inputs=cuda_inputs,
    use_explicit_typing=True,
    min_block_size=1,
)
```

**Discovered:** 2026-03-06, Qwen3-0.6B transform.

---

## Adding a new entry

When you encounter a new issue, add it following this template:

```markdown
### N+1. Short title

**Symptom:** What you observed (error message, wrong output, etc.)

**Root cause:** Why it happened (be specific about the mechanism).

**Fix:** How to fix it (include code snippet if applicable).

**Reference:** File and function where the fix lives.

**Discovered:** Date, which model transform.
```

Also update the quick reference table at the top.
