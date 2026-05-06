# Torch-TRT Debug Lessons

Lessons learned from debugging model transforms in the torch-trt pipeline.
Reference for future model integrations.

---

## Debug methods (ranked by effectiveness)

### 1. Python-side TRT engine probing

Write a standalone Python script that loads engines directly from the bundle
and runs them outside the C++ pipeline. This isolates whether a problem is in
the engines or in the pipeline orchestration (fp16 conversion, mask building,
scheduler, VAE, etc.).

**Example:** `scripts/diagnose_loop.py` runs the full 20-step diffusion
denoising loop in Python using TRT engines extracted from a `.trtfb` bundle.
A two-prompt divergence test (tracking cosine similarity between latents at
each step) proved the engines were correct and narrowed the bug to input
preparation.

**When to use:** Whenever C++ pipeline output is wrong but you're unsure
which stage is at fault. If Python produces the same wrong result, the bug
is in the engine/model wrapping. If Python works but C++ doesn't, the bug is
in the C++ pipeline code.

### 2. Quantitative signal metrics over visual inspection

For diffusion models, track `|conditioned_prediction - unconditioned_prediction|`
(mean absolute difference) at each denoising step. This directly measures how
much the text prompt influences the output — CFG amplifies this difference, so
if it's near zero, the prompt has no effect regardless of how the image looks.

For decoder models, compare logit cosine similarity and top-k overlap between
TRT and HF reference at each generation step.

**When to use:** Always. A single scalar metric often reveals the problem
faster than staring at output images or text. Establish the metric before
iterating on fixes.

### 3. Reasoning about math before empirical debugging

Some bugs are invisible to print-debugging because the values "look fine" in
isolation. The multiplicative masking bug was only diagnosable by working
through the softmax math: with 110 padding tokens at exp(0)=1 and 10 real
tokens at exp(score)~1-3, most probability mass lands on padding. No amount
of logging intermediate tensors would have surfaced this — the tensors were
all finite and reasonable-looking.

**When to use:** When outputs are wrong but all intermediate values look
plausible. Step through the mathematical operations on paper with concrete
small examples.

### 4. Versioned bundle trail for A/B testing

Keep old bundle versions (v4, v5, v6, ...) during iterative debugging. When a
new version regresses (e.g., v7 produced all NaN), immediately re-run the
previous version to confirm the regression is isolated to your change. This
prevents chasing phantom bugs.

---

## Masking pitfalls

### Never use multiplicative masking before softmax

Zeroing out values before softmax does not suppress them — `exp(0) = 1` gives
padding tokens a baseline attention weight. With 110 padding tokens and 10 real
tokens, ~74% of attention weight goes to padding.

**Correct approach:** Additive masking with large negative values (-10000 or
-inf) before softmax. `exp(-10000) ≈ 0`, so padding gets exactly zero weight.

This applies to any model with variable-length inputs and padding tokens
(text encoders, cross-attention in diffusion, encoder-decoder models).

### Multiplicative masking on embeddings is also insufficient

Multiplying `encoder_hidden_states * mask` zeros the embedding values, but
cross-attention Q@K^T still produces finite scores for zero-valued keys.
Softmax then distributes weight to those positions. The only reliable
suppression is additive bias on the attention logits themselves.

---

## Torch-TRT compilation gotchas

### SDPA and fused attention kernels fail with masks

`F.scaled_dot_product_attention` produces NaN when compiled by torch_tensorrt
with attention mask inputs. This is structural — even an all-zeros mask (which
should be a no-op) causes NaN. The fused CUDA kernel that SDPA dispatches to
cannot handle the mask parameter after TRT compilation.

`torch.baddbmm` (used by diffusers' classic `AttnProcessor`) also fails —
torch_tensorrt cannot compile it.

**Solution:** Use explicit basic ops that TRT compiles reliably:
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
scores = scores + additive_mask
probs = scores.softmax(dim=-1)
out = torch.matmul(probs, V)
```

This is slower than fused attention but correct and compiles without issues.

**When to check:** Early in any new model integration. Do not assume
HuggingFace's default attention processor will survive TRT compilation. Test
with a mask input before building the full pipeline.

### Use mixed trace inputs during export

`torch.export` can constant-fold inputs that are uniform during tracing. If
the trace mask is all-ones, the exported graph may hardcode the mask and ignore
the runtime input entirely.

**Solution:** Use non-trivial patterns for any input that varies at runtime.
For attention masks, use a mixed pattern (e.g., half real tokens, half padding):
```python
mask = torch.ones(1, seq_len, dtype=compute_dtype)
mask[0, seq_len // 2:] = 0.0  # half real, half padding
```

### 3D vs 2D mask shapes matter

Many HuggingFace models have internal logic that converts 2D masks `[B, seq]`
to 3D or 4D. If you pre-convert to additive format, pass as 3D `[B, 1, seq]`
to bypass the model's own conversion (which may re-interpret your additive
values as binary).

---

## Diffusion-specific lessons

### Test prompt adherence, not just image quality

A diffusion model can produce beautiful, sharp, coherent images that have
nothing to do with the prompt. Always test with at least two semantically
distinct prompts and verify outputs are visibly different in the expected way.

**Minimum test set:** Two prompts with different subjects (e.g., "a dog" and
"a red car"). If both produce similar images, the text conditioning is broken
regardless of image quality.

### CFG null-text encoding must match HuggingFace

For classifier-free guidance, the unconditional embedding must be encoded the
same way HuggingFace does it. For T5-based models, this means encoding just
the EOS token (id=1) with an all-ones attention mask — not an empty sequence,
not all-zeros.

### EOS tokens may not be added automatically

When the bundle's `tokenizer_add_special_tokens=0`, the tokenizer's `encode()`
call omits special tokens including EOS. T5 was trained to expect EOS at the
end of every input. The pipeline must append it explicitly if missing:
```cpp
if (input_ids.empty() || input_ids.back() != 1) {
    input_ids.push_back(1);  // T5 EOS
}
```

---

## General process lessons

### Isolate components with Python before debugging C++

The C++ pipeline has many interacting parts. A Python script that exercises
one component (encoder, denoiser, VAE) in isolation eliminates variables and
is much faster to iterate on than rebuilding C++.

### Keep debug output behind a flag, not inline

Debug prints (embedding stats, per-step metrics) are invaluable during
investigation but must be removed before merging. Consider adding a
`--verbose` or `TRTMC_DEBUG_DIFFUSION` env var check rather than inline
`std::cerr` statements that need manual cleanup.
