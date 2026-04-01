# Magpie TTS Performance Optimization Worklog

## Goal
Optimize Magpie TTS E2E inference latency on GB300 without quality degradation.

## Platform
- GB300 (compute cap 10.3), TRT 10.15.1, CUDA kernels enabled

## Confirmed Baseline (FP32, CPU sampling) — 2026-03-12

Bundle: `/workspace/users/yifeif/trt-transformers/engines/magpie-tts-357m.trtfb`
Config: sampling (temperature=0.6, top_k=80), CFG scale=2.5, seed=42
Audio: `artifacts/magpie/fp32-baseline/` — **confirmed good quality**

| # | Prompt | Frames | TRT step | Sampling | Pipeline | RTF |
|---|--------|--------|----------|----------|----------|-----|
| 01 | sun | 208 | 2.29 ms | 1.02 ms | 965 ms | 0.100 |
| 02 | coffee | 253 | 2.04 ms | 1.00 ms | 1003 ms | 0.085 |
| 03 | door | 242 | 2.21 ms | 1.01 ms | 1053 ms | 0.094 |
| 04 | cats | 208 | 2.19 ms | 1.01 ms | 911 ms | 0.094 |
| 05 | park | 225 | 2.24 ms | 1.00 ms | 1008 ms | 0.096 |
| 06 | water | 210 | 2.25 ms | 1.00 ms | 960 ms | 0.098 |
| 07 | book | 195 | 2.31 ms | 1.00 ms | 923 ms | 0.102 |
| 08 | sky | 246 | 2.35 ms | 1.02 ms | 1124 ms | 0.098 |
| 09 | street | 245 | 2.16 ms | 1.01 ms | 1018 ms | 0.089 |
| 10 | battery | 222 | 2.23 ms | 1.00 ms | 995 ms | 0.096 |

**Averages: 2.23 ms/TRT step, 1.01 ms/sampling, 996 ms pipeline, RTF 0.095**

### Per-frame breakdown

| Component | Time/frame | % of frame |
|-----------|-----------|------------|
| TRT decoder step (cond + uncond CFG) | 2.23 ms | 69% |
| CPU top-k sampling (8 codebooks) | 1.01 ms | 31% |
| **Total per-frame** | **3.24 ms** | 100% |

### Fixed costs

| Stage | Time |
|-------|------|
| Encoder (×2 for CFG) | ~5 ms |
| Context prefill (110 frames × 2) | ~100 ms |
| Codec | ~15 ms |

## Ruled Out

### FP16 engines (BuilderFlag.FP16)
- **Result**: Audible quality degradation — accuracy loss accumulates over 200+ autoregressive frames
- All 10 prompts hit max frames (750) — EOS never triggered
- Confirmed bad by listening
- The original code comment was correct: "FP16 disabled — causes audio degradation"

### GPU sampling with device-side CFG blend
- **Result**: FP differences in GPU CFG interpolation kernel vs CPU blend cause generation trajectory divergence → bad audio
- Even 1 ULP difference in logit values cascades through autoregressive generation
- Any operation that changes the logit values (GPU embed, GPU CFG blend) breaks quality
- GPU sampling with CPU-identical logits (v8) produces exact frame count match and identical quality, but is slower than CPU sampling due to H2D/D2H overhead

## Optimization Targets (preserving FP32 parity)

The constraint: **logits must be computed identically to CPU baseline**. Only the
sampling step (after logits are finalized) can be optimized.

1. **Faster CPU top-k** (~0.5 ms/frame saved)
   - Current: `std::partial_sort` over 2024 elements × 8 codebooks = 1.01 ms
   - Option A: `std::nth_element` to find threshold + filter (O(n) vs O(n log k))
   - Option B: SIMD-accelerated partial sort (AVX2/NEON)
   - Option C: Pre-sorted index cache (exploit temporal locality in logit rankings)

2. **Reduce per-frame sync overhead in TRT step**
   - Current `run_decoder_step_device` does cudaStreamSynchronize every frame
   - Could pipeline: overlap frame N's sampling with frame N+1's TRT step

3. **Context prefill batching** (~50 ms one-time savings)
   - Currently 110 frames × 2 passes = 220 sequential TRT enqueues
   - Could batch if decoder supports variable-length input

4. **Encoder padding reduction** (minor)
   - Always pads to 2048 positions; most prompts are <200 tokens

## Decoder Deep-Dive (2026-03-12)

### Platform comparison

| Platform | Decoder/frame | TRT step | Sampling | RTF |
|----------|--------------|----------|----------|-----|
| GB300 | 3.24 ms | 2.23 ms | 1.01 ms | 0.095 |
| Orin | 20.57 ms | ~18 ms | ~1 ms | 0.479 |

Decoder is the bottleneck on both, especially Orin (less compute).

### Per-frame anatomy (CPU sampling path, CFG enabled)

```
1. bind_cross_kv()              — 12 layers × 2 setTensorAddress    [host API]
2. prepare_step()               — build attention mask               [host]
3. transfer_decoder_inputs()    — H2D: token, position, mask, embed  [async]
4. bind_decoder_tensors()       — bind ~60 tensor addresses          [host API]
5. TRT enqueueV3 (conditioned)  — 12-layer decoder forward           [GPU async]
6. D2D cache update             — copy present_k/v rows              [async]
7. D2H logits                   — 16192 floats = 64 KB              [async]
8. cudaStreamSynchronize        — BLOCK: wait for 5-7               [sync]
--- REPEAT steps 1-8 for UNCONDITIONAL pass (CFG) ---
9. bind_cross_kv_uncond()       — rebind for null-text encoder output
10. prepare/transfer/bind/enqueue/D2D/D2H/sync (uncond)
11. CPU CFG blend                — loop over 16192 logits
12. CPU top-k sampling           — 8 × partial_sort(2024, k=80)
```

### Key cost drivers

**1. Cross-attention over 2048 positions (most prompts <200 tokens)**
- Encoder pads to max_source_positions=2048, zeros padding
- Each decoder layer does cross-attention Q@K^T: [1, 128] × [2048, 128]^T
- Then softmax over 2048 + weighted sum with V: [1, 2048] × [2048, 128]
- 12 layers × 2 CFG passes = 24 cross-attention blocks per frame
- ~90% of cross-attention compute is wasted on zero-padding
- **Fix: build decoder with actual text length** (or smaller max_source_positions)

**2. Two full TRT passes per frame (CFG)**
- CFG requires conditioned + unconditional forward passes
- Each pass: full 12-layer self-attn + cross-attn + MLP
- Both use the SAME decoder engine but different KV caches + cross-KV
- **Fix: batch both passes in a single TRT call** (batch dim=2)

**3. cross_k/cross_v: 12 separate D2D copies of same data**
- `compute_cross_kv()` copies encoder_output to 12 separate buffers
- All 12 layers receive the IDENTICAL cross_k and cross_v
- The per-layer K/V projection is inside the TRT graph (good)
- But we copy 2048×768×4 = 6.3 MB × 24 = 150 MB of D2D copies at setup
- **Fix: share a single cross_kv buffer** (bind all layers to same address)

**4. Per-frame tensor rebinding**
- `bind_cross_kv()` does 25 setTensorAddress calls per frame (CFG rebind)
- `bind_decoder_tensors()` does ~60 calls per frame
- On Orin, TRT API overhead per call is higher
- **Fix: skip rebinding when addresses haven't changed**

### Optimization priority (decoder-focused)

| # | Optimization | Estimated saving | Platform impact |
|---|-------------|-----------------|-----------------|
| 1 | **Reduce max_source_positions** to actual text length | ~20-40% TRT step | Both (less compute) |
| 2 | **Batch CFG passes** (batch_size=2) | ~30% TRT step | Both (1 enqueue vs 2) |
| 3 | **Share cross-KV buffer** across layers | ~150 MB less D2D | Both (less memory) |
| 4 | **Skip redundant tensor rebinding** | ~1-2 ms/frame | Orin (API overhead) |
| 5 | **Reduce max_cache_length** for short prompts | ~10-20% TRT step | Both (smaller KV) |
