# Research: HuggingFace Model Coverage & 70B Feasibility Audit

## Context

Audit of supported HuggingFace models across all 22 family plugins, and feasibility analysis for running 70B+ parameter models with better GPU hardware.

---

## Part 1: Supported Model Families & HuggingFace Models

### 22 Family Plugins — Full Model Catalog

| # | Plugin | HF `model_type` Match | Notable HF Models | Param Sizes |
|---|--------|----------------------|-------------------|-------------|
| 1 | **Qwen** | `qwen*`, `qwq*` (excl. VL) | Qwen3-0.6B/1.7B/4B/8B/14B/32B, Qwen2-72B, Qwen2.5-72B, QwQ | 0.6B–72B |
| 2 | **LLaMA** | `llama*` | Llama-3.1-8B/70B/405B, Llama-3.3-70B, Llama-2-7B/13B/70B | 7B–**405B** |
| 3 | **Mistral** | `mistral*` | Mistral-7B-v0.1/v0.3, Ministral-3B/8B/14B | 3B–14B |
| 4 | **Gemma** | `gemma*` | Gemma-3-1B/4B/12B/27B, Gemma-2-2B/9B/27B | 1B–27B |
| 5 | **Phi** | `phi*` (excl. phimoe) | Phi-3-mini-4k (3.8B), Phi-3-small (7B), Phi-3-medium (14B) | 3.8B–14B |
| 6 | **Phi-MoE** | `phimoe` | Phi-3.5-MoE (16×3.8B, 6.6B active) | 6.6B active |
| 7 | **Granite** | `granite*` | Granite-4.0-Nano (350M), Granite Code | 350M–8B |
| 8 | **InternLM** | `internlm*` | InternLM2.5-1.8B/7B/20B, InternLM3-8B | 1.8B–20B |
| 9 | **StarCoder2** | `starcoder2` | StarCoder2-3B/7B/15B | 3B–15B |
| 10 | **GPT-2** | `gpt2` | GPT-2 (124M/345M/774M/1.5B) | 124M–1.5B |
| 11 | **OPT** | `opt` | OPT-125M/350M/1.3B/6.7B/13B/30B/66B | 125M–66B |
| 12 | **Falcon** | `falcon*` | Falcon-7B/40B/180B, Falcon-3-1B/3B/7B/10B | 1B–**180B** |
| 13 | **StableLM** | `stablelm*` | StableLM-2-1.6B/12B | 1.6B–12B |
| 14 | **Mamba** | `mamba` | Mamba-130M/370M/790M/1.4B/2.8B, Mamba-2 variants | 130M–2.8B |
| 15 | **Qwen2.5-VL** | `qwen*` + `vl` | Qwen2.5-VL-3B/7B/72B | 3B–72B |
| 16 | **OLMo** | `olmo` | OLMo-3-7B/32B | 7B–32B |
| 17 | **Nemotron** | `nemotron` | Nemotron-4-340B, Nemotron-3-8B | 8B–**340B** |
| 18 | **XGLM** | `xglm` | XGLM-564M/1.7B/4.5B/7.5B | 564M–7.5B |
| 19 | **GPT-NeoX** | `gpt_neox*` | GPT-NeoX-20B, Pythia-70M to 12B | 70M–20B |
| 20 | **GPT-Neo** | `gpt_neo` | GPT-Neo-125M/1.3B/2.7B | 125M–2.7B |
| 21 | **CodeGen** | `codegen` | CodeGen-350M/2B/6B/16B | 350M–16B |
| 22 | **BLOOM** | `bloom` | BLOOM-560M/1.1B/3B/7.1B/176B | 560M–**176B** |
| 23 | **Mixtral** | `mixtral` | Mixtral-8×7B (45B total), Mixtral-8×22B | 45B–**176B total** |

### Models with 70B+ Variants Already Supported by Plugin

| Model | Plugin | Parameters | Notes |
|-------|--------|-----------|-------|
| **Llama-3.1-70B** | llama | 70B | Direct support |
| **Llama-3.1-405B** | llama | 405B | Largest open dense model |
| **Llama-3.3-70B** | llama | 70B | Latest 70B |
| **Qwen2-72B** | qwen | 72B | Direct support |
| **Qwen2.5-72B** | qwen | 72B | Direct support |
| **Qwen2.5-VL-72B** | qwen_vl | 72B | Vision-language |
| **Falcon-180B** | falcon | 180B | Very large |
| **BLOOM-176B** | bloom | 176B | ALiBi attention |
| **Nemotron-4-340B** | nemotron | 340B | Largest supported |
| **OPT-66B** | opt | 66B | Near 70B |
| **Mixtral-8×22B** | mixtral | ~176B total (44B active) | MoE |

**Bottom line: The plugin layer already handles 70B+ architectures. Zero plugin changes needed.**

---

## Part 2: 70B Model Feasibility — Codebase Audit

### Blocking Issues Found

#### 1. **KV Cache is CPU fp32** (CRITICAL)
**File:** `src/runtime/trt/kv_cache_step_state.cpp:17-20`
```cpp
mCacheK.assign(num_layers, std::vector<float>(cache_elems, 0.0F));
mCacheV.assign(num_layers, std::vector<float>(cache_elems, 0.0F));
```
- KV cache lives in **CPU RAM** as `std::vector<float>` (fp32)
- For Llama-3.1-70B (80 layers, 8192 hidden, 4096 cache): `4096 × 8192 × 4 bytes × 2 (K+V) × 80 layers = ~20 GB CPU RAM` just for KV cache
- Each step copies full KV cache to/from GPU via `cudaMemcpy`
- This will dominate both memory and latency for 70B models

#### 2. **max_cache_length Hard Cap at 4096** (CRITICAL)
**File:** `src/cabi/fast_path_config.cpp:53-56`
```cpp
if (cfg.max_cache_length > 4096) {
    cfg.max_cache_length = 4096;
}
```
- Llama-3.1-70B has `max_position_embeddings = 131072`
- This cap silently reduces it to 4096 — no warning logged
- Fix: trivial 1-line change (raise or remove the cap), but must consider memory implications

#### 3. **TRT Workspace Limit: 1 GB** (HIGH RISK)
**File:** `trtf_build/trtf_build/standard_decoder_builder.py:104`
```python
trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
```
- TRT uses workspace for attention score matrices, MLP intermediates during engine building
- 70B model: attention matrix alone is `num_heads × seq_len × seq_len × 4 bytes` per layer
- Likely fails at build time. Fix: increase to 4–8 GB (`1 << 32` or `1 << 33`)

#### 4. **All Inference is fp32** (MAJOR MEMORY CONSTRAINT)
**Files:** `standard_decoder_builder.py`, `kv_cache_step_state.cpp`
- `trt_config.clear_flag(trt.BuilderFlag.TF32)` — disables even TF32
- All weights stored as `np.float32` in checkpoint_mapper
- Engine inputs/outputs all `trt.float32`
- A 70B model in fp32 needs ~280 GB for weights alone
- **fp16 or bf16 is essential for 70B** — currently not supported

#### 5. **Single-Token Autoregressive Loop** (PERFORMANCE)
**File:** `src/runtime/trt/trt_backend_shared.cpp`
- Engine inputs are `(1,)` — single token per step
- No batch dimension, no speculative decoding, no continuous batching
- Generates 1 token → copies full KV cache CPU↔GPU → generates next token
- For 70B, the per-step KV copy overhead will be severe

### Non-Blocking Observations

| Component | Status | Notes |
|-----------|--------|-------|
| int32 parameters | OK | All config fields (vocab_size, num_layers, etc.) fit in int32 |
| Python weight loading | OK | safetensors streams from disk; doesn't need full model in RAM at once |
| Bundle format | OK | No size limits in .trtfb format |
| Plugin layer | OK | All 70B model architectures already have plugins |
| TRT engine building | Needs GPU VRAM | 70B build needs ~80 GB+ VRAM (A100-80GB or H100) |

---

## Part 3: Effort Estimate for 70B Support

### Tier 1: Minimal Changes (make it run, slowly) — ~2-3 days

These changes let you build and run a 70B model in fp32, assuming enough RAM/VRAM:

| Change | File | Effort |
|--------|------|--------|
| Raise `max_cache_length` cap | `fast_path_config.cpp:55` | 1 line |
| Increase workspace to 8 GB | `standard_decoder_builder.py:104` | 1 line |
| Add `--max-cache-length` passthrough to C++ CLI | `trtf_c.cpp` | Already exists (override) |
| Test with small cache (256) on A100-80GB | E2E validation | ~1 day |

**Hardware requirement:** A100-80GB or H100-80GB for engine building. ~120 GB CPU RAM for fp32 KV cache.

### Tier 2: fp16 Inference Support (essential for practical use) — ~1-2 weeks

| Change | Files | Effort |
|--------|-------|--------|
| Add FP16 builder flag | `standard_decoder_builder.py` | ~50 lines |
| Convert KV cache to fp16 (half) | `kv_cache_step_state.cpp`, `trt_decode_runtime.cpp` | ~200 lines |
| fp16 weight loading in checkpoint_mapper | `checkpoint_mapper.py` | ~30 lines |
| Update debug_runner.py for fp16 parity | `debug_runner.py` | ~50 lines |
| Regression testing on all existing models | E2E suite | ~2 days |

**Impact:** Cuts memory in half. 70B KV cache: ~10 GB instead of ~20 GB. Engine size: ~140 GB VRAM instead of ~280 GB.

### Tier 3: GPU-Resident KV Cache (performance) — ~2-3 weeks

| Change | Files | Effort |
|--------|-------|--------|
| Allocate KV cache on GPU (CUDA) | New `gpu_kv_cache.cpp` | ~400 lines |
| Eliminate per-step CPU↔GPU memcpy | `trt_decode_runtime.cpp` | ~200 lines |
| CUDA memory management for cache | `trt_common.cpp` | ~100 lines |
| Update IStepState interface | `step_state.h` | Minor |

**Impact:** Eliminates the biggest latency bottleneck. Required for practical 70B inference.

### Tier 4: Tensor Parallelism (for models > 1 GPU) — ~4-6 weeks

Required for 405B or models that don't fit on a single GPU. Major architectural change — out of scope for initial 70B support if you have an 80GB GPU.

---

## Part 4: Recommended GPU Hardware for 70B

| GPU | VRAM | Can Build 70B? | Can Run 70B (fp32)? | Can Run 70B (fp16)? |
|-----|------|---------------|---------------------|---------------------|
| RTX 4090 | 24 GB | No | No | No |
| A100-40GB | 40 GB | No | No | Tight (with small cache) |
| A100-80GB | 80 GB | Yes (tight) | No | Yes (with ~4k cache) |
| H100-80GB | 80 GB | Yes | No | Yes (with ~4k cache) |
| H200-141GB | 141 GB | Yes | Tight | Yes (comfortably) |
| 2× H100 (TP) | 160 GB | Yes | Yes (needs Tier 4) | Yes (needs Tier 4) |

**Minimum viable for 70B: H100-80GB with fp16 support (Tier 2) and small cache length (~2048-4096).**

---

## Part 5: Summary & Recommendation

### What works today with zero code changes
- All 22 family plugins handle 70B architectures at the **plugin/weight-mapping level**
- Models up to ~13B can build and run on current 24GB GPU hardware in fp32
- Up to ~30B might work on A100-80GB with small cache

### What's needed for practical 70B
1. **Tier 1 (trivial):** Raise hard caps — 1 day
2. **Tier 2 (essential):** fp16 inference — 1-2 weeks, most impactful change
3. **Tier 3 (important):** GPU-resident KV cache — 2-3 weeks for usable latency

### Verification plan
After each tier, run the standard regression gate:
- Tier 1-2 unit tests (existing models stay correct in fp32, new fp16 path validates)
- Tier 3 E2E with Qwen3-0.6B, then LLaMA-3.1-8B, then target 70B model
- Runner parity check (`test_runner_parity.py`) for any decode runtime changes
