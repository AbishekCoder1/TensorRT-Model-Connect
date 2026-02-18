# Architecture Extensibility Assessment

Status of non-standard architecture support. MoE, Mamba/SSM, and vision-language (Qwen-VL) are fully implemented. DeepSeek MLA and hybrid SSM+Attention are planned.

## Executive Summary

With the Python build / C++ runtime split, adding new model families is a **Python-only task** for most architectures. The Python `trtf_build/` package provides a plugin system for family-specific checkpoint mappers and graph builders, while the C++ runtime handles bundle loading and inference with strategy-based dispatch.

As of 2026-02-16, MoE, Mamba/SSM, and vision-language support are **fully implemented**. The standard decoder builder is parameterized to support LayerNorm, GELU, learned positions, and multiple activations. The VL image preprocessor supports 4 strategies with configurable interpolation.

| Architecture Class | Current Support | Effort to Add New Instance | Where Changes Needed |
|---|---|---|---|
| Standard decoder (RMSNorm + RoPE + SwiGLU) | **Works today** (7 families) | ~30 LOC Python | Python plugin only |
| Extended decoder (LayerNorm, GELU, learned positions) | **Works today** (12 families) | ~60 LOC Python | Python plugin only |
| MoE decoder (top-k softmax / SparseMixer routing) | **Works today** (2 families) | ~300 LOC Python | Python graph builder + checkpoint mapper |
| SSM / Mamba | **Works today** (Mamba 130M-2.8B) | ~400 LOC Python | Python graph builder (C++ backend exists) |
| Vision-Language | **Works today** (Qwen-VL) | ~200 LOC Python | Python vision builder + plugin VL config |
| Multi-Latent Attention -- MLA (DeepSeek-V2/V3) | **Not yet implemented** | ~400 LOC Python + C++ | Python graph builder + C++ KV cache shape |
| Hybrid SSM+Attention (Jamba) | **Not yet implemented** | ~500 LOC Python + C++ | Python + C++ hybrid state |

---

## What Is Easy (Python-Only Changes)

### Adding a standard dense decoder family

Create a Python plugin file in `trtf_build/trtf_build/families/` with a checkpoint mapper. Uses the parameterized standard decoder builder. ~30-60 LOC.

**Implemented**: Qwen, LLaMA, Mistral, Gemma, Phi, Granite, InternLM (standard decoder); StarCoder2, GPT-2, OPT, Falcon, StableLM, OLMo, XGLM, GPT-NeoX, GPT-Neo, CodeGen, BLOOM, Nemotron (extended decoder).

### Adding a new MoE family

Write a Python graph builder for the expert routing logic. The C++ runtime uses the same KV-cache backend (routing is handled in the TRT graph). ~300 LOC.

**Implemented**: Phi-MoE (SparseMixer routing), Mixtral (standard top-2 softmax routing). Both use `runtime_strategy="decoder_moe"`.

### Adding a new Mamba/SSM family

Write a Python graph builder for the SSM architecture. The C++ `MambaBackend` and `MambaStepState` are already implemented for the `ssm_recurrent` runtime strategy. ~400 LOC Python.

**Implemented**: Mamba (130M-2.8B, selective scan + conv1d).

### Adding a new Vision-Language family

Write a Python plugin with `build_vision_engine()` for the vision encoder and `get_vl_config()` to specify preprocessing parameters (preprocessor_type, interpolation, prompt template, etc.). The C++ runtime handles 4 preprocessing strategies out of the box. ~200 LOC Python.

**Implemented**: Qwen-VL (Qwen2.5-VL, ViT + 3D RoPE + spatial merge, `runtime_strategy="vision_language"`).

---

## What Requires C++ Changes

### Different state management (done for Mamba/SSM)

The C++ runtime now supports multiple state backends via `runtime_strategy` dispatch in `trtf_c.cpp`:
- `decoder_kv_cache` / `decoder_moe` -> `TrtBackendFastPath` + `DeviceKvCache`
- `ssm_recurrent` -> `MambaBackend` + `MambaStepState`

New state types (e.g., for hybrid architectures) would need:
1. A new `IStepState` implementation in C++
2. A new backend class implementing `IGenerationBackend`
3. A new `runtime_strategy` value and dispatch branch in `trtf_c.cpp`

### Different KV cache shapes (DeepSeek MLA)

Compressed KV caches (e.g., `[cache_len, kv_lora_rank]` instead of `[cache_len, attention_size]`) need changes to `DeviceKvCache` in C++ to support variable cache sizes per layer.

---

## Architecture-Specific Deep Dives

### MoE (Phi-MoE -- IMPLEMENTED)

**Status**: Fully implemented. Phi-MoE plugin with SparseMixer routing.

**Python** (`families/phi_moe.py`):
- Checkpoint mapper: Maps router weights + per-expert gate/up/down projections
- Custom graph builder: SparseMixer routing (independent masked softmax, not standard top-k), per-expert SwiGLU MLPs with gather/scatter dispatch
- LayerNorm with bias, separate Q/K/V/O with biases

**C++**: No changes. Uses `runtime_strategy="decoder_moe"` which dispatches to the same `TrtBackendFastPath` (routing is handled entirely in the TRT graph).

### Mamba / SSM (IMPLEMENTED)

**Status**: Fully implemented. Mamba plugin with selective scan + C++ MambaBackend.

**Python** (`families/mamba.py`):
- Checkpoint mapper: Maps SSM weights (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj)
- Custom graph builder: Selective scan, causal conv1d with cached state, input-dependent discretization
- Engine I/O: token_id + per-layer conv_state/ssm_state inputs, logits + present_conv/present_ssm outputs

**C++** (new files):
- `MambaStepState` (`mamba_step_state.h/cpp`): conv_state + ssm_state per layer (constant memory)
- `MambaStepEngine` + `run_mamba_step()` (`mamba_decode_runtime.h/cpp`)
- `MambaBackend` (`mamba_backend.h/cpp`): autoregressive loop without prefill
- `runtime_strategy="ssm_recurrent"` dispatch in `trtf_c.cpp`

**Debug runner**: `MambaTrtRunner` in `debug_runner.py` for pure-Python Mamba TRT inference.

### Vision-Language -- VL (Qwen-VL -- IMPLEMENTED)

**Status**: Fully implemented. Qwen-VL plugin with vision encoder + text decoder.

**Python** (`families/qwen_vl.py`):
- Checkpoint mapper: Standard Qwen weights for text decoder + vision-specific weights (visual.* prefix)
- Vision engine builder: ViT with 3D RoPE + spatial merge (via `qwen_vl_vision_builder.py`)
- Text decoder: Standard Qwen2.5 with `embed_input=True` for VL prefill
- `get_vl_config()` returns preprocessor_type, interpolation, prompt template, token config

**C++** (`image_preprocessor.h/cpp`):
- 4 image preprocessing strategies: `qwen_merge_group`, `simple_chw`, `center_crop_chw`, `aspect_preserve_chw`
- Configurable interpolation: `bicubic` (default), `bilinear`, `nearest`
- Config parsed from bundle's `config.json` + `preprocessor_config.json`
- `format_vl_prompt()` for prompt template expansion

**Python debug runner** (`debug_runner.py`):
- `VisionTrtRunner`, `VLTrtRunner` for pure-Python VL inference
- `preprocess_image_for_trt()` dispatches to 4 strategies matching C++
- `_resolve_pil_interpolation()` maps mode strings to PIL constants
- Single-image constraint enforced via `NotImplementedError` for multi-image input

**Adding a new VL family**: Create a plugin with `build_vision_engine()` and `get_vl_config()` methods. The `preprocessor_type` and `interpolation` fields in `get_vl_config()` control C++ image preprocessing. See `families/qwen_vl.py` for an example.

### DeepSeek MLA (Multi-Latent Attention)

**Python changes**:
- Checkpoint mapper: Compressed Q/KV projections (`w_dq`, `w_uk`, `w_uv`, `kv_lora_rank`)
- Graph builder: Decomposed attention with low-rank KV

**C++ changes**:
- `DeviceKvCache` generalization for different cache shapes per layer

**Estimated work**: ~300 LOC Python + ~100 LOC C++.

### Hybrid SSM+Attention (Jamba, Zamba)

**Python + C++ changes**: Combination of Mamba and attention approaches.
- Per-layer type: some layers attention (KV cache), some Mamba (recurrent state)
- `HybridStepState` that manages both state types

**Estimated work**: ~500 LOC Python + ~300 LOC C++.

---

## Recommended Approach for New Families

### Tier 1: Python-only, standard builder (implemented for 19 families)
Standard and extended decoders using the parameterized graph builder:
- Already done: Qwen, LLaMA, Mistral, Gemma, Phi, Granite, InternLM, StarCoder2, GPT-2, OPT, Falcon, StableLM, OLMo, XGLM, GPT-NeoX, GPT-Neo, CodeGen, BLOOM, Nemotron
- Candidates: Yi (use llama), Baichuan, DeepSeek-dense, CodeLlama (use llama), Vicuna (use llama)
- ~30-60 LOC each, fully parallelizable

### Tier 2: Python custom graph builder (implemented for 4 families)
Non-standard graph topologies with existing C++ backends:
- Already done: Phi-MoE (MoE, Python only), Mixtral (MoE, Python only), Mamba (SSM, Python + existing C++ backend), Qwen-VL (VL, Python + existing C++ image preprocessor)
- Candidates: Other Mamba variants, LLaVA/InternVL (can reuse simple_chw/aspect_preserve_chw preprocessor)
- ~200-400 LOC each

### Tier 3: Python + new C++ backend (not yet needed)
Fundamentally different architectures requiring new C++ state management:
- DeepSeek MLA (Python + C++ cache shape, ~400 LOC total)
- Jamba hybrid SSM+Attention (Python + C++ hybrid state, ~500 LOC total)
- RWKV (Python + C++ recurrent state)

Each tier can be worked on independently. Tier 1 and 2 families are fully parallelizable since they only touch Python plugins with no shared file edits.
