# Architecture Overview

## Document Control

| Field | Value |
|-------|-------|
| Document ID | ARCH-001 |
| Version | 2.0 |
| Status | NORMATIVE |
| Classification | ISO 26262-6 Section 7 -- Software Architectural Design |
| Author | Safety Architecture Team (yifeif@nvidia.com) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |
| Last Updated | 2026-03-12 |
| Supersedes | ARCH-001 v1.0 (aspirational architecture document) |

This document describes the software architecture of the trt-transformers-cpp system as implemented in the codebase. All file paths referenced in this document have been verified to exist. Aspirational or planned changes are explicitly marked as **PLANNED** in Section 11.

---

## 1. Scope and Purpose

This document defines the software architectural design for the trt-transformers-cpp system: a two-phase inference platform that converts HuggingFace models into optimized TensorRT bundles (Python build phase) and runs autoregressive or single-pass inference from those bundles (C++ runtime phase).

The scope covers:

- The Python builder package (`trtf_build/`) and its plugin-based family architecture.
- The C++ runtime and its centralized factory-based pipeline dispatch.
- The `.trtfb` bundle format that bridges the two phases.
- Core runtime abstractions: `TrtModule`, `KvCache`, `RecurrentState`.
- All 15 concrete pipeline implementations and 19 runtime strategies (including `text_to_text` for T5-style encoder-decoder models).

---

## 2. System Architecture

The system operates in two strictly separated phases:

| Phase | Language | Entry Point | Input | Output |
|-------|----------|-------------|-------|--------|
| **Build** | Python | `trtf-build build` / `trtf_build.build()` | HF repo ID or local directory | `.trtfb` bundle |
| **Run** | C++ | `trtf run` / `trtf::load()` / C ABI | `.trtfb` bundle | Task-specific results |

The bundle is the sole interface between the two phases. The C++ runtime never reads HuggingFace model directories directly. All model-specific architectural decisions (attention type, normalization, activation functions, weight layout) are baked into the TRT engine plan at build time.

```
  HuggingFace Model
        |
        v
  [Python Builder]  -- plugin dispatch, graph construction, engine compilation
        |
        v
    .trtfb Bundle   -- self-describing: engine plan(s) + tokenizer + config JSON
        |
        v
   [C++ Runtime]    -- factory dispatch, pipeline assembly, autoregressive loop
        |
        v
  Task-Specific Output (text, audio, image, segmentation, embedding)
```

---

## 3. Python Builder Architecture

The Python builder is a fully plugin-based system. Adding a new model family requires only a single Python file with zero edits to shared code.

### 3.1 Package Structure

- **Package root**: `trtf_build/trtf_build/`
- **Entry points**: `cli.py` (CLI), `__init__.py` (Python API), `__main__.py`

### 3.2 Orchestration Flow

The orchestrator in `trtf_build/trtf_build/engine_builder.py` executes:

1. **Resolve model** -- download from HuggingFace or use local directory.
2. **Parse config** -- `config.py` reads `config.json` into `ModelConfig`.
3. **Find plugin** -- `families/__init__.py` calls `find_plugin(model_type)`.
4. **Load weights** -- plugin's `load_weights()` calls `checkpoint_mapper.py` to load safetensors into a `WeightDict`, applying family-specific transforms.
5. **Build engine(s)** -- plugin's `build_engine()` constructs TRT networks. For VL models, `build_vision_engine()` builds a second engine. For diffusion models, `build_components()` builds text encoder(s), denoiser, and VAE decoder.
6. **Write bundle** -- `bundle_writer.py` packages engine plan(s), tokenizer files, and config JSON into a `.trtfb` file.

### 3.3 Family Plugin Protocol

Defined in `trtf_build/trtf_build/families/base.py` as a Python `Protocol`:

```python
class FamilyPlugin(Protocol):
    name: str
    def matches(self, model_type: str) -> bool: ...
    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict: ...
    def build_engine(self, config: ModelConfig, weights: WeightDict,
                     max_cache_length: int, *, verbose: bool = False) -> bytes: ...
    # Optional: build_vision_engine(), get_vl_config(),
    #           build_components(), get_diffusion_config()
```

### 3.4 Plugin Auto-Discovery

`trtf_build/trtf_build/families/__init__.py` uses `pkgutil.iter_modules()` to scan all `.py` files in the families directory. Any module exposing a `plugin` attribute is registered automatically. There are currently 56+ Python files in the families directory (54+ plugins + `base.py` protocol + `__init__.py` auto-discovery module). New plugins are added continuously via the autopilot system (`scripts/autopilot/autorun.py`).

Key discovery functions:
- `find_plugin(model_type)` -- matches standard models by HF `model_type`.
- `find_diffusion_plugin(pipeline_class)` -- matches diffusion models by diffusers pipeline class name.

### 3.5 Three-Layer Graph Construction

The TRT graph building is layered:

| Layer | File | Responsibility |
|-------|------|----------------|
| 1. Atomic ops | `trtf_build/trtf_build/graph_ops.py` | Tensor-in/tensor-out ops: RoPE, RMSNorm, attention, ALiBi, conv, etc. |
| 2. Composable blocks | `trtf_build/trtf_build/graph_blocks.py` | Multi-op blocks: SwiGLU MLP, GELU MLP, attention block, apply_norm |
| 3. Standard decoder | `trtf_build/trtf_build/standard_decoder_builder.py` | Full decoder engine: embedding, N transformer layers, LM head |

Most decoder families call into `standard_decoder_builder.py`. Specialized architectures (Mamba, Whisper, Bark, diffusion) build custom graphs in their plugin or dedicated builder modules.

### 3.6 Specialized Builders

Beyond the standard decoder, the build package contains dedicated engine builders:

| Builder | File | Purpose |
|---------|------|---------|
| Vision (Qwen VL) | `qwen_vl_vision_builder.py` | Qwen2.5-VL (3D RoPE) and Qwen3-VL (DeepStack) vision encoders |
| Vision (InternViT) | `internvit_vision_builder.py` | InternVL vision encoder |
| Vision (ONNX) | `onnx_vision_builder.py` | ONNX-traced vision encoders |
| Vision (Phi4) | `phi4mm_vision_builder.py` | Phi-4 multimodal vision encoder |
| Vision (CLIP) | `clip_encoder_builder.py` | CLIP text/image encoder |
| Encoder | `encoder_builder.py` | BERT/encoder-only models |
| Encoder (Mistral) | `mistral_encoder_builder.py` | Mistral encoder for embedding/reranking |
| Encoder (Qwen3) | `qwen3_encoder_builder.py` | Qwen3 text encoder |
| T5 Encoder | `t5_encoder_builder.py` | T5 text encoder for diffusion |
| Causal VAE 3D | `causal_vae_3d_builder.py` | Wan2.1 causal 3D VAE decoder |
| VAE 2D | `vae_2d_builder.py` | 2D VAE decoder |
| FLUX DiT | `flux_dit_builder.py` | FLUX denoiser (DiT) |
| FLUX2 DiT | `flux2_dit_builder.py` | FLUX.2 denoiser variant |
| FLUX VAE | `flux_vae_builder.py` | FLUX VAE decoder |
| Z-Image DiT | `z_image_dit_builder.py` | Z-Image denoiser (DiT) |
| Standard DiT | `standard_dit_builder.py` | Generic DiT builder |
| EnCodec | `encodec_builder.py` | EnCodec audio codec |
| NanoCodec | `nanocodec_builder.py` | NanoCodec audio codec |

### 3.7 Debug Runner

`trtf_build/trtf_build/debug_runner.py` provides pure-Python TRT inference runners that mirror C++ runtime behavior:

- `TrtRunner` -- standard decoder with device-resident KV cache.
- `MambaTrtRunner` -- SSM with device-resident conv + SSM state.
- `VLTrtRunner` -- vision encoder + text decoder with image preprocessing.

These are used by diff-testing tools (`tools/diff_logits.py`, `tools/diff_layers.py`, `tools/diff_vl.py`) and the E2E test harness.

Additionally, `trtf_build/trtf_build/diffusion_runner.py` provides a `DiffusionRunner` for pure-Python TRT inference of diffusion models (text encoding, denoising loop, VAE decode), following the same pattern as the decoder `TrtRunner`.

### 3.8 Scheduler Package

`trtf_build/trtf_build/schedulers/` contains Python diffusion noise schedulers used during build-time validation and debug runner inference. Currently implements `flow_match_euler.py` (Flow Matching Euler Discrete), matching the C++ `FlowMatchEulerScheduler`.

---

## 4. C++ Runtime Architecture

### 4.1 Public API

The public API consists of three headers:

| Header | Contents |
|--------|----------|
| `include/trtf/pipeline.h` | `IPipeline` (abstract base with 14 virtual methods), result types (`TextResult`, `ImageResult`, `AudioResult`, `EmbeddingResult`, `SegmentResult`, `TextEmbedding`), `GenerateConfig`, factory function `trtf::load()`, C ABI functions |
| `include/trtf/bundle.h` | `BundleInfo` struct, `InspectBundle()`, `IsBundle()` |
| `include/trtf/tokenizer.h` | `ITokenizer` interface, factory functions for `VocabTokenizer`, `HfPythonTokenizer`, `IpaTokenizer` |

`IPipeline` defines 14 virtual methods spanning all modalities:

- `generate()` (text-only and text+image overloads)
- `encode_text()`, `generate_image()` (two overloads)
- `generate_audio()`, `transcribe()`, `speak()`
- `embed()`, `rerank()`, `segment()`, `encode()`
- `solve()`, `detect()`

Each method has a default implementation that throws `std::runtime_error`, so pipeline classes only override the methods they support.

### 4.2 C ABI Entry

- **File**: `src/cabi/api/trtf_c.cpp` (~108 LOC)
- Exposes `trtf_create_pipeline_ex()`, `trtf_last_error()`, `trtf_version()`, `trtf_has_trt()`.
- Delegates immediately to `PipelineFactory::from_bundle()`.

### 4.3 Pipeline Factory (Centralized Dispatch)

- **Header**: `include/trtf/runtime/pipeline_factory.h`
- **Implementation**: `src/runtime/pipeline_factory.cpp` (~700 LOC)

The factory is the single entry point for creating pipelines from bundles. It performs:

1. `ReadBundleFile()` -- parse the `.trtfb` file.
2. `find_bundle_sections()` -- locate engine plans, config, tokenizer data.
3. `parse_fast_path_config()` -- parse config JSON into `FastPathModelConfig`.
4. `resolve_family()` -- map `runtime_strategy` string to a `StrategyFamily` enum.
5. `dispatch_pipeline()` -- call the appropriate per-family factory function.

The `StrategyFamily` enum groups 18 runtime strategies into 5 families:

```cpp
enum class StrategyFamily { kText, kEncoder, kVision, kAudio, kDiffusion, kUnknown };
```

Each family has a dedicated factory function:

| Family | Factory Function | Strategies Handled |
|--------|------------------|--------------------|
| `kText` | `create_text_pipeline()` | `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention` |
| `kEncoder` | `create_encoder_pipeline()` | `encoder_only`, `embedding`, `reranking`, `neural_operator` |
| `kVision` | `create_vision_pipeline()` | `vision_language`, `segmentation`, `prompted_segmentation`, `object_detection` |
| `kDiffusion` | `create_diffusion_pipeline()` | `diffusion` |
| `kAudio` | `create_audio_pipeline()` | `speech_to_text`, `text_to_audio`, `speech_to_speech`, `omni_multimodal` |

### 4.4 Configuration

- **Header**: `src/cabi/config/fast_path_config.h`
- **Implementation**: `src/cabi/config/fast_path_config.cpp`

`FastPathModelConfig` is a monolithic struct with ~181 fields spanning all modalities. Fields are grouped by comments indicating which `runtime_strategy` uses them, but there is no compile-time enforcement of which fields apply to which strategy.

### 4.5 Bundle Helpers

- **Header**: `src/cabi/bundle/bundle_helpers.h`
- **Implementation**: `src/cabi/bundle/bundle_helpers.cpp`

Provides `BundleSections` (pointers into parsed bundle data for engine plans, tokenizer files, config JSON, preprocessor weights) and tokenizer extraction utilities.

---

## 5. Bundle Format

- **Header**: `src/bundle/bundle_format.h`
- **Implementation**: `src/bundle/bundle_format.cpp`
- **Writer**: `trtf_build/trtf_build/bundle_writer.py`

### 5.1 Binary Layout

```
Bytes 0-7:    Magic "TRTFB\x00\x01\x00"
Bytes 8-15:   uint64_t json_header_length (little-endian)
Bytes 16..N:  JSON metadata header (UTF-8)
Bytes N..EOF: Binary sections referenced by offset in the header
```

### 5.2 Sections

The JSON header declares named sections with byte offsets and sizes. Common sections include:

| Section | Present In | Contents |
|---------|-----------|----------|
| `engine_plan` | All | Primary TRT engine plan bytes |
| `config.json` | All | Model config (runtime_strategy, dimensions, tokenizer settings) |
| `tokenizer_dir` | Most | HF tokenizer files (tokenizer.json, vocab, merges) |
| `vision_engine_plan` | VL, SAM | Vision encoder TRT engine plan |
| `denoiser_plan` | Diffusion | DiT/UNet TRT engine plan |
| `vae_decoder_plan` | Diffusion | VAE decoder TRT engine plan |
| `text_encoder_N` | Diffusion | Text encoder TRT engine plan(s) |
| `preprocessor_weights` | Diffusion | Preprocessing weight tensors (timestep embedder, patchify, etc.) |
| `preprocessor_config` | VL | Image preprocessing configuration |
| `semantic_engine_plan` | Bark | Semantic model engine |
| `coarse_engine_plan` | Bark | Coarse acoustic model engine |
| `fine_engine_plan` | Bark | Fine acoustic model engine |
| `codec_engine_plan` | Bark | EnCodec decoder engine |
| `talker_engine_plan` | Omni | Talker decoder engine |
| `code2wav_engine_plan` | Omni | Code-to-waveform engine |

### 5.3 Self-Describing Config

The bundle's `config.json` section carries all build-time decisions. The C++ runtime reads `runtime_strategy` to select the pipeline type, `max_cache_length` for cache sizing, `tokenizer_add_special_tokens` for tokenizer behavior, and modality-specific fields (vision dimensions, audio parameters, diffusion scheduler config). No external configuration files are needed at runtime.

---

## 6. Runtime Strategy Dispatch

### 6.1 Complete Dispatch Flow

```
trtf::load(bundle_path)
  -> PipelineFactory::from_bundle()
    -> ReadBundleFile()                      [src/bundle/bundle_format.cpp]
    -> find_bundle_sections()                [src/cabi/bundle/bundle_helpers.cpp]
    -> parse_fast_path_config()              [src/cabi/config/fast_path_config.cpp]
    -> resolve_family(runtime_strategy)      [src/runtime/pipeline_factory.cpp]
    -> dispatch_pipeline(family, ...)        [src/runtime/pipeline_factory.cpp]
      -> create_{text,encoder,vision,diffusion,audio}_pipeline()
        -> load_trt_module_from_plan()       -- deserialize engine, create TrtModule
        -> create_tokenizer_from_bundle()    -- extract and create ITokenizer
        -> create KvCache / RecurrentState   -- if applicable
        -> construct concrete Pipeline class
  -> return unique_ptr<IPipeline>
```

### 6.2 Runtime Strategy to Pipeline Class Mapping

| `runtime_strategy` | `StrategyFamily` | Pipeline Class | State Management |
|--------------------|------------------|----------------|------------------|
| `decoder_kv_cache` | kText | `TextGenerationPipeline` | `KvCache` |
| `decoder_moe` | kText | `TextGenerationPipeline` | `KvCache` |
| `ssm_recurrent` | kText | `RecurrentPipeline` | `RecurrentState` (conv + ssm) |
| `rwkv_recurrent` | kText | `RecurrentPipeline` | `RecurrentState` (5 state vectors) |
| `hybrid_mamba_attention` | kText | `RecurrentPipeline` | `KvCache` + `RecurrentState` (hybrid) |
| `encoder_only` | kEncoder | `EncoderPipeline` | None (single pass) |
| `embedding` | kEncoder | `EncoderPipeline` | None (single pass) |
| `reranking` | kEncoder | `EncoderPipeline` | None (single pass) |
| `neural_operator` | kEncoder | `EncoderPipeline` | None (single pass) |
| `vision_language` | kVision | `VLPipeline` | `KvCache` + vision `TrtModule` |
| `segmentation` | kVision | `SegmentPipeline` | None (single pass) |
| `prompted_segmentation` | kVision | `SamPipeline` | None (two-pass: encoder + decoder) |
| `object_detection` | kVision | (routed to encoder) | None (single pass) |
| `diffusion` (wan_3d) | kDiffusion | `WanPipeline` | None (iterative denoising) |
| `diffusion` (flux_2d) | kDiffusion | `FluxPipeline` | None (iterative denoising) |
| `diffusion` (z_image_2d) | kDiffusion | `ZImagePipeline` | None (iterative denoising) |
| `text_to_text` | kText | `T5Pipeline` | `KvCache` + encoder `TrtModule` |
| `speech_to_text` | kAudio | `WhisperPipeline` | Legacy `WhisperBackend` |
| `text_to_audio` (bark) | kAudio | `BarkPipeline` | Legacy `BarkBackend` |
| `text_to_audio` (magpie) | kAudio | `MagpiePipeline` | Legacy `MagpieTTSBackend` |
| `speech_to_speech` | kAudio | `SpeechPipeline` | Legacy `SpeechToSpeechBackend` |
| `omni_multimodal` | kAudio | `OmniPipeline` | `TrtModule` + `KvCache` (new runtime) |

Note: Diffusion sub-dispatch is based on the `diffusion_backend_type` field in `FastPathModelConfig`, not on `runtime_strategy` alone. Audio sub-dispatch for `text_to_audio` uses the `is_magpie_tts` boolean flag.

---

## 7. Concrete Pipeline Implementations

All 14 pipeline classes implement `IPipeline` and reside in `src/runtime/pipelines/`:

### 7.1 Text Generation

| Class | Header | Composition |
|-------|--------|-------------|
| `TextGenerationPipeline` | `text_generation_pipeline.h` | `TrtModule` (decoder) + `KvCache` + `ITokenizer` |
| `RecurrentPipeline` | `recurrent_pipeline.h` | `TrtModule` + `IStateManager` + `ITokenizer` |

`TextGenerationPipeline` serves all standard decoder and MoE models. The model-specific architecture (GQA, RoPE, SwiGLU, etc.) is baked into the TRT engine.

`RecurrentPipeline` uses an `IStateManager` abstraction (defined in `recurrent_pipeline.h`) with two implementations:
- `RecurrentStateManager` -- wraps `RecurrentState` for pure-recurrent models (Mamba, RWKV). Does not produce attention masks.
- `HybridStateManager` -- wraps both `KvCache` + `RecurrentState` for hybrid Mamba-attention models. Produces attention masks via the KvCache.

### 7.2 Vision-Language

| Class | Header | Composition |
|-------|--------|-------------|
| `VLPipeline` | `vl_pipeline.h` | `TrtModule` (text decoder) + `TrtModule` (vision encoder, optional) + `KvCache` + `ImagePreprocessor` + `ITokenizer` |

### 7.3 Encoder / Perception

| Class | Header | Composition |
|-------|--------|-------------|
| `EncoderPipeline` | `encoder_pipeline.h` | `TrtModule` + `ITokenizer`; mode string selects behavior |
| `SegmentPipeline` | `encoder_pipeline.h` | `TrtModule` (single-pass segmentation) |
| `SamPipeline` | `encoder_pipeline.h` | `TrtModule` (image encoder) + `TrtModule` (mask decoder) |

### 7.4 Diffusion

| Class | Header | Composition |
|-------|--------|-------------|
| `WanPipeline` | `diffusion_pipeline.h` | `TrtModule` (T5 encoder) + `TrtModule` (denoiser) + `TrtModule` (VAE) |
| `FluxPipeline` | `diffusion_pipeline.h` | `TrtModule`(s) (T5 + CLIP) + `TrtModule` (denoiser) + `TrtModule` (VAE) |
| `ZImagePipeline` | `diffusion_pipeline.h` | `TrtModule` (text encoder) + `TrtModule` (denoiser) + `TrtModule` (VAE) |

### 7.5 Audio

| Class | Header | Composition |
|-------|--------|-------------|
| `WhisperPipeline` | `audio_pipeline.h` | Legacy `WhisperBackend` + mel filterbank + `ITokenizer` |
| `BarkPipeline` | `audio_pipeline.h` | Legacy `BarkBackend` + `ITokenizer` |
| `MagpiePipeline` | `audio_pipeline.h` | Legacy `MagpieTTSBackend` + `ITokenizer` |
| `SpeechPipeline` | `audio_pipeline.h` | Legacy `SpeechToSpeechBackend` |
| `OmniPipeline` | `audio_pipeline.h` | `TrtModule` (thinker) + `KvCache` + `TrtModule` (talker) + `KvCache` + `TrtModule` (code2wav) + `ITokenizer` |

Note: Whisper, Bark, Magpie, and Speech pipelines delegate to legacy backend classes in `src/runtime/trt/audio/`. `OmniPipeline` is fully migrated to the `TrtModule` + `KvCache` composition pattern.

---

## 8. Core Abstraction Inventory

### 8.1 TrtModule

- **Header**: `include/trtf/runtime/trt_module.h`
- **Implementation**: `src/runtime/trt/core/trt_module.cpp`

The `model.forward()` abstraction for TensorRT engines. Wraps an engine + execution context. Provides:

- `forward(TensorMap)` -- CPU-to-GPU-to-CPU synchronous execution.
- `forward_device(DeviceTensorMap)` -- GPU-only execution, no host transfers.
- `forward_async()` / `sync()` -- asynchronous execution.
- `bind_external()` -- allows `KvCache` to bind cache device pointers directly.
- `device_ptr()` -- direct device buffer access.
- `keep_alive()` -- opaque resource ownership (engine, stream lifetime).

### 8.2 KvCache

- **Header**: `include/trtf/runtime/kv_cache.h`
- **Implementation**: `src/runtime/trt/core/kv_cache.cpp`

Autoregressive KV cache state manager. HF equivalent: `DynamicCache` / `past_key_values`. Manages:

- Per-layer K/V device tensors of shape `[max_length, kv_dim]`.
- Position tracking and causal attention mask construction.
- `bind_to(TrtModule)` -- binds `cache_k_{i}`, `cache_v_{i}` (inputs) and `present_k_{i}`, `present_v_{i}` (outputs).
- `advance()` -- copies present K/V into cache, advances position.
- `reset()` -- zeros all buffers for a new sequence.

### 8.3 RecurrentState

- **Header**: `include/trtf/runtime/recurrent_state.h`
- **Implementation**: `src/runtime/trt/core/recurrent_state.cpp`

Generic config-driven SSM/RWKV state manager. Replaces the original separate `MambaStepState` and `RwkvStepState` with a single class parameterized by `TensorSpec` vectors:

- Mamba: `{{"conv_state", {d_inner*conv_kernel}}, {"ssm_state", {state_size*d_inner}}}`
- RWKV: 5 state vectors per layer: `attn_state`, `ff_state`, `num_state`, `den_state`, `max_state`.

### 8.4 IScheduler

- **Header**: `include/trtf/runtime/scheduler.h`
- **Implementation**: `src/runtime/trt/core/flow_match_euler_scheduler.cpp`

Diffusion noise scheduler interface. HF equivalent: `SchedulerMixin` / `FlowMatchEulerDiscreteScheduler`. Provides:

- `set_timesteps(num_steps)` -- configure the timestep schedule.
- `timesteps()` / `sigmas()` -- access the schedule.
- `step()` -- single scheduler step: update latents in-place.

One concrete implementation: `FlowMatchEulerScheduler` (used by FLUX, Wan, Z-Image). Factory function `create_scheduler(name, shift)` instantiates by name.

### 8.5 ITokenizer

- **Header**: `include/trtf/tokenizer.h`
- **Implementations**: `src/tokenizer/`

Three concrete implementations:

| Class | File | Mechanism |
|-------|------|-----------|
| `VocabTokenizer` | `vocab_tokenizer.cpp` | Word-to-ID lookup from vocabulary list |
| `HfPythonTokenizer` | `hf_python_tokenizer.cpp` | Subprocess bridge to HuggingFace tokenizers |
| `IpaTokenizer` | `ipa_tokenizer.cpp` | Phoneme tokenizer for MagpieTTS |

Users should include `trtf/tokenizer.h` (the public API). `trtf/runtime/tokenizer_interface.h` is the abstract interface definition re-exported by `tokenizer.h`.

---

## 9. Backend Executor Organization

Backend executor code lives in `src/runtime/trt/` organized by modality:

| Directory | Contents |
|-----------|----------|
| `src/runtime/trt/core/` | Shared infrastructure: `TrtModule`, `KvCache`, `RecurrentState`, `DeviceTensor`, TRT common utilities, decode runtime (argmax, mask building), engine lifecycle |
| `src/runtime/trt/audio/` | Whisper, Bark, MagpieTTS, Speech-to-Speech, Omni backends and supporting types |
| `src/runtime/trt/diffusion/` | Diffusion denoising step seam, generation plan, scheduler helpers, preprocessor weights, math utilities |
| `src/runtime/trt/encoder/` | Encoder, embedding, and reranking backends |
| `src/runtime/trt/multimodal/` | VL backend, vision engine, image preprocessor (4 strategies: `qwen_merge_group`, `simple_chw`, `center_crop_chw`, `aspect_preserve_chw`) |
| `src/runtime/trt/perception/` | Segmentation, SAM, detection, neural operator backends |
| `src/runtime/trt/recurrent/` | Mamba, RWKV, hybrid backends and their decode runtimes and step states |

---

## 10. Known Architectural Debt

### 10.1 FastPathModelConfig God Struct

`FastPathModelConfig` in `src/cabi/config/fast_path_config.h` is a monolithic struct with ~181 fields spanning all modalities (text, vision, audio, diffusion, segmentation, detection, neural operators). Every field is always present regardless of runtime strategy. There is no compile-time or type-system enforcement of which fields are valid for a given strategy.

**Impact**: Any new modality adds more fields to this single struct. Readers must rely on comments to know which fields apply to which strategy.

### 10.2 Centralized Pipeline Factory

`src/runtime/pipeline_factory.cpp` (~700 LOC) is a centralized dispatch function containing all pipeline construction logic. Adding a new strategy family requires editing this single file.

**Impact**: The factory function grows linearly with the number of strategies. Factory logic for unrelated modalities (text, audio, diffusion) lives in a single compilation unit.

### 10.3 Legacy Audio Backends

Whisper, Bark, Magpie, and Speech pipelines delegate to legacy backend classes (`WhisperBackend`, `BarkBackend`, `MagpieTTSBackend`, `SpeechToSpeechBackend` in `src/runtime/trt/audio/`) that predate the `TrtModule` + `KvCache` composition pattern. `OmniPipeline` is the only audio pipeline that has been migrated to the new pattern.

**Impact**: The legacy backends duplicate patterns (engine loading, cache management, decode loops) that are now handled generically by `TrtModule` and `KvCache`.

### 10.4 Perception Backends vs Pipeline Classes

The perception backends (`DetectionBackend`, `NeuralOperatorBackend`, `SegmentationBackend`, `SamBackend` in `src/runtime/trt/perception/`) coexist with the pipeline-level classes (`SegmentPipeline`, `SamPipeline` in `src/runtime/pipelines/encoder_pipeline.h`). The relationship between backend and pipeline is not always clear. `EncoderPipeline` handles `object_detection` and `neural_operator` strategies but the corresponding dedicated pipeline classes do not exist at the `IPipeline` level.

---

## 11. Planned Evolution

**STATUS: PLANNED -- not yet implemented in the codebase.**

The following architectural changes are under consideration:

### 11.1 Plugin Registry for C++ Runtime

Replace the centralized `pipeline_factory.cpp` dispatch with a strategy-builder registry pattern where each strategy family registers a builder plugin. This would allow adding new strategies without modifying a central file.

### 11.2 FastPathModelConfig Decomposition

Split `FastPathModelConfig` into a base config plus per-strategy config structs (e.g., `DiffusionConfig` already exists at the pipeline level; similar typed configs could be introduced for audio, vision, and perception).

### 11.3 Legacy Audio Migration

Migrate remaining audio backends (Whisper, Bark, Magpie, Speech) to the `TrtModule` + `KvCache` composition pattern, following the `OmniPipeline` precedent.

### 11.4 Service-Oriented Runtime

Decompose pipeline implementations into service interfaces (`ITextService`, `IAudioService`, etc.) behind a `PipelineRouter` that translates `IPipeline` calls into service requests. This would separate task orchestration from TRT execution.

---

## Appendix A: File Path Reference

All paths below are relative to the repository root and have been verified to exist.

### Python Builder
| Path | Purpose |
|------|---------|
| `trtf_build/trtf_build/engine_builder.py` | Build orchestrator |
| `trtf_build/trtf_build/config.py` | HF config.json parser |
| `trtf_build/trtf_build/checkpoint_mapper.py` | Safetensors weight loader |
| `trtf_build/trtf_build/bundle_writer.py` | Bundle file writer |
| `trtf_build/trtf_build/graph_ops.py` | Atomic TRT graph ops |
| `trtf_build/trtf_build/graph_blocks.py` | Composable graph blocks |
| `trtf_build/trtf_build/standard_decoder_builder.py` | Standard decoder engine builder |
| `trtf_build/trtf_build/debug_runner.py` | Python TRT inference runners (decoder, Mamba, VL) |
| `trtf_build/trtf_build/diffusion_runner.py` | Python TRT diffusion runner |
| `trtf_build/trtf_build/pipeline.py` | Subprocess wrapper around C++ trtf binary |
| `trtf_build/trtf_build/schedulers/` | Python diffusion schedulers (flow_match_euler) |
| `trtf_build/trtf_build/families/__init__.py` | Plugin auto-discovery |
| `trtf_build/trtf_build/families/base.py` | FamilyPlugin protocol |
| `trtf_build/trtf_build/cli.py` | CLI entry point |

### C++ Runtime -- Public API
| Path | Purpose |
|------|---------|
| `include/trtf/pipeline.h` | IPipeline interface, result types, C ABI |
| `include/trtf/bundle.h` | BundleInfo, InspectBundle |
| `include/trtf/tokenizer.h` | ITokenizer interface |
| `include/trtf/runtime/trt_module.h` | TrtModule abstraction |
| `include/trtf/runtime/kv_cache.h` | KvCache state manager |
| `include/trtf/runtime/recurrent_state.h` | RecurrentState manager |
| `include/trtf/runtime/scheduler.h` | IScheduler interface, FlowMatchEulerScheduler |
| `include/trtf/runtime/tensor.h` | Tensor, TensorMap, TensorInfo types |
| `include/trtf/runtime/device_tensor.h` | DeviceTensor, DeviceTensorMap types |
| `include/trtf/runtime/pipeline_factory.h` | PipelineFactory |
| `include/trtf/runtime/tokenizer_interface.h` | ITokenizer abstract interface (re-exported by `tokenizer.h`) |
| `include/trtf/runtime/trt/audio/speech_decode_stop_policy.h` | Speech decode stop policy for audio pipelines |
| `include/trtf/runtime/trt/audio/subprocess_runner.h` | Subprocess runner utility for tokenizer bridge |
| `include/trtf/runtime/trt/multimodal/image_transform_helper.h` | Image transformation utilities for VL preprocessing |

### C++ Runtime -- Implementation
| Path | Purpose |
|------|---------|
| `src/cabi/api/trtf_c.cpp` | C ABI entry point |
| `src/cabi/config/fast_path_config.h` | FastPathModelConfig struct |
| `src/cabi/config/fast_path_config.cpp` | Config JSON parser |
| `src/cabi/bundle/bundle_helpers.h` | BundleSections, tokenizer extraction |
| `src/cabi/bundle/bundle_helpers.cpp` | Bundle section implementation |
| `src/bundle/bundle_format.h` | Bundle magic, section types |
| `src/bundle/bundle_format.cpp` | Bundle reader |
| `src/runtime/pipeline_factory.cpp` | Centralized factory dispatch |
| `src/runtime/trt/core/trt_module.cpp` | TrtModule implementation |
| `src/runtime/trt/core/kv_cache.cpp` | KvCache implementation |
| `src/runtime/trt/core/recurrent_state.cpp` | RecurrentState implementation |
| `src/runtime/trt/core/flow_match_euler_scheduler.cpp` | FlowMatchEulerScheduler implementation |
| `src/runtime/pipelines/text_generation_pipeline.h` | TextGenerationPipeline |
| `src/runtime/pipelines/recurrent_pipeline.h` | RecurrentPipeline, IStateManager |
| `src/runtime/pipelines/vl_pipeline.h` | VLPipeline |
| `src/runtime/pipelines/encoder_pipeline.h` | EncoderPipeline, SegmentPipeline, SamPipeline |
| `src/runtime/pipelines/diffusion_pipeline.h` | FluxPipeline, WanPipeline, ZImagePipeline |
| `src/runtime/pipelines/audio_pipeline.h` | WhisperPipeline, BarkPipeline, MagpiePipeline, SpeechPipeline, OmniPipeline |
| `src/tokenizer/vocab_tokenizer.cpp` | VocabTokenizer |
| `src/tokenizer/hf_python_tokenizer.cpp` | HfPythonTokenizer |
| `src/tokenizer/ipa_tokenizer.cpp` | IpaTokenizer |
