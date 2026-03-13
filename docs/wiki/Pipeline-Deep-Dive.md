# Pipeline Deep Dive

| Field | Value |
|-------|-------|
| Document ID | PDD-001 |
| ISO 26262-6 clause | 7.4.5 (Software unit design and implementation) |
| Applicable to | trt-transformers-cpp C++ runtime |
| Revision | 2.0 |
| Date | 2026-03-12 |
| Status | Living document -- reflects code as of this revision date |
| Author | Safety Architecture Team (yifeif@nvidia.com) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |

---

This page is a detailed walkthrough of the real pipeline creation and request
handling code.  Every class, function, enum, and file path referenced here
exists in the codebase.  There is no `PipelineRouter`, `PipelineServices`,
`BuildContext`, or `StrategyBuilder`.

---

## 1. Entry Point: `trtf_create_pipeline_ex()`

**File:** `src/cabi/api/trtf_c.cpp` (108 LOC)

This is the C ABI entry point. Its responsibilities are deliberately narrow:

1. Validate `bundle_path` is non-null and non-empty.
2. Call `trtf::IsBundle(path)` to verify the file has `.trtfb` magic bytes.
3. Extract `hf_python` from `TrtfPipelineOptions` if provided.
4. Delegate entirely to `trtf::PipelineFactory::from_bundle(path, hf_python)`.
5. On success: log timing, return `pipeline.release()` (raw pointer transfer).
6. On exception: store error in `thread_local g_last_error`, return `nullptr`.

The backward-compatible `trtf_create_pipeline(bundle_path, flags)` is a thin
wrapper that calls `trtf_create_pipeline_ex` with default options.

Additional C ABI functions:
- `trtf_last_error()` -- returns the thread-local error string.
- `trtf_version()` -- returns `TRTF_VERSION_STRING`.
- `trtf_has_trt()` -- returns 1 if compiled with TRT, 0 otherwise.

**Note:** `trtf_create_pipeline_ex` does NOT own any modality-specific logic.
All construction logic is in `pipeline_factory.cpp`.

---

## 2. Pipeline Factory: `PipelineFactory::from_bundle()`

**File:** `src/runtime/pipeline_factory.cpp` (~700 LOC)
**Header:** `include/trtf/runtime/pipeline_factory.h`

This single static method is the entire pipeline assembly path:

```text
PipelineFactory::from_bundle(bundle_path, hf_python)
  |
  +-> ReadBundleFile(bundle_path)             [src/bundle/bundle_format.cpp]
  |     Returns BundleFile{info, sections[]}
  |
  +-> find_bundle_sections(bundle)            [src/cabi/bundle/bundle_helpers.cpp]
  |     Returns BundleSections (non-owning pointers into section data)
  |
  +-> parse_bundle_config(sections, info)     [local in pipeline_factory.cpp]
  |     If config.json section exists:
  |       parse_fast_path_config(text, max_cache_length)  [src/cabi/config/fast_path_config.cpp]
  |     Otherwise: populate FastPathModelConfig from BundleInfo header fields
  |     Returns FastPathModelConfig
  |
  +-> resolve_family(cfg.runtime_strategy)    [local anonymous namespace]
  |     Returns StrategyFamily enum
  |
  +-> dispatch_pipeline(family, bundle, sections, cfg, strategy, path, hf_python)
        Switch on StrategyFamily, calls per-family factory function
        Returns unique_ptr<IPipeline>
```

A free function `trtf::load()` is also provided as a convenience alias:
```cpp
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python)
{
    return PipelineFactory::from_bundle(bundle_path, hf_python);
}
```

---

## 3. Strategy Families

**Defined in:** `src/runtime/pipeline_factory.cpp` (anonymous namespace)

```cpp
enum class StrategyFamily { kText, kEncoder, kVision, kAudio, kDiffusion, kUnknown };
```

The `resolve_family()` function maps `runtime_strategy` strings to families:

| StrategyFamily | runtime_strategy values |
|---------------|------------------------|
| `kText` | `decoder_kv_cache`, `decoder_moe`, `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention` |
| `kEncoder` | `encoder_only`, `embedding`, `reranking`, `neural_operator` |
| `kVision` | `vision_language`, `segmentation`, `prompted_segmentation`, `object_detection` |
| `kAudio` | `speech_to_text`, `text_to_audio`, `speech_to_speech`, `omni_multimodal` |
| `kDiffusion` | `diffusion` |

If `runtime_strategy` is empty (old bundles), it defaults to `"decoder_kv_cache"`.
If the strategy string is not in the map, `resolve_family` returns `kUnknown`
and `dispatch_pipeline` throws `std::runtime_error`.

---

## 4. Per-Family Pipeline Construction

### 4.1 Text Family (`create_text_pipeline`)

All text strategies share a common pattern:
1. `load_trt_module(sections)` -- deserialize the engine plan into a `TrtModule`.
2. `create_tokenizer_from_bundle(sections, hf_python)` -- extract tokenizer files to temp dir, create `HfPythonTokenizer`.

Then dispatch by strategy:

**`decoder_kv_cache` / `decoder_moe`** -> `create_decoder_pipeline()`:
- `compute_kv_dim(cfg)` -> KV dimension from config.
- `KvCache(num_layers, max_cache_length, kv_dim, stream)`.
- Returns `TextGenerationPipeline(decoder, cache, config, stream, tokenizer, model_id)`.

**`ssm_recurrent`** -> `create_recurrent_pipeline()`:
- `RecurrentState` with specs: `conv_state` [d_inner * conv_kernel], `ssm_state` [state_size * d_inner].
- `RecurrentStateManager(state)`.
- Returns `RecurrentPipeline(decoder, state_mgr, config, stream, "MambaPipeline", tokenizer, model_id)`.

**`rwkv_recurrent`** -> `create_recurrent_pipeline()`:
- `RecurrentState` with 5 specs per layer: `attn_state`, `ff_state`, `num_state`, `den_state`, `max_state` (each [hidden_size]).
- Returns `RecurrentPipeline(..., "RwkvPipeline", ...)`.

**`hybrid_mamba_attention`** -> `create_hybrid_pipeline()`:
- `KvCache` for attention layers (num_attention_layers).
- `RecurrentState` for Mamba layers (num_mamba_layers) with `conv_state` and `ssm_state`.
- `HybridStateManager(kv, ssm)` -- implements `IStateManager` by delegating to both.
- Returns `RecurrentPipeline(..., "HybridPipeline", ...)`.

**Key files:**
- `src/runtime/pipelines/text_generation_pipeline.h` -- `TextGenerationPipeline`
- `src/runtime/pipelines/recurrent_pipeline.h` -- `RecurrentPipeline`, `IStateManager`, `RecurrentStateManager`, `HybridStateManager`
- `include/trtf/runtime/kv_cache.h` -- `KvCache`
- `include/trtf/runtime/recurrent_state.h` -- `RecurrentState`

### 4.2 Vision Family (`create_vision_pipeline`)

Segmentation and prompted segmentation are routed to `create_encoder_pipeline()`.
For `vision_language`:

1. Load text decoder TrtModule from `plan_data`.
2. `try_load_trt_module_from_plan(vision_plan_data)` -- optional vision encoder (soft failure).
3. `KvCache` for text decoder.
4. `build_vl_preprocess_config(sections)` -- parse preprocessor_config.json + config.json.
5. Returns `VLPipeline(text_decoder, vision_encoder, cache, vlc, vl_preprocess, stream, tokenizer, model_id)`.

**Key files:**
- `src/runtime/pipelines/vl_pipeline.h` -- `VLPipeline`, `VLConfig`
- `src/runtime/trt/multimodal/image_preprocessor.h` -- `VLPreprocessConfig`, `parse_vl_preprocess_config()`
- `src/runtime/trt/multimodal/vision_engine.h` -- vision engine helpers

### 4.3 Diffusion Family (`create_diffusion_pipeline`)

1. `load_diffusion_parts(sections, cfg, hf_python)`:
   - Creates a shared `CudaStream` for all engines.
   - Loads `denoiser_plan_data` and `vae_decoder_plan_data` as TrtModules.
   - Loads `text_encoder_plans[0..N]` as TrtModules. Falls back to `plan_data` if no dedicated text encoder sections exist.
   - Parses `DiffusionConfig` from `FastPathModelConfig`.
   - Parses `PreprocessorWeights` from `preprocessor_weights_data` section.
   - Creates tokenizer.

2. Dispatch on `cfg.diffusion_backend_type`:
   - `"flux_2d"` or contains `"flux"` -> `create_flux_pipeline()`:
     - Also extracts a CLIP tokenizer from bundle (dual tokenizer).
     - Returns `FluxPipeline(text_encoders[], denoiser, vae, config, weights, tokenizer, clip_tokenizer, model_id)`.
   - `"z_image_2d"` or contains `"z_image"` -> `create_zimage_pipeline()`:
     - Parses `ZImagePreprocessorWeights` (timestep embedder, caption embedder, patch embedder).
     - Returns `ZImagePipeline(text_encoder, denoiser, vae, config, weights, z_weights, tokenizer, model_id, hf_python, bundle_path)`.
   - Default (including `"wan_3d"`) -> `create_wan_t2v_pipeline()`:
     - Returns `WanPipeline(text_encoder, denoiser, vae, config, weights, tokenizer, model_id)`.

**Key files:**
- `src/runtime/pipelines/diffusion_pipeline.h` -- `FluxPipeline`, `WanPipeline`, `ZImagePipeline`, `ZImagePreprocessorWeights`
- `src/runtime/pipelines/wan_pipeline.cpp`, `flux_pipeline.cpp`, `z_image_pipeline.cpp` -- implementations
- `src/runtime/trt/diffusion/diffusion_types.h` -- `DiffusionConfig`, `PreprocessorWeights`

### 4.4 Audio Family (`create_audio_pipeline`)

Dispatches by strategy string:

- `"speech_to_text"` -> `make_whisper_pipeline_from_bundle()` -> `WhisperPipeline`.
- `"text_to_audio"`:
  - If `cfg.is_magpie_tts` -> `make_magpie_pipeline_from_bundle()` -> `MagpiePipeline`.
  - Otherwise -> `make_bark_pipeline_from_bundle()` -> `BarkPipeline`.
- `"speech_to_speech"` -> `make_speech_pipeline_from_bundle()` -> `SpeechPipeline`.
- `"omni_multimodal"` -> `create_omni_pipeline()` -> `OmniPipeline`.

The `create_omni_pipeline()` function is defined inline in `pipeline_factory.cpp`:
1. Load thinker TrtModule from `plan_data` + KvCache.
2. Load optional talker TrtModule from `talker_engine_plan_data` + its own KvCache.
3. Load optional code2wav TrtModule from `code2wav_engine_plan_data`.
4. Build `OmniConfig` from `FastPathModelConfig` fields.
5. Returns `OmniPipeline(thinker, thinker_cache, talker, talker_cache, code2wav, config, stream, tokenizer, model_id)`.

The other audio factory functions live in `src/runtime/pipelines/audio_backend_factory.cpp`.

**Key files:**
- `src/runtime/pipelines/audio_pipeline.h` -- all audio pipeline classes
- `src/runtime/pipelines/audio_backend_factory.h` -- factory function declarations
- `src/runtime/pipelines/audio_backend_factory.cpp` -- factory function implementations
- `src/runtime/trt/audio/` -- backend implementations (WhisperBackend, BarkBackend, etc.)

### 4.5 Encoder Family (`create_encoder_pipeline`)

- `"segmentation"` -> `SegmentPipeline(module, model_id)`.
- `"prompted_segmentation"`:
  - Try loading mask decoder from `vision_plan_data`.
  - If successful: `SamPipeline(image_encoder, mask_decoder, model_id)`.
  - If not: fall back to `SegmentPipeline`.
- `"encoder_only"`, `"embedding"`, `"reranking"` -> `EncoderPipeline(module, strategy, tokenizer, model_id)`.

**Key files:**
- `src/runtime/pipelines/encoder_pipeline.h` -- `EncoderPipeline`, `SegmentPipeline`, `SamPipeline`
- `src/runtime/pipelines/encoder_pipeline.cpp` -- implementations

---

## 5. TrtModule: The Forward Pass Abstraction

**Header:** `include/trtf/runtime/trt_module.h`
**Implementation:** `src/runtime/trt/core/trt_module.cpp`

`TrtModule` wraps a TRT `ICudaEngine` + `IExecutionContext`. It pre-allocates
all device buffers at construction and provides three forward pass modes:

### 5.1 `forward(const TensorMap& inputs) -> TensorMap`
Synchronous CPU-to-CPU path:
1. **H2D upload:** For each input in the TensorMap, `cudaMemcpyAsync(HostToDevice)` into pre-allocated device buffers.
2. **Execute:** `ctx_->enqueueV3(stream_)`.
3. **Sync:** `cudaStreamSynchronize(stream_)`.
4. **D2H download:** For each output buffer, `cudaMemcpy(DeviceToHost)` into pre-allocated host staging buffers.
5. Returns a TensorMap where each `Tensor.data` points to the host staging buffer.

This is the primary path used by `TextGenerationPipeline::run_step()`.

### 5.2 `forward_device(const DeviceTensorMap& inputs) -> DeviceTensorMap`
GPU-to-GPU path (no host copies):
1. **D2D copy:** For each input DeviceTensor, `cudaMemcpyAsync(DeviceToDevice)` if the pointer differs from the internal buffer.
2. **Execute + sync.**
3. Returns references to internal output buffers (no copy).

### 5.3 `forward_async(inputs)` + `sync()`
Split async path:
1. `forward_async()` -- H2D upload + enqueueV3 (no sync).
2. `sync()` -- `cudaStreamSynchronize()`.
3. Caller then reads outputs via forward or device_ptr().

### 5.4 `bind_external(name, device_ptr)`
Replaces the pre-allocated buffer for a tensor with an external pointer:
1. Frees the old buffer (if owned).
2. Sets `is_external = true` (so destructor does not free it).
3. Calls `ctx_->setTensorAddress(name, external_device_ptr)`.

This is how `KvCache::bind_to()` and `RecurrentState::bind_to()` inject
their state buffers into the TrtModule's execution context.

### 5.5 `keep_alive(shared_ptr<void>)`
Stores opaque resource ownership (engine, stream) so they outlive the module's
execution context. Called by `pipeline_factory.cpp` after engine deserialization.

---

## 6. KvCache Lifecycle

**Header:** `include/trtf/runtime/kv_cache.h`
**Implementation:** `src/runtime/trt/core/kv_cache.cpp`

### 6.1 Construction

```cpp
KvCache(num_layers, max_length, kv_dim, stream)
```
Allocates per-layer device buffers:
- `cache_k_[i]`: DeviceTensor shape `[max_length, kv_dim]` (persistent K cache).
- `cache_v_[i]`: DeviceTensor shape `[max_length, kv_dim]` (persistent V cache).
- `present_k_[i]`: DeviceTensor shape `[1, kv_dim]` (single-step K output).
- `present_v_[i]`: DeviceTensor shape `[1, kv_dim]` (single-step V output).

Calls `reset()` to zero all buffers and set position to 0.

### 6.2 `bind_to(TrtModule& module)`

For each layer `i`:
```cpp
module.bind_external("cache_k_" + i, cache_k_[i].data());
module.bind_external("cache_v_" + i, cache_v_[i].data());
module.bind_external("present_k_" + i, present_k_[i].data());
module.bind_external("present_v_" + i, present_v_[i].data());
```
After this call, the TRT engine reads from `cache_k/v` and writes to `present_k/v`
as part of its execution.

### 6.3 `build_attention_mask(vector<float>& mask)`

Builds a causal mask of size `max_length + 1`:
- Positions `[0, position)` = `0.0f` (visible/attended).
- Positions `[position, max_length)` = `-1e4f` (masked/future).
- Position `max_length` = `0.0f` (current token slot -- always visible).

### 6.4 `advance()`

After each decoder step:
1. D2D async copy: `present_k_[i] -> cache_k_[i][position_, :]` for all layers.
2. D2D async copy: `present_v_[i] -> cache_v_[i][position_, :]` for all layers.
3. `position_++`.
4. If `position_ >= max_length_`, clamp to `max_length_ - 1` (sliding window behavior).

### 6.5 `reset()`

Zeros all `cache_k_`, `cache_v_`, `present_k_`, `present_v_` buffers via `cudaMemsetAsync`.
Sets `position_ = 0`. Synchronizes the stream.

---

## 7. RecurrentState Lifecycle

**Header:** `include/trtf/runtime/recurrent_state.h`
**Implementation:** `src/runtime/trt/core/recurrent_state.cpp`

Generic state manager for SSM (Mamba) and RWKV models.

### 7.1 Construction

```cpp
RecurrentState(num_layers, specs, stream)
```
Where `specs` is a vector of `TensorSpec{name, shape, output_prefix}`. For each spec
and each layer, allocates two DeviceTensors:
- `state_[spec_idx][layer_idx]` -- persistent state (input to the engine).
- `present_[spec_idx][layer_idx]` -- single-step output from the engine.

### 7.2 `bind_to(TrtModule& module)`

For each spec and layer `i`:
```cpp
module.bind_external("{spec.name}_{i}", state_[spec_idx][i].data());
module.bind_external("{spec.output_prefix}_{i}", present_[spec_idx][i].data());
```

### 7.3 `advance()`

D2D async copy: `present_[spec][layer] -> state_[spec][layer]` for all specs and layers.

### 7.4 `reset()`

Zeros all state and present buffers.

---

## 8. IStateManager: Unifying KvCache and RecurrentState

**Defined in:** `src/runtime/pipelines/recurrent_pipeline.h`

```cpp
struct IStateManager {
    virtual void reset() = 0;
    virtual void bind_to(TrtModule& module) = 0;
    virtual void advance() = 0;
    virtual void build_mask(vector<float>& mask) const = 0;
    virtual int32_t position() const = 0;
    virtual bool has_mask() const = 0;
};
```

Three concrete implementations:

| Class | Used by | State | Mask? |
|-------|---------|-------|-------|
| `RecurrentStateManager` | Mamba, RWKV | `RecurrentState` | No (no-op `build_mask`) |
| `HybridStateManager` | Nemotron-H | `KvCache` + `RecurrentState` | Yes (delegates to `KvCache::build_attention_mask`) |
| (implicit) | Standard decoders | `KvCache` directly | Yes (called directly in `TextGenerationPipeline`) |

Note: `TextGenerationPipeline` does NOT use `IStateManager` -- it owns a
`KvCache` directly. Only `RecurrentPipeline` uses the `IStateManager` interface.

---

## 9. IPipeline: The Public API

**Header:** `include/trtf/pipeline.h`

`IPipeline` is a pure virtual interface with default implementations that throw
`std::runtime_error` for unsupported operations. Each pipeline type overrides
only the methods it supports:

| Pipeline class | Overrides |
|---------------|-----------|
| `TextGenerationPipeline` | `generate(prompt, cfg)` |
| `RecurrentPipeline` | `generate(prompt, cfg)` |
| `VLPipeline` | `generate(prompt, cfg)`, `generate(prompt, image, h, w, cfg)` |
| `FluxPipeline` | `generate_image(prompt, cfg)` |
| `WanPipeline` | `generate_image(prompt, cfg)` |
| `ZImagePipeline` | `generate_image(prompt, cfg)` |
| `WhisperPipeline` | `transcribe(audio, n, max_tokens, sample_rate)` |
| `BarkPipeline` | `generate_audio(prompt, cfg)` |
| `MagpiePipeline` | `generate_audio(prompt, cfg)` |
| `SpeechPipeline` | `speak(audio_in, n, cfg, sample_rate)` |
| `OmniPipeline` | `generate_audio(prompt, cfg)` |
| `EncoderPipeline` | `embed(text)`, `encode(text)`, `rerank(query, doc)` |
| `SegmentPipeline` | `segment(pixels, h, w)` |
| `SamPipeline` | `segment(pixels, h, w)` |

All pipelines also implement `model_id()` and `pipeline_type()` (pure virtual).

---

## 10. BundleSections: Non-Owning Section Pointers

**Header:** `src/cabi/bundle/bundle_helpers.h`

`BundleSections` is a struct of `const std::vector<char>*` pointers into the
`BundleFile::sections` vector. It is populated by `find_bundle_sections()` which
scans section names and assigns pointers:

| Section name pattern | Pointer field |
|---------------------|---------------|
| `engine_plan` | `plan_data` |
| `vision_engine_plan` | `vision_plan_data` |
| `config.json` | `config_json_data` |
| `preprocessor_config.json` | `preprocessor_config_data` |
| `tokenizer.json` | `tokenizer_json_data` |
| `tokenizer_config.json` | `tokenizer_config_data` |
| `denoiser_plan` | `denoiser_plan_data` |
| `vae_decoder_plan` | `vae_decoder_plan_data` |
| `text_encoder_N_plan` | `text_encoder_plans[N]` |
| `coarse_engine_plan` | `coarse_engine_plan_data` |
| `fine_engine_plan` | `fine_engine_plan_data` |
| `codec_engine_plan` | `codec_engine_plan_data` |
| `talker_engine_plan` | `talker_engine_plan_data` |
| `code2wav_engine_plan` | `code2wav_engine_plan_data` |
| `mel_filterbank` | `mel_filterbank_data` |
| ... (30+ section types) | ... |

The `BundleFile` must outlive any use of `BundleSections`.

---

## 11. FastPathModelConfig: The Monolithic Config Struct

**Header:** `src/cabi/config/fast_path_config.h` (~181 fields)

`FastPathModelConfig` holds ALL configuration fields for ALL modalities in a single
flat struct. Each runtime strategy uses a subset of the fields:

| Strategy | Fields used |
|----------|------------|
| `decoder_kv_cache` | `vocab_size`, `hidden_size`, `num_layers`, `num_heads`, `num_kv_heads`, `head_dim`, `max_cache_length`, `id_bos`, `id_eos`, `runtime_strategy` |
| `ssm_recurrent` | Above + `d_inner`, `state_size`, `conv_kernel` |
| `rwkv_recurrent` | Above + `hidden_size` (for state dimensions) |
| `hybrid_mamba_attention` | Above + `num_mamba_layers`, `num_attention_layers`, `mamba_d_state`, `mamba_d_conv`, `mamba_nheads`, `mamba_head_dim`, `conv_dim` |
| `vision_language` | Text decoder fields + `has_vision_engine`, `embed_input`, `image_token_id`, `vision_output_dim`, `fixed_image_size`, VL prompt template fields |
| `diffusion` | `scheduler`, `num_inference_steps`, `guidance_scale`, `flow_shift`, `video_height`, `video_width`, `video_num_frames`, `z_dim`, `dit_dim`, `dit_num_heads`, `freq_dim`, `text_seq_len`, `diffusion_backend_type`, RoPE fields, VAE fields |
| `text_to_audio` | Audio/Bark fields (~30 fields), MagpieTTS fields |
| `speech_to_text` | `num_mel_bins`, `max_source_positions`, `mel_*` fields, `eot_token_id` |
| `omni_multimodal` | `omni_*` fields (sample_rate, experts, talker, codebooks) |

Parsed by `parse_fast_path_config()` in `src/cabi/config/fast_path_config.cpp`
from the `config.json` section text.

---

## 12. Known Limitations

1. **FastPathModelConfig is monolithic.** All ~181 fields live in one struct,
   even though each strategy uses only 10-30 of them. This works today because
   the parser simply ignores unknown fields, but it grows linearly with every
   new modality.

2. **Pipeline factory is centralized.** All strategy construction logic lives
   in one 700-line file (`pipeline_factory.cpp`). Adding a new strategy requires
   editing this file. There is no plugin registration mechanism on the C++ side
   (unlike the Python build side where family plugins are auto-discovered).

3. **No pipeline reuse.** Each `trtf_create_pipeline_ex()` call deserializes
   engines from scratch. There is no caching of deserialized engines across
   pipeline instances.

4. **Single-sequence inference only.** KvCache and RecurrentState manage state
   for one sequence at a time. There is no batching support.

5. **Synchronous execution.** All pipeline `generate()` methods are fully
   synchronous and block until completion. There is no async/streaming API.

---

## 13. File Reference

| Component | File |
|-----------|------|
| C ABI entry point | `src/cabi/api/trtf_c.cpp` |
| Pipeline factory | `src/runtime/pipeline_factory.cpp` |
| Pipeline factory header | `include/trtf/runtime/pipeline_factory.h` |
| IPipeline interface | `include/trtf/pipeline.h` |
| TextGenerationPipeline | `src/runtime/pipelines/text_generation_pipeline.h`, `.cpp` |
| RecurrentPipeline | `src/runtime/pipelines/recurrent_pipeline.h`, `.cpp` |
| VLPipeline | `src/runtime/pipelines/vl_pipeline.h`, `.cpp` |
| Diffusion pipelines | `src/runtime/pipelines/diffusion_pipeline.h`, `wan_pipeline.cpp`, `flux_pipeline.cpp`, `z_image_pipeline.cpp` |
| Audio pipelines | `src/runtime/pipelines/audio_pipeline.h`, `.cpp` |
| Audio factory | `src/runtime/pipelines/audio_backend_factory.h`, `.cpp` |
| Encoder pipelines | `src/runtime/pipelines/encoder_pipeline.h`, `.cpp` |
| TrtModule | `include/trtf/runtime/trt_module.h`, `src/runtime/trt/core/trt_module.cpp` |
| KvCache | `include/trtf/runtime/kv_cache.h`, `src/runtime/trt/core/kv_cache.cpp` |
| RecurrentState | `include/trtf/runtime/recurrent_state.h`, `src/runtime/trt/core/recurrent_state.cpp` |
| Bundle format | `src/bundle/bundle_format.h`, `.cpp` |
| Bundle helpers | `src/cabi/bundle/bundle_helpers.h`, `.cpp` |
| Config parser | `src/cabi/config/fast_path_config.h`, `.cpp` |
| Image preprocessor | `src/runtime/trt/multimodal/image_preprocessor.h`, `.cpp` |
| Diffusion types | `src/runtime/trt/diffusion/diffusion_types.h` |
| Scheduler | `src/runtime/trt/core/flow_match_euler_scheduler.cpp` |
| Tokenizer interface | `include/trtf/runtime/tokenizer_interface.h` |
| HF Python tokenizer | `src/tokenizer/hf_python_tokenizer.cpp` |
| Vocab tokenizer | `src/tokenizer/vocab_tokenizer.cpp` |
