# Pipeline Deep Dive

This page traces the complete execution path across both phases: Python build and C++ runtime.

## Phase 1: Building a Bundle (Python)

The Python CLI `trtf-build build` produces a `.trtfb` bundle from an HF model directory.

```
trtf-build build path/to/Qwen3-0.6B -o qwen3.trtfb --max-cache-length 256
  |
  +-- Parse config.json
  |     - Extract model_type ("qwen3"), architectures, dims, heads, etc.
  |
  +-- Match family plugin
  |     - Registry checks model_type against registered plugins
  |     - Qwen plugin matches: model_type starts with "qwen"
  |
  +-- Load weights
  |     - Open model.safetensors (or sharded files via index.json)
  |     - Family checkpoint mapper translates HF tensor keys to canonical format:
  |       - Transpose [out, in] -> [in, out]
  |       - Expand K/V for GQA
  |       - Handle tied lm_head
  |       - Gemma: +1.0 to RMSNorm gamma, scale embedding
  |
  +-- Build TRT network (via TensorRT Python API)
  |     - Create IBuilder, INetworkDefinition, IBuilderConfig
  |     - Add inputs: token_id, position_id, attention_mask, per-layer cache_k/v
  |     - Add embedding gather
  |     - Precompute RoPE tables as constants
  |     - For each decoder layer:
  |       - RMSNorm -> QKV proj -> optional QK norm -> RoPE
  |       - KV cache concat -> GQA attention -> output proj -> residual
  |       - RMSNorm -> SwiGLU MLP -> residual
  |     - Final RMSNorm + LM head -> logits
  |     - Mark outputs: logits + per-layer present_k/v
  |
  +-- Compile engine
  |     - buildEngineWithConfig(network, config)
  |     - GPU kernel compilation (30-300s depending on model size)
  |     - Serialize engine plan to bytes
  |
  +-- Package bundle
        - Write .trtfb file containing:
          - Serialized engine plan bytes
          - tokenizer.json, tokenizer_config.json, config.json
          - Bundle metadata (TRT version, GPU, timestamp, architecture)
```

## Phase 2: Running from a Bundle (C++)

The C++ runtime loads a `.trtfb` bundle and runs inference. This is the only path the C++ runtime supports -- it does not load raw HF model directories.

### The Call Chain

```
User: trtf_create_pipeline("qwen3.trtfb", TRTF_FORCE_TRT)
  |     src/cabi/trtf_c.cpp
  |
  +-- Detect bundle (HasBundleMagic)
  |
  +-- ReadBundleFile("qwen3.trtfb")
  |     src/bundle/bundle_format.cpp
  |     - Parse JSON header (metadata, section offsets)
  |     - Extract engine plan bytes
  |     - Extract tokenizer files to temp directory
  |
  +-- Deserialize TRT engine
  |     - createInferRuntime(logger)
  |     - deserializeCudaEngine(plan_bytes)
  |     - createExecutionContext()
  |     - Populate DecoderStepEngine metadata from bundle header
  |
  +-- Create tokenizer
  |     src/tokenizer/hf_python_tokenizer.cpp
  |     - HfPythonTokenizer(temp_dir) if tokenizer.json exists
  |     - VocabTokenizer otherwise
  |
  +-- Create TrtBackendFastPath
  |     src/runtime/trt/trt_backend_shared.cpp
  |     - Wraps engine + generation loop
  |
  +-- Return PipelineImpl
        - Owns tokenizer + backend
        - Ready for generate() calls
```

### Generation: `pipeline->generate("Hello")`

```
pipeline->generate("Hello", 30)
  |
  +-- tokenizer->encode("Hello") -> [token_ids]
  |
  +-- backend->generate(token_ids, config)
  |     TrtBackendFastPath::generate()
  |     |
  |     +-- Create DeviceKvCache(engine) + DeviceResources
  |     |     - Allocate per-layer cache_k, cache_v on GPU (device-resident)
  |     |     - Pre-allocate per-step I/O buffers (DeviceResources)
  |     |
  |     +-- Prefill phase
  |     |     For each input token (except last):
  |     |       - Build causal attention mask
  |     |       - run_decoder_step_device(token, position, mask, cache)
  |     |       - D2D cache update internal to DeviceKvCache
  |     |       - Advance position counter
  |     |
  |     +-- Decode phase
  |           For step = 0 to max_new_tokens:
  |             - run_decoder_step_device(current_token, position, mask, cache)
  |             - select_argmax_token(logits) -> next_token
  |             - If next_token == eos_token: break
  |             - Append to output, update cache
  |
  +-- tokenizer->decode(output_ids) -> "Hello Answer is a..."
  |
  +-- Return generated text
```

## Key Data Structures (C++ Runtime)

### DecoderStepEngine

The deserialized TRT engine with binding metadata:

```cpp
struct DecoderStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    // Tensor binding names
    std::string token_id_name;
    std::string position_id_name;
    std::string attention_mask_name;
    std::vector<std::string> cache_k_names;     // Per-layer
    std::vector<std::string> cache_v_names;     // Per-layer
    std::vector<std::string> present_k_names;   // Per-layer outputs
    std::vector<std::string> present_v_names;   // Per-layer outputs
    std::string logits_name;

    // Metadata (from bundle header)
    int32_t num_layers;
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t cache_state_size;
    int32_t max_cache_length;
};
```

### DeviceKvCache

Device-resident KV cache for autoregressive generation. Keeps KV cache on GPU; only small inputs (token ID, position, mask) are transferred H2D per step. D2D cache update is internal.

```cpp
class DeviceKvCache {
    int32_t mCacheStateSize;
    int32_t mMaxCacheLength;
    int32_t mNumLayers;
    int32_t mCacheLength;                 // Current filled length
    std::vector<CudaBuffer> mCacheK;      // Per-layer [max_cache, attn_size] on GPU
    std::vector<CudaBuffer> mCacheV;      // Per-layer [max_cache, attn_size] on GPU
};

struct DeviceResources {
    CudaStream stream;
    CudaBuffer token_id_buf;     // Pre-allocated per-step I/O
    CudaBuffer position_id_buf;
    CudaBuffer mask_buf;
    CudaBuffer logits_buf;
};
```

### RuntimeAssembly

The output consumed by the PipelineImpl constructor:

```cpp
struct RuntimeAssembly {
    std::unique_ptr<ITokenizer> tokenizer;
    std::unique_ptr<IGenerationBackend> backend;
    std::string backend_name;  // "trt" or "hf-transformers"
};
```

## Bundle Format (`.trtfb`)

The bundle is a binary file with:

```
[4 bytes: magic "TRTF"]
[JSON header: metadata + section descriptors]
[section 1: engine_plan - serialized TRT engine plan bytes]
[section 2: config.json - model architecture config]
[section 3: tokenizer.json - HF tokenizer definition]
[section 4: tokenizer_config.json - HF tokenizer config]
```

Metadata in the JSON header includes:
- TRT version used during compilation
- GPU name the engine was compiled for
- Build timestamp
- Architecture info (family, hidden_size, num_layers, etc.)
- Max cache length

## VL Image Preprocessing (Vision-Language)

For vision-language models, the runtime preprocesses images before feeding pixel values to the vision TRT engine. The preprocessing strategy is configured via `preprocessor_type` in the bundle's config.json (set by the family plugin's `get_vl_config()`).

### Supported Strategies

| Strategy | Output Layout | Description | Use Case |
|----------|--------------|-------------|----------|
| `qwen_merge_group` | `[C*T, H, W]` | Merge-group patch permutation + temporal channel duplication | Qwen2.5-VL |
| `simple_chw` | `[C, H, W]` | Standard resize to square + normalize | LLaVA, InternVL, Phi-3-Vision |
| `center_crop_chw` | `[C, H, W]` | Center-crop to square, then resize + normalize | CLIP, DINOv2-based models |
| `aspect_preserve_chw` | `[C, H, W]` | Aspect-ratio-preserving resize + zero-pad to square | InternVL v2 |

### Interpolation

The `interpolation` field controls resize filtering:
- `"bicubic"` (default) -- matches PIL BICUBIC / Catmull-Rom
- `"bilinear"` -- matches PIL BILINEAR / triangle filter
- `"nearest"` -- matches PIL NEAREST / point sample

The C++ implementation uses `stb_image_resize2` filters; the Python debug runner uses PIL constants. The mode is read from `config.json` (set by the engine builder). If not explicitly set, it falls back to the HF `preprocessor_config.json` `resample` integer (PIL enum: 0=NEAREST, 2=BILINEAR, 3=BICUBIC).

### Config Flow

```
FamilyPlugin.get_vl_config()  -->  bundle config.json  -->  parse_vl_preprocess_config()
                                                              |
                                                              +-- preprocessor_type
                                                              +-- interpolation
                                                              +-- fixed_image_size
                                                              +-- image_mean / image_std
                                                              +-- image_token_id
                                                              +-- vl_prompt_template
```

### Parity

Both C++ (`image_preprocessor.cpp`) and Python (`debug_runner.py`) implement the same four preprocessing strategies with the same interpolation dispatch. The `diff_vl.py` script compares TRT vision encoder output against HuggingFace reference features to validate parity.

---

## Configuration

### Python CLI (`trtf-build`) Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--max-cache-length N` | from config (capped at 4096) | Cap KV cache length in the compiled engine |

### C++ CLI (`trtf run`) Flags / TrtfPipelineOptions

| CLI Flag | Options Field | Default | Effect |
|----------|--------------|---------|--------|
| `--max-new-tokens N` | `max_new_tokens` | 20 | Max generation tokens |
| `--hf-python PATH` | `hf_python` | auto-detect | Path to Python for HF tokenizer |
| `--force-trt` | `flags = TRTF_FORCE_TRT` | prefer | Require TRT backend |

### Remaining Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `TRTF_DATA_DIR` | auto-detected | Override source directory for built-in model/script paths |
| `TRTF_TRT_LOG_STDERR` | 0 | Enable TRT logger output to stderr |
| `TRTF_TRT_LOG_MIN_SEVERITY` | 2 (warning) | Minimum TRT log severity |
