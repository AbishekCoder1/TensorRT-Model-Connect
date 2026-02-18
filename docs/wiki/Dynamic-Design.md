# Dynamic Design: Sequence Diagrams and Data Flow

This page shows the runtime behavior of the system through sequence diagrams and data flow descriptions. The system is split: Python builds bundles, C++ runs them.

## Table of Contents

1. [Bundle Build (Python)](#1-bundle-build-python)
2. [Bundle Load and Pipeline Creation (C++)](#2-bundle-load-and-pipeline-creation-c)
3. [Autoregressive Generation (C++)](#3-autoregressive-generation-c)
4. [Single Decode Step (TRT)](#4-single-decode-step-trt)
5. [Data Transformation Pipeline](#5-data-transformation-pipeline)

---

## 1. Bundle Build (Python)

End-to-end sequence for `trtf-build build path/to/Qwen3-0.6B -o qwen3.trtfb`.

```mermaid
sequenceDiagram
    actor User
    participant CLI as trtf-build CLI
    participant Reg as FamilyRegistry
    participant Plug as QwenPlugin
    participant ST as safetensors lib
    participant CM as CheckpointMapper
    participant GB as GraphBuilder
    participant TRT as TensorRT Python API
    participant Bun as BundleWriter

    User->>CLI: trtf-build build path/to/Qwen3-0.6B -o qwen3.trtfb

    CLI->>CLI: parse config.json -> model_type="qwen3", dims, heads
    CLI->>Reg: match_family("qwen3")
    Reg->>Plug: QwenPlugin.matches("qwen3") -> true

    Note over CLI: Load weights
    CLI->>ST: open model.safetensors
    CLI->>CM: QwenCheckpointMapper.map(safetensors, config)
    CM->>ST: read tensors (embedding, per-layer weights, lm_head)
    CM->>CM: transpose [out,in] -> [in,out]
    CM->>CM: expand K/V for GQA
    CM->>CM: handle q_norm/k_norm (Qwen3)
    CM-->>CLI: canonical weights dict

    Note over CLI: Build TRT network
    CLI->>GB: StandardGraphBuilder.build(weights, config)
    GB->>TRT: createBuilder + createNetwork + createBuilderConfig
    GB->>TRT: add inputs (token_id, position_id, mask, cache_k/v)
    GB->>TRT: add embedding gather
    GB->>TRT: precompute RoPE tables
    loop each decoder layer
        GB->>TRT: add_rms_norm + QKV proj + QK norm + RoPE
        GB->>TRT: KV concat + attention + output proj + residual
        GB->>TRT: add_rms_norm + SwiGLU MLP + residual
    end
    GB->>TRT: final RMSNorm + LM head
    GB->>TRT: mark outputs (logits, present_k/v)

    Note over CLI: Compile engine
    GB->>TRT: buildEngineWithConfig(network, config)
    Note over TRT: GPU kernel compilation (30-300s)
    TRT-->>GB: ICudaEngine
    GB->>TRT: serialize() -> plan bytes
    GB-->>CLI: engine plan bytes

    Note over CLI: Package bundle
    CLI->>Bun: write .trtfb (plan + tokenizer.json + config.json)
    Bun-->>User: qwen3.trtfb
```

---

## 2. Bundle Load and Pipeline Creation (C++)

Sequence for `trtf_create_pipeline("qwen3.trtfb", TRTF_FORCE_TRT)`.

```mermaid
sequenceDiagram
    actor User
    participant CABI as trtf_c.cpp
    participant BF as bundle_format.cpp
    participant TRT as TensorRT Runtime
    participant Tok as HfPythonTokenizer
    participant BE as TrtBackendFastPath

    User->>CABI: trtf_create_pipeline("qwen3.trtfb", TRTF_FORCE_TRT)

    CABI->>CABI: HasBundleMagic("qwen3.trtfb") -> true

    CABI->>BF: ReadBundleFile("qwen3.trtfb")
    BF->>BF: parse JSON header (metadata, section offsets)
    BF->>BF: extract engine plan bytes
    BF->>BF: extract tokenizer files to temp dir
    BF-->>CABI: engine plan + temp dir path + metadata

    CABI->>TRT: createInferRuntime(logger)
    CABI->>TRT: deserializeCudaEngine(plan_bytes)
    TRT-->>CABI: ICudaEngine
    CABI->>TRT: createExecutionContext()
    TRT-->>CABI: IExecutionContext

    Note over CABI: Build DecoderStepEngine from metadata

    CABI->>Tok: CreateHfPythonTokenizer(temp_dir)
    Tok-->>CABI: tokenizer

    CABI->>BE: CreateTrtBackendFromEngine(engine)
    BE-->>CABI: TrtBackendFastPath

    CABI-->>User: PipelineImpl (ready)
```

---

## 3. Autoregressive Generation (C++)

How `TrtBackendFastPath::generate()` runs the prefill + decode loop.

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant T as ITokenizer
    participant B as TrtBackendFastPath
    participant S as DeviceKvCache
    participant E as DecoderStepEngine
    participant GPU as CUDA GPU

    P->>T: encode("Hello world")
    T-->>P: [token_1, token_2]
    P->>B: generate([token_1, token_2], config)

    Note over B: Create device-resident KV cache + I/O buffers
    B->>S: DeviceKvCache(engine)
    Note over S: Allocates per-layer cache_k, cache_v on GPU

    Note over B: Prefill phase
    loop for each input token except last
        B->>S: prepare_step(position_id, mask)
        S-->>B: position_id, attention_mask
        B->>E: run_decoder_step_device(token, position, mask, cache)
        E->>GPU: enqueueV3(stream)
        GPU-->>E: logits[vocab_size], D2D cache update internal
        Note over S: cache_length++
    end

    Note over B: Decode phase
    loop step = 0..max_new_tokens
        B->>S: prepare_step(position_id, mask)
        S-->>B: position_id, attention_mask
        B->>E: run_decoder_step_device(current_token, position, mask, cache)
        E->>GPU: enqueueV3(stream)
        GPU-->>E: logits[vocab_size], D2D cache update internal
        B->>B: select_argmax_token(logits) -> next_token
        alt next_token == eos_token
            Note over B: Stop generation
        end
        B->>B: output.push_back(next_token)
    end

    B-->>P: [token_1, token_2, gen_1, gen_2, ...]
    P->>T: decode(output_ids)
    T-->>P: "Hello world The answer is..."
    P-->>P: return generated_text
```

### Key implementation details

- **State management is abstracted via `IStepState`**. `DeviceKvCache` implements it for standard attention models, keeping KV cache resident on GPU.
- **KV cache is a fixed-size circular buffer** per layer: `[max_cache_length, attention_size]`, held in device memory. D2D cache update is internal to `DeviceKvCache` (no host round-trip). Only small inputs (token ID, position, mask) are transferred H2D per step.
- **Attention mask** is built each step: `0.0` for visible positions, `-1e9` for masked. Grows by 1 each step.
- **Greedy sampling** via `select_argmax_token()` -- scans logits array for maximum value.

---

## 4. Single Decode Step (TRT)

Detailed view of what happens inside `run_decoder_step_device()`.

```mermaid
sequenceDiagram
    participant RT as run_decoder_step_device()
    participant Ctx as IExecutionContext
    participant GPU as CUDA Buffers
    participant Stream as CudaStream

    Note over RT: Set tensor addresses for inputs (small H2D transfers)
    RT->>GPU: cudaMemcpyAsync(token_id, H->D, DeviceResources.token_id_buf)
    RT->>GPU: cudaMemcpyAsync(position_id, H->D, DeviceResources.position_id_buf)
    RT->>GPU: cudaMemcpyAsync(attention_mask, H->D, DeviceResources.mask_buf)

    Note over RT: Bind device-resident KV cache (already on GPU)
    RT->>Ctx: setTensorAddress("token_id", device_token_buf)
    RT->>Ctx: setTensorAddress("position_id", device_position_buf)
    RT->>Ctx: setTensorAddress("attention_mask", device_mask_buf)
    loop each layer
        RT->>Ctx: setTensorAddress("cache_k_{i}", DeviceKvCache.cache_k[i])
        RT->>Ctx: setTensorAddress("cache_v_{i}", DeviceKvCache.cache_v[i])
    end

    Note over RT: Set tensor addresses for outputs
    RT->>Ctx: setTensorAddress("logits", device_logits)
    loop each layer
        RT->>Ctx: setTensorAddress("present_k_{i}", device_present_k)
        RT->>Ctx: setTensorAddress("present_v_{i}", device_present_v)
    end

    Note over RT: Execute TRT inference
    RT->>Ctx: enqueueV3(stream)
    Note over GPU: TRT runs fused CUDA kernels:<br/>embedding -> N x (RMSNorm+attn+MLP) -> logits

    Note over RT: D2D cache update internal to DeviceKvCache
    Note over RT: Copy only logits from GPU
    RT->>GPU: cudaMemcpyAsync(logits, D->H)
    RT->>Stream: cudaStreamSynchronize()

    RT-->>RT: return logits (cache stays on device)
```

---

## 5. Data Transformation Pipeline

End-to-end data transformation from HF files to generated text, showing the Python/C++ boundary.

```mermaid
flowchart TD
    A[("HF Model Directory<br/>config.json + model.safetensors<br/>+ tokenizer.json")] --> B

    subgraph "Python: trtf-build"
        B["config.json Parser"] --> C["Architecture Config<br/><i>family, dims, heads, rope_theta</i>"]
        A --> D["safetensors library"]
        D --> E["Checkpoint Mapper<br/><i>Per-family plugin</i>"]
        C --> E
        E --> F["Canonical Weights<br/><i>Transposed, GQA-expanded</i>"]
        F --> G["TRT Graph Builder<br/><i>TensorRT Python API</i>"]
        C --> G
        G --> H["INetworkDefinition"]
        H --> I["Engine Compilation<br/><i>buildEngineWithConfig</i>"]
        I --> J["Serialized Plan Bytes"]
    end

    J --> K[(".trtfb Bundle<br/><i>plan + tokenizer files</i>")]

    subgraph "C++: trtf runtime"
        K --> L["ReadBundleFile()"]
        L --> M["deserializeCudaEngine()"]
        M --> N["DecoderStepEngine"]
        K --> O["HfPythonTokenizer<br/><i>or VocabTokenizer</i>"]
        N --> P["TrtBackendFastPath::generate()<br/><i>Prefill + Decode loop</i>"]
        O -->|"encode(prompt)"| P
        P --> Q["run_decoder_step_device()<br/><i>GPU kernel execution</i>"]
        Q --> R["select_argmax_token()"]
        R -->|"loop until EOS"| P
        P -->|"token IDs"| O
        O -->|"decode(ids)"| S(["Generated Text"])
    end

    style A fill:#f1f5f9,stroke:#94a3b8
    style K fill:#dbeafe,stroke:#2563eb
    style N fill:#dcfce7,stroke:#16a34a
    style S fill:#fef3c7,stroke:#f59e0b
```

### VL Pipeline Data Flow (Vision-Language Models)

For vision-language bundles, an additional image preprocessing step occurs before text generation:

```
Image file
  |
  +-- load_and_preprocess_image(path, VLPreprocessConfig)
  |     Dispatches to preprocessor_type strategy:
  |       "qwen_merge_group" -> [C*T, H, W] with patch permutation
  |       "simple_chw"       -> [C, H, W] standard resize
  |       "center_crop_chw"  -> [C, H, W] center-crop + resize
  |       "aspect_preserve_chw" -> [C, H, W] aspect-preserve + pad
  |     Uses configurable interpolation (bicubic/bilinear/nearest)
  |
  +-- Vision TRT engine -> image features [N, dim]
  |
  +-- format_vl_prompt(user_prompt, config) -> formatted prompt
  |     Replaces {image_pads} and {prompt} in template
  |
  +-- Text decoder with embed_input mode (features replace pad tokens)
  |
  +-- Autoregressive generation (same as decoder pipeline)
```

### Data formats at each stage

| Stage | Location | Data Format |
|-------|----------|-------------|
| HF files | Input | Safetensors binary (F32/F16/BF16) + JSON config |
| After checkpoint mapper | Python | NumPy arrays, transposed to `[in, out]`, GQA KV expanded |
| TRT Network | Python | `INetworkDefinition` with weights as constant tensors |
| Compiled engine | `.trtfb` bundle | Serialized plan bytes (optimized CUDA kernels) |
| Deserialized engine | C++ runtime | `ICudaEngine` + `IExecutionContext` |
| Runtime | C++ | Host `int32_t` token IDs, device-resident `float` KV caches (`DeviceKvCache`), small H2D per-step I/O via `DeviceResources` |
