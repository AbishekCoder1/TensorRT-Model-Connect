# Dynamic Design: Sequence Diagrams and Data Flow

This page shows the runtime behavior of the system through sequence diagrams and data flow descriptions.

## Table of Contents

1. [Pipeline Creation (Full Sequence)](#1-pipeline-creation-full-sequence)
2. [Pipeline Creation — Fast Path (Cached Engine)](#2-pipeline-creation--fast-path-cached-engine)
3. [HF Family Resolution Detail](#3-hf-family-resolution-detail)
4. [Checkpoint Loading and Mapping](#4-checkpoint-loading-and-mapping)
5. [TRT Engine Build](#5-trt-engine-build)
6. [Autoregressive Generation](#6-autoregressive-generation)
7. [Single Decode Step (TRT)](#7-single-decode-step-trt)
8. [Data Transformation Pipeline](#8-data-transformation-pipeline)

---

## 1. Pipeline Creation (Full Sequence)

End-to-end sequence from `trtf_create_pipeline("QWEN3", TRTF_FORCE_TRT)` to a ready-to-use IPipeline object. The C ABI factory in `trtf_c.cpp` delegates to the same internal stages.

```mermaid
sequenceDiagram
    actor User
    participant CABI as trtf_c.cpp
    participant P as Pipeline
    participant R as ModelResolver
    participant FR as HfFamilyRegistry
    participant QR as QwenRegistration
    participant ML as ModelLoader
    participant CM as CheckpointMapper
    participant RF as RuntimeFactory
    participant TB as TrtBackend

    User->>CABI: trtf_create_pipeline("QWEN3", TRTF_FORCE_TRT)
    CABI->>P: ResolveTextGenerationModel + BuildRuntimeForTextGeneration

    Note over P: Stage 1: Model Resolution
    P->>R: ResolveTextGenerationModel("QWEN3")
    R->>FR: ResolveHfModelViaFamilyRegistry("QWEN3")
    FR->>FR: resolve_model_dir_from_alias("QWEN3") → path
    FR->>FR: is_hf_transformers_model_dir(path) → true
    FR->>FR: RegisterBuiltinHfModelFamilies() [once]

    Note over FR: Stage 2: Family Dispatch
    FR->>FR: load_hf_metadata(path) → {model_type: "qwen3"}
    FR->>QR: matcher({model_type: "qwen3"}) → true
    QR->>ML: LoadDecoderModel(model_dir)
    ML->>ML: parse config.json → architecture
    ML->>ML: detect model.safetensors → TensorSource
    ML->>CM: FindCheckpointMapper("qwen3")
    CM-->>ML: QwenCheckpointMapper
    ML->>CM: map_checkpoint(reader, vocab, path, arch)
    CM-->>ML: DecoderCheckpoint (per-layer weights)
    ML-->>QR: DecoderModel
    QR->>QR: cap max_cache_length if > 4096
    QR-->>FR: DecoderModel
    FR-->>R: ResolvedModelSpec{kDecoderDefinition}
    R-->>P: ResolvedModelSpec

    Note over P: Stage 3: Runtime Assembly
    P->>RF: BuildRuntimeForTextGeneration(spec, selection)
    RF->>RF: Create HfPythonTokenizer(hf_tokenizer_dir)
    RF->>TB: CreateTrtBackend(tokenizer, model)
    TB->>TB: FindTrtGraphBuilder("qwen3") → StandardDecoderGraphBuilder
    TB->>TB: BuildTrtDecoderWeights(model) (inlined conversion)
    TB->>TB: builder.build_decoder_step_engine(weights, logger)
    TB-->>RF: TrtBackendShared
    RF-->>P: RuntimeAssembly{tokenizer, backend}

    P-->>User: Pipeline (ready)
```

### What happens at each stage

**Stage 1** resolves the model ID. For `"QWEN3"`, alias resolution maps it to a local directory. The resolver tries custom resolvers first (none registered), then the HF family registry.

**Stage 2** parses `config.json` to get `model_type`, matches it against registered families (Qwen matches), and the family loader calls `LoadDecoderModel()` which reads safetensors through the checkpoint mapper registry.

**Stage 3** creates the tokenizer (HfPythonTokenizer if `tokenizer.json` exists) and backend (TRT preferred, with HF-Python fallback). The TRT backend finds the graph builder, converts model weights to TRT format, builds the TensorRT engine, and wraps it in `TrtBackendShared`.

---

## 2. Pipeline Creation -- Fast Path (Cached Engine)

When a TRT engine has been previously built and cached to disk, the fast path bypasses all weight loading, checkpoint mapping, and TRT graph building. This reduces cached startup from ~260s to ~7s.

The fast path is attempted **before** the slow path. On cache miss, it returns `nullptr` and the slow path runs normally. On the first successful slow-path build, the engine cache index is saved so subsequent runs take the fast path.

```mermaid
sequenceDiagram
    actor User
    participant CABI as trtf_c.cpp
    participant Dir as resolve_model_dir_lightweight
    participant Cfg as parse_fast_path_config
    participant Idx as BuildModelDirIndexKey
    participant Lkp as LookupModelDirIndex
    participant Load as LoadTrtEnginePlanFromCache
    participant TRT as TensorRT Runtime
    participant Tok as HfPythonTokenizer
    participant BE as CreateTrtBackendFromEngine

    User->>CABI: trtf_create_pipeline("QWEN3", TRTF_FORCE_TRT)

    Note over CABI: Fast path attempt
    CABI->>Dir: resolve_model_dir_lightweight("QWEN3")
    Dir-->>CABI: models/hf/Qwen__Qwen3-0.6B

    CABI->>Cfg: parse_fast_path_config(config.json text, env override)
    Cfg-->>CABI: FastPathModelConfig (dims, max_cache_length, eos/bos)

    CABI->>Idx: BuildModelDirIndexKey(model_dir, max_cache_length)
    Idx-->>CABI: index_key (FNV-1a hash)

    CABI->>Lkp: LookupModelDirIndex(index_key)
    alt cache HIT (index + .plan file exist)
        Lkp-->>CABI: cache_key

        CABI->>Load: LoadTrtEnginePlanFromCache(cache_key)
        Load->>Load: mmap .plan file (~2.5 GB)
        Load-->>CABI: plan bytes

        CABI->>TRT: createInferRuntime(logger)
        CABI->>TRT: deserializeCudaEngine(plan)
        TRT-->>CABI: ICudaEngine
        CABI->>TRT: createExecutionContext()
        TRT-->>CABI: IExecutionContext

        Note over CABI: Build DecoderStepEngine from config metadata<br/>(no weight loading — zero allocation)

        CABI->>Tok: CreateHfPythonTokenizer(model_dir)
        Tok-->>CABI: tokenizer

        CABI->>BE: CreateTrtBackendFromEngine(engine)
        BE-->>CABI: TrtBackend (fast path)

        CABI-->>User: Pipeline (ready in ~7s)
    else cache MISS
        Lkp-->>CABI: nullopt
        Note over CABI: Fall through to slow path (Section 1)
    end
```

### What makes the fast path fast

| Slow path step | Time | Fast path equivalent |
|---------------|------|---------------------|
| Load safetensors (600M+ FP32 weights) | ~120s | **Skipped entirely** |
| Checkpoint mapping (transpose, GQA expand) | ~40s | **Skipped entirely** |
| TRT engine build (or cache load) | ~100s | mmap .plan file (~400ms) |
| Engine deserialization | ~5s | Same (~5s) |
| Tokenizer init | ~3s | Same (~3s) |

### Model-dir index key

`BuildModelDirIndexKey()` hashes:
- Canonical model directory path
- `config.json` file content (captures all architecture changes)
- `model.safetensors` file size (catches weight file changes without reading data)
- `model.safetensors.index.json` content (for sharded models)
- `max_cache_length` (different cache lengths need different engines)

This ensures any change to the model files invalidates the cache automatically.

### Cache invalidation

`LookupModelDirIndex()` verifies the `.plan` file still exists before returning a cache hit. If the plan was deleted (e.g., TRT version upgrade), it returns `nullopt` and the slow path rebuilds.

---

## 3. HF Family Resolution Detail

Detailed view of how the family registry matches and dispatches.

```mermaid
sequenceDiagram
    participant FR as HfFamilyRegistry
    participant Meta as load_hf_metadata
    participant Qwen as QwenFamily
    participant LLaMA as LLaMAFamily

    FR->>FR: RegisterBuiltinHfModelFamilies()
    FR->>Qwen: RegisterQwenFamily()
    Note over Qwen: Registers into 3 registries
    FR->>LLaMA: RegisterLlamaFamily()
    Note over LLaMA: Registers into 3 registries

    FR->>Meta: load_hf_metadata(model_dir)
    Meta->>Meta: read config.json
    Meta-->>FR: {model_type, architectures}

    FR->>FR: Sort families by priority (highest first)

    FR->>Qwen: matcher(metadata)
    alt model_type starts with "qwen" or "qwq"
        Qwen-->>FR: true
        FR->>Qwen: model_definition_loader(metadata)
        Qwen-->>FR: DecoderModel
    else model_type starts with "llama"
        Qwen-->>FR: false
        FR->>LLaMA: matcher(metadata)
        LLaMA-->>FR: true
        FR->>LLaMA: model_definition_loader(metadata)
        LLaMA-->>FR: DecoderModel
    end
```

---

## 4. Checkpoint Loading and Mapping

How HF safetensors tensor keys are translated into the canonical `DecoderCheckpoint`.

```mermaid
sequenceDiagram
    participant ML as ModelLoader
    participant TS as TensorSource
    participant SR as SafetensorReader
    participant CM as StandardCheckpointMapper

    ML->>ML: detect model.safetensors or index.json
    alt single file
        ML->>TS: TensorSource(single_path)
        TS->>SR: SafetensorReader(model.safetensors)
        SR->>SR: parse JSON header
    else sharded
        ML->>TS: TensorSource(model_dir, index_path)
        TS->>TS: parse model.safetensors.index.json
        loop each shard file
            TS->>SR: SafetensorReader(shard_N.safetensors)
        end
    end

    ML->>ML: FindCheckpointMapper(arch) → family mapper
    ML->>CM: map_checkpoint(reader, vocab_size, path, arch)

    Note over CM: Read embedding
    CM->>TS: read_f32("model.embed_tokens.weight")
    TS-->>CM: [vocab × hidden] (row-major)

    Note over CM: Read per-layer weights
    loop for each layer 0..N-1
        CM->>TS: read_f32("model.layers.{i}.input_layernorm.weight")
        CM->>TS: read_f32("model.layers.{i}.self_attn.q_proj.weight")
        Note over CM: Transpose [out,in] → [in,out]
        CM->>TS: read_f32("model.layers.{i}.self_attn.k_proj.weight")
        Note over CM: Transpose + expand_kv_projection() for GQA
        CM->>TS: read_f32("model.layers.{i}.self_attn.v_proj.weight")
        Note over CM: Transpose + expand_kv_projection() for GQA
        CM->>TS: read_f32("model.layers.{i}.self_attn.o_proj.weight")
        CM->>TS: read_f32("model.layers.{i}.mlp.gate_proj.weight")
        CM->>TS: read_f32("model.layers.{i}.mlp.up_proj.weight")
        CM->>TS: read_f32("model.layers.{i}.mlp.down_proj.weight")
        opt if q_norm exists (Qwen3)
            CM->>TS: read_f32("model.layers.{i}.self_attn.q_norm.weight")
            Note over CM: repeat_head_norm() to full attention_size
        end
    end

    Note over CM: Read final norm + lm_head
    CM->>TS: read_f32("model.norm.weight")
    CM->>TS: read_f32("lm_head.weight")
    Note over CM: If absent → tied to embedding

    CM-->>ML: DecoderCheckpoint
```

### Key transformations during mapping

| Transformation | When | Why |
|---------------|------|-----|
| **Transpose** | All weight matrices | HF stores `[out, in]`, TRT matmul needs `[in, out]` for right-side constant |
| **GQA KV expansion** | `num_kv_heads < num_attention_heads` | `expand_kv_projection()` repeats KV heads to match Q heads, so the TRT graph builder doesn't need special GQA logic |
| **Per-head norm repeat** | When `q_norm`/`k_norm` present | `repeat_head_norm()` expands `[head_dim]` to `[num_heads × head_dim]` for per-head RMSNorm |
| **Tied lm_head** | When `lm_head.weight` absent | Reuses embedding matrix (transposed) as LM head projection |

---

## 5. TRT Engine Build

How `StandardDecoderGraphBuilder` constructs the TensorRT network graph.

```mermaid
sequenceDiagram
    participant SDB as StandardDecoderGraphBuilder
    participant NV as TensorRT API
    participant OPS as trt_graph_ops
    participant LC as EngineLifecycle
    participant Cache as EngineCache

    SDB->>NV: createInferBuilder(logger)
    SDB->>NV: createNetworkV2(flags)
    SDB->>NV: createBuilderConfig()
    SDB->>NV: setMemoryPoolLimit(1GB)

    Note over SDB: Add network inputs
    SDB->>NV: addInput("token_id", INT32, [1])
    SDB->>NV: addInput("position_id", INT32, [1])
    SDB->>NV: addInput("attention_mask", FLOAT, [1, window])
    loop each layer
        SDB->>NV: addInput("cache_k_{i}", FLOAT, [max_cache, attn_size])
        SDB->>NV: addInput("cache_v_{i}", FLOAT, [max_cache, attn_size])
    end

    Note over SDB: Add embedding lookup
    SDB->>OPS: add_constant_tensor(embedding)
    SDB->>NV: addGather(embedding, token_id)

    Note over SDB: Precompute RoPE tables as constants
    SDB->>OPS: make_rope_table(cos) → constant
    SDB->>OPS: make_rope_table(sin) → constant
    SDB->>OPS: make_rotate_half_matrix() → constant

    Note over SDB: Build decoder layers
    loop each layer 0..N-1
        SDB->>SDB: add_standard_decoder_layer_block()
        Note over SDB: RMSNorm → QKV proj → QK norm → RoPE →<br/>KV cache concat → GQA attention →<br/>output proj → residual →<br/>RMSNorm → SwiGLU MLP → residual
    end

    Note over SDB: Final norm + LM head
    SDB->>OPS: add_rms_norm(hidden, final_norm)
    SDB->>OPS: add_matmul_rhs_constant(hidden, w_out)
    SDB->>NV: markOutput(logits)
    loop each layer
        SDB->>NV: markOutput(present_k_{i}, present_v_{i})
    end

    Note over SDB: Build engine (or load from cache)
    SDB->>LC: finalize_decoder_step_engine(builder, network, config)
    LC->>Cache: check cache for serialized plan
    alt cache hit
        Cache-->>LC: serialized plan bytes
        LC->>NV: deserializeCudaEngine(plan)
    else cache miss
        LC->>NV: buildEngineWithConfig(network, config)
        Note over NV: Kernel compilation (30+ seconds)
        LC->>Cache: store serialized plan
    end
    LC-->>SDB: DecoderStepEngine
```

---

## 6. Autoregressive Generation

How `TrtBackendShared::generate()` runs the prefill + decode loop.

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant T as ITokenizer
    participant B as TrtBackendShared
    participant S as KvCacheStepState
    participant E as DecoderStepEngine
    participant GPU as CUDA GPU

    P->>T: encode("Hello world")
    T-->>P: [token_1, token_2]
    P->>B: generate([token_1, token_2], config)

    Note over B: Create step state (IStepState)
    B->>S: KvCacheStepState(engine)
    Note over S: Allocates per-layer cache_k, cache_v

    Note over B: Prefill phase
    loop for each input token except last
        B->>S: prepare_step(position_id, mask)
        S-->>B: position_id, attention_mask
        B->>E: run_decoder_step(token, position, mask, state.caches)
        E->>GPU: enqueueV3(stream) → execute TRT engine
        GPU-->>E: logits[vocab_size] + present_k/v
        B->>S: update_after_step(present_k, present_v)
        Note over S: append_cache_state(), cache_length++
    end

    Note over B: Decode phase
    loop step = 0..max_new_tokens
        B->>S: prepare_step(position_id, mask)
        S-->>B: position_id, attention_mask
        B->>E: run_decoder_step(current_token, position, mask, state.caches)
        E->>GPU: enqueueV3(stream)
        GPU-->>E: logits[vocab_size] + present_k/v
        B->>S: update_after_step(present_k, present_v)
        B->>B: select_argmax_token(logits) → next_token
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

- **State management is abstracted via `IStepState`**. `KvCacheStepState` implements it for standard attention models. Mamba/SSM models can provide `RecurrentStepState`, hybrid models can combine both.
- **KV cache is a fixed-size circular buffer** per layer: `[max_cache_length, attention_size]`. `append_cache_state()` writes new K/V at `position % max_cache_length`.
- **Attention mask** is built each step: `0.0` for visible positions, `-1e9` for masked. Grows by 1 each step.
- **Greedy sampling** via `select_argmax_token()` — scans logits array for maximum value.
- **Greedy sampling** via `select_argmax_token()` -- scans logits array for maximum value.

---

## 7. Single Decode Step (TRT)

Detailed view of what happens inside `run_decoder_step()`.

```mermaid
sequenceDiagram
    participant RT as run_decoder_step()
    participant Ctx as IExecutionContext
    participant GPU as CUDA Buffers
    participant Stream as CudaStream

    Note over RT: Set tensor addresses for inputs
    RT->>Ctx: setTensorAddress("token_id", &host_token)
    RT->>Ctx: setTensorAddress("position_id", &host_position)
    RT->>Ctx: setTensorAddress("attention_mask", device_mask)
    loop each layer
        RT->>Ctx: setTensorAddress("cache_k_{i}", device_cache_k[i])
        RT->>Ctx: setTensorAddress("cache_v_{i}", device_cache_v[i])
    end

    Note over RT: Set tensor addresses for outputs
    RT->>Ctx: setTensorAddress("logits", device_logits)
    loop each layer
        RT->>Ctx: setTensorAddress("present_k_{i}", device_present_k)
        RT->>Ctx: setTensorAddress("present_v_{i}", device_present_v)
    end

    Note over RT: Copy input data to GPU
    RT->>GPU: cudaMemcpyAsync(token_id, H→D)
    RT->>GPU: cudaMemcpyAsync(position_id, H→D)
    RT->>GPU: cudaMemcpyAsync(attention_mask, H→D)

    Note over RT: Execute TRT inference
    RT->>Ctx: enqueueV3(stream)
    Note over GPU: TRT runs fused CUDA kernels:<br/>embedding → N×(RMSNorm+attn+MLP) → logits

    Note over RT: Copy outputs from GPU
    RT->>GPU: cudaMemcpyAsync(logits, D→H)
    RT->>GPU: cudaMemcpyAsync(present_k/v, D→H)
    RT->>Stream: cudaStreamSynchronize()

    RT-->>RT: return logits + present_k/v
```

---

## 8. Data Transformation Pipeline

End-to-end data transformation from HF files to generated text.

```mermaid
flowchart TD
    A[("HF Model Directory<br/>config.json + model.safetensors<br/>+ tokenizer.json")] --> B

    subgraph "Stage 1+2: Loading"
        B["config.json Parser<br/><i>model_loader.cpp</i>"] --> C["DecoderArchitectureConfig<br/><i>family, dims, heads, rope_theta</i>"]
        A --> D["SafetensorReader<br/><i>safetensors_loader.cpp</i>"]
        D --> E["TensorSource<br/><i>Single or sharded access</i>"]
        E --> F["CheckpointMapper<br/><i>(Registry 2)</i>"]
        C --> F
        F --> G["DecoderCheckpoint<br/><i>Per-layer: input_norm, w_qkvo,<br/>post_attn_norm, w_gate/up/down</i>"]
    end

    C --> H["DecoderModel<br/><i>Unified representation</i>"]
    G --> H

    subgraph "Stage 3a: TRT Definition"
        H --> I["BuildTrtDecoderWeights()<br/><i>(inlined conversion)</i>"]
        I --> J["TrtDecoderDefinition<br/><i>TRT-ready weights + arch params</i>"]
    end

    subgraph "Stage 3b: TRT Engine"
        J --> K["StandardDecoderGraphBuilder<br/><i>(Registry 3)</i>"]
        K --> L["INetworkDefinition<br/><i>TRT graph: embed → N layers → logits</i>"]
        L --> M["finalize_decoder_step_engine()<br/><i>Compile or load from cache</i>"]
        M --> N["DecoderStepEngine<br/><i>ICudaEngine + IExecutionContext</i>"]
    end

    subgraph "Stage 3c: Generation"
        N --> O["TrtBackendShared::generate()<br/><i>Prefill + Decode loop</i>"]
        O --> P["run_decoder_step()<br/><i>GPU kernel execution</i>"]
        P --> Q["select_argmax_token()<br/><i>Greedy sampling</i>"]
        Q -->|"loop until EOS"| O
    end

    A --> R["HfPythonTokenizer<br/><i>or VocabTokenizer</i>"]
    R -->|"encode(prompt)"| O
    O -->|"token IDs"| R
    R -->|"decode(ids)"| S(["Generated Text"])

    style A fill:#f1f5f9,stroke:#94a3b8
    style H fill:#dbeafe,stroke:#2563eb
    style J fill:#fdf2f8,stroke:#db2777
    style N fill:#dcfce7,stroke:#16a34a
    style S fill:#fef3c7,stroke:#f59e0b
```

### Data formats at each stage

| Stage | Data Format | Key Properties |
|-------|-------------|---------------|
| HF files | Safetensors binary (F32/F16/BF16) + JSON config | HF tensor naming, `[out, in]` weight layout |
| After CheckpointMapper | `DecoderCheckpoint` with `vector<float>` per tensor | Transposed to `[in, out]`, GQA KV expanded, canonical field names |
| After BuildTrtDecoderWeights | `TrtDecoderDefinition` with per-layer `TrtDecoderLayerDefinition` | Same data as checkpoint but organized for TRT graph builder consumption, with validated sizes |
| TRT Network | `INetworkDefinition` with `ITensor*` nodes | Weights baked as constant tensors, ops fused by TRT compiler |
| Compiled Engine | `ICudaEngine` serialized plan | Optimized CUDA kernels, can be cached to disk |
| Runtime | Host `int32_t` token IDs + device `float` KV caches | Prefill processes input tokens, decode generates one token per step |
