# Pipeline Deep Dive

This page traces the complete execution path from user code to generated text, with file references and key data structures.

## The Call Chain

The primary entry point is `trtf_create_pipeline()` / `trtf_create_pipeline_ex()` (C ABI, `src/cabi/trtf_c.cpp`).

```
User: trtf_create_pipeline("QWEN3", TRTF_FORCE_TRT)
  │     src/cabi/trtf_c.cpp
  │
  ├── Stage 1: ResolveTextGenerationModel("QWEN3")
  │     src/model/model_resolver.cpp
  │     │
  │     └── ResolveHfModelViaFamilyRegistry("QWEN3")
  │           src/model/hf_family_registry.cpp
  │           │
  │           ├── resolve_model_dir_from_alias("QWEN3")
  │           │     → "models/hf/Qwen__Qwen3-0.6B" (or demo fallback)
  │           │
  │           ├── RegisterBuiltinHfModelFamilies() [once]
  │           │     ├── qwen::RegisterQwenFamily()
  │           │     ├── llama::RegisterLlamaFamily()
  │           │     ├── mistral::RegisterMistralFamily()
  │           │     └── gemma::RegisterGemmaFamily()
  │     │           │
  │     │           ├── load_hf_metadata() → {model_type: "qwen3", architectures: [...]}
  │     │           │
  │     │           └── For each family (priority-sorted):
  │     │                 qwen matcher(metadata) → true
  │     │                 qwen loader(metadata):
  │     │                   │
  │     │                   └── Stage 2: LoadDecoderModel(model_dir)
  │     │                         src/model/model_loader.cpp
  │     │                         │
  │     │                         ├── Parse config.json → architecture, dims
  │     │                         ├── Detect safetensors → create TensorSource
  │     │                         ├── FindCheckpointMapper("qwen3") → QwenCheckpointMapper
  │                         └── map_checkpoint() → DecoderCheckpoint
  │                               src/models/qwen/checkpoint_mapper.cpp
  │                               (inherits StandardCheckpointMapper)
  │
  ├── Stage 3: BuildRuntimeForTextGeneration(spec, selection)
  │     src/runtime/runtime_factory.cpp
  │           │
  │           ├── CreateHfPythonTokenizer(hf_tokenizer_dir)
  │           │     src/tokenizer/hf_python_tokenizer.cpp
  │           │
  │           └── CreateTrtBackend(tokenizer, model)
  │                 src/runtime/trt_backend.cpp
  │                 │
  │                 ├── FindTrtGraphBuilder("qwen3") → StandardDecoderGraphBuilder
  │                 │
  │                 └── CreateTrtBackendWithBuilder(tokenizer, model, builder)
  │                       src/runtime/trt/trt_backend_shared.cpp
  │                       │
  │                       ├── BuildTrtDecoderWeights(model) → TrtDecoderDefinition
  │                       │     (inlined conversion in trt_model_definition.cpp)
  │                       │
  │                       ├── builder.build_decoder_step_engine(definition, logger)
  │                       │     src/runtime/trt/standard_decoder_graph_builder.cpp
  │                       │     → creates ICudaEngine
  │                       │
  │                       └── TrtBackendShared (holds engine, CUDA buffers)
  │
  └── pipeline.generate("Hello")
        │
        ├── tokenizer->encode("Hello") → [token_ids]
        ├── backend->generate(token_ids, config)
        │     TrtBackendShared::generate()
        │     ├── Prefill: run_decoder_step() for each input token
        │     └── Decode: run_decoder_step() + argmax sampling, repeat
        └── tokenizer->decode(output_ids) → "Hello Answer is a..."
```

## Key Data Structures

### ResolvedModelSpec

The output of Stage 1. Tells Stage 3 what kind of model we have:

```cpp
struct ResolvedModelSpec {
    std::string model_id;            // Original user-provided ID
    ResolvedModelKind kind;          // kDecoderDefinition | kHuggingFaceLocal
    DecoderModel decoder_model;      // Populated for kDecoderDefinition
    std::string huggingface_model_dir; // Populated for kHuggingFaceLocal
};
```

### DecoderModel

The unified internal representation. All model sources (HF, built-in, custom) converge here:

```cpp
struct DecoderModel {
    std::string model_id;
    std::vector<std::string> vocab;                        // Token vocabulary
    int32_t max_cache_length;
    DecoderArchitectureConfig architecture;                // Family, dims, RoPE, etc.
    bool prefer_hf_tokenizer;
    std::string hf_tokenizer_dir;
    bool has_checkpoint;
    DecoderCheckpoint checkpoint;                          // All weight tensors
};
```

### DecoderCheckpoint

Contains all loaded weight tensors:

```cpp
struct DecoderCheckpoint {
    int32_t hidden_size, attention_size, mlp_size;

    // Multi-layer fields (for HF models)
    bool has_decoder_layers;
    std::vector<DecoderLayerCheckpoint> decoder_layers;  // Per-layer weights
    std::vector<float> final_norm;                       // Final RMSNorm weights
};

struct DecoderLayerCheckpoint {
    std::vector<float> input_norm;     // Pre-attention RMSNorm gamma
    std::vector<float> q_norm, k_norm; // Per-head QK norm (empty for LLaMA)
    std::vector<float> w_q, w_k, w_v, w_o;  // Attention projections [in, out]
    std::vector<float> post_attn_norm; // Post-attention RMSNorm gamma
    std::vector<float> w_gate, w_up, w_down; // SwiGLU MLP weights
};
```

### DecoderArchitectureConfig

Model architecture metadata parsed from `config.json`:

```cpp
struct DecoderArchitectureConfig {
    std::string family;          // "qwen3", "llama", etc.
    int32_t num_layers;
    int32_t num_attention_heads;
    int32_t num_key_value_heads; // < num_attention_heads for GQA
    int32_t bos_token_id, eos_token_id, pad_token_id;
    float rms_norm_eps;
    float rope_theta;
};
```

### TrtDecoderDefinition

TRT-ready version of the model weights, produced by `BuildTrtDecoderWeights()`:

```cpp
struct TrtDecoderDefinition {
    int32_t vocab_size, hidden_size, attention_size, mlp_size;
    int32_t max_cache_length;
    int32_t bos_token_id, eos_token_id;
    int32_t num_attention_heads, num_key_value_heads;
    float rms_norm_eps, rope_theta;

    std::vector<float> embedding;  // [vocab, hidden]
    std::vector<float> w_out;      // [hidden, vocab] (lm_head)
    std::vector<float> b_out;      // [vocab] (bias, may be zeros)
    std::vector<float> final_norm; // [hidden]

    bool has_decoder_layers;
    std::vector<TrtDecoderLayerDefinition> decoder_layers;
};
```

### RuntimeAssembly

The output of Stage 3, consumed by the Pipeline constructor:

```cpp
struct RuntimeAssembly {
    std::unique_ptr<ITokenizer> tokenizer;
    std::unique_ptr<IGenerationBackend> backend;
    std::string backend_name;  // "trt" or "hf-transformers"
};
```

## SafeTensors Loading

The `SafetensorReader` (`src/model/safetensors_loader.cpp`) parses the safetensors binary format:

```
[8 bytes: header_size (LE uint64)]
[header_size bytes: JSON header mapping tensor names to metadata]
[remaining bytes: raw tensor data]
```

Each tensor entry in the header:
```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "F32",
    "shape": [896, 896],
    "data_offsets": [0, 3211264]
  }
}
```

Supported dtypes: `F32`, `F16`, `BF16`. F16/BF16 are converted to F32 during loading.

For **sharded models**, `TensorSource` reads `model.safetensors.index.json` which maps tensor names to shard files:
```json
{
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
    "model.layers.27.mlp.down_proj.weight": "model-00003-of-00003.safetensors"
  }
}
```

## Config.json Parsing

The model loader reads these fields from `config.json`:

| JSON Field | Maps to | Used for |
|-----------|---------|----------|
| `model_type` | `architecture.family` | Family detection in Registry 1 |
| `hidden_size` | `checkpoint.hidden_size` | Weight dimension validation |
| `intermediate_size` | `checkpoint.mlp_size` | MLP weight dimensions |
| `num_hidden_layers` | `architecture.num_layers` | Layer count |
| `num_attention_heads` | `architecture.num_attention_heads` | Attention head count |
| `num_key_value_heads` | `architecture.num_key_value_heads` | GQA head count |
| `rms_norm_eps` | `architecture.rms_norm_eps` | RMSNorm epsilon |
| `rope_theta` | `architecture.rope_theta` | RoPE base frequency |
| `vocab_size` | Used for placeholder vocab | Embedding/lm_head dimensions |
| `bos_token_id` | `architecture.bos_token_id` | Sequence start marker |
| `eos_token_id` | `architecture.eos_token_id` | Stop generation |
| `max_position_embeddings` | `max_cache_length` | KV cache size |

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `TRTF_MAX_NEW_TOKENS` | 20 | Override max generation tokens |
| `TRTF_MAX_CACHE_LENGTH` | from config | Cap KV cache length (saves GPU memory) |
| `TRTF_HF_PYTHON` | none | Path to Python for HF tokenizer/backend |
| `TRTF_TRT_ENGINE_CACHE_DIR` | none | Directory for serialized TRT engine plans |
| `TRTF_DISABLE_ENGINE_CACHE` | 0 | Force engine rebuild every run |
| `TRTF_TRT_LOG_STDERR` | 0 | Enable TRT logger output to stderr |
