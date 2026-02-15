# Static Design: Class-Level UML and Logical Descriptions

This page decomposes the architecture into software units, showing class-level UML diagrams and describing the logical role and implementation of each class.

## Table of Contents

1. [Unit 1: Public API Layer](#unit-1-public-api-layer)
2. [Unit 2: Model Data Structures](#unit-2-model-data-structures)
3. [Unit 3: Model Resolution and Loading](#unit-3-model-resolution-and-loading)
4. [Unit 4: Registry System](#unit-4-registry-system)
5. [Unit 5: TRT Backend Infrastructure](#unit-5-trt-backend-infrastructure)
6. [Unit 6: Tokenization](#unit-6-tokenization)
7. [Unit 7: Alternative Backends](#unit-7-alternative-backends)

---

## Unit 1: Public API Layer

The public API has two tiers: (1) the **C ABI entry point** (`trtf_create_pipeline` → `IPipeline*`) for library consumers, and (2) the **legacy Pipeline class** retained for backward compatibility with existing examples.

### Class Diagram

```mermaid
classDiagram
    class IPipeline {
        <<interface>>
        +generate(prompt, max_new_tokens)* const char*
        +model_id()* const char*
        +backend_name()* const char*
        +save_bundle(output_path)* bool
    }

    class PipelineImpl {
        -string mModelId
        -unique_ptr~ITokenizer~ mTokenizer
        -unique_ptr~IGenerationBackend~ mBackend
        -string mBackendName
        -GenerationConfig mGenConfig
        -string mLastOutput
        +generate(prompt, max_new_tokens) const char*
        +model_id() const char*
        +backend_name() const char*
        +save_bundle(output_path) bool
    }

    class Pipeline {
        -string mTask
        -string mModelId
        -unique_ptr~ITokenizer~ mTokenizer
        -unique_ptr~IGenerationBackend~ mBackend
        -string mBackendName
        -GenerationConfig mGenerationConfig
        +CreateTextGeneration(model_id, prefer_trt, force_trt)$ Pipeline
        +LoadModel(model_id, prefer_trt, force_trt)$ Pipeline
        +operator()(prompt) vector~GenerationResult~
        +generate(prompt) string
        +task() string
        +model_id() string
        +backend_name() string
    }

    IPipeline <|.. PipelineImpl

    note for IPipeline "C ABI: trtf_create_pipeline() returns IPipeline*\nconst char* only, no std::string"

    class ITokenizer {
        <<interface>>
        +encode(text)* vector~int32_t~
        +decode(ids)* string
        +id_for_token(token)* int32_t
        +token_for_id(id)* string
    }

    class IGenerationBackend {
        <<interface>>
        +is_available()* bool
        +name()* string
        +unavailable_reason() string
        +generate(input_ids, config)* vector~int32_t~
        +supports_text_generation() bool
        +generate_text(prompt, config) string
    }

    class GenerationConfig {
        +size_t max_new_tokens = 20
        +bool do_sample = false
        +float temperature = 1.0
    }

    class GenerationResult {
        +string generated_text
    }

    Pipeline *-- ITokenizer : owns
    Pipeline *-- IGenerationBackend : owns
    Pipeline *-- GenerationConfig : holds
    Pipeline ..> GenerationResult : produces
```

### Logical Description

| Class | Role | Implementation |
|-------|------|---------------|
| **Pipeline** | Orchestration facade. Resolves model, assembles runtime, and provides generation API. | Static factory `CreateTextGeneration()` calls `ResolveTextGenerationModel()` (Stage 1) then `BuildRuntimeForTextGeneration()` (Stage 3). `operator()` encodes prompt via tokenizer, calls backend's `generate()`, decodes output. Handles `TRTF_MAX_NEW_TOKENS` env override. File: `src/pipeline/pipeline.cpp`. |
| **ITokenizer** | Abstract interface for text-to-token-ids and back. | Pure virtual. Two concrete implementations: `ToyTokenizer` (vocab.txt lookup) and `HfPythonTokenizer` (subprocess bridge). See [Unit 6](#unit-6-tokenization). |
| **IGenerationBackend** | Abstract interface for token-level autoregressive generation. | Pure virtual `generate()` takes input token IDs and returns input + generated IDs. `is_available()` for runtime capability check. Optional `generate_text()` for text-level backends (HF Python). Three implementations: `TrtBackendShared`, `CpuReferenceBackend`, `HfPythonBackend`. |
| **GenerationConfig** | Value struct for generation parameters. | Currently supports `max_new_tokens`, `do_sample`, `temperature`. Passed through from Pipeline to backend. |

---

## Unit 2: Model Data Structures

These structs form the internal data model. All model sources (HF safetensors, built-in, custom) converge to `DecoderModel`.

### Class Diagram

```mermaid
classDiagram
    class DecoderModel {
        +string model_id
        +vector~string~ vocab
        +vector~pair~ transitions
        +string default_next_token
        +int32_t max_cache_length
        +DecoderArchitectureConfig architecture
        +bool prefer_hf_tokenizer
        +string hf_tokenizer_dir
        +bool has_checkpoint
        +DecoderCheckpoint checkpoint
    }

    class DecoderArchitectureConfig {
        +string family
        +int32_t num_layers
        +int32_t num_attention_heads
        +int32_t num_key_value_heads
        +int32_t bos_token_id
        +int32_t eos_token_id
        +int32_t pad_token_id
        +float rms_norm_eps
        +float rope_theta
        +map~string,int32_t~ extra_int_params
        +map~string,float~ extra_float_params
        +map~string,string~ extra_string_params
    }

    class DecoderCheckpoint {
        +int32_t hidden_size
        +int32_t attention_size
        +int32_t mlp_size
        +vector~float~ embedding
        +vector~float~ w_q, w_k, w_v
        +vector~float~ w1, b1, w2, b2
        +vector~float~ w_out, b_out
        +bool has_decoder_layers
        +vector~DecoderLayerCheckpoint~ decoder_layers
        +vector~float~ final_norm
    }

    class DecoderLayerCheckpoint {
        +vector~float~ input_norm
        +vector~float~ q_norm
        +vector~float~ k_norm
        +vector~float~ w_q, w_k, w_v, w_o
        +vector~float~ post_attn_norm
        +vector~float~ w_gate, w_up, w_down
        +map~string,vector~float~~ extra_tensors
    }

    class ResolvedModelSpec {
        +string model_id
        +ResolvedModelKind kind
        +DecoderModel decoder_model
        +string huggingface_model_dir
        +string custom_type
        +string custom_model_dir
    }

    class HfModelMetadata {
        +string model_dir
        +string model_type
        +vector~string~ architectures
    }

    DecoderModel *-- DecoderArchitectureConfig
    DecoderModel *-- DecoderCheckpoint
    DecoderCheckpoint *-- "0..*" DecoderLayerCheckpoint
    ResolvedModelSpec *-- DecoderModel
```

### Logical Description

| Struct | Role | Key Details |
|--------|------|-------------|
| **DecoderModel** | Unified representation of any loaded model. Consumed by all backends. | Contains vocab (for CPU-reference transitions), architecture metadata (parsed from `config.json`), and optional checkpoint weight tensors. `has_checkpoint=true` when safetensors/weights.txt loaded. `prefer_hf_tokenizer` set when `tokenizer.json` detected. File: `include/trtf/model.h`. |
| **DecoderArchitectureConfig** | Architecture metadata from `config.json`. | `family` drives registry dispatch (e.g., `"qwen3"`, `"llama"`). `num_key_value_heads < num_attention_heads` indicates GQA. `rope_theta` configures RoPE base frequency. Parsed in `model_loader.cpp`. |
| **DecoderCheckpoint** | All weight tensors in canonical format. | Legacy flat fields (`w_q`, `w1`, etc.) for tiny built-in models. Multi-layer fields (`decoder_layers`, `final_norm`) for real HF models. `attention_size` may differ from `hidden_size` after GQA KV expansion. |
| **DecoderLayerCheckpoint** | Per-layer weights for one transformer decoder layer. | `q_norm`/`k_norm` are empty for models without per-head norms (LLaMA, Mistral). Non-empty for Qwen3. All projection matrices stored as `[in_features, out_features]` (transposed from HF's `[out, in]` during checkpoint mapping). |
| **ResolvedModelSpec** | Output of Stage 1 (model resolution). | `kind` determines which Stage 3 path runs: `kDecoderDefinition` (TRT/CPU), `kHuggingFaceLocal` (HF Python subprocess), `kCustom` (user extension). |
| **HfModelMetadata** | Parsed metadata from an HF `config.json`. | Used by family matchers. `model_type` is the primary dispatch key (e.g., `"qwen3"`, `"llama"`). `architectures` is secondary (e.g., `["Qwen2ForCausalLM"]`). |

---

## Unit 3: Model Resolution and Loading

These components turn a `model_id` string into a fully-loaded `DecoderModel`.

### Class Diagram

```mermaid
classDiagram
    class SafetensorReader {
        -vector~SafetensorEntry~ mEntries
        -string mHeaderJson
        -filesystem_path mPath
        +SafetensorReader(path)
        +tensor_names() vector~string~
        +read_f32(name) vector~float~
        +shape(name) vector~int64_t~
        +has_tensor(name) bool
    }

    class TensorSource {
        -vector~SafetensorReader~ mReaders
        -map~string,size_t~ mTensorToReader
        +TensorSource(model_dir, index_path)
        +TensorSource(single_file_path)
        +tensor_names() vector~string~
        +read_f32(name) vector~float~
        +shape(name) vector~int64_t~
        +has_tensor(name) bool
    }

    class SafetensorEntry {
        +string name
        +string dtype
        +vector~int64_t~ shape
        +uint64_t data_offset_begin
        +uint64_t data_offset_end
    }

    class HfModelFamilyRegistration {
        +string family_name
        +int priority
        +HfFamilyMatcher matcher
        +HfFamilyModelDefinitionLoader model_definition_loader
    }

    TensorSource o-- "1..*" SafetensorReader : contains
    SafetensorReader *-- "0..*" SafetensorEntry : parsed from header

    note for TensorSource "Handles both single-file and\nsharded (index.json) safetensors"
```

### Logical Description

| Class/Struct | Role | Implementation |
|-------------|------|---------------|
| **SafetensorReader** | Parses a single `.safetensors` file. | Reads the 8-byte header length, parses JSON header to extract tensor names/shapes/offsets, then reads raw data on demand. Supports F32, F16, BF16 dtypes with conversion to float32. File: `src/model/safetensors_loader.cpp`. |
| **TensorSource** | Unified access to single or sharded safetensors. | For single files: wraps one `SafetensorReader`. For sharded models: reads `model.safetensors.index.json`, creates one `SafetensorReader` per shard file, and routes `read_f32(name)` to the correct shard. File: `src/model/safetensors_loader.cpp`. |
| **HfModelFamilyRegistration** | Registration entry for an HF model family. | `matcher` is a `std::function<bool(HfModelMetadata)>` — typically checks `model_type` prefix. `model_definition_loader` is a `std::function<DecoderModel(HfModelMetadata)>` — typically calls `LoadDecoderModel()` after checkpoint mapper is registered. |
| **LoadDecoderModel()** | Free function: generic model loader. | Reads `config.json`, derives architecture, detects vocab source (vocab.txt or placeholder), detects checkpoint format (weights.txt, single safetensors, sharded safetensors), dispatches to `FindCheckpointMapper()` for HF tensor key translation. File: `src/model/model_loader.cpp`. |

---

## Unit 4: Registry System

Four registries provide the plug-and-play extension mechanism. All use priority-sorted dispatch with mutex-protected registration.

### Class Diagram

```mermaid
classDiagram
    class ICheckpointMapper {
        <<interface>>
        +can_map(architecture)* bool
        +map_checkpoint(reader, vocab_size, path, architecture)* DecoderCheckpoint
    }

    class StandardCheckpointMapper {
        +map_checkpoint(reader, vocab_size, path, architecture) DecoderCheckpoint
    }

    class QwenCheckpointMapper {
        +can_map(architecture) bool
    }

    class LlamaCheckpointMapper {
        +can_map(architecture) bool
    }

    class ITrtModelDefinitionPopulator {
        <<interface>>
        +can_populate(model)* bool
        +populate(definition, model)* bool
    }

    class StandardTrtModelDefinitionPopulator {
        +can_populate(model) bool
        +populate(definition, model) bool
    }

    class ITrtGraphBuilder {
        <<interface>>
        +build_decoder_step_engine(weights, logger)* unique_ptr~DecoderStepEngine~
    }

    class StandardDecoderGraphBuilder {
        +build_decoder_step_engine(weights, logger) unique_ptr~DecoderStepEngine~
    }

    ICheckpointMapper <|-- StandardCheckpointMapper
    StandardCheckpointMapper <|-- QwenCheckpointMapper
    StandardCheckpointMapper <|-- LlamaCheckpointMapper

    ITrtModelDefinitionPopulator <|-- StandardTrtModelDefinitionPopulator

    ITrtGraphBuilder <|-- StandardDecoderGraphBuilder

    note for StandardCheckpointMapper "Handles standard HF tensor naming.\nTransposes weights, expands GQA KV,\ndetects tied lm_head, handles q_norm/k_norm."
    note for StandardTrtModelDefinitionPopulator "Priority=0 fallback. Handles any model\nwith has_decoder_layers. Validates tensor\nsizes and copies to TrtDecoderDefinition."
    note for StandardDecoderGraphBuilder "Pre-RMSNorm + GQA + RoPE + SwiGLU.\nTwo paths: multi-layer (real) and\nlegacy (tiny built-in)."
```

### Logical Description

| Class | Role | Implementation |
|-------|------|---------------|
| **ICheckpointMapper** | Registry 2 interface. Translates HF safetensors tensor keys to canonical `DecoderCheckpoint`. | `can_map()` checks if this mapper handles the given `architecture.family`. `map_checkpoint()` reads tensors from `TensorSource` and populates `DecoderCheckpoint`. Priority-sorted registry in `checkpoint_mapper.cpp`. |
| **StandardCheckpointMapper** | Base class for the standard HF tensor naming convention. | Handles `model.embed_tokens.weight`, `model.layers.N.self_attn.{q,k,v,o}_proj.weight`, `model.layers.N.mlp.{gate,up,down}_proj.weight`, `model.norm.weight`, `lm_head.weight`. Transposes all weight matrices (`[out,in]` → `[in,out]`). Expands K/V for GQA via `expand_kv_projection()`. Repeats per-head norms via `repeat_head_norm()`. Detects tied `lm_head`. File: `src/model/standard_checkpoint_mapper.cpp` (260 LOC). |
| **QwenCheckpointMapper** | Qwen family mapper. | Inherits all logic from `StandardCheckpointMapper`. Only overrides `can_map()` to match `starts_with(family, "qwen")` or `"qwq"`. 15 LOC. File: `src/models/qwen/checkpoint_mapper.cpp`. |
| **LlamaCheckpointMapper** | LLaMA family mapper. | Same pattern: inherits `StandardCheckpointMapper`, overrides `can_map()` for `"llama"`. 15 LOC. File: `src/models/llama/checkpoint_mapper.cpp`. |
| **ITrtModelDefinitionPopulator** | Registry 3 interface. Converts `DecoderModel` → `TrtDecoderDefinition`. | `can_populate()` checks if the model is compatible. `populate()` transfers checkpoint data into TRT-ready format with size validation. Priority-sorted registry in `trt_model_definition_populator.cpp`. |
| **StandardTrtModelDefinitionPopulator** | Family-agnostic populator for any model with `has_decoder_layers`. | Validates all tensor sizes match expected dimensions. Copies per-layer weights into `TrtDecoderLayerDefinition` vectors. Sets RoPE theta, RMSNorm eps, attention heads from architecture config. Registered at priority 0 as automatic fallback. File: `src/model/standard_trt_model_definition_populator.cpp` (105 LOC). |
| **ITrtGraphBuilder** | Registry 4 interface. Builds TRT `INetworkDefinition` from `TrtDecoderDefinition`. | `build_decoder_step_engine()` returns a compiled `DecoderStepEngine`. Name-based registry in `trt_graph_builder.cpp`. Lookup: exact family name → `"standard-decoder"` fallback. |
| **StandardDecoderGraphBuilder** | Builds the dominant LLM decoder pattern in TRT. | Creates `INetworkDefinition` with embedding gather, N decoder layers (each: RMSNorm → QKV → optional QK norm → RoPE → GQA attention → residual → RMSNorm → SwiGLU MLP → residual), final RMSNorm, LM head projection. Uses reusable ops from `trt_graph_ops.h`. Two paths: `create_decoder_step_engine_multi_layer()` for real models, `create_decoder_step_engine_legacy()` for tiny built-in. File: `src/runtime/trt/standard_decoder_graph_builder.cpp` (620 LOC). |

### Registry Free Functions

```mermaid
classDiagram
    class RegistryFunctions {
        <<module>>
        +RegisterCheckpointMapper(family, priority, mapper)
        +FindCheckpointMapper(architecture) ICheckpointMapper*
        +RegisterTrtModelDefinitionPopulator(family, priority, populator)
        +PopulateViaRegistry(definition, model) bool
        +RegisterTrtGraphBuilder(family, builder)
        +FindTrtGraphBuilder(family) ITrtGraphBuilder*
        +RegisterHfModelFamily(registration)
        +ResolveHfModelViaFamilyRegistry(model_id) optional~ResolvedModelSpec~
        +RegisterBuiltinHfModelFamilies()
    }
```

---

## Unit 5: TRT Backend Infrastructure

The TRT backend comprises RAII resource wrappers, the compiled engine, decode runtime utilities, and the autoregressive generation loop.

### Class Diagram

```mermaid
classDiagram
    class TrtBackendShared {
        -const ITokenizer& mTokenizer
        -bool mAvailable
        -string mInitError
        -TrtLogger mLogger
        -TrtDecoderDefinition mWeights
        -unique_ptr~DecoderStepEngine~ mDecoderStepEngine
        +is_available() bool
        +name() string
        +unavailable_reason() string
        +generate(input_ids, config) vector~int32_t~
    }

    class DecoderStepEngine {
        +TrtUniquePtr~ICudaEngine~ engine
        +TrtUniquePtr~IExecutionContext~ context
        +string token_input_name
        +string position_input_name
        +string mask_input_name
        +string logits_output_name
        +vector~string~ cache_k_input_names
        +vector~string~ cache_v_input_names
        +vector~string~ present_k_output_names
        +vector~string~ present_v_output_names
        +vector~TensorBinding~ extra_bindings
        +int32_t num_layers
        +bool requires_position_input
        +int32_t vocab_size
        +int32_t hidden_size
        +int32_t cache_state_size
        +int32_t max_cache_length
    }

    class IStepState {
        <<interface>>
        +prepare_step(position_id, mask)* bool
        +cache_k_by_layer()* vector~vector~float~~
        +cache_v_by_layer()* vector~vector~float~~
        +update_after_step(present_k, present_v)*
    }

    class KvCacheStepState {
        -int32_t mCacheStateSize
        -int32_t mMaxCacheLength
        -int32_t mNumLayers
        -int32_t mCacheLength
        -vector~vector~float~~ mCacheK
        -vector~vector~float~~ mCacheV
        +prepare_step(position_id, mask) bool
        +cache_k_by_layer() vector~vector~float~~
        +cache_v_by_layer() vector~vector~float~~
        +update_after_step(present_k, present_v)
    }

    class TrtDecoderDefinition {
        +int32_t vocab_size
        +int32_t hidden_size
        +int32_t attention_size
        +int32_t mlp_size
        +int32_t max_cache_length
        +int32_t num_attention_heads
        +int32_t num_key_value_heads
        +float rms_norm_eps
        +float rope_theta
        +vector~float~ embedding
        +vector~float~ w_out, b_out
        +vector~float~ final_norm
        +bool has_decoder_layers
        +vector~TrtDecoderLayerDefinition~ decoder_layers
    }

    class TrtDecoderLayerDefinition {
        +vector~float~ input_norm
        +vector~float~ q_norm, k_norm
        +vector~float~ w_q, w_k, w_v, w_o
        +vector~float~ post_attn_norm
        +vector~float~ w_gate, w_up, w_down
        +map~string,vector~float~~ extra_tensors
    }

    class TrtLogger {
        -string mLastError
        +log(severity, msg)
        +last_error() string
        +clear_error()
    }

    class CudaStream {
        -cudaStream_t mStream
        -cudaError_t mStatus
        +CudaStream()
        +ok() bool
        +get() cudaStream_t
    }

    class CudaBuffer {
        -void* mPtr
        -size_t mBytes
        -cudaError_t mStatus
        +CudaBuffer(bytes)
        +ok() bool
        +data() void*
    }

    class FastPathModelConfig {
        +int32_t vocab_size
        +int32_t hidden_size
        +int32_t num_layers
        +int32_t num_heads
        +int32_t num_kv_heads
        +int32_t head_dim
        +int32_t attention_size
        +int32_t max_cache_length
        +int32_t id_bos
        +int32_t id_eos
    }

    IGenerationBackend <|-- TrtBackendShared
    TrtBackendShared *-- DecoderStepEngine : owns
    TrtBackendShared *-- TrtDecoderDefinition : holds
    TrtBackendShared *-- TrtLogger : holds
    TrtBackendShared ..> IStepState : creates during generate()
    TrtBackendShared ..> CudaStream : uses during generate()
    TrtBackendShared ..> CudaBuffer : uses during generate()
    IStepState <|-- KvCacheStepState
    TrtDecoderDefinition *-- "0..*" TrtDecoderLayerDefinition
    DecoderStepEngine ..> CudaStream : executes on

    note for FastPathModelConfig "Used by try_create_from_cached_engine()\nto populate DecoderStepEngine metadata\nfrom config.json without loading weights.\nFile: src/cabi/fast_path_config.h"
```

### Logical Description

| Class | Role | Implementation |
|-------|------|---------------|
| **TrtBackendShared** | `IGenerationBackend` implementation for TRT. Owns the compiled engine and runs the autoregressive loop. | Constructor calls `BuildTrtDecoderWeights()` (Registry 3 dispatch) then the provided engine factory (Registry 4 dispatch) to produce a `DecoderStepEngine`. `generate()` creates a `KvCacheStepState` (via `IStepState` interface), runs prefill then decode loop (greedy argmax until EOS or max tokens). Supports `TRTF_DEBUG_LOGITS_TOPK` for debugging. File: `src/runtime/trt/trt_backend_shared.cpp`. |
| **IStepState** | Abstract interface for per-step state management during autoregressive generation. | Defines `prepare_step()` (compute position/mask), `cache_k/v_by_layer()` (access current state), `update_after_step()` (incorporate new outputs). Enables Mamba/SSM models to provide recurrent state instead of KV cache. File: `src/runtime/trt/step_state.h`. |
| **KvCacheStepState** | KV-cache state implementation for standard attention-based decoders. | Manages per-layer `cache_k`/`cache_v` vectors, tracks `cache_length`, builds attention masks, appends new state via `append_cache_state()`. Extracted from the previous inline cache management in `generate()`. File: `src/runtime/trt/kv_cache_step_state.cpp`. |
| **DecoderStepEngine** | Compiled TRT engine + execution context + tensor binding metadata. | Created by `finalize_decoder_step_engine()` which builds `ICudaEngine` from `INetworkDefinition` (or loads from cache). Stores all tensor names for inputs (token_id, position_id, attention_mask, per-layer cache_k/v) and outputs (logits, per-layer present_k/v). Also supports `extra_bindings` for non-KV-cache models. File: `src/runtime/trt/trt_engine_lifecycle.h`. |
| **TrtDecoderDefinition** | TRT-ready model weights + architecture parameters. | Produced by `BuildTrtDecoderWeights()` which calls `PopulateViaRegistry()` (Registry 3). Contains all data needed by the graph builder: embedding, per-layer weights, final norm, LM head, dims, special token IDs, RoPE/RMSNorm params. File: `src/model/trt_model_definition.h`. |
| **TrtLogger** | TRT `ILogger` implementation. | Forwards TRT log messages to stderr when `TRTF_TRT_LOG_STDERR=1`. Captures last error for diagnostic reporting. Filters by `TRTF_TRT_LOG_MIN_SEVERITY`. File: `src/runtime/trt/trt_common.cpp`. |
| **CudaStream** | RAII wrapper for `cudaStream_t`. | Constructor calls `cudaStreamCreate()`, destructor calls `cudaStreamDestroy()`. Non-copyable. `ok()` checks `cudaSuccess`. File: `src/runtime/trt/trt_common.cpp`. |
| **CudaBuffer** | RAII wrapper for GPU memory allocation. | Constructor calls `cudaMalloc()`, destructor calls `cudaFree()`. Non-copyable. `ok()` checks `cudaSuccess`. File: `src/runtime/trt/trt_common.cpp`. |
| **FastPathModelConfig** | Lightweight config struct for the zero-weight fast path. | Parsed from `config.json` text by `parse_fast_path_config()` without loading any safetensors data. Contains dimensions, cache length (with 4096 cap), and special token IDs needed to populate `DecoderStepEngine` metadata. File: `src/cabi/fast_path_config.h/cpp`. |
| **CreateTrtBackendFromEngine()** | Creates a TRT backend from a pre-built `DecoderStepEngine` (fast path). | Wraps the engine in a `TrtBackendShared` without calling any graph builder or weight populator. Used when the engine was deserialized from a cached `.plan` file. File: `src/runtime/trt/trt_backend_shared.cpp`. |

### TRT Graph Ops (Reusable Building Blocks)

These free functions in `trt_graph_ops.h/cpp` build individual TRT network layers. Used by `StandardDecoderGraphBuilder` and available for custom builders:

| Function | Builds | Input → Output |
|----------|--------|---------------|
| `add_constant_tensor()` | Constant weight tensor | dims + float data → `ITensor*` |
| `add_matmul_rhs_constant()` | Matrix multiply with constant weights | `[1, M]` × `[M, N]` → `[1, N]` |
| `add_bias_sum()` | Elementwise add bias | `[1, N]` + `[N]` → `[1, N]` |
| `add_rms_norm()` | RMS normalization with gamma | `[1, H]` → `[1, H]` |
| `add_rms_norm_per_head()` | Per-head RMS norm (Qwen3 QK norms) | `[1, A]` → `[1, A]` where A = num_heads × head_dim |
| `add_apply_rope()` | Rotary position embedding | Q/K `[1, A]` + position → `[1, A]` |
| `make_rope_table()` | Precompute cos/sin tables | (max_pos, hidden, heads, theta) → `vector<float>` |
| `make_rotate_half_matrix()` | Build rotation matrix for RoPE | (hidden, heads) → `vector<float>` |

---

## Unit 6: Tokenization

Two tokenizer implementations, both implementing `ITokenizer`.

### Class Diagram

```mermaid
classDiagram
    class ITokenizer {
        <<interface>>
        +encode(text)* vector~int32_t~
        +decode(ids)* string
        +id_for_token(token)* int32_t
        +token_for_id(id)* string
    }

    class ToyTokenizer {
        -vector~string~ mVocab
        -unordered_map~string,int32_t~ mTokenToId
        -int32_t mUnkId
        +encode(text) vector~int32_t~
        +decode(ids) string
        +id_for_token(token) int32_t
        +token_for_id(id) string
    }

    class HfPythonTokenizer {
        -string mModelDir
        -string mPythonCommand
        -filesystem_path mScript
        +encode(text) vector~int32_t~
        +decode(ids) string
        +id_for_token(token) int32_t
        +token_for_id(id) string
    }

    ITokenizer <|-- ToyTokenizer
    ITokenizer <|-- HfPythonTokenizer
```

### Logical Description

| Class | Role | Implementation |
|-------|------|---------------|
| **ToyTokenizer** | Simple word-to-ID lookup for built-in models. | Loads vocabulary from `vector<string>`. `encode()` splits text on spaces/punctuation, lowercases, looks up each token. Unknown tokens map to `<unk>` ID. `decode()` joins tokens with spaces. File: `src/tokenizer/toy_tokenizer.cpp`. |
| **HfPythonTokenizer** | Bridges to HuggingFace `tokenizers` library via Python subprocess. | Each `encode()`/`decode()` call spawns a Python process running `scripts/hf_tokenizer.py`. Sanitizes output to strip library warnings. Guarantees exact parity with HF tokenization. Requires `TRTF_HF_PYTHON` env var. File: `src/tokenizer/hf_python_tokenizer.cpp`. |

---

## Unit 7: Alternative Backends

Two non-TRT backends for fallback and compatibility.

### Class Diagram

```mermaid
classDiagram
    class IGenerationBackend {
        <<interface>>
        +is_available()* bool
        +name()* string
        +generate(input_ids, config)* vector~int32_t~
    }

    class CpuReferenceBackend {
        -const ITokenizer& mTokenizer
        -unordered_map~int32_t,int32_t~ mNextTokenById
        -int32_t mDefaultNextId
        -int32_t mBosId
        -int32_t mEosId
        +is_available() bool
        +name() string
        +generate(input_ids, config) vector~int32_t~
    }

    class HfPythonBackend {
        -string mModelDir
        -string mPythonCommand
        -filesystem_path mRunnerScript
        -bool mAvailable
        -string mInitError
        +is_available() bool
        +name() string
        +generate(input_ids, config) vector~int32_t~
        +supports_text_generation() bool
        +generate_text(prompt, config) string
    }

    IGenerationBackend <|-- CpuReferenceBackend
    IGenerationBackend <|-- HfPythonBackend
```

### Logical Description

| Class | Role | Implementation |
|-------|------|---------------|
| **CpuReferenceBackend** | Deterministic CPU fallback using transition tables. Always available. | Constructor builds `mNextTokenById` map from `DecoderModel.transitions`. `generate()` iterates: look up current token → get next token → append. Stops on EOS or max tokens. No neural network computation. File: `src/runtime/cpu_reference_backend.cpp` (94 LOC). |
| **HfPythonBackend** | Delegates generation to HuggingFace Python subprocess. | `generate_text()` writes prompt to temp file, spawns `scripts/hf_generate.py` with model dir and config args, reads generated text from stdout. Maximum compatibility with any HF model. `is_available()` checks that Python and model dir exist. File: `src/runtime/hf_python_backend.cpp` (236 LOC). |
