# Runtime Target Architecture

| Field | Value |
|-------|-------|
| **Document ID** | ARCH-RT-001 |
| **Status** | DRAFT / PLANNED |
| **Applies to** | C++ runtime (`src/`) |
| **Author** | Safety Architecture Team (yifeif@nvidia.com) |
| **Reviewer** | Independent Review Required (TBD — assign before merge) |
| **Review Status** | Pending independent review |
| **Last updated** | 2026-03-12 |
| **ISO 26262 relevance** | ASIL-QM (non-safety, design improvement) |

---

> **STATUS: PLANNED -- NOT YET IMPLEMENTED**
>
> This document describes the **target** architecture for the C++ runtime.
> It is **NOT** the current architecture. See [Architecture Overview](Architecture-Overview.md)
> for the implemented system.
>
> Nothing described below exists in the codebase today. All sections describe
> future work that has not been scheduled. Do not use this document as a
> reference for how the runtime currently operates.

---

## 1. Motivation

The current C++ runtime uses a centralized dispatch model:

- `PipelineFactory::from_bundle()` in `src/runtime/pipeline_factory.cpp` is the single assembly point for all strategies.
- `FastPathModelConfig` in `src/cabi/config/fast_path_config.h` is a monolithic struct (~230 fields) that carries configuration for every strategy family (decoder, encoder, vision, audio, diffusion, segmentation, detection, neural operator, speech-to-speech, omni, etc.).
- `resolve_family()` maps `runtime_strategy` strings to a `StrategyFamily` enum and dispatches to per-family factory functions within the same file.

This works, but it has scaling limitations:

1. **Adding a new runtime strategy requires editing shared files.** You must add fields to `FastPathModelConfig`, add a case to `resolve_family()`, and add a factory function in `pipeline_factory.cpp`.
2. **The god struct grows with every strategy.** Fields for unrelated strategies share the same struct, making it hard to validate that a bundle provides exactly the config a strategy needs.
3. **Unit testing factory logic requires including all strategy headers.** There is no way to test one strategy's assembly in isolation.

## 2. Current State (As-Is)

```text
trtf_create_pipeline_ex(bundle_path)
  -> ReadBundleFile()
  -> parse_fast_path_config()          // populates ALL fields in FastPathModelConfig
  -> resolve_family(runtime_strategy)  // returns StrategyFamily enum
  -> switch on StrategyFamily:
       kText      -> create_text_pipeline()
       kEncoder   -> create_encoder_pipeline()
       kVision    -> create_vision_pipeline()
       kAudio     -> create_audio_pipeline()
       kDiffusion -> create_diffusion_pipeline()
  -> IPipeline*
```

Key files in the current implementation:

| File | Role |
|------|------|
| `include/trtf/runtime/pipeline_factory.h` | `PipelineFactory::from_bundle()` declaration |
| `src/runtime/pipeline_factory.cpp` | Central dispatch: `resolve_family()` + all `create_*_pipeline()` functions |
| `src/cabi/config/fast_path_config.h` | `FastPathModelConfig` -- monolithic config struct |
| `src/cabi/config/fast_path_config.cpp` | JSON parsing into `FastPathModelConfig` |
| `src/cabi/api/trtf_c.cpp` | C ABI entry point, calls `PipelineFactory::from_bundle()` |
| `src/runtime/pipelines/*.h/*.cpp` | Concrete `IPipeline` implementations per strategy |

## 3. Target State (To-Be)

The target architecture replaces centralized dispatch with a plugin registry.

```text
trtf_create_pipeline_ex(bundle_path)
  -> ReadBundleFile()
  -> extract runtime_strategy string from config.json (lightweight parse)
  -> StrategyRegistry::resolve(runtime_strategy)
  -> IRuntimeStrategyPlugin::parse_config(config_json)   // per-strategy config
  -> IRuntimeStrategyPlugin::create_pipeline(config, bundle, hf_python)
  -> IPipeline*
```

### 3.1 IRuntimeStrategyPlugin

```cpp
// PLANNED -- does not exist today
class IRuntimeStrategyPlugin {
public:
    virtual ~IRuntimeStrategyPlugin() = default;

    // Strategies this plugin handles (e.g., {"decoder_kv_cache", "decoder_moe"}).
    virtual std::vector<std::string> supported_strategies() const = 0;

    // Parse only the config fields this strategy needs.
    // Returns an opaque config object owned by the plugin.
    virtual std::unique_ptr<IStrategyConfig> parse_config(
        const std::string& config_json) const = 0;

    // Validate that the bundle has the required sections.
    virtual bool validate_bundle(
        const BundleSections& sections,
        const IStrategyConfig& config,
        std::string& error_message) const = 0;

    // Create the pipeline. This is the composition root for the strategy.
    virtual std::unique_ptr<IPipeline> create_pipeline(
        const IStrategyConfig& config,
        const BundleSections& sections,
        const std::string& hf_python) const = 0;
};
```

### 3.2 StrategyRegistry

```cpp
// PLANNED -- does not exist today
class StrategyRegistry {
public:
    static StrategyRegistry& instance();

    void register_plugin(std::unique_ptr<IRuntimeStrategyPlugin> plugin);

    // Returns the plugin that handles the given runtime_strategy,
    // or nullptr if no plugin is registered for it.
    const IRuntimeStrategyPlugin* resolve(const std::string& strategy) const;

    // List all registered strategies (for diagnostics and testing).
    std::vector<std::string> registered_strategies() const;
};
```

### 3.3 Per-Strategy Config Structs

Each strategy plugin defines its own config struct instead of sharing `FastPathModelConfig`:

```cpp
// PLANNED -- does not exist today

struct DecoderKvConfig : IStrategyConfig {
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t num_layers;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t max_cache_length;
    int32_t id_bos;
    int32_t id_eos;
    bool tokenizer_add_special_tokens;
};

struct WhisperConfig : IStrategyConfig {
    int32_t num_mel_bins;
    int32_t max_source_positions;
    int32_t encoder_layers;
    int32_t decoder_layers;
    // ... only Whisper-relevant fields
};

struct DiffusionConfig : IStrategyConfig {
    std::string scheduler;
    int32_t num_inference_steps;
    float guidance_scale;
    // ... only diffusion-relevant fields
};
```

### 3.4 Self-Contained Plugins

Each strategy becomes a self-contained plugin that owns:

- Its config parsing logic
- Its bundle validation logic
- Its pipeline construction logic (tokenizer, engine loading, state allocation)
- Its own header dependencies (no need to include audio headers for text strategies)

```text
src/runtime/strategies/
  decoder_kv/
    decoder_kv_plugin.h
    decoder_kv_plugin.cpp
    decoder_kv_config.h
  ssm_recurrent/
    ssm_plugin.h
    ssm_plugin.cpp
    ssm_config.h
  diffusion/
    diffusion_plugin.h
    diffusion_plugin.cpp
    diffusion_config.h
  ...
```

## 4. Migration Phases

All phases below are **PLANNED and not yet started**.

### Phase 1: Introduce IRuntimeStrategyPlugin + StrategyRegistry

- Define the `IRuntimeStrategyPlugin` interface and `IStrategyConfig` base class.
- Implement `StrategyRegistry` with `register_plugin()` and `resolve()`.
- `PipelineFactory::from_bundle()` continues to work as before -- the registry exists alongside but is not yet the primary path.
- **Risk:** Low. Purely additive, no existing code changes.

### Phase 2: Decompose FastPathModelConfig

- Extract per-strategy config structs (DecoderKvConfig, WhisperConfig, DiffusionConfig, etc.).
- Implement `parse_config()` for each, reading only the fields the strategy needs.
- `FastPathModelConfig` remains as a compatibility layer during transition.
- **Risk:** Medium. Config parsing changes can cause subtle regressions. Each extraction needs E2E validation.

### Phase 3: Migrate strategies one-by-one to plugins

- Convert each `create_*_pipeline()` function into an `IRuntimeStrategyPlugin` implementation.
- Start with the simplest strategies (encoder_only, embedding, reranking) as proof-of-concept.
- Move to higher-complexity strategies (decoder_kv_cache, vision_language, diffusion) once the pattern is validated.
- Each migration is independently testable and deployable.
- **Risk:** Medium-high for complex strategies. Each migration requires full E2E regression.

### Phase 4: Remove centralized dispatch from pipeline_factory.cpp

- Once all strategies are migrated, `PipelineFactory::from_bundle()` becomes a thin wrapper: parse `runtime_strategy`, call `StrategyRegistry::resolve()`, call plugin's `create_pipeline()`.
- Delete `resolve_family()` enum dispatch.
- Delete `FastPathModelConfig` (all config parsing now lives in plugins).
- **Risk:** Low if Phase 3 is complete. This is cleanup.

### Phase 5: Plugin self-registration

- Each plugin registers itself via static initialization or an explicit `register_all_plugins()` call.
- External plugins (out-of-tree strategies) become possible.
- **Risk:** Low. Static initialization order is the main concern, mitigated by explicit registration as a fallback.

## 5. C ABI Stability Constraint

The public C ABI defined in `include/trtf/pipeline.h` **must remain stable throughout the entire migration**:

- `trtf_create_pipeline()` and `trtf_create_pipeline_ex()` continue to take a bundle path and return an `IPipeline*`.
- `TrtfPipelineOptions` struct is not changed.
- The `IPipeline` virtual interface (generate, embed, segment, transcribe, etc.) is not changed.
- All changes are internal to the factory and strategy assembly layer.

Callers of the C ABI will not need any changes at any phase of the migration.

## 6. Testing Strategy for Migration

Each migration phase must maintain the existing test gates:

| Gate | What it validates |
|------|-------------------|
| C++ unit tests (`ctest`) | Bundle parsing, tokenizers, CUDA wrappers, KV cache |
| Python builder tests (`pytest tests/builder/`) | Config parsing, weight mapping, graph ops |
| CCN gate (`tools/check_cyclomatic_complexity.py`) | No function exceeds CCN 10 |
| E2E suite (`pytest tests/test_e2e.py`) | Full pipeline correctness for all 50+ models |

Additionally, each new plugin should have:

- **Config parsing unit tests** -- verify that the plugin's `parse_config()` extracts the correct fields and rejects invalid configs.
- **Bundle validation unit tests** -- verify that `validate_bundle()` catches missing sections.
- **Construction unit tests** -- verify that `create_pipeline()` produces a working pipeline (may require TRT harness tests).

## 7. What This Document Is NOT

- This is **not** the current architecture. See [Architecture Overview](Architecture-Overview.md).
- This is **not** an approved migration plan with a schedule. It is a design target.
- This does **not** describe PipelineRouter, PipelineServices, StrategyBuilder, or service-composed runtime patterns. Those concepts do not exist in the codebase and are not part of this target.
- This does **not** imply that the current `PipelineFactory` approach is broken. It works correctly for all 50+ supported models. The target architecture addresses long-term maintainability as the strategy count grows.
