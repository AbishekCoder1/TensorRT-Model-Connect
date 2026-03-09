# TASK-04: IPipeline interface + PipelineFactory

## Status: ready (after TASK-01)
## Phase: 1 (Foundation)
## Risk: low — defines interfaces and factory skeleton, no backends ported yet

## Goal

Define the `IPipeline` interface (what the C API calls) and `PipelineFactory`
(config-driven assembly like HF's `from_pretrained()`). This replaces the current
adapter→port→service→router chain with a flat: factory → pipeline → C API.

## IPipeline interface

```cpp
class IPipeline {
public:
    virtual ~IPipeline() = default;

    // Text generation (decoder, mamba, rwkv, hybrid, VL)
    virtual GenerationResult generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config);

    // Media generation (diffusion — image/video)
    virtual MediaResult generate_media(
        const std::string& prompt,
        const MediaConfig& config);

    // Transcription (whisper)
    virtual TranscriptionResult transcribe(
        const float* audio_data, int32_t num_samples,
        int32_t max_new_tokens);

    // Audio generation (bark, magpie, personaplex, omni)
    virtual AudioResult generate_audio(
        const std::string& prompt,
        const AudioConfig& config);

    // Single-pass encode (BERT, embedding, reranking)
    virtual EncodeResult encode(
        const std::vector<int32_t>& input_ids);

    // Segmentation (segformer, SAM)
    virtual SegmentResult segment(
        const float* image_data, int32_t height, int32_t width);

    // Metadata
    virtual const char* pipeline_name() const = 0;
    virtual const char* runtime_strategy() const = 0;
};
```

All methods have default implementations that throw "unsupported". Each concrete
pipeline overrides only the methods it supports. This is simpler than the current
11-interface port system.

## PipelineFactory

```cpp
class PipelineFactory {
public:
    // The single entry point — like HF's Pipeline.from_pretrained()
    static std::unique_ptr<IPipeline> from_bundle(
        const std::string& bundle_path,
        const PipelineOptions& options = {});

private:
    // First-level dispatch: runtime_strategy → family
    // Second-level: family-specific sub-dispatch (e.g., diffusion_backend_type)

    static std::unique_ptr<IPipeline> create_text_generation(
        BundleFile& bundle, const FastPathModelConfig& cfg,
        nvinfer1::IRuntime* rt, cudaStream_t stream,
        const PipelineOptions& opts);

    static std::unique_ptr<IPipeline> create_diffusion(
        BundleFile& bundle, const FastPathModelConfig& cfg,
        nvinfer1::IRuntime* rt, cudaStream_t stream,
        const PipelineOptions& opts);

    static std::unique_ptr<IPipeline> create_audio(
        BundleFile& bundle, const FastPathModelConfig& cfg,
        nvinfer1::IRuntime* rt, cudaStream_t stream,
        const PipelineOptions& opts);

    // ... etc for vision, encoder, perception

    // Helper: load TRT engine section → TrtModule
    static std::unique_ptr<TrtModule> load_module(
        BundleFile& bundle, nvinfer1::IRuntime* rt,
        const std::string& section_name, cudaStream_t stream);
};
```

## PipelineOptions

```cpp
struct PipelineOptions {
    std::string hf_python_path;          // for HfPythonTokenizer
    int32_t max_new_tokens_override{0};  // 0 = use default
    int32_t max_cache_length_override{0};
};
```

## How it replaces current code

Current flow (7 layers):
```
trtf_create_pipeline_ex() → resolve_strategy_family() → TextStrategyBuilder::build()
  → create_decoder_backend() → GenerationBackendAdapter → GenerationBackendPort
  → GenerationTextService → PipelineRouter
```

New flow (3 layers):
```
trtf_create_pipeline_ex() → PipelineFactory::from_bundle()
  → create_text_generation() → TextGenerationPipeline
```

## Integration with C API

`trtf_c.cpp` changes are minimal:
```cpp
TrtfPipeline trtf_create_pipeline_ex(const char* path, const TrtfPipelineOptions* opts) {
    PipelineOptions po;
    if (opts) { po.hf_python_path = opts->hf_python; ... }
    auto pipeline = PipelineFactory::from_bundle(path, po);
    return reinterpret_cast<TrtfPipeline>(pipeline.release());
}
```

## Files to create

- `include/trtf/runtime/pipeline.h` — IPipeline, result types
- `include/trtf/runtime/pipeline_factory.h` — PipelineFactory
- `src/runtime/pipeline_factory.cpp` — dispatch logic (skeleton, calls TODO stubs)
- `tests/cpp/test_pipeline_factory.cpp` — verify dispatch for each strategy

## Phase 1 scope

Factory is a skeleton: each `create_*` method returns nullptr with a TODO comment.
Concrete pipeline implementations come in TASK-05 through TASK-10.
But the factory dispatch logic and IPipeline interface are finalized here.

## Dependencies

TASK-01 (TrtModule — used by load_module helper)
