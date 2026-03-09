# TASK-05: TextGenerationPipeline — serves 25+ decoder-only models

## Status: blocked (needs TASK-01, TASK-02, TASK-03, TASK-04)
## Phase: 2 (Core pipelines)
## Risk: medium — this is the most-used pipeline, must match existing behavior exactly

## Goal

Implement `TextGenerationPipeline` — the single pipeline that serves ALL
decoder-only LLMs (Qwen, LLaMA, Mistral, GPT-2, OPT, Phi, Gemma, Bloom,
Falcon, Granite, StarCoder, XGLM, GPT-Neo, Codegen, OLMo, Nemotron, etc.).

Like HF's `TextGenerationPipeline`: one class, many models. The model-specific
differences (GQA vs MQA, RoPE vs ALiBi, SwiGLU vs GELU) are baked into the TRT
engine at build time. The pipeline just calls `module.forward()` in a loop.

## Class design

```cpp
class TextGenerationPipeline final : public IPipeline {
public:
    TextGenerationPipeline(
        std::unique_ptr<TrtModule> decoder,
        std::unique_ptr<KvCache> cache,
        std::unique_ptr<ITokenizer> tokenizer,
        TextGenConfig config);

    GenerationResult generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) override;

    const char* pipeline_name() const override { return "TextGenerationPipeline"; }

private:
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<KvCache> cache_;
    std::unique_ptr<ITokenizer> tokenizer_;
    TextGenConfig config_;  // vocab_size, bos, eos
    cudaStream_t stream_;

    // Shared generation loop
    void prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    int32_t decode_step(int32_t token_id, std::vector<float>& logits);
};
```

## Generate flow

```
generate(input_ids, config):
    cache_.reset()
    cache_.bind_to(*decoder_)

    // Prefill: process all input tokens
    for i in 0..input_ids.size()-1:
        mask = cache_.build_attention_mask()
        logits = decoder_.forward({
            {"token_id",       Tensor(input_ids[i])},
            {"position_id",    Tensor(cache_.position())},
            {"attention_mask", Tensor(mask)}
        })  // cache_k/v already bound via bind_external
        cache_.advance()

    // Decode: autoregressive loop
    current_token = argmax(logits)
    output = [current_token]
    while len(output) < config.max_new_tokens:
        logits = decoder_.forward({same pattern})
        cache_.advance()
        next_token = argmax(logits)
        output.push_back(next_token)
        if next_token == config_.eos: break

    return output
```

## What it replaces

- `TrtBackendFastPath` (trt_backend_shared.cpp — 130 lines)
- `run_prefill_steps()` / `run_decode_steps()` helpers
- `run_decoder_step_device()` (the 200-line function in device_kv_cache.cpp)
- `GenerationBackendAdapter` + `GenerationBackendPort` + `GenerationTextService`

All replaced by ~150 lines of clean pipeline code that composes TrtModule + KvCache.

## Wire into factory

```cpp
// In PipelineFactory::create_text_generation():
auto module = load_module(bundle, rt, "plan", stream);
auto cache = std::make_unique<KvCache>(cfg.num_layers, cfg.max_cache_length,
                                        cfg.num_kv_heads * cfg.head_dim, stream);
auto tokenizer = create_tokenizer_from_bundle(bundle, opts.hf_python_path);
return std::make_unique<TextGenerationPipeline>(
    std::move(module), std::move(cache), std::move(tokenizer),
    TextGenConfig{cfg.vocab_size, cfg.id_bos, cfg.id_eos});
```

## Files to create

- `src/runtime/pipelines/text_generation_pipeline.h`
- `src/runtime/pipelines/text_generation_pipeline.cpp`
- `tests/cpp/test_text_generation_pipeline.cpp`

## Acceptance criteria

- [ ] E2E: `trtf run qwen3-0.6b.trtfb --prompt "Hello"` produces same output as before
- [ ] E2E: GPT-2, OPT, Bloom also produce identical output (strategy=decoder_kv_cache)
- [ ] Unit test: prefill + decode with mock TrtModule
- [ ] Old path still works (both paths coexist during migration)

## Dependencies

TASK-01 (TrtModule), TASK-02 (KvCache), TASK-03 (ITokenizer), TASK-04 (IPipeline/Factory)
