# TASK-08: VLPipeline — vision-language generation

## Status: blocked (needs TASK-01, TASK-02, TASK-05)
## Phase: 2 (Core pipelines)
## Risk: medium — VL has image preprocessing + embedding injection into decoder

## Goal

Implement `VLPipeline` for vision-language models (Qwen2.5-VL, Qwen3-VL,
InternVL3, Phi4-multimodal). Composes a vision encoder TrtModule + text decoder
TrtModule + KvCache + ImagePreprocessor.

## Class design

```cpp
class VLPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> text_decoder_;
    std::unique_ptr<TrtModule> vision_encoder_;
    std::unique_ptr<KvCache> cache_;
    std::unique_ptr<ITokenizer> tokenizer_;
    ImagePreprocessor image_preprocessor_;
    VLConfig config_;  // image_token_id, vision_output_dim, prompt_template, etc.

    GenerationResult generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) override;

    // VL-specific: generate with image
    GenerationResult generate_vl(
        const std::vector<int32_t>& input_ids,
        const float* image_data, int32_t height, int32_t width,
        const GenerationConfig& config);
};
```

## Generate flow

```
generate_vl(input_ids, image, config):
    // 1. Vision encode: image → features
    preprocessed = image_preprocessor_.preprocess(image, h, w)
    vision_out = vision_encoder_->forward_device({{"pixel_values", preprocessed}})
    vision_features = vision_out["image_features"]  // stays on GPU

    // 2. Text decode with vision injection
    cache_.reset()
    cache_.bind_to(*text_decoder_)

    for i in 0..input_ids.size()-1:
        token = input_ids[i]
        if token == config_.image_token_id:
            // Inject vision features via input_embed
            decoder_->forward({
                {"token_id", token}, {"position_id", pos}, {"attention_mask", mask},
                {"input_embed", vision_features},
                {"use_input_embed", 1.0f}
            })
        else:
            decoder_->forward({normal token inputs})
        cache_.advance()

    // 3. Standard decode loop (same as TextGeneration)
    ...
```

## DeepStack support (Qwen3-VL)

Qwen3-VL injects multi-level vision features at specific early decoder layers.
This is handled by binding DeviceTensors to the `deepstack_embed_{i}` inputs:

```cpp
if (config_.has_deepstack) {
    for (int i = 0; i < config_.num_deepstack_levels; ++i) {
        decoder_->bind_external("deepstack_embed_" + std::to_string(i),
                                vision_features_by_level[i].data());
    }
    decoder_->bind_external("deepstack_active", &active_flag);
}
```

## What it replaces

- `VlBackend` (vl_backend.cpp — ~400 lines)
- `VisionEngine` (vision_engine.cpp — ~300 lines)
- Vision strategy builder VL path
- VL adapter/port/service wrappers

## Files to create

- `src/runtime/pipelines/vl_pipeline.h` + `.cpp`
- `tests/cpp/test_vl_pipeline.cpp`

## Dependencies

TASK-01, TASK-02, TASK-05 (reuses TextGeneration's prefill/decode pattern)
