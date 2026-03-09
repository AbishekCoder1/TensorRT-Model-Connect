# TASK-07: EncoderPipeline + SegmentPipeline + SamPipeline

## Status: blocked (needs TASK-01, TASK-04)
## Phase: 2 (Core pipelines)
## Risk: very low — simplest backends, single forward pass

## Goal

Implement the simplest pipeline types: single-pass encoder models and perception
models. These don't have loops, caches, or state — just `module.forward()`.

## Classes

```cpp
// Serves: BERT (encoder_only), Eagle-embed (embedding), Eagle-rerank (reranking)
class EncoderPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::string mode_;  // "encoder_only", "embedding", "reranking"

    EncodeResult encode(const std::vector<int32_t>& input_ids) override {
        auto out = encoder_->forward({{"input_ids", ...}, {"attention_mask", ...}});
        // Mode-specific postprocess: pooling, normalization, score extraction
        return result;
    }
};

// Serves: SegFormer
class SegmentPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> model_;
    ImagePreprocessor preprocessor_;

    SegmentResult segment(const float* image, int h, int w) override {
        auto input = preprocessor_.preprocess(image, h, w);
        auto out = model_->forward({{"pixel_values", input}});
        return postprocess(out);
    }
};

// Serves: SAM (two-stage: encoder + decoder)
class SamPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> image_encoder_;
    std::unique_ptr<TrtModule> mask_decoder_;
    ImagePreprocessor preprocessor_;

    SegmentResult segment(const float* image, int h, int w) override {
        auto features = image_encoder_->forward_device({{"pixel_values", ...}});
        auto masks = mask_decoder_->forward({
            {"image_embeddings", features["image_embeddings"]},
            {"point_coords", ...}, {"point_labels", ...}
        });
        return postprocess(masks);
    }
};
```

## What it replaces

- `EncoderBackend` (encoder_backend.cpp — ~300 lines)
- `EmbeddingBackend` (embedding_backend.cpp — ~400 lines)
- `RerankingBackend` (reranking_backend.cpp — ~200 lines)
- `SegmentationBackend` (segmentation_backend.cpp — ~300 lines)
- `SamBackend` (sam_backend.cpp — ~500 lines)
- Plus their adapter/port/service wrappers

## Files to create

- `src/runtime/pipelines/encoder_pipeline.h` + `.cpp`
- `src/runtime/pipelines/segment_pipeline.h` + `.cpp`
- `src/runtime/pipelines/sam_pipeline.h` + `.cpp`
- `tests/cpp/test_encoder_pipeline.cpp`
- `tests/cpp/test_segment_pipeline.cpp`

## Dependencies

TASK-01 (TrtModule), TASK-04 (Factory). No KvCache or scheduler needed.
