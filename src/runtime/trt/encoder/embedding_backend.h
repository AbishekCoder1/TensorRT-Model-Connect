#pragma once

#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "cabi/config/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct EmbeddingResult {
    std::vector<float> embedding;  // [embedding_dim] L2-normalized vector
    int32_t embedding_dim{0};
};

struct EmbeddingConfig {
    int32_t max_seq_length{512};
    int32_t hidden_size{2048};
    int32_t embedding_dim{2048};  // output dimension (may differ from hidden)
    bool has_input_embed{false};  // engine supports input_embed bypass
    int32_t img_context_token_id{128258};  // placeholder token for image patches
};

class EmbeddingBackend {
public:
    EmbeddingBackend(
        TrtUniquePtr<nvinfer1::ICudaEngine> engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> context,
        EmbeddingConfig config);

    ~EmbeddingBackend();

    bool is_available() const;

    // Encode token IDs into a normalized embedding vector.
    // Mean pools over non-padding positions and L2-normalizes.
    EmbeddingResult embed(const std::vector<int32_t>& input_ids);

    // Embed with an image: preprocess -> vision encode -> inject features -> text backbone.
    EmbeddingResult embed_with_image(const std::vector<int32_t>& input_ids,
                                     const std::string& image_path);

    // Set optional vision engine for VL embedding.
    void set_vision_engine(std::unique_ptr<VisionStepEngine> vision_engine,
                           VLPreprocessConfig vl_config);

    bool has_vision() const;

    const EmbeddingConfig& config() const { return mConfig; }
    const VLPreprocessConfig& vl_config() const { return mVLConfig; }

private:
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext;
    EmbeddingConfig mConfig;
    CudaStream mStream;
    std::unique_ptr<VisionStepEngine> mVisionEngine;
    VLPreprocessConfig mVLConfig;
};

// Create from engine + fast path config.
std::unique_ptr<EmbeddingBackend> CreateEmbeddingBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> context,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
