#pragma once

#include "runtime/trt/core/trt_common.h"
#include "cabi/config/fast_path_config.h"

#if TRTF_HAS_TRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

struct SamConfig {
    int32_t image_size{1024};
    int32_t image_embedding_size{64};  // image_size / patch_size
    int32_t decoder_hidden_size{256};
    int32_t num_mask_outputs{4};       // num_multimask + 1
    int32_t num_multimask_outputs{3};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};

    // Prompt encoder embeddings (loaded from config.json)
    std::vector<float> point_embed_fg;           // foreground point [decoder_hidden_size]
    std::vector<float> point_embed_bg;           // background point [decoder_hidden_size]
    std::vector<float> not_a_point_embed;        // padding point [decoder_hidden_size]
    std::vector<float> shared_image_pe;          // [2, num_pos_feats] flattened
};

struct SamResult {
    std::vector<float> masks;     // [num_masks, 256, 256]
    std::vector<float> iou_scores; // [num_masks]
    int32_t num_masks{0};
    int32_t mask_height{256};
    int32_t mask_width{256};
};

class SamBackend {
public:
    SamBackend(
        TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx,
        TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine,
        TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx,
        SamConfig config);

    ~SamBackend();

    bool is_available() const;

    // Encode an image. Stores embeddings in GPU memory.
    bool encode_image(const std::string& image_path);

    // Segment with a single point prompt. Returns masks + IoU scores.
    // point_x, point_y: normalized coordinates [0, 1]
    // is_foreground: true for foreground, false for background
    SamResult segment_point(float point_x, float point_y, bool is_foreground);

    // Segment with a box prompt.
    // x1, y1, x2, y2: normalized coordinates [0, 1]
    SamResult segment_box(float x1, float y1, float x2, float y2);

    const SamConfig& config() const { return mConfig; }

private:
    // Run the mask decoder with given sparse prompt embeddings
    SamResult run_decoder(const std::vector<float>& sparse_prompt);

    // Build sparse prompt from point
    std::vector<float> encode_point(float x, float y, bool is_foreground);

    // Build sparse prompt from box (two corner points)
    std::vector<float> encode_box(float x1, float y1, float x2, float y2);

    TrtUniquePtr<nvinfer1::ICudaEngine> mEncoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mEncoderCtx;
    TrtUniquePtr<nvinfer1::ICudaEngine> mDecoderEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext> mDecoderCtx;
    SamConfig mConfig;
    CudaStream mStream;

    // Cached image embeddings on GPU
    CudaBuffer mImageEmbeddings;
    bool mHasImage{false};

    // Rescaled image dimensions (for point coordinate transformation)
    int32_t mRescaledW{0};
    int32_t mRescaledH{0};
};

// Create from engines + config
std::unique_ptr<SamBackend> CreateSamBackend(
    TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx,
    TrtUniquePtr<nvinfer1::ICudaEngine> decoder_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> decoder_ctx,
    const FastPathModelConfig& cfg);

} // namespace trtf

#endif // TRTF_HAS_TRT
