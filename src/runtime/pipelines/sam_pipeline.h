#pragma once

// SamPipeline: two-stage segmentation (SAM -- encoder + decoder).
// Uses TrtModule(image_encoder) + TrtModule(mask_decoder).

#include "trtf/pipeline.h"
#include "trtf/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

namespace trtf {

class SamPipeline final : public IPipeline {
public:
    SamPipeline(std::unique_ptr<TrtModule> image_encoder,
                std::unique_ptr<TrtModule> mask_decoder,
                std::string model_id_str = "");

    SegmentResult segment(const float* pixels, int32_t height, int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SamPipeline"; }

private:
    std::unique_ptr<TrtModule> image_encoder_;
    std::unique_ptr<TrtModule> mask_decoder_;
    std::string model_id_;
};

} // namespace trtf

#endif // TRTF_HAS_TRT
