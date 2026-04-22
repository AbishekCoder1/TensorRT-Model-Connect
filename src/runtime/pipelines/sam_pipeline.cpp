#include "runtime/pipelines/sam_pipeline.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf {

SamPipeline::SamPipeline(std::unique_ptr<TrtModule> image_encoder,
                         std::unique_ptr<TrtModule> mask_decoder, std::string model_id_str)
    : image_encoder_(std::move(image_encoder)), mask_decoder_(std::move(mask_decoder)),
      model_id_(std::move(model_id_str)) {
    if (!image_encoder_ || !image_encoder_->ok())
        throw std::runtime_error("SamPipeline: invalid image_encoder");
    if (!mask_decoder_ || !mask_decoder_->ok())
        throw std::runtime_error("SamPipeline: invalid mask_decoder");
}

SegmentResult SamPipeline::segment(const float* pixels, int32_t height, int32_t width) {
    Tensor img_t;
    img_t.data = const_cast<float*>(pixels);
    img_t.shape = {3, height, width};
    img_t.dtype = DType::kFloat32;

    auto enc_out = image_encoder_->forward({{"pixel_values", img_t}});

    TensorMap decoder_inputs;
    for (auto& [name, tensor] : enc_out)
        decoder_inputs[name] = tensor;

    auto dec_out = mask_decoder_->forward(decoder_inputs);

    SegmentResult result;
    result.height = height;
    result.width = width;

    for (auto& [name, tensor] : dec_out) {
        if (name.find("mask") != std::string::npos || name.find("output") != std::string::npos) {
            auto n = tensor.numel();
            result.mask.resize(static_cast<std::size_t>(n));
            const auto* data = static_cast<const float*>(tensor.data);
            for (int64_t i = 0; i < n; ++i)
                result.mask[static_cast<std::size_t>(i)] = static_cast<int32_t>(data[i]);
            break;
        }
    }

    return result;
}

} // namespace trtf
