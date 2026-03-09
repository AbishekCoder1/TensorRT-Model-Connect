#pragma once

#include "runtime/trt/core/generation_backend.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/core/trt_common.h"
#include "cabi/config/fast_path_config.h"

#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Vision-Language backend: owns a text decoder engine + vision encoder engine.
// Image preprocessing is done in pure C++ (stb_image). The only remaining Python
// dependency is the HF tokenizer subprocess.
class VLBackendFastPath final : public IGenerationBackend {
public:
    VLBackendFastPath(
        std::unique_ptr<DecoderStepEngine> decoder_engine,
        std::unique_ptr<VisionStepEngine> vision_engine,
        const FastPathModelConfig& config,
        VLPreprocessConfig vl_config);

    bool is_available() const override;
    const char* name() const override;
    bool supports_vision() const override;

    // Text-only generation (no image)
    std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) override;

    // VL generation with pre-computed image features
    std::vector<int32_t> generate_vl(
        const std::vector<int32_t>& input_ids,
        const float* image_features, int32_t num_features, int32_t feature_dim,
        const std::vector<int32_t>& image_positions,
        const GenerationConfig& config) override;

    // High-level native pipeline:
    // 1. load_and_preprocess_image (C++ stb)
    // 2. run_vision_encoder (C++ TRT)
    // 3. format_vl_prompt (C++ string)
    // 4. tokenize (Python subprocess, via ITokenizer passed at call site)
    // Returns image features and formatted prompt for the caller to tokenize.
    bool prepare_image(
        const runtime::adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error);

    const VLPreprocessConfig& vl_config() const { return mVLConfig; }

    // DeepStack: store per-level features after vision encoding.
    // Set by prepare_image() when the vision engine has deepstack outputs.
    std::vector<std::vector<float>> deepstack_features;

private:
    std::unique_ptr<DecoderStepEngine> mDecoderEngine;
    std::unique_ptr<VisionStepEngine> mVisionEngine;
    FastPathModelConfig mConfig;
    VLPreprocessConfig mVLConfig;
    bool mHasVision{false};
};

// Factory function
std::unique_ptr<IGenerationBackend> CreateVLBackendFromEngines(
    std::unique_ptr<DecoderStepEngine> decoder_engine,
    std::unique_ptr<VisionStepEngine> vision_engine,
    const FastPathModelConfig& config,
    VLPreprocessConfig vl_config);

#endif // TRTF_HAS_TRT

} // namespace trtf
