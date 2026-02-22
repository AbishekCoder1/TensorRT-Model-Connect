#pragma once

// Shared helpers for bundle-based pipeline creation.
// Extracts duplicated tokenizer/engine setup logic from trtf_c.cpp
// so each backend factory can reuse it without copy-paste.

#include "trtf/tokenizer.h"
#include "bundle/bundle_format.h"
#include "cabi/fast_path_config.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_common.h"

#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Pointers into a BundleFile's sections. Non-owning — the BundleFile must
// outlive any use of these pointers.
struct BundleSections {
    const std::vector<char>* plan_data{nullptr};
    const std::vector<char>* vision_plan_data{nullptr};
    const std::vector<char>* config_json_data{nullptr};
    const std::vector<char>* preprocessor_config_data{nullptr};
    const std::vector<char>* tokenizer_json_data{nullptr};
    const std::vector<char>* tokenizer_config_data{nullptr};
    const std::vector<char>* vocab_json_data{nullptr};
    const std::vector<char>* merges_txt_data{nullptr};
    const std::vector<char>* special_tokens_data{nullptr};
    const std::vector<char>* tokenizer_model_data{nullptr};

    // Bark/multi-engine sections
    const std::vector<char>* coarse_engine_plan_data{nullptr};
    const std::vector<char>* fine_engine_plan_data{nullptr};
    const std::vector<char>* codec_engine_plan_data{nullptr};

    // Diffusion sections
    std::vector<const std::vector<char>*> text_encoder_plans;  // text_encoder_0_plan, ...
    const std::vector<char>* denoiser_plan_data{nullptr};
    const std::vector<char>* vae_decoder_plan_data{nullptr};
    const std::vector<char>* preprocessor_weights_data{nullptr};
};

// Scan bundle sections and populate pointers by name.
BundleSections find_bundle_sections(const BundleFile& bundle);

// Result of extracting a tokenizer from a bundle.
struct TokenizerResult {
    std::unique_ptr<ITokenizer> tokenizer;
    std::string temp_dir;  // caller must transfer ownership to PipelineImpl
};

// Write tokenizer files from bundle sections to a temp dir, then create
// an HfPythonTokenizer. Throws on failure.
TokenizerResult extract_tokenizer_from_bundle(
    const BundleSections& sections,
    const std::string& hf_python,
    bool add_special_tokens = false);

// Build a DecoderStepEngine from a TRT engine + config.
std::unique_ptr<DecoderStepEngine> make_decoder_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const FastPathModelConfig& cfg);

#endif // TRTF_HAS_TRT

} // namespace trtf
