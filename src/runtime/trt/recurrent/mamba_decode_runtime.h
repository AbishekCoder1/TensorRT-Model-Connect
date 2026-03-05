#pragma once

#include "runtime/trt/core/trt_engine_lifecycle.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Mamba/SSM engine configuration (different tensor names from DecoderStepEngine).
struct MambaStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string token_input_name{"token_id"};
    std::string logits_output_name{"logits"};
    std::vector<std::string> conv_state_input_names;
    std::vector<std::string> ssm_state_input_names;
    std::vector<std::string> present_conv_output_names;
    std::vector<std::string> present_ssm_output_names;

    int32_t num_layers{1};
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t d_inner{0};
    int32_t state_size{16};
    int32_t conv_kernel{4};
    int32_t id_bos{0};
    int32_t id_eos{0};
};

bool has_all_required_mamba_tensors(const MambaStepEngine& engine);

bool run_mamba_step(
    const MambaStepEngine& engine,
    int32_t token_id,
    const std::vector<std::vector<float>>& conv_state_by_layer,
    const std::vector<std::vector<float>>& ssm_state_by_layer,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_conv_by_layer,
    std::vector<std::vector<float>>& present_ssm_by_layer,
    std::string& error);

#endif // TRTF_HAS_TRT

} // namespace trtf
