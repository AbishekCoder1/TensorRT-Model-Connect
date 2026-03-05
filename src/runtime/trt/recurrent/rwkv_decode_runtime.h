#pragma once

#include "runtime/trt/core/trt_engine_lifecycle.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// RWKV engine configuration (5 state tensors per layer).
struct RwkvStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string token_input_name{"token_id"};
    std::string logits_output_name{"logits"};

    std::vector<std::string> attn_state_input_names;
    std::vector<std::string> ff_state_input_names;
    std::vector<std::string> num_state_input_names;
    std::vector<std::string> den_state_input_names;
    std::vector<std::string> max_state_input_names;

    std::vector<std::string> present_attn_output_names;
    std::vector<std::string> present_ff_output_names;
    std::vector<std::string> present_num_output_names;
    std::vector<std::string> present_den_output_names;
    std::vector<std::string> present_max_output_names;

    int32_t num_layers{1};
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
};

bool has_all_required_rwkv_tensors(const RwkvStepEngine& engine);

bool run_rwkv_step(
    const RwkvStepEngine& engine,
    int32_t token_id,
    const std::vector<std::vector<float>>& attn_state_by_layer,
    const std::vector<std::vector<float>>& ff_state_by_layer,
    const std::vector<std::vector<float>>& num_state_by_layer,
    const std::vector<std::vector<float>>& den_state_by_layer,
    const std::vector<std::vector<float>>& max_state_by_layer,
    std::vector<float>& logits,
    std::vector<std::vector<float>>& present_attn_by_layer,
    std::vector<std::vector<float>>& present_ff_by_layer,
    std::vector<std::vector<float>>& present_num_by_layer,
    std::vector<std::vector<float>>& present_den_by_layer,
    std::vector<std::vector<float>>& present_max_by_layer,
    std::string& error);

#endif // TRTF_HAS_TRT

} // namespace trtf
