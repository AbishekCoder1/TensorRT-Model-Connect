#pragma once

#include "runtime/core/trt_common.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

// Expand a layer-name pattern by replacing {i}, {2i}, {2i+1}, {2i+2} tokens.
// Pure string logic — no TRT dependency.
std::string expand_layer_name(const std::string& pattern, int32_t layer);

constexpr int32_t kDefaultMaxCacheLength = 32;
constexpr float kMaskedScore = -1.0e4F;

struct DecoderStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string token_input_name{"token_id"};
    std::string position_input_name{"position_id"};
    std::string mask_input_name{"attention_mask"};
    std::string logits_output_name{"logits"};
    std::vector<std::string> cache_k_input_names;
    std::vector<std::string> cache_v_input_names;
    std::vector<std::string> present_k_output_names;
    std::vector<std::string> present_v_output_names;

    int32_t num_layers{1};
    bool requires_position_input{false};
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t cache_state_size{0};
    int32_t attention_mask_size{0};
    int32_t max_cache_length{kDefaultMaxCacheLength};
    int32_t id_bos{0};
    int32_t id_eos{0};
};

bool has_io_tensor(const nvinfer1::ICudaEngine& engine, const std::string& tensor_name);
bool has_all_required_tensors(const DecoderStepEngine& engine);

std::string layer_tensor_name(const char* stem, int32_t layer);

} // namespace trtf
