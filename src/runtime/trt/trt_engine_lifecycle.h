#pragma once

#include "runtime/trt/trt_common.h"
#include "model/trt_model_definition.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#endif

namespace trtf {

#if TRTF_HAS_TRT

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

    // Generic tensor bindings for non-KV-cache models (Mamba, MLA, etc.)
    struct TensorBinding {
        std::string logical_name;
        std::string engine_name;
        bool is_input{true};
        int32_t element_count{0};
    };
    std::vector<TensorBinding> extra_bindings;

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

std::vector<const DecoderStepEngine::TensorBinding*> find_extra_bindings(
    const DecoderStepEngine& engine, const std::string& prefix, bool is_input);

// Try to load a cached engine without building the TRT graph.
// Returns nullptr if no cache hit (caller should build graph and call finalize_decoder_step_engine).
std::unique_ptr<DecoderStepEngine> try_load_cached_engine(TrtLogger& logger,
    const TrtDecoderDefinition& weights, int32_t num_layers, bool requires_position_input);

std::unique_ptr<DecoderStepEngine> finalize_decoder_step_engine(nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, TrtLogger& logger,
    const TrtDecoderDefinition& weights, const std::vector<std::string>& cache_k_input_names,
    const std::vector<std::string>& cache_v_input_names, const std::vector<std::string>& present_k_output_names,
    const std::vector<std::string>& present_v_output_names, bool requires_position_input);

std::unique_ptr<DecoderStepEngine> finalize_decoder_step_engine(nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, TrtLogger& logger,
    const TrtDecoderDefinition& weights, const std::vector<std::string>& cache_k_input_names,
    const std::vector<std::string>& cache_v_input_names, const std::vector<std::string>& present_k_output_names,
    const std::vector<std::string>& present_v_output_names, bool requires_position_input,
    const std::vector<DecoderStepEngine::TensorBinding>& extra_bindings);

// Serialize an engine plan to a byte vector for bundling.
std::vector<char> SerializeEnginePlan(const DecoderStepEngine& engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
