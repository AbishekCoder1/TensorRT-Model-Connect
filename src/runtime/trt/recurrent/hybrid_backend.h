#pragma once

#include "runtime/trt/core/generation_backend.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/recurrent/mamba_decode_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Engine configuration for the hybrid Mamba-2 + Attention model.
// A single TRT engine handles all layer types; this struct tracks which
// I/O tensors belong to Mamba state vs KV cache.
struct HybridStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string token_input_name{"token_id"};
    std::string position_input_name{"position_id"};
    std::string mask_input_name{"attention_mask"};
    std::string logits_output_name{"logits"};

    // Mamba-2 state tensors (indexed by mamba layer index, not global)
    std::vector<std::string> conv_state_input_names;
    std::vector<std::string> ssm_state_input_names;
    std::vector<std::string> present_conv_output_names;
    std::vector<std::string> present_ssm_output_names;

    // Attention KV cache tensors (indexed by attention layer index, not global)
    std::vector<std::string> cache_k_input_names;
    std::vector<std::string> cache_v_input_names;
    std::vector<std::string> present_k_output_names;
    std::vector<std::string> present_v_output_names;

    // Layer type metadata
    std::vector<std::string> layer_types;  // "mamba2", "mlp", "attention"
    int32_t num_mamba_layers{0};
    int32_t num_attention_layers{0};

    // Model dimensions
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t attention_size{0};
    int32_t max_cache_length{32};

    // Mamba-2 dimensions
    int32_t d_inner{0};
    int32_t d_state{128};
    int32_t d_conv{4};
    int32_t nheads{0};
    int32_t head_dim{0};           // per-head dimension (d_inner / nheads)
    int32_t conv_dim{0};           // conv1d channels (d_inner + 2*n_groups*d_state)

    int32_t id_bos{0};
    int32_t id_eos{0};
};

bool has_all_required_hybrid_tensors(const HybridStepEngine& engine);

// Creates a hybrid Mamba-2/Attention backend from a pre-built engine.
std::unique_ptr<IGenerationBackend> CreateHybridBackendFromEngine(
    std::unique_ptr<HybridStepEngine> engine);

#endif // TRTF_HAS_TRT

} // namespace trtf
