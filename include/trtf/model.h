#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtf {

struct DecoderLayerCheckpoint {
    std::vector<float> input_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    std::vector<float> w_q;
    std::vector<float> w_k;
    std::vector<float> w_v;
    std::vector<float> q_bias;
    std::vector<float> k_bias;
    std::vector<float> v_bias;
    std::vector<float> w_o;
    std::vector<float> post_attn_norm;
    std::vector<float> w_gate;
    std::vector<float> w_up;
    std::vector<float> w_down;

    // Extension: arbitrary named tensors for non-standard layers (MoE experts, SSM weights, etc.)
    std::unordered_map<std::string, std::vector<float>> extra_tensors;
};

struct DecoderCheckpoint {
    int32_t hidden_size{0};
    int32_t attention_size{0};
    int32_t mlp_size{0};

    std::vector<float> embedding;
    std::vector<float> w_q;
    std::vector<float> w_k;
    std::vector<float> w_v;
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    std::vector<float> b2;
    std::vector<float> w_out;
    std::vector<float> b_out;

    bool has_decoder_layers{false};
    std::vector<DecoderLayerCheckpoint> decoder_layers;
    std::vector<float> final_norm;
};

struct DecoderArchitectureConfig {
    std::string family{"toy-decoder"};
    int32_t num_layers{1};
    int32_t num_attention_heads{1};
    int32_t num_key_value_heads{1};
    int32_t bos_token_id{-1};
    int32_t eos_token_id{-1};
    int32_t pad_token_id{-1};
    float rms_norm_eps{1.0e-5F};
    float rope_theta{10000.0F};

    // Extension: arbitrary architecture parameters for non-standard configs
    std::unordered_map<std::string, int32_t> extra_int_params;
    std::unordered_map<std::string, float> extra_float_params;
    std::unordered_map<std::string, std::string> extra_string_params;
};

struct DecoderModel {
    std::string model_id;
    std::vector<std::string> vocab;
    std::vector<std::pair<std::string, std::string>> transitions;
    std::string default_next_token{"to"};
    int32_t max_cache_length{32};
    DecoderArchitectureConfig architecture;
    bool prefer_hf_tokenizer{false};
    std::string hf_tokenizer_dir;
    bool has_checkpoint{false};
    DecoderCheckpoint checkpoint;
};

DecoderModel LoadDecoderModel(const std::string& model_id);

} // namespace trtf
