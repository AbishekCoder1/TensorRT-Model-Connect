#pragma once

#include "trtf/model.h"
#include "trtf/tokenizer.h"

#include <cstdint>
#include <vector>

namespace trtf {

struct TrtDecoderLayerDefinition {
    std::vector<float> input_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    std::vector<float> w_q;
    std::vector<float> w_k;
    std::vector<float> w_v;
    std::vector<float> w_o;
    std::vector<float> post_attn_norm;
    std::vector<float> w_gate;
    std::vector<float> w_up;
    std::vector<float> w_down;
};

struct TrtDecoderDefinition {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t attention_size{0};
    int32_t mlp_size{0};
    int32_t max_cache_length{32};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_qwen_layers{false};
    float rms_norm_eps{1.0e-5F};
    int32_t num_attention_heads{1};
    int32_t num_key_value_heads{1};
    float rope_theta{10000.0F};

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
    std::vector<float> final_norm;
    std::vector<TrtDecoderLayerDefinition> qwen_layers;
};

TrtDecoderDefinition BuildTrtDecoderWeights(const ITokenizer& tokenizer, const DecoderModel& model);

} // namespace trtf
