#include "qwen3_trt_model_definition.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace trtf {

bool PopulateQwen3TrtModelDefinition(TrtDecoderDefinition& definition, const DecoderModel& model)
{
    if (!model.checkpoint.has_qwen_layers || model.checkpoint.qwen_layers.empty())
    {
        return false;
    }

    const auto expect_size = [](const std::vector<float>& tensor, std::size_t expected, const char* name) {
        if (tensor.size() != expected)
        {
            throw std::runtime_error(
                std::string("Invalid checkpoint tensor size for ") + name + ": expected "
                + std::to_string(expected) + ", got " + std::to_string(tensor.size()));
        }
    };

    const std::size_t hidden = static_cast<std::size_t>(definition.hidden_size);
    const std::size_t mlp = static_cast<std::size_t>(definition.mlp_size);

    const int32_t checkpoint_attention
        = model.checkpoint.attention_size > 0 ? model.checkpoint.attention_size : definition.attention_size;
    if (checkpoint_attention <= 0)
    {
        throw std::runtime_error("Qwen3 checkpoint is missing a valid attention size.");
    }

    definition.has_qwen_layers = true;
    definition.attention_size = checkpoint_attention;
    definition.rms_norm_eps = std::max(model.architecture.rms_norm_eps, 1.0e-9F);
    definition.num_attention_heads = std::max(model.architecture.num_attention_heads, 1);
    definition.num_key_value_heads = std::max(model.architecture.num_key_value_heads, 1);
    definition.rope_theta = model.architecture.rope_theta > 0.0F ? model.architecture.rope_theta : 10000.0F;
    definition.final_norm = model.checkpoint.final_norm;

    if (definition.attention_size % definition.num_attention_heads != 0)
    {
        throw std::runtime_error("Qwen3 attention_size must be divisible by num_attention_heads.");
    }

    const std::size_t attention = static_cast<std::size_t>(definition.attention_size);

    if (definition.final_norm.empty())
    {
        definition.final_norm.assign(hidden, 1.0F);
    }
    expect_size(definition.final_norm, hidden, "final_norm");

    definition.qwen_layers.reserve(model.checkpoint.qwen_layers.size());
    for (std::size_t i = 0; i < model.checkpoint.qwen_layers.size(); ++i)
    {
        const DecoderLayerCheckpoint& src = model.checkpoint.qwen_layers[i];
        TrtDecoderLayerDefinition layer;
        layer.input_norm = src.input_norm;
        layer.q_norm = src.q_norm;
        layer.k_norm = src.k_norm;
        layer.w_q = src.w_q;
        layer.w_k = src.w_k;
        layer.w_v = src.w_v;
        layer.w_o = src.w_o;
        layer.post_attn_norm = src.post_attn_norm;
        layer.w_gate = src.w_gate;
        layer.w_up = src.w_up;
        layer.w_down = src.w_down;

        const std::string prefix = "qwen_layers[" + std::to_string(i) + "].";
        expect_size(layer.input_norm, hidden, (prefix + "input_norm").c_str());
        if (layer.q_norm.empty())
        {
            layer.q_norm.assign(attention, 1.0F);
        }
        if (layer.k_norm.empty())
        {
            layer.k_norm.assign(attention, 1.0F);
        }
        expect_size(layer.q_norm, attention, (prefix + "q_norm").c_str());
        expect_size(layer.k_norm, attention, (prefix + "k_norm").c_str());
        expect_size(layer.w_q, hidden * attention, (prefix + "w_q").c_str());
        expect_size(layer.w_k, hidden * attention, (prefix + "w_k").c_str());
        expect_size(layer.w_v, hidden * attention, (prefix + "w_v").c_str());
        expect_size(layer.w_o, attention * hidden, (prefix + "w_o").c_str());
        expect_size(layer.post_attn_norm, hidden, (prefix + "post_attn_norm").c_str());
        expect_size(layer.w_gate, hidden * mlp, (prefix + "w_gate").c_str());
        expect_size(layer.w_up, hidden * mlp, (prefix + "w_up").c_str());
        expect_size(layer.w_down, mlp * hidden, (prefix + "w_down").c_str());

        definition.qwen_layers.push_back(std::move(layer));
    }
    return true;
}

} // namespace trtf
