#include "model/standard_checkpoint_mapper.h"
#include "utils/text_parsers.h"
#include "utils/tensor_math.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf {
namespace {

int32_t checked_i32(int64_t value, const std::string& context)
{
    if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::runtime_error("Invalid dimension for " + context + ": " + std::to_string(value));
    }
    return static_cast<int32_t>(value);
}

std::string layer_tensor_key(int32_t layer_idx, const std::string& suffix)
{
    return "model.layers." + std::to_string(layer_idx) + "." + suffix;
}

} // namespace

DecoderCheckpoint StandardCheckpointMapper::map_checkpoint(
    const TensorSource& reader, std::size_t vocab_size,
    const std::filesystem::path& path,
    const DecoderArchitectureConfig& architecture) const
{
    const std::string key_embedding = "model.embed_tokens.weight";
    const std::string key_lm_head = "lm_head.weight";
    const std::string key_final_norm = "model.norm.weight";

    const auto require_2d = [&](const std::string& name) -> std::pair<int32_t, int32_t> {
        if (!reader.has(name))
        {
            throw std::runtime_error("Missing required safetensors tensor " + name + " in " + path.string());
        }
        const SafetensorEntry& entry = reader.entry(name);
        if (entry.shape.size() != 2)
        {
            throw std::runtime_error("Expected rank-2 safetensors tensor for " + name + " in " + path.string());
        }
        return {checked_i32(entry.shape[0], name), checked_i32(entry.shape[1], name)};
    };
    const auto require_1d = [&](const std::string& name) -> int32_t {
        if (!reader.has(name))
        {
            throw std::runtime_error("Missing required safetensors tensor " + name + " in " + path.string());
        }
        const SafetensorEntry& entry = reader.entry(name);
        if (entry.shape.size() != 1)
        {
            throw std::runtime_error("Expected rank-1 safetensors tensor for " + name + " in " + path.string());
        }
        return checked_i32(entry.shape[0], name);
    };

    const auto embedding_shape = require_2d(key_embedding);
    const int32_t vocab = embedding_shape.first;
    const int32_t hidden = embedding_shape.second;
    if (vocab != static_cast<int32_t>(vocab_size))
    {
        throw std::runtime_error("Safetensors vocab size mismatch. embedding rows="
            + std::to_string(vocab) + ", vocab size=" + std::to_string(vocab_size)
            + " in " + path.string());
    }

    const int32_t num_layers = std::max(architecture.num_layers, 1);
    const int32_t num_attention_heads = std::max(architecture.num_attention_heads, 1);
    const int32_t num_kv_heads = std::max(architecture.num_key_value_heads, 1);

    DecoderCheckpoint checkpoint;
    checkpoint.hidden_size = hidden;
    checkpoint.has_decoder_layers = true;
    checkpoint.embedding = reader.load_f32(key_embedding);
    checkpoint.decoder_layers.reserve(static_cast<std::size_t>(num_layers));

    int32_t inferred_mlp_size = 0;
    int32_t inferred_attention_size = 0;
    for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        const std::string input_norm_key = layer_tensor_key(layer_idx, "input_layernorm.weight");
        const std::string q_norm_key = layer_tensor_key(layer_idx, "self_attn.q_norm.weight");
        const std::string k_norm_key = layer_tensor_key(layer_idx, "self_attn.k_norm.weight");
        const std::string q_key = layer_tensor_key(layer_idx, "self_attn.q_proj.weight");
        const std::string k_key = layer_tensor_key(layer_idx, "self_attn.k_proj.weight");
        const std::string v_key = layer_tensor_key(layer_idx, "self_attn.v_proj.weight");
        const std::string o_key = layer_tensor_key(layer_idx, "self_attn.o_proj.weight");
        const std::string post_norm_key = layer_tensor_key(layer_idx, "post_attention_layernorm.weight");
        const std::string gate_key = layer_tensor_key(layer_idx, "mlp.gate_proj.weight");
        const std::string up_key = layer_tensor_key(layer_idx, "mlp.up_proj.weight");
        const std::string down_key = layer_tensor_key(layer_idx, "mlp.down_proj.weight");

        const int32_t input_norm = require_1d(input_norm_key);
        const auto q_shape = require_2d(q_key);
        const auto k_shape = require_2d(k_key);
        const auto v_shape = require_2d(v_key);
        const auto o_shape = require_2d(o_key);
        const int32_t post_norm = require_1d(post_norm_key);
        const auto gate_shape = require_2d(gate_key);
        const auto up_shape = require_2d(up_key);
        const auto down_shape = require_2d(down_key);

        const int32_t q_hidden = q_shape.first;
        const int32_t kv_hidden = k_shape.first;

        if (input_norm != hidden || post_norm != hidden)
        {
            throw std::runtime_error("RMSNorm shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }
        if (q_shape.second != hidden || o_shape.first != hidden || o_shape.second != q_hidden)
        {
            throw std::runtime_error("Attention projection shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }
        if (k_shape.second != hidden || v_shape.second != hidden || kv_hidden != v_shape.first)
        {
            throw std::runtime_error("K/V projection shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }
        if (gate_shape.second != hidden || up_shape.second != hidden || gate_shape.first != up_shape.first
            || down_shape.first != hidden || down_shape.second != gate_shape.first)
        {
            throw std::runtime_error("MLP projection shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }

        if (inferred_mlp_size == 0)
        {
            inferred_mlp_size = gate_shape.first;
        }
        else if (inferred_mlp_size != gate_shape.first)
        {
            throw std::runtime_error("Inconsistent MLP size across layers in " + path.string());
        }

        if (inferred_attention_size == 0)
        {
            inferred_attention_size = q_hidden;
        }
        else if (inferred_attention_size != q_hidden)
        {
            throw std::runtime_error("Inconsistent attention projection size across layers in " + path.string());
        }

        if (q_hidden % num_attention_heads != 0)
        {
            throw std::runtime_error("q_proj out_features must be divisible by num_attention_heads at layer "
                + std::to_string(layer_idx) + " in " + path.string());
        }
        const int32_t head_dim = q_hidden / num_attention_heads;
        const bool grouped_kv_layout = num_attention_heads % num_kv_heads == 0
            && kv_hidden == head_dim * num_kv_heads;
        if (!grouped_kv_layout && kv_hidden != q_hidden)
        {
            throw std::runtime_error("K/V projection out_features do not match grouped-KV or full-attention layout at layer "
                + std::to_string(layer_idx) + " in " + path.string());
        }

        DecoderLayerCheckpoint layer;
        layer.input_norm = reader.load_f32(input_norm_key);

        // Per-head QKV biases: load if present (Qwen2), leave empty otherwise.
        // Empty bias tells the graph builder to skip the bias addition.
        // K/V biases are expanded from kv_hidden to q_hidden (same GQA expansion as weights).
        const std::string q_bias_key = layer_tensor_key(layer_idx, "self_attn.q_proj.bias");
        const std::string k_bias_key = layer_tensor_key(layer_idx, "self_attn.k_proj.bias");
        const std::string v_bias_key = layer_tensor_key(layer_idx, "self_attn.v_proj.bias");
        if (reader.has(q_bias_key))
        {
            layer.q_bias = reader.load_f32(q_bias_key);
        }
        if (reader.has(k_bias_key))
        {
            const std::vector<float> raw_k_bias = reader.load_f32(k_bias_key);
            if (static_cast<int32_t>(raw_k_bias.size()) == kv_hidden && kv_hidden != q_hidden && head_dim > 0)
            {
                // Expand KV bias from [kv_hidden] to [q_hidden] by repeating each KV head's block.
                layer.k_bias.resize(static_cast<std::size_t>(q_hidden));
                for (int32_t qh = 0; qh < num_attention_heads; ++qh)
                {
                    const int32_t kvh = std::min(num_kv_heads - 1, qh / (num_attention_heads / num_kv_heads));
                    std::copy_n(raw_k_bias.data() + static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim),
                        static_cast<std::size_t>(head_dim),
                        layer.k_bias.data() + static_cast<std::size_t>(qh) * static_cast<std::size_t>(head_dim));
                }
            }
            else
            {
                layer.k_bias = raw_k_bias;
            }
        }
        if (reader.has(v_bias_key))
        {
            const std::vector<float> raw_v_bias = reader.load_f32(v_bias_key);
            if (static_cast<int32_t>(raw_v_bias.size()) == kv_hidden && kv_hidden != q_hidden && head_dim > 0)
            {
                layer.v_bias.resize(static_cast<std::size_t>(q_hidden));
                for (int32_t qh = 0; qh < num_attention_heads; ++qh)
                {
                    const int32_t kvh = std::min(num_kv_heads - 1, qh / (num_attention_heads / num_kv_heads));
                    std::copy_n(raw_v_bias.data() + static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim),
                        static_cast<std::size_t>(head_dim),
                        layer.v_bias.data() + static_cast<std::size_t>(qh) * static_cast<std::size_t>(head_dim));
                }
            }
            else
            {
                layer.v_bias = raw_v_bias;
            }
        }

        // Per-head q_norm/k_norm: load if present (Qwen3), leave empty otherwise.
        // Empty q_norm/k_norm tells the graph builder to skip per-head RMS norm entirely,
        // which is correct for LLaMA, Mistral, and other models without QK norm.
        // RMSNorm with gamma=1.0 is NOT identity — it normalizes by dividing by RMS.
        if (reader.has(q_norm_key))
        {
            const int32_t q_norm_size = require_1d(q_norm_key);
            if (q_norm_size != head_dim)
            {
                throw std::runtime_error("q_norm shape mismatch at layer " + std::to_string(layer_idx)
                    + " in " + path.string());
            }
            layer.q_norm = repeat_head_norm(reader.load_f32(q_norm_key), num_attention_heads);
        }

        if (reader.has(k_norm_key))
        {
            const int32_t k_norm_size = require_1d(k_norm_key);
            if (k_norm_size != head_dim)
            {
                throw std::runtime_error("k_norm shape mismatch at layer " + std::to_string(layer_idx)
                    + " in " + path.string());
            }
            layer.k_norm = repeat_head_norm(reader.load_f32(k_norm_key), num_attention_heads);
        }

        layer.w_q = transpose_2d(reader.load_f32(q_key), q_hidden, hidden, "q_proj");

        const std::vector<float> k_t = transpose_2d(reader.load_f32(k_key), kv_hidden, hidden, "k_proj");
        const std::vector<float> v_t = transpose_2d(reader.load_f32(v_key), kv_hidden, hidden, "v_proj");
        layer.w_k = expand_kv_projection(k_t, hidden, kv_hidden, q_hidden, architecture, "k_proj");
        layer.w_v = expand_kv_projection(v_t, hidden, kv_hidden, q_hidden, architecture, "v_proj");

        layer.w_o = transpose_2d(reader.load_f32(o_key), hidden, q_hidden, "o_proj");
        layer.post_attn_norm = reader.load_f32(post_norm_key);
        layer.w_gate = transpose_2d(reader.load_f32(gate_key), inferred_mlp_size, hidden, "gate_proj");
        layer.w_up = transpose_2d(reader.load_f32(up_key), inferred_mlp_size, hidden, "up_proj");
        layer.w_down = transpose_2d(reader.load_f32(down_key), hidden, inferred_mlp_size, "down_proj");
        checkpoint.decoder_layers.push_back(std::move(layer));
    }

    checkpoint.attention_size = inferred_attention_size > 0 ? inferred_attention_size : hidden;
    checkpoint.mlp_size = inferred_mlp_size;

    if (reader.has(key_final_norm))
    {
        if (require_1d(key_final_norm) != hidden)
        {
            throw std::runtime_error("model.norm.weight shape mismatch in " + path.string());
        }
        checkpoint.final_norm = reader.load_f32(key_final_norm);
    }
    else
    {
        checkpoint.final_norm.assign(static_cast<std::size_t>(hidden), 1.0F);
    }

    if (reader.has(key_lm_head))
    {
        const auto lm_shape = require_2d(key_lm_head);
        if (lm_shape.first != vocab || lm_shape.second != hidden)
        {
            throw std::runtime_error("lm_head shape mismatch in " + path.string());
        }
        checkpoint.w_out = transpose_2d(reader.load_f32(key_lm_head), vocab, hidden, "lm_head");
    }
    else
    {
        checkpoint.w_out = transpose_2d(checkpoint.embedding, vocab, hidden, "embedding");
    }

    // Populate layer-0 compatibility tensors for legacy paths.
    if (!checkpoint.decoder_layers.empty())
    {
        const DecoderLayerCheckpoint& layer0 = checkpoint.decoder_layers.front();
        checkpoint.w_q = layer0.w_q;
        checkpoint.w_k = layer0.w_k;
        checkpoint.w_v = layer0.w_v;
        checkpoint.w1 = layer0.w_up;
        checkpoint.b1.assign(static_cast<std::size_t>(checkpoint.mlp_size), 0.0F);
        checkpoint.w2 = layer0.w_down;
        checkpoint.b2.assign(static_cast<std::size_t>(hidden), 0.0F);
    }

    checkpoint.b_out.assign(static_cast<std::size_t>(vocab), 0.0F);
    return checkpoint;
}

} // namespace trtf
