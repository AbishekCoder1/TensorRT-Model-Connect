#include "trtf/model.h"
#include "safetensors_loader.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"
#include "utils/tensor_math.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef TRTF_SOURCE_DIR
#define TRTF_SOURCE_DIR "."
#endif

namespace trtf {
namespace {

struct ParsedTensor {
    std::vector<int32_t> dims;
    std::vector<float> data;
};

std::size_t checked_element_count(const std::vector<int32_t>& dims, const std::filesystem::path& path, int line_number)
{
    std::size_t count = 1;
    for (const int32_t dim : dims)
    {
        if (dim <= 0)
        {
            throw std::runtime_error(
                "Tensor dimension must be > 0 at " + path.string() + ":" + std::to_string(line_number));
        }
        if (count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))
        {
            throw std::runtime_error("Tensor element count overflow at " + path.string() + ":" + std::to_string(line_number));
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::size_t flatten_index(const std::vector<int32_t>& dims, const std::vector<int32_t>& indices,
    const std::filesystem::path& path, int line_number)
{
    if (dims.size() != indices.size())
    {
        throw std::runtime_error(
            "Invalid index rank at " + path.string() + ":" + std::to_string(line_number));
    }

    std::size_t offset = 0;
    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= dims[i])
        {
            throw std::runtime_error(
                "Index out of range at " + path.string() + ":" + std::to_string(line_number));
        }
        offset *= static_cast<std::size_t>(dims[i]);
        offset += static_cast<std::size_t>(indices[i]);
    }
    return offset;
}

DecoderCheckpoint load_checkpoint_text(const std::filesystem::path& path, std::size_t vocab_size)
{
    std::vector<SourceLine> lines = read_clean_lines(path);

    int32_t hidden_size = 0;
    int32_t mlp_size = 0;
    std::unordered_map<std::string, ParsedTensor> tensors;

    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        const SourceLine& line = lines[i];
        const std::vector<std::string> words = split_words(line.text);
        if (words.empty())
        {
            continue;
        }

        if (words[0] == "hidden_size")
        {
            if (words.size() != 2)
            {
                throw std::runtime_error(
                    "Invalid hidden_size syntax at " + path.string() + ":" + std::to_string(line.number));
            }
            hidden_size = parse_int(words[1], path, line.number, "hidden_size");
            continue;
        }

        if (words[0] == "mlp_size")
        {
            if (words.size() != 2)
            {
                throw std::runtime_error(
                    "Invalid mlp_size syntax at " + path.string() + ":" + std::to_string(line.number));
            }
            mlp_size = parse_int(words[1], path, line.number, "mlp_size");
            continue;
        }

        if (words[0] != "tensor")
        {
            throw std::runtime_error(
                "Unexpected token in checkpoint at " + path.string() + ":" + std::to_string(line.number));
        }

        if (words.size() < 4)
        {
            throw std::runtime_error(
                "Invalid tensor header at " + path.string() + ":" + std::to_string(line.number));
        }

        const std::string name = words[1];
        const int32_t rank = parse_int(words[2], path, line.number, "rank");
        if (rank <= 0 || static_cast<std::size_t>(rank) + 3 != words.size())
        {
            throw std::runtime_error("Invalid tensor rank/dims at " + path.string() + ":" + std::to_string(line.number));
        }

        ParsedTensor tensor;
        tensor.dims.reserve(static_cast<std::size_t>(rank));
        for (int32_t d = 0; d < rank; ++d)
        {
            tensor.dims.push_back(parse_int(words[static_cast<std::size_t>(d) + 3], path, line.number, "dimension"));
        }
        tensor.data.assign(checked_element_count(tensor.dims, path, line.number), 0.0F);

        bool saw_end = false;
        for (++i; i < lines.size(); ++i)
        {
            const SourceLine& op_line = lines[i];
            const std::vector<std::string> op = split_words(op_line.text);
            if (op.empty())
            {
                continue;
            }

            if (op[0] == "end")
            {
                saw_end = true;
                break;
            }

            if (op[0] == "fill")
            {
                if (op.size() != 2)
                {
                    throw std::runtime_error(
                        "Invalid fill syntax at " + path.string() + ":" + std::to_string(op_line.number));
                }
                std::fill(tensor.data.begin(), tensor.data.end(), parse_float(op[1], path, op_line.number, "fill"));
                continue;
            }

            if (op[0] == "identity")
            {
                if (op.size() != 2 || tensor.dims.size() != 2 || tensor.dims[0] != tensor.dims[1])
                {
                    throw std::runtime_error(
                        "Invalid identity syntax at " + path.string() + ":" + std::to_string(op_line.number));
                }
                const float value = parse_float(op[1], path, op_line.number, "identity");
                const std::size_t dim = static_cast<std::size_t>(tensor.dims[0]);
                for (std::size_t d = 0; d < dim; ++d)
                {
                    tensor.data[d * dim + d] = value;
                }
                continue;
            }

            if (op[0] == "set")
            {
                const std::size_t expected = 2 + tensor.dims.size();
                if (op.size() != expected)
                {
                    throw std::runtime_error(
                        "Invalid set syntax at " + path.string() + ":" + std::to_string(op_line.number));
                }
                std::vector<int32_t> indices;
                indices.reserve(tensor.dims.size());
                for (std::size_t d = 0; d < tensor.dims.size(); ++d)
                {
                    indices.push_back(parse_int(op[d + 1], path, op_line.number, "set_index"));
                }
                const float value = parse_float(op.back(), path, op_line.number, "set_value");
                const std::size_t offset = flatten_index(tensor.dims, indices, path, op_line.number);
                tensor.data[offset] = value;
                continue;
            }

            if (op[0] == "row_transition")
            {
                if (op.size() != 5 || tensor.dims.size() != 2)
                {
                    throw std::runtime_error(
                        "Invalid row_transition syntax at " + path.string() + ":" + std::to_string(op_line.number));
                }
                const int32_t row = parse_int(op[1], path, op_line.number, "row");
                const int32_t col = parse_int(op[2], path, op_line.number, "col");
                const float high = parse_float(op[3], path, op_line.number, "high");
                const float low = parse_float(op[4], path, op_line.number, "low");
                if (row < 0 || row >= tensor.dims[0] || col < 0 || col >= tensor.dims[1])
                {
                    throw std::runtime_error(
                        "row_transition index out of range at " + path.string() + ":" + std::to_string(op_line.number));
                }
                const std::size_t cols = static_cast<std::size_t>(tensor.dims[1]);
                const std::size_t row_start = static_cast<std::size_t>(row) * cols;
                std::fill(tensor.data.begin() + static_cast<std::ptrdiff_t>(row_start),
                    tensor.data.begin() + static_cast<std::ptrdiff_t>(row_start + cols), low);
                tensor.data[row_start + static_cast<std::size_t>(col)] = high;
                continue;
            }

            if (op[0] == "all_rows_transition")
            {
                if (op.size() != 4 || tensor.dims.size() != 2)
                {
                    throw std::runtime_error("Invalid all_rows_transition syntax at " + path.string() + ":"
                        + std::to_string(op_line.number));
                }
                const int32_t col = parse_int(op[1], path, op_line.number, "col");
                const float high = parse_float(op[2], path, op_line.number, "high");
                const float low = parse_float(op[3], path, op_line.number, "low");
                if (col < 0 || col >= tensor.dims[1])
                {
                    throw std::runtime_error("all_rows_transition column out of range at " + path.string() + ":"
                        + std::to_string(op_line.number));
                }
                const std::size_t rows = static_cast<std::size_t>(tensor.dims[0]);
                const std::size_t cols = static_cast<std::size_t>(tensor.dims[1]);
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const std::size_t start = row * cols;
                    std::fill(tensor.data.begin() + static_cast<std::ptrdiff_t>(start),
                        tensor.data.begin() + static_cast<std::ptrdiff_t>(start + cols), low);
                    tensor.data[start + static_cast<std::size_t>(col)] = high;
                }
                continue;
            }

            throw std::runtime_error(
                "Unknown tensor op at " + path.string() + ":" + std::to_string(op_line.number) + ": " + op[0]);
        }

        if (!saw_end)
        {
            throw std::runtime_error("Missing 'end' for tensor " + name + " in " + path.string());
        }

        tensors[name] = std::move(tensor);
    }

    if (hidden_size <= 0 || mlp_size <= 0)
    {
        throw std::runtime_error("Checkpoint missing valid hidden_size/mlp_size: " + path.string());
    }

    const int32_t vocab = static_cast<int32_t>(vocab_size);
    const auto require_tensor = [&](const std::string& name, std::initializer_list<int32_t> expected_dims) -> const ParsedTensor& {
        const auto it = tensors.find(name);
        if (it == tensors.end())
        {
            throw std::runtime_error("Checkpoint missing tensor: " + name + " in " + path.string());
        }
        const std::vector<int32_t> expected(expected_dims.begin(), expected_dims.end());
        if (it->second.dims != expected)
        {
            std::ostringstream oss;
            oss << "Tensor " << name << " has invalid shape in " << path.string();
            throw std::runtime_error(oss.str());
        }
        return it->second;
    };

    DecoderCheckpoint checkpoint;
    checkpoint.hidden_size = hidden_size;
    checkpoint.mlp_size = mlp_size;
    checkpoint.embedding = require_tensor("embedding", {vocab, hidden_size}).data;
    checkpoint.w_q = require_tensor("w_q", {hidden_size, hidden_size}).data;
    checkpoint.w_k = require_tensor("w_k", {hidden_size, hidden_size}).data;
    checkpoint.w_v = require_tensor("w_v", {hidden_size, hidden_size}).data;
    checkpoint.w1 = require_tensor("w1", {hidden_size, mlp_size}).data;
    checkpoint.b1 = require_tensor("b1", {mlp_size}).data;
    checkpoint.w2 = require_tensor("w2", {mlp_size, hidden_size}).data;
    checkpoint.b2 = require_tensor("b2", {hidden_size}).data;
    checkpoint.w_out = require_tensor("w_out", {hidden_size, vocab}).data;
    checkpoint.b_out = require_tensor("b_out", {vocab}).data;
    return checkpoint;
}

int32_t checked_i32(int64_t value, const std::string& context)
{
    if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::runtime_error("Invalid dimension for " + context + ": " + std::to_string(value));
    }
    return static_cast<int32_t>(value);
}

bool is_qwen_architecture(const DecoderArchitectureConfig& architecture)
{
    const std::string family = to_lower_ascii(architecture.family);
    return starts_with(family, "qwen");
}

std::string qwen_layer_tensor_key(int32_t layer_idx, const std::string& suffix)
{
    return "model.layers." + std::to_string(layer_idx) + "." + suffix;
}

DecoderCheckpoint load_checkpoint_from_safetensors_direct(
    const TensorSource& reader, std::size_t vocab_size, const std::filesystem::path& path)
{
    const auto require_2d = [&](const std::string& name) -> std::pair<int32_t, int32_t> {
        const SafetensorEntry& entry = reader.entry(name);
        if (entry.shape.size() != 2)
        {
            throw std::runtime_error("Expected rank-2 safetensors tensor for " + name + " in " + path.string());
        }
        return {checked_i32(entry.shape[0], name), checked_i32(entry.shape[1], name)};
    };
    const auto require_1d = [&](const std::string& name) -> int32_t {
        const SafetensorEntry& entry = reader.entry(name);
        if (entry.shape.size() != 1)
        {
            throw std::runtime_error("Expected rank-1 safetensors tensor for " + name + " in " + path.string());
        }
        return checked_i32(entry.shape[0], name);
    };

    const auto embedding_shape = require_2d("embedding");
    const int32_t vocab = embedding_shape.first;
    const int32_t hidden = embedding_shape.second;
    if (vocab != static_cast<int32_t>(vocab_size))
    {
        throw std::runtime_error("Safetensors embedding vocab size mismatch in " + path.string());
    }

    const auto q_shape = require_2d("w_q");
    const auto k_shape = require_2d("w_k");
    const auto v_shape = require_2d("w_v");
    const auto w1_shape = require_2d("w1");
    const int32_t b1 = require_1d("b1");
    const auto w2_shape = require_2d("w2");
    const int32_t b2 = require_1d("b2");
    const auto w_out_shape = require_2d("w_out");
    const int32_t b_out = require_1d("b_out");

    if (q_shape.first != hidden || q_shape.second != hidden || k_shape.first != hidden || k_shape.second != hidden
        || v_shape.first != hidden || v_shape.second != hidden || w1_shape.first != hidden || w2_shape.second != hidden
        || w2_shape.first != w1_shape.second || b1 != w1_shape.second || b2 != hidden || w_out_shape.first != hidden
        || w_out_shape.second != vocab || b_out != vocab)
    {
        throw std::runtime_error("Safetensors checkpoint tensor shape mismatch in " + path.string());
    }

    DecoderCheckpoint checkpoint;
    checkpoint.hidden_size = hidden;
    checkpoint.mlp_size = w1_shape.second;
    checkpoint.embedding = reader.load_f32("embedding");
    checkpoint.w_q = reader.load_f32("w_q");
    checkpoint.w_k = reader.load_f32("w_k");
    checkpoint.w_v = reader.load_f32("w_v");
    checkpoint.w1 = reader.load_f32("w1");
    checkpoint.b1 = reader.load_f32("b1");
    checkpoint.w2 = reader.load_f32("w2");
    checkpoint.b2 = reader.load_f32("b2");
    checkpoint.w_out = reader.load_f32("w_out");
    checkpoint.b_out = reader.load_f32("b_out");
    return checkpoint;
}

DecoderCheckpoint load_checkpoint_from_safetensors_qwen3(const TensorSource& reader, std::size_t vocab_size,
    const std::filesystem::path& path, const DecoderArchitectureConfig& architecture)
{
    const std::string key_embedding = "model.embed_tokens.weight";
    const std::string key_lm_head = "lm_head.weight";
    const std::string key_final_norm = "model.norm.weight";

    const auto require_2d = [&](const std::string& name) -> std::pair<int32_t, int32_t> {
        if (!reader.has(name))
        {
            throw std::runtime_error("Missing required Qwen3 safetensors tensor " + name + " in " + path.string());
        }
        const SafetensorEntry& entry = reader.entry(name);
        if (entry.shape.size() != 2)
        {
            throw std::runtime_error("Expected rank-2 Qwen3 safetensors tensor for " + name + " in " + path.string());
        }
        return {checked_i32(entry.shape[0], name), checked_i32(entry.shape[1], name)};
    };
    const auto require_1d = [&](const std::string& name) -> int32_t {
        if (!reader.has(name))
        {
            throw std::runtime_error("Missing required Qwen3 safetensors tensor " + name + " in " + path.string());
        }
        const SafetensorEntry& entry = reader.entry(name);
        if (entry.shape.size() != 1)
        {
            throw std::runtime_error("Expected rank-1 Qwen3 safetensors tensor for " + name + " in " + path.string());
        }
        return checked_i32(entry.shape[0], name);
    };

    const auto embedding_shape = require_2d(key_embedding);
    const int32_t vocab = embedding_shape.first;
    const int32_t hidden = embedding_shape.second;
    if (vocab != static_cast<int32_t>(vocab_size))
    {
        throw std::runtime_error("Qwen3 safetensors vocab size mismatch. embedding rows="
            + std::to_string(vocab) + ", vocab.txt size=" + std::to_string(vocab_size));
    }

    const int32_t num_layers = std::max(architecture.num_layers, 1);
    const int32_t num_attention_heads = std::max(architecture.num_attention_heads, 1);
    const int32_t num_kv_heads = std::max(architecture.num_key_value_heads, 1);

    DecoderCheckpoint checkpoint;
    checkpoint.hidden_size = hidden;
    checkpoint.has_qwen_layers = true;
    checkpoint.embedding = reader.load_f32(key_embedding);
    checkpoint.qwen_layers.reserve(static_cast<std::size_t>(num_layers));

    int32_t inferred_mlp_size = 0;
    int32_t inferred_attention_size = 0;
    for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        const std::string input_norm_key = qwen_layer_tensor_key(layer_idx, "input_layernorm.weight");
        const std::string q_norm_key = qwen_layer_tensor_key(layer_idx, "self_attn.q_norm.weight");
        const std::string k_norm_key = qwen_layer_tensor_key(layer_idx, "self_attn.k_norm.weight");
        const std::string q_key = qwen_layer_tensor_key(layer_idx, "self_attn.q_proj.weight");
        const std::string k_key = qwen_layer_tensor_key(layer_idx, "self_attn.k_proj.weight");
        const std::string v_key = qwen_layer_tensor_key(layer_idx, "self_attn.v_proj.weight");
        const std::string o_key = qwen_layer_tensor_key(layer_idx, "self_attn.o_proj.weight");
        const std::string post_norm_key = qwen_layer_tensor_key(layer_idx, "post_attention_layernorm.weight");
        const std::string gate_key = qwen_layer_tensor_key(layer_idx, "mlp.gate_proj.weight");
        const std::string up_key = qwen_layer_tensor_key(layer_idx, "mlp.up_proj.weight");
        const std::string down_key = qwen_layer_tensor_key(layer_idx, "mlp.down_proj.weight");

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
            throw std::runtime_error("Qwen3 RMSNorm shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }
        if (q_shape.second != hidden || o_shape.first != hidden || o_shape.second != q_hidden)
        {
            throw std::runtime_error("Qwen3 attention projection shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }
        if (k_shape.second != hidden || v_shape.second != hidden || kv_hidden != v_shape.first)
        {
            throw std::runtime_error("Qwen3 k/v projection shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }
        if (gate_shape.second != hidden || up_shape.second != hidden || gate_shape.first != up_shape.first
            || down_shape.first != hidden || down_shape.second != gate_shape.first)
        {
            throw std::runtime_error("Qwen3 MLP projection shape mismatch at layer " + std::to_string(layer_idx)
                + " in " + path.string());
        }

        if (inferred_mlp_size == 0)
        {
            inferred_mlp_size = gate_shape.first;
        }
        else if (inferred_mlp_size != gate_shape.first)
        {
            throw std::runtime_error("Qwen3 inconsistent MLP size across layers in " + path.string());
        }

        if (inferred_attention_size == 0)
        {
            inferred_attention_size = q_hidden;
        }
        else if (inferred_attention_size != q_hidden)
        {
            throw std::runtime_error("Qwen3 inconsistent attention projection size across layers in " + path.string());
        }

        if (q_hidden % num_attention_heads != 0)
        {
            throw std::runtime_error("Qwen3 q_proj out_features must be divisible by num_attention_heads at layer "
                + std::to_string(layer_idx) + " in " + path.string());
        }
        const int32_t head_dim = q_hidden / num_attention_heads;
        const bool grouped_kv_layout = num_attention_heads % num_kv_heads == 0
            && kv_hidden == head_dim * num_kv_heads;
        if (!grouped_kv_layout && kv_hidden != q_hidden)
        {
            throw std::runtime_error("Qwen3 k/v projection out_features do not match grouped-KV or full-attention layout at layer "
                + std::to_string(layer_idx) + " in " + path.string());
        }

        DecoderLayerCheckpoint layer;
        layer.input_norm = reader.load_f32(input_norm_key);

        std::vector<float> q_norm_head(static_cast<std::size_t>(head_dim), 1.0F);
        if (reader.has(q_norm_key))
        {
            const int32_t q_norm_size = require_1d(q_norm_key);
            if (q_norm_size != head_dim)
            {
                throw std::runtime_error("Qwen3 q_norm shape mismatch at layer " + std::to_string(layer_idx)
                    + " in " + path.string());
            }
            q_norm_head = reader.load_f32(q_norm_key);
        }
        layer.q_norm = repeat_head_norm(q_norm_head, num_attention_heads);

        std::vector<float> k_norm_head(static_cast<std::size_t>(head_dim), 1.0F);
        if (reader.has(k_norm_key))
        {
            const int32_t k_norm_size = require_1d(k_norm_key);
            if (k_norm_size != head_dim)
            {
                throw std::runtime_error("Qwen3 k_norm shape mismatch at layer " + std::to_string(layer_idx)
                    + " in " + path.string());
            }
            k_norm_head = reader.load_f32(k_norm_key);
        }
        layer.k_norm = repeat_head_norm(k_norm_head, num_attention_heads);

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
        checkpoint.qwen_layers.push_back(std::move(layer));
    }

    checkpoint.attention_size = inferred_attention_size > 0 ? inferred_attention_size : hidden;
    checkpoint.mlp_size = inferred_mlp_size;

    if (reader.has(key_final_norm))
    {
        if (require_1d(key_final_norm) != hidden)
        {
            throw std::runtime_error("Qwen3 model.norm.weight shape mismatch in " + path.string());
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
            throw std::runtime_error("Qwen3 lm_head shape mismatch in " + path.string());
        }
        checkpoint.w_out = transpose_2d(reader.load_f32(key_lm_head), vocab, hidden, "lm_head");
    }
    else
    {
        checkpoint.w_out = transpose_2d(checkpoint.embedding, vocab, hidden, "embedding");
    }

    // Populate layer-0 compatibility tensors for legacy paths.
    if (!checkpoint.qwen_layers.empty())
    {
        const DecoderLayerCheckpoint& layer0 = checkpoint.qwen_layers.front();
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

DecoderCheckpoint load_checkpoint_from_safetensors(
    const std::filesystem::path& path, std::size_t vocab_size, const DecoderArchitectureConfig& architecture)
{
    const TensorSource reader(path);

    const bool has_direct = reader.has("embedding") && reader.has("w_q") && reader.has("w_k") && reader.has("w_v")
        && reader.has("w1") && reader.has("b1") && reader.has("w2") && reader.has("b2") && reader.has("w_out")
        && reader.has("b_out");
    if (has_direct)
    {
        return load_checkpoint_from_safetensors_direct(reader, vocab_size, path);
    }

    if (is_qwen_architecture(architecture))
    {
        return load_checkpoint_from_safetensors_qwen3(reader, vocab_size, path, architecture);
    }

    throw std::runtime_error("Unsupported safetensors checkpoint layout in " + path.string()
        + ". Provide direct trtf tensor names or set architecture_family=qwen3 for Qwen mapping.");
}

DecoderCheckpoint load_checkpoint(
    const std::filesystem::path& path, std::size_t vocab_size, const DecoderArchitectureConfig& architecture)
{
    if (path.extension() == ".safetensors" || is_safetensors_index_file(path))
    {
        return load_checkpoint_from_safetensors(path, vocab_size, architecture);
    }
    return load_checkpoint_text(path, vocab_size);
}

std::filesystem::path builtin_model_dir()
{
    return std::filesystem::path(TRTF_SOURCE_DIR) / "models" / "tiny-cake-v1";
}

std::vector<std::string> make_placeholder_vocab(int32_t vocab_size)
{
    if (vocab_size <= 0)
    {
        throw std::runtime_error("Invalid vocab_size for placeholder vocab: " + std::to_string(vocab_size));
    }

    std::vector<std::string> vocab(static_cast<std::size_t>(vocab_size));
    for (int32_t i = 0; i < vocab_size; ++i)
    {
        vocab[static_cast<std::size_t>(i)] = "token_" + std::to_string(i);
    }

    if (vocab_size > 0)
    {
        vocab[0] = "<unk>";
    }
    if (vocab_size > 1)
    {
        vocab[1] = "<bos>";
    }
    if (vocab_size > 2)
    {
        vocab[2] = "<eos>";
    }
    if (vocab_size > 3)
    {
        vocab[3] = "<pad>";
    }
    return vocab;
}

std::vector<std::pair<std::string, std::string>> default_fallback_transitions()
{
    return {{"<eos>", "<eos>"}};
}

DecoderModel load_model_from_dir(const std::filesystem::path& model_dir, const std::string& model_id)
{
    const std::filesystem::path config_path = model_dir / "config.json";
    const std::filesystem::path vocab_path = model_dir / "vocab.txt";
    const std::filesystem::path transitions_path = model_dir / "transitions.txt";

    const std::string config_text = read_file(config_path);
    const std::string weights_file = extract_json_string(config_text, "weights_file", "");
    std::filesystem::path checkpoint_path;
    if (!weights_file.empty())
    {
        checkpoint_path = model_dir / weights_file;
    }
    if (checkpoint_path.empty() || !std::filesystem::exists(checkpoint_path))
    {
        const std::filesystem::path hf_safetensors = model_dir / "model.safetensors";
        if (std::filesystem::exists(hf_safetensors))
        {
            checkpoint_path = hf_safetensors;
        }
        else
        {
            const std::filesystem::path sharded_index = model_dir / "model.safetensors.index.json";
            if (std::filesystem::exists(sharded_index))
            {
                checkpoint_path = sharded_index;
            }
        }
    }
    if (checkpoint_path.empty())
    {
        checkpoint_path = model_dir / "weights.txt";
    }
    const bool has_checkpoint_file = std::filesystem::exists(checkpoint_path);

    DecoderModel model;
    model.model_id = model_id;

    const std::string model_type = extract_json_string(config_text, "model_type", "");
    model.architecture.family = extract_json_string(config_text, "architecture_family", "");
    if (model.architecture.family.empty())
    {
        const std::string lowered = to_lower_ascii(model_type);
        model.architecture.family = starts_with(lowered, "qwen") ? "qwen3" : "toy-decoder";
    }

    if (std::filesystem::exists(vocab_path))
    {
        model.vocab = load_vocab(vocab_path);
    }
    else if (has_checkpoint_file)
    {
        const int32_t vocab_size = extract_json_int(config_text, "vocab_size", 0);
        if (vocab_size <= 0)
        {
            throw std::runtime_error("Missing vocab.txt and no valid vocab_size in config for checkpoint-backed model: "
                + model_dir.string());
        }
        model.vocab = make_placeholder_vocab(vocab_size);
    }
    else
    {
        throw std::runtime_error("Failed to open vocab file: " + vocab_path.string());
    }

    if (std::filesystem::exists(transitions_path))
    {
        model.transitions = load_transitions(transitions_path);
    }
    else if (has_checkpoint_file)
    {
        model.transitions = default_fallback_transitions();
    }
    else
    {
        throw std::runtime_error("Failed to open transitions file: " + transitions_path.string());
    }

    model.default_next_token = extract_json_string(config_text, "default_next_token", "<eos>");
    if (std::find(model.vocab.begin(), model.vocab.end(), model.default_next_token) == model.vocab.end())
    {
        model.default_next_token = "<eos>";
        if (std::find(model.vocab.begin(), model.vocab.end(), model.default_next_token) == model.vocab.end())
        {
            model.default_next_token = model.vocab.front();
        }
    }

    const int32_t configured_max_cache = extract_json_int(config_text, "max_cache_length", -1);
    model.max_cache_length = configured_max_cache;
    if (model.max_cache_length <= 0)
    {
        model.max_cache_length = extract_json_int(config_text, "max_position_embeddings", 32);
    }

    const int32_t env_max_cache = parse_positive_env_int("TRTF_MAX_CACHE_LENGTH", -1);
    if (env_max_cache > 0)
    {
        model.max_cache_length = env_max_cache;
    }
    else if (configured_max_cache <= 0)
    {
        const std::string family_lower = to_lower_ascii(model.architecture.family);
        if (starts_with(family_lower, "qwen") && model.max_cache_length > 4096)
        {
            // Default cap keeps TRT step-engine memory practical for large upstream checkpoints.
            model.max_cache_length = 4096;
        }
    }

    model.architecture.num_layers = std::max(extract_json_int(config_text, "num_hidden_layers", 1), 1);
    model.architecture.num_attention_heads = std::max(extract_json_int(config_text, "num_attention_heads", 1), 1);
    model.architecture.num_key_value_heads = std::max(extract_json_int(config_text, "num_key_value_heads", 1), 1);
    model.architecture.bos_token_id = extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    model.architecture.eos_token_id = extract_json_int_or_first_array(config_text, "eos_token_id", -1);
    model.architecture.pad_token_id = extract_json_int_or_first_array(config_text, "pad_token_id", -1);
    model.architecture.rms_norm_eps = extract_json_float(config_text, "rms_norm_eps", 1.0e-5F);
    model.architecture.rope_theta = extract_json_float(config_text, "rope_theta", 10000.0F);

    if (model.max_cache_length <= 0)
    {
        model.max_cache_length = 32;
    }

    if (std::filesystem::exists(model_dir / "tokenizer.json"))
    {
        model.prefer_hf_tokenizer = true;
        model.hf_tokenizer_dir = model_dir.string();
    }

    if (has_checkpoint_file)
    {
        model.checkpoint = load_checkpoint(checkpoint_path, model.vocab.size(), model.architecture);
        model.has_checkpoint = true;
    }
    return model;
}

} // namespace

DecoderModel LoadDecoderModel(const std::string& model_id)
{
    static constexpr char kBuiltinModelId[] = "trtf/tiny-cake-v1";

    if (!model_id.empty())
    {
        std::filesystem::path requested(model_id);
        if (std::filesystem::exists(requested) && std::filesystem::is_directory(requested))
        {
            return load_model_from_dir(requested, model_id);
        }

        if (model_id != kBuiltinModelId)
        {
            throw std::runtime_error("Unknown model_id: " + model_id
                + ". Use \"" + std::string(kBuiltinModelId) + "\" or a local model directory path.");
        }
    }

    const std::filesystem::path builtin = builtin_model_dir();
    if (!std::filesystem::exists(builtin))
    {
        throw std::runtime_error("Built-in model directory not found: " + builtin.string());
    }
    return load_model_from_dir(builtin, kBuiltinModelId);
}

} // namespace trtf
