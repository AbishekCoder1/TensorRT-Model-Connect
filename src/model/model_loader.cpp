#include "trtf/model.h"
#include "checkpoint_mapper.h"
#include "safetensors_loader.h"
#include "utils/data_dir.h"
#include "utils/text_parsers.h"
#include "utils/json_helpers.h"

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

    ICheckpointMapper* mapper = FindCheckpointMapper(architecture);
    if (mapper)
    {
        return mapper->map_checkpoint(reader, vocab_size, path, architecture);
    }

    throw std::runtime_error("No checkpoint mapper for family: " + architecture.family
        + " in " + path.string()
        + ". Provide direct trtf tensor names or register a checkpoint mapper.");
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
    const bool has_checkpoint_file = !checkpoint_path.empty() && std::filesystem::exists(checkpoint_path);

    DecoderModel model;
    model.model_id = model_id;

    const std::string model_type = extract_json_string(config_text, "model_type", "");
    model.architecture.family = extract_json_string(config_text, "architecture_family", "");
    if (model.architecture.family.empty())
    {
        model.architecture.family = model_type.empty() ? "unknown" : to_lower_ascii(model_type);
    }

    const int32_t vocab_size = extract_json_int(config_text, "vocab_size", 0);
    if (vocab_size > 0)
    {
        model.vocab = make_placeholder_vocab(vocab_size);
    }
    else if (!has_checkpoint_file)
    {
        throw std::runtime_error("Missing vocab_size in config.json for model: " + model_dir.string());
    }

    model.transitions = default_fallback_transitions();
    model.default_next_token = "<eos>";

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

    model.architecture.num_layers = std::max(extract_json_int(config_text, "num_hidden_layers", 1), 1);
    model.architecture.num_attention_heads = std::max(extract_json_int(config_text, "num_attention_heads", 1), 1);
    model.architecture.num_key_value_heads = std::max(extract_json_int(config_text, "num_key_value_heads", 1), 1);
    model.architecture.bos_token_id = extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    model.architecture.eos_token_id = extract_json_int_or_first_array(config_text, "eos_token_id", -1);
    model.architecture.pad_token_id = extract_json_int_or_first_array(config_text, "pad_token_id", -1);
    model.architecture.rms_norm_eps = extract_json_float(config_text, "rms_norm_eps", 1.0e-5F);
    model.architecture.rope_theta = extract_json_float(config_text, "rope_theta", 10000.0F);

    const int32_t intermediate_size = extract_json_int(config_text, "intermediate_size", -1);
    if (intermediate_size > 0)
    {
        model.architecture.extra_int_params["intermediate_size"] = intermediate_size;
    }

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
        model.checkpoint = load_checkpoint_from_safetensors(checkpoint_path, model.vocab.size(), model.architecture);
        model.has_checkpoint = true;
    }
    return model;
}

} // namespace

DecoderModel LoadDecoderModel(const std::string& model_id)
{
    if (model_id.empty())
    {
        throw std::runtime_error("model_id must not be empty.");
    }

    std::filesystem::path requested(model_id);
    if (std::filesystem::exists(requested) && std::filesystem::is_directory(requested))
    {
        return load_model_from_dir(requested, model_id);
    }

    throw std::runtime_error("Unknown model_id: " + model_id
        + ". Use a local model directory path or an HF model alias.");
}

} // namespace trtf
