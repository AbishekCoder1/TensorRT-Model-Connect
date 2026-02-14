#include "trtf/model_resolver.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdlib.h>

namespace {

struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

std::filesystem::path make_temp_dir_or_throw(const char* pattern)
{
    char buffer[256];
    std::strncpy(buffer, pattern, sizeof(buffer));
    buffer[sizeof(buffer) - 1] = '\0';
    char* created = mkdtemp(buffer);
    if (created == nullptr)
    {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(created);
}

void write_file(const std::filesystem::path& path, const std::string& content)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out << content;
}

void write_u64_le(std::ofstream& out, uint64_t value)
{
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i)
    {
        bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFFU);
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

void write_safetensors_f32(const std::filesystem::path& path, const std::vector<TensorSpec>& specs)
{
    std::string header = "{";
    uint64_t offset = 0;
    for (std::size_t i = 0; i < specs.size(); ++i)
    {
        const auto& spec = specs[i];
        const uint64_t bytes = static_cast<uint64_t>(spec.data.size() * sizeof(float));
        if (i != 0)
        {
            header += ",";
        }
        header += "\"" + spec.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t d = 0; d < spec.shape.size(); ++d)
        {
            if (d != 0)
            {
                header += ",";
            }
            header += std::to_string(spec.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    header += "}";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("Failed to write safetensors file: " + path.string());
    }

    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& spec : specs)
    {
        out.write(reinterpret_cast<const char*>(spec.data.data()),
            static_cast<std::streamsize>(spec.data.size() * sizeof(float)));
    }
}

void write_llama_root_checkpoint(const std::filesystem::path& dir)
{
    const int32_t vocab = 8;
    const int32_t hidden = 8;
    const int32_t q_hidden = 16;
    const int32_t kv_hidden = 8;
    const int32_t mlp = 16;
    const int32_t layers = 2;

    std::vector<float> embedding(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden), 0.0F);
    for (int32_t i = 0; i < hidden; ++i)
    {
        embedding[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> q_proj(static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden), 0.0F);
    for (int32_t i = 0; i < hidden; ++i)
    {
        q_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> o_proj(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(q_hidden), 0.0F);
    for (int32_t i = 0; i < hidden; ++i)
    {
        o_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(q_hidden) + static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> k_proj(static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> v_proj(static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden), 0.0F);
    for (int32_t i = 0; i < kv_hidden; ++i)
    {
        k_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(i)] = 1.0F;
        v_proj[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<float> norm(static_cast<std::size_t>(hidden), 1.0F);
    std::vector<float> up_proj(static_cast<std::size_t>(mlp) * static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> gate_proj(static_cast<std::size_t>(mlp) * static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> down_proj(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(mlp), 0.0F);
    std::vector<float> lm_head(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden), -1.0F);
    for (int32_t i = 0; i < vocab && i < hidden; ++i)
    {
        lm_head[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(i)] = 1.0F;
    }

    // LLaMA checkpoint: same as Qwen but WITHOUT q_norm/k_norm tensors.
    std::vector<TensorSpec> tensors;
    tensors.push_back({"model.embed_tokens.weight", {vocab, hidden}, embedding});
    for (int32_t layer = 0; layer < layers; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        tensors.push_back({prefix + "input_layernorm.weight", {hidden}, norm});
        // No q_norm / k_norm for LLaMA.
        tensors.push_back({prefix + "self_attn.q_proj.weight", {q_hidden, hidden}, q_proj});
        tensors.push_back({prefix + "self_attn.k_proj.weight", {kv_hidden, hidden}, k_proj});
        tensors.push_back({prefix + "self_attn.v_proj.weight", {kv_hidden, hidden}, v_proj});
        tensors.push_back({prefix + "self_attn.o_proj.weight", {hidden, q_hidden}, o_proj});
        tensors.push_back({prefix + "post_attention_layernorm.weight", {hidden}, norm});
        tensors.push_back({prefix + "mlp.gate_proj.weight", {mlp, hidden}, gate_proj});
        tensors.push_back({prefix + "mlp.up_proj.weight", {mlp, hidden}, up_proj});
        tensors.push_back({prefix + "mlp.down_proj.weight", {hidden, mlp}, down_proj});
    }
    tensors.push_back({"model.norm.weight", {hidden}, norm});
    tensors.push_back({"lm_head.weight", {vocab, hidden}, lm_head});
    write_safetensors_f32(dir / "model.safetensors", tensors);
}

void write_hf_llama_root(const std::filesystem::path& dir)
{
    write_file(dir / "config.json",
        "{\n"
        "  \"model_type\": \"llama\",\n"
        "  \"architectures\": [\"LlamaForCausalLM\"],\n"
        "  \"vocab_size\": 8,\n"
        "  \"hidden_size\": 8,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"rms_norm_eps\": 1e-5,\n"
        "  \"rope_theta\": 10000.0\n"
        "}\n");
    write_llama_root_checkpoint(dir);
}

} // namespace

int main()
{
    std::filesystem::path llama_dir;

    try
    {
        llama_dir = make_temp_dir_or_throw("/tmp/trtf_llama_family_XXXXXX");

        write_hf_llama_root(llama_dir);

        // Verify that the LLaMA model resolves correctly through the family registry.
        const trtf::ResolvedModelSpec spec = trtf::ResolveTextGenerationModel(llama_dir.string());
        if (spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected llama hf dir to resolve as decoder-definition" << std::endl;
            return 1;
        }
        if (spec.decoder_model.model_id != llama_dir.string())
        {
            std::cerr << "expected resolved model_id to track original hf dir" << std::endl;
            return 1;
        }
        if (!spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected llama safetensors bridge to load checkpoint" << std::endl;
            return 1;
        }

        // Verify the family is detected as llama.
        {
            const std::string family = spec.decoder_model.architecture.family;
            if (family.substr(0, 5) != "llama")
            {
                std::cerr << "expected llama architecture family, got " << family << std::endl;
                return 1;
            }
        }

        // Verify multi-layer checkpoint was loaded.
        if (!spec.decoder_model.checkpoint.has_decoder_layers
            || spec.decoder_model.checkpoint.decoder_layers.size() != 2)
        {
            std::cerr << "expected llama bridge to load full multi-layer checkpoint tensors" << std::endl;
            return 1;
        }

        // Verify final_norm was loaded.
        if (spec.decoder_model.checkpoint.final_norm.size() != 8)
        {
            std::cerr << "expected llama bridge to load final_norm tensor" << std::endl;
            return 1;
        }

        // Verify attention_size is preserved (q_proj has q_hidden=16 with hidden=8).
        if (spec.decoder_model.checkpoint.attention_size != 16)
        {
            std::cerr << "expected llama bridge to preserve non-square q attention width, got "
                      << spec.decoder_model.checkpoint.attention_size << std::endl;
            return 1;
        }

        // Verify q_norm/k_norm are empty since LLaMA has no per-head norms.
        // Empty q_norm tells the graph builder to skip per-head RMS norm entirely.
        {
            const auto& layer0 = spec.decoder_model.checkpoint.decoder_layers[0];
            if (!layer0.q_norm.empty())
            {
                std::cerr << "expected llama q_norm to be empty (no QK norm), got size " << layer0.q_norm.size() << std::endl;
                return 1;
            }
            if (!layer0.k_norm.empty())
            {
                std::cerr << "expected llama k_norm to be empty (no QK norm), got size " << layer0.k_norm.size() << std::endl;
                return 1;
            }
        }

        std::cout << "test_llama_family passed" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_llama_family failed: " << e.what() << std::endl;
        std::filesystem::remove_all(llama_dir);
        return 1;
    }

    std::filesystem::remove_all(llama_dir);
    return 0;
}
