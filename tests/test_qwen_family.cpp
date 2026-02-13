#include "trtf/model_resolver.h"
#include "trtf/pipeline.h"

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

void write_qwen_root_checkpoint(const std::filesystem::path& dir)
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
    std::vector<float> qk_norm(static_cast<std::size_t>(q_hidden / 2), 1.0F);
    std::vector<float> up_proj(static_cast<std::size_t>(mlp) * static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> gate_proj(static_cast<std::size_t>(mlp) * static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> down_proj(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(mlp), 0.0F);
    std::vector<float> lm_head(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden), -1.0F);
    for (int32_t i = 0; i < vocab && i < hidden; ++i)
    {
        lm_head[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(i)] = 1.0F;
    }

    std::vector<TensorSpec> tensors;
    tensors.push_back({"model.embed_tokens.weight", {vocab, hidden}, embedding});
    for (int32_t layer = 0; layer < layers; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        tensors.push_back({prefix + "input_layernorm.weight", {hidden}, norm});
        tensors.push_back({prefix + "self_attn.q_norm.weight", {q_hidden / 2}, qk_norm});
        tensors.push_back({prefix + "self_attn.k_norm.weight", {q_hidden / 2}, qk_norm});
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

void write_hf_qwen_root(const std::filesystem::path& dir)
{
    write_file(dir / "config.json",
        "{\n"
        "  \"model_type\": \"qwen2\",\n"
        "  \"architectures\": [\"Qwen2ForCausalLM\"],\n"
        "  \"vocab_size\": 8,\n"
        "  \"hidden_size\": 8,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"rms_norm_eps\": 1e-5,\n"
        "  \"rope_theta\": 10000.0\n"
        "}\n");
    write_qwen_root_checkpoint(dir);
}

void write_decoder_definition(const std::filesystem::path& dir)
{
    const std::filesystem::path decoder_dir = dir / "trtf_decoder";
    std::filesystem::create_directories(decoder_dir);

    write_file(decoder_dir / "config.json",
        "{\n"
        "  \"default_next_token\": \"from\",\n"
        "  \"max_cache_length\": 16\n"
        "}\n");

    write_file(decoder_dir / "vocab.txt",
        "<unk>\n"
        "<bos>\n"
        "<eos>\n"
        "<pad>\n"
        "hello\n"
        "from\n"
        "qwen\n"
        ".\n");

    write_file(decoder_dir / "transitions.txt",
        "hello from\n"
        "from qwen\n"
        "qwen .\n"
        ". <eos>\n");
}

} // namespace

int main()
{
    std::filesystem::path with_definition;
    std::filesystem::path without_definition;

    try
    {
        with_definition = make_temp_dir_or_throw("/tmp/trtf_qwen_family_with_def_XXXXXX");
        without_definition = make_temp_dir_or_throw("/tmp/trtf_qwen_family_without_def_XXXXXX");

        write_hf_qwen_root(with_definition);
        write_decoder_definition(with_definition);

        write_hf_qwen_root(without_definition);

        const trtf::ResolvedModelSpec qwen_spec = trtf::ResolveTextGenerationModel(with_definition.string());
        if (qwen_spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected qwen+decoder-definition to resolve as decoder-definition" << std::endl;
            return 1;
        }
        if (qwen_spec.decoder_model.model_id != with_definition.string())
        {
            std::cerr << "expected resolved model_id to track original hf dir" << std::endl;
            return 1;
        }

        auto pipeline = trtf::Pipeline::CreateTextGeneration(with_definition.string(), false, false);
        if (pipeline.backend_name() != "cpu-reference")
        {
            std::cerr << "expected cpu-reference backend for qwen family definition path, got "
                      << pipeline.backend_name() << std::endl;
            return 1;
        }
        const auto out = pipeline("hello");
        if (out.size() != 1 || out[0].generated_text.find("hello from qwen.") == std::string::npos)
        {
            std::cerr << "unexpected output for qwen family definition path: "
                      << (out.empty() ? std::string("<empty>") : out[0].generated_text) << std::endl;
            return 1;
        }

        const trtf::ResolvedModelSpec fallback_spec = trtf::ResolveTextGenerationModel(without_definition.string());
        if (fallback_spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected qwen hf dir without decoder-definition to resolve through qwen safetensors bridge"
                      << std::endl;
            return 1;
        }
        if (!fallback_spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected qwen root safetensors bridge path to load checkpoint" << std::endl;
            return 1;
        }
        if (fallback_spec.decoder_model.architecture.family != "qwen3")
        {
            std::cerr << "expected qwen root path to set architecture_family=qwen3" << std::endl;
            return 1;
        }
        if (!fallback_spec.decoder_model.checkpoint.has_qwen_layers
            || fallback_spec.decoder_model.checkpoint.qwen_layers.size() != 2)
        {
            std::cerr << "expected qwen root bridge to load full multi-layer qwen checkpoint tensors" << std::endl;
            return 1;
        }
        if (fallback_spec.decoder_model.checkpoint.final_norm.size() != 8)
        {
            std::cerr << "expected qwen root bridge to load final_norm tensor" << std::endl;
            return 1;
        }
        if (fallback_spec.decoder_model.checkpoint.attention_size != 16)
        {
            std::cerr << "expected qwen root bridge to preserve non-square q attention width" << std::endl;
            return 1;
        }

        const trtf::ResolvedModelSpec builtin_alias_spec = trtf::ResolveTextGenerationModel("QWEN3");
        if (builtin_alias_spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected built-in QWEN3 alias to resolve as decoder-definition" << std::endl;
            return 1;
        }
        if (builtin_alias_spec.decoder_model.model_id != "QWEN3")
        {
            std::cerr << "expected built-in QWEN3 alias to preserve requested model_id" << std::endl;
            return 1;
        }
        if (!builtin_alias_spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected built-in QWEN3 alias to load checkpoint tensors" << std::endl;
            return 1;
        }

        auto qwen3_model = trtf::loadModel("QWEN3", false, false);
        const std::string qwen3_out = qwen3_model.generate("Hello");
        if (qwen3_model.backend_name() != "cpu-reference")
        {
            std::cerr << "expected cpu-reference backend for built-in QWEN3 in cpu-only selection, got "
                      << qwen3_model.backend_name() << std::endl;
            return 1;
        }
        if (qwen3_out.empty())
        {
            std::cerr << "expected non-empty generated text for built-in QWEN3 alias" << std::endl;
            return 1;
        }

        std::cout << "test_qwen_family passed" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_qwen_family failed: " << e.what() << std::endl;
        std::filesystem::remove_all(with_definition);
        std::filesystem::remove_all(without_definition);
        return 1;
    }

    std::filesystem::remove_all(with_definition);
    std::filesystem::remove_all(without_definition);
    return 0;
}
