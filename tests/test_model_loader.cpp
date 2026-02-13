#include "trtf/model.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <stdlib.h>

namespace {

struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

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
        throw std::runtime_error("failed to open safetensors test file: " + path.string());
    }

    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& spec : specs)
    {
        out.write(reinterpret_cast<const char*>(spec.data.data()),
            static_cast<std::streamsize>(spec.data.size() * sizeof(float)));
    }
}

void write_safetensors_index(const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& weight_map)
{
    std::ofstream out(path, std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("failed to open safetensors index test file: " + path.string());
    }

    out << "{\n";
    out << "  \"metadata\": {},\n";
    out << "  \"weight_map\": {\n";
    for (std::size_t i = 0; i < weight_map.size(); ++i)
    {
        out << "    \"" << weight_map[i].first << "\": \"" << weight_map[i].second << "\"";
        out << (i + 1 == weight_map.size() ? "\n" : ",\n");
    }
    out << "  }\n";
    out << "}\n";
}

void validate_checkpoint_shapes(const trtf::DecoderModel& model, const char* context)
{
    if (!model.has_checkpoint)
    {
        throw std::runtime_error(std::string("expected checkpoint tensors for ") + context);
    }
    if (model.checkpoint.hidden_size <= 0 || model.checkpoint.mlp_size <= 0)
    {
        throw std::runtime_error(std::string("invalid hidden_size/mlp_size for ") + context);
    }

    const std::size_t vocab = model.vocab.size();
    const std::size_t hidden = static_cast<std::size_t>(model.checkpoint.hidden_size);
    const std::size_t mlp = static_cast<std::size_t>(model.checkpoint.mlp_size);

    if (model.checkpoint.embedding.size() != vocab * hidden
        || model.checkpoint.w_q.size() != hidden * hidden
        || model.checkpoint.w_k.size() != hidden * hidden
        || model.checkpoint.w_v.size() != hidden * hidden
        || model.checkpoint.w1.size() != hidden * mlp
        || model.checkpoint.b1.size() != mlp
        || model.checkpoint.w2.size() != mlp * hidden
        || model.checkpoint.b2.size() != hidden
        || model.checkpoint.w_out.size() != hidden * vocab
        || model.checkpoint.b_out.size() != vocab)
    {
        throw std::runtime_error(std::string("checkpoint tensor size mismatch for ") + context);
    }
}

std::filesystem::path make_temp_dir()
{
    char temp_dir_template[] = "/tmp/trtf_model_loader_XXXXXX";
    char* created = mkdtemp(temp_dir_template);
    if (created == nullptr)
    {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(created);
}

} // namespace

int main()
{
    try
    {
        const trtf::DecoderModel builtin = trtf::LoadDecoderModel("trtf/tiny-cake-v1");
        validate_checkpoint_shapes(builtin, "built-in weights.txt");

        const std::filesystem::path tmp_dir = make_temp_dir();
        const std::filesystem::path cfg = tmp_dir / "config.json";
        {
            std::ofstream out(cfg);
            out << "{\n"
                << "  \"model_type\": \"toy_decoder_block\",\n"
                << "  \"weights_file\": \"weights.safetensors\",\n"
                << "  \"vocab_size\": 4,\n"
                << "  \"max_cache_length\": 8\n"
                << "}\n";
        }

        const std::vector<TensorSpec> tensors = {
            {"embedding", {4, 4}, {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            }},
            {"w_q", {4, 4}, {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            }},
            {"w_k", {4, 4}, std::vector<float>(16, 0.0F)},
            {"w_v", {4, 4}, std::vector<float>(16, 0.0F)},
            {"w1", {4, 8}, std::vector<float>(32, 0.0F)},
            {"b1", {8}, std::vector<float>(8, 0.0F)},
            {"w2", {8, 4}, std::vector<float>(32, 0.0F)},
            {"b2", {4}, std::vector<float>(4, 0.0F)},
            {"w_out", {4, 4}, std::vector<float>(16, -1.0F)},
            {"b_out", {4}, std::vector<float>(4, 0.0F)},
        };
        write_safetensors_f32(tmp_dir / "weights.safetensors", tensors);

        const trtf::DecoderModel safetensors_model = trtf::LoadDecoderModel(tmp_dir.string());
        validate_checkpoint_shapes(safetensors_model, "safetensors");
        if (safetensors_model.vocab.size() != 4)
        {
            throw std::runtime_error("expected placeholder vocab size=4 for safetensors model");
        }
        if (safetensors_model.transitions.empty())
        {
            throw std::runtime_error("expected fallback transitions for safetensors model");
        }

        const std::filesystem::path sharded_dir = tmp_dir / "sharded";
        std::filesystem::create_directories(sharded_dir);
        {
            std::ofstream out(sharded_dir / "config.json");
            out << "{\n"
                << "  \"model_type\": \"toy_decoder_block\",\n"
                << "  \"vocab_size\": 4,\n"
                << "  \"max_cache_length\": 8\n"
                << "}\n";
        }

        write_safetensors_f32(sharded_dir / "model-00001-of-00002.safetensors",
            {
                {"embedding", {4, 4}, tensors[0].data},
                {"w_q", {4, 4}, tensors[1].data},
                {"w_k", {4, 4}, tensors[2].data},
                {"w_v", {4, 4}, tensors[3].data},
                {"w1", {4, 8}, tensors[4].data},
                {"b1", {8}, tensors[5].data},
            });
        write_safetensors_f32(sharded_dir / "model-00002-of-00002.safetensors",
            {
                {"w2", {8, 4}, tensors[6].data},
                {"b2", {4}, tensors[7].data},
                {"w_out", {4, 4}, tensors[8].data},
                {"b_out", {4}, tensors[9].data},
            });
        write_safetensors_index(sharded_dir / "model.safetensors.index.json",
            {
                {"embedding", "model-00001-of-00002.safetensors"},
                {"w_q", "model-00001-of-00002.safetensors"},
                {"w_k", "model-00001-of-00002.safetensors"},
                {"w_v", "model-00001-of-00002.safetensors"},
                {"w1", "model-00001-of-00002.safetensors"},
                {"b1", "model-00001-of-00002.safetensors"},
                {"w2", "model-00002-of-00002.safetensors"},
                {"b2", "model-00002-of-00002.safetensors"},
                {"w_out", "model-00002-of-00002.safetensors"},
                {"b_out", "model-00002-of-00002.safetensors"},
            });

        const trtf::DecoderModel sharded_model = trtf::LoadDecoderModel(sharded_dir.string());
        validate_checkpoint_shapes(sharded_model, "sharded safetensors");

        std::filesystem::remove_all(tmp_dir);
        std::cout << "test_model_loader passed with hidden_size=" << builtin.checkpoint.hidden_size
                  << " vocab=" << builtin.vocab.size() << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_model_loader failed: " << e.what() << std::endl;
        return 1;
    }
}
