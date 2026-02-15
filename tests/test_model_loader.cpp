// Test: Generic model loading from safetensors checkpoint formats.
// Verifies: Single safetensors file and sharded safetensors with
// model.safetensors.index.json. Validates checkpoint tensor shapes.

#include "test_helpers.h"
#include "trtf/model.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

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

} // namespace

int main()
{
    try
    {
        const std::filesystem::path tmp_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_model_loader_XXXXXX");
        trtf_test::write_file(tmp_dir / "config.json",
            "{\n"
            "  \"model_type\": \"toy_decoder_block\",\n"
            "  \"weights_file\": \"weights.safetensors\",\n"
            "  \"vocab_size\": 4,\n"
            "  \"max_cache_length\": 8\n"
            "}\n");

        const std::vector<trtf_test::TensorSpec> tensors = {
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
        trtf_test::write_safetensors_f32(tmp_dir / "weights.safetensors", tensors);

        const trtf::DecoderModel safetensors_model = trtf::LoadDecoderModel(tmp_dir.string());
        validate_checkpoint_shapes(safetensors_model, "safetensors");
        if (safetensors_model.vocab.size() != 4)
        {
            throw std::runtime_error("expected placeholder vocab size=4 for safetensors model");
        }

        const std::filesystem::path sharded_dir = tmp_dir / "sharded";
        std::filesystem::create_directories(sharded_dir);
        trtf_test::write_file(sharded_dir / "config.json",
            "{\n"
            "  \"model_type\": \"toy_decoder_block\",\n"
            "  \"vocab_size\": 4,\n"
            "  \"max_cache_length\": 8\n"
            "}\n");

        trtf_test::write_safetensors_f32(sharded_dir / "model-00001-of-00002.safetensors",
            {
                {"embedding", {4, 4}, tensors[0].data},
                {"w_q", {4, 4}, tensors[1].data},
                {"w_k", {4, 4}, tensors[2].data},
                {"w_v", {4, 4}, tensors[3].data},
                {"w1", {4, 8}, tensors[4].data},
                {"b1", {8}, tensors[5].data},
            });
        trtf_test::write_safetensors_f32(sharded_dir / "model-00002-of-00002.safetensors",
            {
                {"w2", {8, 4}, tensors[6].data},
                {"b2", {4}, tensors[7].data},
                {"w_out", {4, 4}, tensors[8].data},
                {"b_out", {4}, tensors[9].data},
            });
        trtf_test::write_safetensors_index(sharded_dir / "model.safetensors.index.json",
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
        std::cout << "test_model_loader passed" << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_model_loader failed: " << e.what() << std::endl;
        return 1;
    }
}
