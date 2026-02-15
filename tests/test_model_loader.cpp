// =============================================================================
// Test: Generic model loading from safetensors checkpoint formats.
//
// Purpose:
//   Validates the low-level DecoderModel loading pipeline (LoadDecoderModel)
//   for both single-file and sharded safetensors checkpoint formats. This tests
//   the safetensors parser (SafetensorReader), the generic checkpoint mapper,
//   and tensor shape validation -- independent of any specific model family.
//
// What is verified:
//   - Single safetensors loading: A model with weights_file pointing to a
//     single .safetensors file loads all tensors with correct shapes.
//   - Sharded safetensors loading: A model with model.safetensors.index.json
//     pointing to multiple shard files loads and reassembles all tensors
//     correctly.
//   - Tensor shape validation: All loaded tensors (embedding, attention, MLP,
//     output) match the expected dimensions derived from config parameters.
//   - Vocab placeholder: The placeholder vocabulary has the correct size.
//
// Dependencies:
//   - test_helpers.h: make_temp_dir_or_throw, write_file, write_safetensors_f32,
//     write_safetensors_index, TensorSpec
//   - trtf/model.h: DecoderModel, LoadDecoderModel
//
// Approach:
//   1. Creates a temporary directory with a config.json and a single
//      weights.safetensors file containing 10 tensors (embedding, attention
//      weights, MLP weights, output projection). Loads the model and validates
//      all tensor shapes.
//   2. Creates a subdirectory with the same tensors split across two shard files
//      and a model.safetensors.index.json mapping file. Loads the sharded model
//      and validates all tensor shapes match the single-file case.
// =============================================================================

#include "test_helpers.h"
#include "trtf/model.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// validate_checkpoint_shapes
//
// Intention: Validate that all checkpoint tensors have the correct sizes given
//   the model's vocab_size, hidden_size, and mlp_size.
// Setup: Receives a loaded DecoderModel and a context string for error messages.
// Mechanism: Computes expected sizes from vocab * hidden, hidden * hidden,
//   hidden * mlp, etc. and compares against actual tensor vector sizes. Throws
//   on any mismatch. Checks all 10 core tensors: embedding, w_q, w_k, w_v,
//   w1, b1, w2, b2, w_out, b_out.
// ---------------------------------------------------------------------------
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
        // =================================================================
        // Section 1: Single safetensors file loading
        //
        // Intention: Verify that LoadDecoderModel correctly reads a single
        //   safetensors file referenced via the "weights_file" config key.
        // Setup: Creates a temp dir with:
        //   - config.json: model_type="toy_decoder_block", vocab_size=4,
        //     max_cache_length=8, weights_file="weights.safetensors"
        //   - weights.safetensors: 10 tensors with shapes matching a
        //     hidden=4, mlp=8, vocab=4 toy model. Embedding and w_q use
        //     identity matrices; w_out uses -1.0 fill with identity diagonal.
        // Mechanism: Calls LoadDecoderModel, then validate_checkpoint_shapes
        //   to verify all 10 tensors have the correct dimensionality.
        // =================================================================
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

        // -----------------------------------------------------------------
        // Assertion: Single-file model loads with correct shapes and vocab
        //
        // Intention: Confirm LoadDecoderModel parses the safetensors header,
        //   reads the tensor data, and populates the checkpoint struct with
        //   correctly-sized tensors.
        // Mechanism: validate_checkpoint_shapes checks all 10 tensors;
        //   vocab size is checked separately (should be 4 from config).
        // -----------------------------------------------------------------
        const trtf::DecoderModel safetensors_model = trtf::LoadDecoderModel(tmp_dir.string());
        validate_checkpoint_shapes(safetensors_model, "safetensors");
        if (safetensors_model.vocab.size() != 4)
        {
            throw std::runtime_error("expected placeholder vocab size=4 for safetensors model");
        }

        // =================================================================
        // Section 2: Sharded safetensors loading
        //
        // Intention: Verify that LoadDecoderModel correctly reads sharded
        //   safetensors files via model.safetensors.index.json.
        // Setup: Creates a "sharded" subdirectory with:
        //   - config.json: same toy model config (no weights_file key, which
        //     triggers the sharded loading path).
        //   - model-00001-of-00002.safetensors: first 6 tensors (embedding
        //     through b1).
        //   - model-00002-of-00002.safetensors: remaining 4 tensors (w2
        //     through b_out).
        //   - model.safetensors.index.json: maps each tensor name to its
        //     shard file.
        // Mechanism: Calls LoadDecoderModel on the sharded dir, then
        //   validate_checkpoint_shapes to verify all tensors reassembled
        //   correctly from the two shards.
        // =================================================================
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

        // -----------------------------------------------------------------
        // Assertion: Sharded model loads with correct shapes
        //
        // Intention: Confirm that the sharded loading path (triggered by
        //   model.safetensors.index.json) reads from multiple shard files
        //   and produces the same tensor shapes as the single-file case.
        // Mechanism: validate_checkpoint_shapes checks all 10 tensors.
        // -----------------------------------------------------------------
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
