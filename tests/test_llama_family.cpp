// Test: LLaMA model family registration and checkpoint loading.
// Verifies: HF family detection for model_type "llama", multi-layer safetensors
// bridge WITHOUT q_norm/k_norm, GQA layout with num_kv_heads < num_attention_heads,
// and empty q_norm/k_norm vectors.
// Approach: Creates synthetic 2-layer LLaMA checkpoint in a temp dir, resolves it
// through the family registry, and validates checkpoint structure.

#include "test_helpers.h"
#include "trtf/model_resolver.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

void write_hf_llama_root(const std::filesystem::path& dir)
{
    trtf_test::write_file(dir / "config.json",
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
    trtf_test::write_standard_decoder_checkpoint(dir, 8, 8, 16, 8, 16, 2, false);
}

} // namespace

int main()
{
    std::filesystem::path llama_dir;

    try
    {
        llama_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_llama_family_XXXXXX");

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
