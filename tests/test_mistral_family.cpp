// Test: Mistral model family registration and checkpoint loading.

#include "test_helpers.h"
#include "trtf/model_resolver.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

void write_hf_mistral_root(const std::filesystem::path& dir)
{
    trtf_test::write_file(dir / "config.json",
        "{\n"
        "  \"model_type\": \"mistral\",\n"
        "  \"architectures\": [\"MistralForCausalLM\"],\n"
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
    std::filesystem::path family_dir;

    try
    {
        family_dir = trtf_test::make_temp_dir_or_throw("/tmp/trtf_mistral_family_XXXXXX");
        write_hf_mistral_root(family_dir);

        const trtf::ResolvedModelSpec spec = trtf::ResolveTextGenerationModel(family_dir.string());
        if (spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
        {
            std::cerr << "expected mistral hf dir to resolve as decoder-definition" << std::endl;
            return 1;
        }
        if (!spec.decoder_model.has_checkpoint)
        {
            std::cerr << "expected mistral safetensors bridge to load checkpoint" << std::endl;
            return 1;
        }

        const std::string family = spec.decoder_model.architecture.family;
        if (family.substr(0, 7) != "mistral")
        {
            std::cerr << "expected mistral architecture family, got " << family << std::endl;
            return 1;
        }

        if (!spec.decoder_model.checkpoint.has_decoder_layers
            || spec.decoder_model.checkpoint.decoder_layers.size() != 2)
        {
            std::cerr << "expected mistral bridge to load full multi-layer checkpoint" << std::endl;
            return 1;
        }

        if (spec.decoder_model.checkpoint.final_norm.size() != 8)
        {
            std::cerr << "expected mistral bridge to load final_norm tensor" << std::endl;
            return 1;
        }

        if (spec.decoder_model.checkpoint.attention_size != 16)
        {
            std::cerr << "expected mistral bridge to preserve attention width, got "
                      << spec.decoder_model.checkpoint.attention_size << std::endl;
            return 1;
        }

        {
            const auto& layer0 = spec.decoder_model.checkpoint.decoder_layers[0];
            if (!layer0.q_norm.empty())
            {
                std::cerr << "expected mistral q_norm to be empty (no QK norm)" << std::endl;
                return 1;
            }
            if (!layer0.k_norm.empty())
            {
                std::cerr << "expected mistral k_norm to be empty (no QK norm)" << std::endl;
                return 1;
            }
        }

        std::cout << "test_mistral_family passed" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_mistral_family failed: " << e.what() << std::endl;
        std::filesystem::remove_all(family_dir);
        return 1;
    }

    std::filesystem::remove_all(family_dir);
    return 0;
}
