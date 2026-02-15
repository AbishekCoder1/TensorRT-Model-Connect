// Test: Qwen model family registration and checkpoint loading.
// Verifies: HF family detection for model_type "qwen2"/"qwen3", multi-layer safetensors
// bridge with q_norm/k_norm, decoder-definition precedence, and built-in QWEN3 alias.
// Approach: Creates synthetic 2-layer Qwen checkpoint in a temp dir, resolves it
// through the family registry, and validates checkpoint structure.

#include "test_helpers.h"
#include "trtf/model_resolver.h"
#include "trtf/pipeline_legacy.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

void write_hf_qwen_root(const std::filesystem::path& dir)
{
    trtf_test::write_file(dir / "config.json",
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
    trtf_test::write_standard_decoder_checkpoint(dir, 8, 8, 16, 8, 16, 2, true);
}

void write_decoder_definition(const std::filesystem::path& dir)
{
    const std::filesystem::path decoder_dir = dir / "trtf_decoder";
    std::filesystem::create_directories(decoder_dir);

    trtf_test::write_file(decoder_dir / "config.json",
        "{\n"
        "  \"default_next_token\": \"from\",\n"
        "  \"max_cache_length\": 16\n"
        "}\n");

    trtf_test::write_file(decoder_dir / "vocab.txt",
        "<unk>\n"
        "<bos>\n"
        "<eos>\n"
        "<pad>\n"
        "hello\n"
        "from\n"
        "qwen\n"
        ".\n");

    trtf_test::write_file(decoder_dir / "transitions.txt",
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
        with_definition = trtf_test::make_temp_dir_or_throw("/tmp/trtf_qwen_family_with_def_XXXXXX");
        without_definition = trtf_test::make_temp_dir_or_throw("/tmp/trtf_qwen_family_without_def_XXXXXX");

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
        {
            const std::string family = fallback_spec.decoder_model.architecture.family;
            if (family.substr(0, 4) != "qwen" && family.substr(0, 3) != "qwq")
            {
                std::cerr << "expected qwen root path to set a qwen/qwq architecture family, got "
                          << family << std::endl;
                return 1;
            }
        }
        if (!fallback_spec.decoder_model.checkpoint.has_decoder_layers
            || fallback_spec.decoder_model.checkpoint.decoder_layers.size() != 2)
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

        const bool builtin_alias_is_large_checkpoint
            = builtin_alias_spec.decoder_model.checkpoint.embedding.size() > 5000000ULL;
        if (!builtin_alias_is_large_checkpoint)
        {
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
