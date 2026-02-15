// Test: HF family registry with priority ordering and metadata parsing.
// Verifies: Mock family registration, priority-based matching, metadata extraction
// from config.json, and end-to-end pipeline with family-defined decoder model.

#include "trtf/hf_family_registry.h"
#include "trtf/model_resolver.h"
#include "trtf/pipeline_legacy.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <stdlib.h>

namespace {

trtf::DecoderModel make_mock_model(const std::string& model_id, const std::string& default_next)
{
    trtf::DecoderModel model;
    model.model_id = model_id;
    model.vocab = {"<unk>", "<bos>", "<eos>", "hello", "from", "mock", "."};
    model.transitions = {
        {"hello", "from"},
        {"from", "mock"},
        {"mock", "."},
        {".", "<eos>"},
    };
    model.default_next_token = default_next;
    model.max_cache_length = 16;
    return model;
}

} // namespace

int main()
{
    char temp_dir_template[] = "/tmp/trtf_hf_family_registry_XXXXXX";
    char* created_dir = mkdtemp(temp_dir_template);
    if (created_dir == nullptr)
    {
        std::cerr << "mkdtemp failed: " << std::strerror(errno) << std::endl;
        return 1;
    }

    const std::filesystem::path hf_dir(created_dir);
    {
        std::ofstream config(hf_dir / "config.json");
        config << "{\n"
               << "  \"model_type\": \"mock-family\",\n"
               << "  \"architectures\": [\"MockForCausalLM\"]\n"
               << "}\n";
    }
    {
        std::ofstream safetensors(hf_dir / "model.safetensors");
        safetensors << "";
    }

    bool saw_expected_metadata = false;

    trtf::RegisterHfModelFamily({
        "mock-low-priority",
        10,
        [](const trtf::HfModelMetadata& metadata) { return metadata.model_type == "mock-family"; },
        [](const trtf::HfModelMetadata& metadata) { return make_mock_model(metadata.model_dir, "<unk>"); },
    });

    trtf::RegisterHfModelFamily({
        "mock-high-priority",
        100,
        [&saw_expected_metadata](const trtf::HfModelMetadata& metadata) {
            if (metadata.model_type != "mock-family")
            {
                return false;
            }
            if (metadata.architectures.size() == 1 && metadata.architectures[0] == "MockForCausalLM")
            {
                saw_expected_metadata = true;
            }
            return true;
        },
        [](const trtf::HfModelMetadata& metadata) { return make_mock_model(metadata.model_dir, "<eos>"); },
    });

    const trtf::ResolvedModelSpec spec = trtf::ResolveTextGenerationModel(hf_dir.string());
    if (spec.kind != trtf::ResolvedModelKind::kDecoderDefinition)
    {
        std::cerr << "expected hf family registry to resolve as decoder-definition" << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }
    if (!saw_expected_metadata)
    {
        std::cerr << "expected matcher to observe parsed hf metadata" << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }
    if (spec.decoder_model.default_next_token != "<eos>")
    {
        std::cerr << "expected higher-priority family registration to win" << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }

    auto pipeline = trtf::Pipeline::CreateTextGeneration(hf_dir.string(), false, false);
    if (pipeline.backend_name() != "cpu-reference")
    {
        std::cerr << "expected cpu-reference backend for mock family, got " << pipeline.backend_name() << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }

    const auto out = pipeline("hello");
    if (out.size() != 1 || out[0].generated_text.find("hello from mock.") == std::string::npos)
    {
        std::cerr << "unexpected output for family-defined decoder model: "
                  << (out.empty() ? std::string("<empty>") : out[0].generated_text) << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }

    std::filesystem::remove_all(hf_dir);
    std::cout << "test_hf_family_registry passed" << std::endl;
    return 0;
}
