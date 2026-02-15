// =============================================================================
// Test suite: HF family registry — priority ordering and metadata parsing
// =============================================================================
//
// Purpose:
//   Validates the HuggingFace model family registry (Registry 1), which is
//   responsible for matching HF model directories (containing config.json +
//   safetensors) to registered model families. This test ensures that:
//     1. Multiple families can register for the same model_type.
//     2. The highest-priority registration wins when multiple matchers match.
//     3. The matcher callback receives correctly parsed HfModelMetadata
//        (model_type, architectures) extracted from config.json.
//     4. The winning family's loader produces the DecoderModel that ends up
//        in the resolved model spec.
//
// Dependencies:
//   - trtf/hf_family_registry.h (RegisterHfModelFamily, HfModelMetadata)
//   - trtf/model_resolver.h (ResolveTextGenerationModel, ResolvedModelSpec)
//   - trtf/pipeline.h (DecoderModel)
//   - Filesystem: creates a temp directory with config.json + model.safetensors
//
// Approach:
//   Creates a temporary HF model directory on disk with a minimal config.json.
//   Registers two mock families with different priorities (10 and 100) that
//   both match "mock-family" model_type. Resolves the model and verifies the
//   higher-priority family won by checking the default_next_token field
//   (set differently by each family's loader). Also verifies that the
//   matcher received the expected metadata via a side-effect flag.
// =============================================================================

#include "trtf/hf_family_registry.h"
#include "trtf/model_resolver.h"
#include "trtf/pipeline.h"

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

// -----------------------------------------------------------------------------
// Helper: Create a minimal DecoderModel with a small vocab and transitions.
//
// The default_next_token parameter is used as a discriminator to identify
// which family registration produced this model. The low-priority family
// sets it to "<unk>", while the high-priority family sets it to "<eos>".
// -----------------------------------------------------------------------------
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
    // -------------------------------------------------------------------------
    // Setup: Create a temporary HF model directory with config.json
    //
    // The directory mimics a HuggingFace model checkout with:
    //   - config.json containing model_type="mock-family" and
    //     architectures=["MockForCausalLM"]
    //   - An empty model.safetensors file (required for the resolver to
    //     recognize this as an HF model directory)
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Registration 1: Low-priority mock family (priority=10)
    //
    // Intention:
    //   Register a family that matches "mock-family" but at low priority.
    //   Its loader sets default_next_token to "<unk>". If this family wins
    //   (incorrectly), the priority assertion below will catch it.
    //
    // Mechanism:
    //   Uses RegisterHfModelFamily with priority=10 and a simple matcher
    //   that checks model_type == "mock-family".
    // -------------------------------------------------------------------------
    trtf::RegisterHfModelFamily({
        "mock-low-priority",
        10,
        [](const trtf::HfModelMetadata& metadata) { return metadata.model_type == "mock-family"; },
        [](const trtf::HfModelMetadata& metadata) { return make_mock_model(metadata.model_dir, "<unk>"); },
    });

    // -------------------------------------------------------------------------
    // Registration 2: High-priority mock family (priority=100)
    //
    // Intention:
    //   Register a second family for the same model_type at higher priority.
    //   Its loader sets default_next_token to "<eos>". This family should
    //   win during resolution.
    //
    // Mechanism:
    //   Uses RegisterHfModelFamily with priority=100. The matcher also
    //   validates that HfModelMetadata contains the correct architectures
    //   array (["MockForCausalLM"]) and sets the saw_expected_metadata
    //   flag as a side effect to confirm metadata was parsed correctly.
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Test: Resolve the mock HF directory and verify outcomes
    //
    // Intention:
    //   Verify three properties of the resolution pipeline:
    //     1. The HF directory resolves as kDecoderDefinition (meaning a
    //        registered family matched and produced a DecoderModel).
    //     2. The high-priority matcher observed correct metadata from
    //        config.json (architectures array parsed correctly).
    //     3. The high-priority family's loader won (default_next_token is
    //        "<eos>", not "<unk>" from the low-priority family).
    //
    // Setup:
    //   Both mock families are already registered above.
    //
    // Mechanism:
    //   Calls ResolveTextGenerationModel with the temp directory path.
    //   Checks spec.kind, the saw_expected_metadata flag, and
    //   spec.decoder_model.default_next_token.
    // -------------------------------------------------------------------------
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

    std::filesystem::remove_all(hf_dir);
    std::cout << "test_hf_family_registry passed" << std::endl;
    return 0;
}
