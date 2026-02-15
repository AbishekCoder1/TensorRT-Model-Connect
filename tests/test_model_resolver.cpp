// =============================================================================
// Test: Model resolution pipeline with multiple resolution strategies.
//
// Purpose:
//   Validates the model resolver (ResolveTextGenerationModel) which is the first
//   stage of the pipeline flow. The resolver must correctly classify model_id
//   strings into resolution kinds: kHuggingFaceLocal for valid HF directories,
//   kDecoderDefinition for registered families, and throw for unresolvable IDs.
//
// What is verified:
//   - HF directory detection: A directory containing config.json + model.safetensors
//     (but no registered family match) resolves as kHuggingFaceLocal with the
//     correct huggingface_model_dir path.
//   - Unknown model error: A completely invalid model_id string throws a
//     runtime_error with "Unknown model_id" in the message.
//
// Dependencies:
//   - trtf/model_resolver.h: ResolveTextGenerationModel, ResolvedModelSpec,
//     ResolvedModelKind
//   - Standard library: filesystem, fstream (for temp dir and file creation)
//
// Approach:
//   1. Creates a temporary directory with an empty config.json and an empty
//      model.safetensors file (minimal HF directory structure). Resolves it
//      and verifies kHuggingFaceLocal kind and correct path.
//   2. Attempts to resolve a nonsense model_id string and verifies that a
//      runtime_error with "Unknown model_id" is thrown.
//
// Note: This test creates its temp dir directly via mkdtemp rather than using
//   test_helpers.h, and writes files via std::ofstream. This is an older test
//   that predates the shared test helpers.
// =============================================================================

#include "trtf/model_resolver.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stdlib.h>

int main()
{
    // =====================================================================
    // Section 1: HF directory resolves as kHuggingFaceLocal
    //
    // Intention: Verify that a directory containing config.json +
    //   model.safetensors (with no matching registered model family) is
    //   resolved as a HuggingFace local directory, which will later be
    //   handled by the HF-Python backend.
    // Setup: Creates a temp dir with:
    //   - config.json: empty JSON object "{}" (no model_type, so no family
    //     registry match occurs).
    //   - model.safetensors: empty file (presence is enough to trigger
    //     HF directory detection).
    // Mechanism: Calls ResolveTextGenerationModel and checks that kind is
    //   kHuggingFaceLocal and huggingface_model_dir matches the temp path.
    // =====================================================================
    char temp_dir_template[] = "/tmp/trtf_hf_resolver_XXXXXX";
    char* created_dir = mkdtemp(temp_dir_template);
    if (created_dir == nullptr)
    {
        std::cerr << "mkdtemp failed: " << std::strerror(errno) << std::endl;
        return 1;
    }

    const std::filesystem::path hf_dir(created_dir);
    {
        std::ofstream config(hf_dir / "config.json");
        config << "{}";
    }
    {
        std::ofstream safetensors(hf_dir / "model.safetensors");
        safetensors << "";
    }

    // -----------------------------------------------------------------
    // Assertion: HF directory resolves with correct kind and path
    //
    // Intention: The resolver should detect the config.json + safetensors
    //   combination and classify this as a local HuggingFace model dir.
    // Mechanism: Checks spec.kind == kHuggingFaceLocal and
    //   spec.huggingface_model_dir == hf_dir path string.
    // -----------------------------------------------------------------
    const trtf::ResolvedModelSpec hf = trtf::ResolveTextGenerationModel(hf_dir.string());
    if (hf.kind != trtf::ResolvedModelKind::kHuggingFaceLocal)
    {
        std::cerr << "expected hf directory to resolve as huggingface-local" << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }
    if (hf.huggingface_model_dir != hf_dir.string())
    {
        std::cerr << "unexpected huggingface_model_dir in resolved model spec" << std::endl;
        std::filesystem::remove_all(hf_dir);
        return 1;
    }

    std::filesystem::remove_all(hf_dir);

    // =====================================================================
    // Section 2: Unknown model_id throws runtime_error
    //
    // Intention: Verify that the resolver rejects completely invalid model_id
    //   strings with a descriptive error. This ensures that typos or
    //   misconfigured model paths are caught early rather than silently
    //   producing incorrect behavior.
    // Setup: Uses the string "trtf/definitely-not-a-real-model" which does
    //   not correspond to any file path, built-in alias, or registered model.
    // Mechanism: Wraps the resolve call in a try/catch and checks that a
    //   runtime_error is thrown whose what() message contains "Unknown model_id".
    // =====================================================================
    bool saw_unknown_model = false;
    try
    {
        (void) trtf::ResolveTextGenerationModel("trtf/definitely-not-a-real-model");
    }
    catch (const std::runtime_error& e)
    {
        if (std::string(e.what()).find("Unknown model_id") != std::string::npos)
        {
            saw_unknown_model = true;
        }
    }

    // -----------------------------------------------------------------
    // Assertion: Unknown model_id was rejected with correct error
    //
    // Intention: Confirms the catch block above was triggered and the error
    //   message contained the expected "Unknown model_id" substring.
    // Mechanism: Checks the saw_unknown_model flag set in the catch block.
    // -----------------------------------------------------------------
    if (!saw_unknown_model)
    {
        std::cerr << "expected unknown model_id error for unresolvable model id" << std::endl;
        return 1;
    }

    std::cout << "test_model_resolver passed" << std::endl;
    return 0;
}
