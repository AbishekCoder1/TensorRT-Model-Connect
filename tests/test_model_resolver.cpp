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
    const trtf::ResolvedModelSpec builtin = trtf::ResolveTextGenerationModel("trtf/tiny-cake-v1");
    if (builtin.kind != trtf::ResolvedModelKind::kDecoderDefinition)
    {
        std::cerr << "expected built-in model to resolve as decoder-definition" << std::endl;
        return 1;
    }
    if (builtin.decoder_model.vocab.empty())
    {
        std::cerr << "expected built-in decoder model vocab to be non-empty" << std::endl;
        return 1;
    }

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

    if (!saw_unknown_model)
    {
        std::cerr << "expected unknown model_id error for unresolvable model id" << std::endl;
        return 1;
    }

    std::cout << "test_model_resolver passed" << std::endl;
    return 0;
}
