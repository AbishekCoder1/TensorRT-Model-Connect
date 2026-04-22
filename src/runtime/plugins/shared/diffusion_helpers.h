#pragma once
#include "runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h"
#include "runtime/domains/diffusion/diffusion_types.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "utils/json_helpers.h"

#if TRTF_HAS_TRT

namespace trtf {

DiffusionConfig make_diffusion_config(const std::string& json);

// Shared diffusion resources loaded once, then dispatched to per-model factory.
struct DiffusionParts {
    LoadedModule denoiser;
    LoadedModule vae;
    std::vector<LoadedModule> text_encoders;
    DiffusionConfig config;
    PreprocessorWeights weights;
    std::shared_ptr<ITokenizer> tokenizer;
};

DiffusionParts load_diffusion_parts(IBackend* backend, const BundleFile& bundle,
                                    const std::string& json,
                                    const ModuleCreateOptions& options = {});

} // namespace trtf

#endif // TRTF_HAS_TRT
