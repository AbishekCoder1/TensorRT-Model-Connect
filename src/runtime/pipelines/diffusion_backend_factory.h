#pragma once

// Factory function that creates a diffusion IPipeline from bundle sections.
// Isolated from pipeline.h's AudioResult definition in bark_backend.h.

#include "trtf/pipeline.h"
#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"

#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

std::unique_ptr<IPipeline> make_diffusion_pipeline_from_bundle(
    const BundleSections& sections,
    const FastPathModelConfig& cfg,
    const std::string& bundle_path,
    const std::string& hf_python,
    const std::string& model_id);

#endif // TRTF_HAS_TRT

} // namespace trtf
