#pragma once

// Factory functions that create audio IPipeline instances from bundle sections.
// Isolated from pipeline.h's AudioResult definition in bark_backend.h.

#include "trtf/pipeline.h"
#include "cabi/bundle/bundle_helpers.h"
#include "cabi/config/fast_path_config.h"

#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

std::unique_ptr<IPipeline> make_whisper_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id);

std::unique_ptr<IPipeline> make_bark_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id);

std::unique_ptr<IPipeline> make_magpie_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id);

std::unique_ptr<IPipeline> make_speech_pipeline_from_bundle(
    const BundleSections& sections, const FastPathModelConfig& cfg,
    const std::string& hf_python, const std::string& model_id);

#endif // TRTF_HAS_TRT

} // namespace trtf
