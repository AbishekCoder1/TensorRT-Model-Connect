#pragma once

// PipelineFactory: config-driven pipeline assembly.
// Reads config.json from a .trtfb bundle, dispatches on runtime_strategy,
// loads TRT engines, creates tokenizers, and assembles the appropriate pipeline.
//
// This is the single entry point for creating pipelines from bundles.

#include "trtf/pipeline.h"

#include <memory>
#include <string>

namespace trtf {

class PipelineFactory {
public:
    static std::unique_ptr<IPipeline> from_bundle(
        const std::string& bundle_path,
        const std::string& hf_python = "");
};

} // namespace trtf
