#pragma once

#include "trtf/backend.h"
#include "trtf/model_resolver.h"
#include "trtf/tokenizer.h"

#include <memory>
#include <string>

namespace trtf {

struct BackendSelection {
    bool prefer_trt{true};
    bool force_trt{false};
    std::string hf_python;  // empty = auto-detect
};

struct RuntimeAssembly {
    std::unique_ptr<ITokenizer> tokenizer;
    std::unique_ptr<IGenerationBackend> backend;
    std::string backend_name;
};

RuntimeAssembly BuildRuntimeForTextGeneration(const ResolvedModelSpec& model_spec, const BackendSelection& selection);

} // namespace trtf
