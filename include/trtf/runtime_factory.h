#pragma once

#include "trtf/backend.h"
#include "trtf/model_resolver.h"
#include "trtf/tokenizer.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace trtf {

struct BackendSelection {
    bool prefer_trt{true};
    bool force_trt{false};
};

struct RuntimeAssembly {
    std::unique_ptr<ITokenizer> tokenizer;
    std::unique_ptr<IGenerationBackend> backend;
    std::string backend_name;
};

using TextGenerationRuntimeAssembler
    = std::function<std::optional<RuntimeAssembly>(const ResolvedModelSpec& model_spec, const BackendSelection& selection)>;

void RegisterTextGenerationRuntimeAssembler(TextGenerationRuntimeAssembler assembler);

RuntimeAssembly BuildRuntimeForTextGeneration(const ResolvedModelSpec& model_spec, const BackendSelection& selection);

} // namespace trtf
