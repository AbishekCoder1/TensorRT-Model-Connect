#pragma once

#include "trtf/model.h"

#include <memory>
#include <optional>
#include <string>

namespace trtf {

enum class ResolvedModelKind {
    kDecoderDefinition,
    kHuggingFaceLocal,
};

struct ResolvedModelSpec {
    std::string model_id;
    ResolvedModelKind kind{ResolvedModelKind::kDecoderDefinition};

    // Valid when kind == kDecoderDefinition.
    DecoderModel decoder_model;

    // Valid when kind == kHuggingFaceLocal.
    std::string huggingface_model_dir;
};

ResolvedModelSpec ResolveTextGenerationModel(const std::string& model_id);

} // namespace trtf
