#pragma once

#include "trtf/model.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace trtf {

enum class ResolvedModelKind {
    kDecoderDefinition,
    kHuggingFaceLocal,
    kCustom,
};

struct ResolvedModelSpec {
    std::string model_id;
    ResolvedModelKind kind{ResolvedModelKind::kDecoderDefinition};

    // Valid when kind == kDecoderDefinition.
    DecoderModel decoder_model;

    // Valid when kind == kHuggingFaceLocal.
    std::string huggingface_model_dir;

    // Valid when kind == kCustom.
    std::string custom_type;
    std::string custom_model_dir;
    std::shared_ptr<const void> custom_payload;
};

using TextGenerationModelResolver
    = std::function<std::optional<ResolvedModelSpec>(const std::string& model_id)>;

void RegisterTextGenerationModelResolver(TextGenerationModelResolver resolver);

ResolvedModelSpec ResolveTextGenerationModel(const std::string& model_id);

} // namespace trtf
