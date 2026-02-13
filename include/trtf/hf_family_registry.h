#pragma once

#include "trtf/model_resolver.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace trtf {

struct HfModelMetadata {
    std::string model_dir;
    std::string model_type;
    std::vector<std::string> architectures;
};

using HfFamilyMatcher = std::function<bool(const HfModelMetadata& metadata)>;
using HfFamilyModelDefinitionLoader = std::function<DecoderModel(const HfModelMetadata& metadata)>;

struct HfModelFamilyRegistration {
    std::string family_name;
    int priority{0};
    HfFamilyMatcher matcher;
    HfFamilyModelDefinitionLoader model_definition_loader;
};

void RegisterHfModelFamily(HfModelFamilyRegistration registration);

// Built-in family set that ships with trtf_core (currently qwen-style, requiring
// a normalized decoder definition at <hf-model-dir>/trtf_decoder/).
void RegisterBuiltinHfModelFamilies();

// Returns kDecoderDefinition when a matching family loader is found.
std::optional<ResolvedModelSpec> ResolveHfModelViaFamilyRegistry(const std::string& model_id);

} // namespace trtf
