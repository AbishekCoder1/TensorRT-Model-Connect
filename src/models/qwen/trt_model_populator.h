#pragma once

#include "model/standard_trt_model_definition_populator.h"

namespace trtf {
namespace qwen {

// Qwen-specific populator — currently identical to StandardTrtModelDefinitionPopulator.
// Kept as a subclass in case Qwen-specific overrides are needed in the future.
using QwenTrtModelDefinitionPopulator = StandardTrtModelDefinitionPopulator;

} // namespace qwen
} // namespace trtf
