#pragma once

#include "trt_model_definition.h"

namespace trtf {

// Returns true when model checkpoint contains a Qwen-style layer stack and the definition is populated.
bool PopulateQwen3TrtModelDefinition(TrtDecoderDefinition& definition, const DecoderModel& model);

} // namespace trtf

