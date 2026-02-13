#pragma once

#include "trtf/model.h"

#include <string>

namespace trtf {

// Qwen-family owned loader entrypoint. This is the seam where family-specific
// config/checkpoint loading can evolve without touching shared loader dispatch.
DecoderModel LoadQwen3DecoderModel(const std::string& model_dir);

} // namespace trtf
