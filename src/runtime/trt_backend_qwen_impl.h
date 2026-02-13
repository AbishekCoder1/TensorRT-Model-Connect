#pragma once

#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"

#include <memory>

namespace trtf {

// Qwen-style and legacy decoder implementation currently backed by the TRT step engine.
std::unique_ptr<IGenerationBackend> CreateTrtQwenBackend(const ITokenizer& tokenizer, const DecoderModel& model);

} // namespace trtf
