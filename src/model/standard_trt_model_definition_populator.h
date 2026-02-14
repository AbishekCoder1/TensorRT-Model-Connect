#pragma once

#include "model/trt_model_definition_populator.h"

namespace trtf {

// Family-agnostic populator for any model with has_decoder_layers.
// Handles the standard Pre-RMSNorm + GQA + RoPE + SwiGLU layout.
// Registered as a low-priority fallback so family-specific populators take precedence.
class StandardTrtModelDefinitionPopulator final : public ITrtModelDefinitionPopulator {
public:
    bool can_populate(const DecoderModel& model) const override;
    bool populate(TrtDecoderDefinition& definition,
                  const DecoderModel& model) const override;
};

} // namespace trtf
