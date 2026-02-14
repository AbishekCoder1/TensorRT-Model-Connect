#pragma once

#include "model/trt_model_definition_populator.h"

namespace trtf {
namespace qwen {

class QwenTrtModelDefinitionPopulator final : public ITrtModelDefinitionPopulator {
public:
    bool can_populate(const DecoderModel& model) const override;
    bool populate(TrtDecoderDefinition& definition,
                  const DecoderModel& model) const override;
};

} // namespace qwen
} // namespace trtf
