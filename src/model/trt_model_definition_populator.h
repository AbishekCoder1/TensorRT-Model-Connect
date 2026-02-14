#pragma once

#include "trt_model_definition.h"
#include "trtf/model.h"

#include <memory>
#include <string>

namespace trtf {

class ITrtModelDefinitionPopulator {
public:
    virtual ~ITrtModelDefinitionPopulator() = default;
    virtual bool can_populate(const DecoderModel& model) const = 0;
    virtual bool populate(TrtDecoderDefinition& definition,
                          const DecoderModel& model) const = 0;
};

void RegisterTrtModelDefinitionPopulator(const std::string& family, int priority,
    std::unique_ptr<ITrtModelDefinitionPopulator> populator);
bool PopulateViaRegistry(TrtDecoderDefinition& definition, const DecoderModel& model);

} // namespace trtf
