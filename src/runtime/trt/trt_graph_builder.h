#pragma once

#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "model/trt_model_definition.h"

#include <memory>
#include <string>

namespace trtf {

#if TRTF_HAS_TRT

class ITrtGraphBuilder {
public:
    virtual ~ITrtGraphBuilder() = default;
    virtual std::unique_ptr<DecoderStepEngine> build_decoder_step_engine(
        const TrtDecoderDefinition& weights, TrtLogger& logger) = 0;
};

void RegisterTrtGraphBuilder(const std::string& family, std::unique_ptr<ITrtGraphBuilder> builder);
ITrtGraphBuilder* FindTrtGraphBuilder(const std::string& family);

#endif // TRTF_HAS_TRT

} // namespace trtf
