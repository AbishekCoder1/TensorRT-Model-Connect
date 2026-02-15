#pragma once

#include "runtime/trt/trt_graph_builder.h"

namespace trtf {

#if TRTF_HAS_TRT

class StandardDecoderGraphBuilder final : public ITrtGraphBuilder {
public:
    std::unique_ptr<DecoderStepEngine> build_decoder_step_engine(
        const TrtDecoderDefinition& weights, TrtLogger& logger) override;
};

#endif // TRTF_HAS_TRT

} // namespace trtf
