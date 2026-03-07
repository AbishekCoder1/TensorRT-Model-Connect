#pragma once

#include "trtf/runtime/contracts/contracts.h"
#include "trtf/runtime/ports/bundle_port.h"
#include "trtf/runtime/ports/trt_port.h"

#include <functional>

namespace nvinfer1 {
class IRuntime;
}

namespace trtf::runtime::builders::encoder {

class EncoderStrategyBuilder final : public IStrategyBuilder {
public:
    using ComposeEncoderServicesFn = std::function<BuildResult(
        const BuildContext& context,
        const trtf::FastPathModelConfig& config,
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime)>;

    EncoderStrategyBuilder(
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime,
        ComposeEncoderServicesFn compose_encoder_services = {});

    BuildResult build(const BuildContext& context) override;

private:
    BuildResult compose_encoder_services(
        const BuildContext& context,
        const trtf::FastPathModelConfig& config) const;

    const IBundlePort& mBundlePort;
    const ITrtPort& mTrtPort;
    nvinfer1::IRuntime* mRuntime;
    ComposeEncoderServicesFn mComposeEncoderServices;
};

} // namespace trtf::runtime::builders::encoder
