#pragma once

#include "trtf/runtime/contracts/contracts.h"
#include "trtf/runtime/ports/bundle_port.h"
#include "trtf/runtime/ports/trt_port.h"

#include <functional>

namespace nvinfer1 {
class IRuntime;
}

namespace trtf::runtime::builders::vision {

class VisionStrategyBuilder final : public IStrategyBuilder {
public:
    using ComposeVisionServiceFn = std::function<BuildResult(
        const BuildContext& context,
        const trtf::FastPathModelConfig& config,
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime)>;

    VisionStrategyBuilder(
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime,
        ComposeVisionServiceFn compose_vision_service = {});

    BuildResult build(const BuildContext& context) override;

private:
    BuildResult compose_vision_service(
        const BuildContext& context,
        const trtf::FastPathModelConfig& config) const;

    const IBundlePort& mBundlePort;
    const ITrtPort& mTrtPort;
    nvinfer1::IRuntime* mRuntime;
    ComposeVisionServiceFn mComposeVisionService;
};

} // namespace trtf::runtime::builders::vision
