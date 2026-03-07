#pragma once

#include "trtf/runtime/contracts/contracts.h"
#include "trtf/runtime/ports/bundle_port.h"
#include "trtf/runtime/ports/trt_port.h"

#include <functional>

namespace nvinfer1 {
class IRuntime;
}

namespace trtf::runtime::builders::text {

class TextStrategyBuilder final : public IStrategyBuilder {
public:
    using ComposeTextServiceFn = std::function<BuildResult(
        const BuildContext& context,
        const trtf::FastPathModelConfig& config,
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime)>;

    TextStrategyBuilder(
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime,
        ComposeTextServiceFn compose_text_service = {});

    BuildResult build(const BuildContext& context) override;

private:
    BuildResult compose_text_service(
        const BuildContext& context,
        const trtf::FastPathModelConfig& config) const;

    const IBundlePort& mBundlePort;
    const ITrtPort& mTrtPort;
    nvinfer1::IRuntime* mRuntime;
    ComposeTextServiceFn mComposeTextService;
};

} // namespace trtf::runtime::builders::text
