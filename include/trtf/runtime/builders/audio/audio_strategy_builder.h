#pragma once

#include "trtf/runtime/contracts/contracts.h"
#include "trtf/runtime/ports/bundle_port.h"
#include "trtf/runtime/ports/trt_port.h"

namespace nvinfer1 {
class IRuntime;
}

namespace trtf::runtime::builders::audio {

class AudioStrategyBuilder final : public IStrategyBuilder {
public:
    AudioStrategyBuilder(
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime);

    BuildResult build(const BuildContext& context) override;

private:
    const IBundlePort& mBundlePort;
    const ITrtPort& mTrtPort;
    nvinfer1::IRuntime* mRuntime;
};

} // namespace trtf::runtime::builders::audio
