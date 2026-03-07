#pragma once

#include "trtf/runtime/contracts/contracts.h"
#include "trtf/runtime/ports/bundle_port.h"
#include "trtf/runtime/ports/trt_port.h"

namespace nvinfer1 {
class IRuntime;
}

namespace trtf::runtime::builders::diffusion {

class DiffusionStrategyBuilder final : public IStrategyBuilder {
public:
    using VideoServiceFactory = std::unique_ptr<IVideoService>(*)(
        const BuildContext& context, nvinfer1::IRuntime* runtime);

    DiffusionStrategyBuilder(
        const IBundlePort& bundle_port,
        const ITrtPort& trt_port,
        nvinfer1::IRuntime* runtime);

    BuildResult build(const BuildContext& context) override;

    static void set_video_service_factory_for_tests(VideoServiceFactory factory);

private:
    const IBundlePort& mBundlePort;
    const ITrtPort& mTrtPort;
    nvinfer1::IRuntime* mRuntime;
};

} // namespace trtf::runtime::builders::diffusion
