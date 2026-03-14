// ObjectDetectionPlugin: handles "object_detection" strategy.
// Uses EncoderPipeline in "object_detection" mode (no tokenizer needed).

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/pipelines/encoder_pipeline.h"

#if TRTF_HAS_TRT

namespace trtf {

class ObjectDetectionPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");

        return std::make_unique<EncoderPipeline>(
            std::move(loaded.module), "object_detection",
            nullptr, ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_ObjectDetectionPlugin = 0;

} // namespace trtf

static trtf::ObjectDetectionPlugin g_ObjectDetectionPlugin_instance;
static trtf::PluginRegistrar g_ObjectDetectionPlugin_reg("object_detection", &g_ObjectDetectionPlugin_instance);

#endif // TRTF_HAS_TRT
