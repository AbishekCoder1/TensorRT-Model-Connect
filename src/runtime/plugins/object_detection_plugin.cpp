// ObjectDetectionPlugin: handles "object_detection" strategy.
// Uses EncoderPipeline in "object_detection" mode (no tokenizer needed).

#include "runtime/pipelines/encoder_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class ObjectDetectionPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);

        return std::make_unique<EncoderPipeline>(std::move(loaded.module), "object_detection",
                                                 nullptr, ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_ObjectDetectionPlugin = 0;

} // namespace trtf

static trtf::ObjectDetectionPlugin g_ObjectDetectionPlugin_instance;
static trtf::PluginRegistrar g_ObjectDetectionPlugin_reg("object_detection",
                                                         &g_ObjectDetectionPlugin_instance);

#endif // TRTF_HAS_TRT
