// SegmentationPlugin: handles "segmentation" and "prompted_segmentation"
// strategies. SegFormer (single encoder) and SAM (encoder + mask decoder).

#include "runtime/pipelines/sam_pipeline.h"
#include "runtime/pipelines/segment_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

namespace trtf {

class SegmentationPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);

        if (ctx.config.runtime_strategy == "prompted_segmentation") {
            auto decoder = try_load_trt_module_from_plan(
                ctx.backend, find_section(ctx.bundle, "vision_engine_plan"),
                "vision_plan (SAM mask_decoder)", opts);
            if (decoder.module && decoder.module->ok())
                return std::make_unique<SamPipeline>(
                    std::move(loaded.module), std::move(decoder.module), ctx.bundle.info.model_id);
            // Fallback to single-encoder segmentation if decoder failed
            return std::make_unique<SegmentPipeline>(std::move(loaded.module),
                                                     ctx.bundle.info.model_id);
        }

        // strategy == "segmentation"
        return std::make_unique<SegmentPipeline>(std::move(loaded.module),
                                                 ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_SegmentationPlugin = 0;

} // namespace trtf

static trtf::SegmentationPlugin g_SegmentationPlugin_instance;
static trtf::PluginRegistrar g_SegmentationPlugin_reg1("segmentation",
                                                       &g_SegmentationPlugin_instance);
static trtf::PluginRegistrar g_SegmentationPlugin_reg2("prompted_segmentation",
                                                       &g_SegmentationPlugin_instance);
