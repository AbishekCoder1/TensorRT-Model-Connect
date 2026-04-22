// PatchTSMixerPlugin: handles "patchtsmixer_torchtrt" strategy.
// Loads a single TRT engine and dispatches to PatchTSMixerPipeline.

#include "runtime/pipelines/patchtsmixer_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class PatchTSMixerPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto cfg = parse_patchtsmixer_config(ctx.config_json, ctx.config.max_cache_length);
        return std::make_unique<PatchTSMixerPipeline>(std::move(loaded.module), std::move(cfg),
                                                      ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_PatchTSMixerPlugin = 0;

} // namespace trtf

static trtf::PatchTSMixerPlugin g_PatchTSMixerPlugin_instance;
static trtf::PluginRegistrar g_PatchTSMixerPlugin_reg("patchtsmixer_torchtrt",
                                                      &g_PatchTSMixerPlugin_instance);

#endif // TRTF_HAS_TRT
