// TorchTrtDiffusionPlugin: handles "torchtrt_diffusion" strategy.
// Uses TorchTrtDiffusionPipeline for torch-trt compiled diffusion models.
// Unlike WanPlugin, no preprocessor_weights are needed since the torch-trt
// engines include all preprocessing internally.

#include "runtime/pipelines/torchtrt_diffusion_pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class TorchTrtDiffusionPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto shared_stream = std::make_shared<CudaStream>();

        // Load the three component engines
        auto te = load_trt_module_from_plan(find_section(ctx.bundle, "text_encoder_0_plan"),
                                            "text_encoder_0_plan", shared_stream);
        auto denoiser = load_trt_module_from_plan(find_section(ctx.bundle, "denoiser_plan"),
                                                  "denoiser_plan", shared_stream);
        auto vae = load_trt_module_from_plan(find_section(ctx.bundle, "vae_decoder_plan"),
                                             "vae_decoder_plan", shared_stream);

        auto config = make_diffusion_config(ctx.config_json);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<TorchTrtDiffusionPipeline>(
            std::move(te.module), std::move(denoiser.module), std::move(vae.module),
            std::move(config), std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_TorchTrtDiffusionPlugin = 0;

} // namespace trtf

static trtf::TorchTrtDiffusionPlugin g_TorchTrtDiffusionPlugin_instance;
static trtf::PluginRegistrar g_TorchTrtDiffusionPlugin_reg("torchtrt_diffusion",
                                                           &g_TorchTrtDiffusionPlugin_instance);

#endif // TRTF_HAS_TRT
