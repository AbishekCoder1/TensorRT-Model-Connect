// PixArtPlugin: handles "diffusion_pixart" strategy.
// PixArt-Sigma/Alpha via TRT Network API. Same engine format as Wan
// (preprocessor_weights, T5 text encoder, DiT denoiser, VAE decoder).
// Standalone plugin — no delegation to WanPlugin.

#include "runtime/pipelines/pixart_pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class PixArtPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto parts = load_diffusion_parts(ctx.bundle, ctx.config_json);

        // Extract first text encoder
        std::unique_ptr<TrtModule> te_module;
        if (!parts.text_encoders.empty())
            te_module = std::move(parts.text_encoders[0].module);

        return std::make_unique<PixArtPipeline>(
            std::move(te_module), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(parts.tokenizer),
            ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_PixArtPlugin = 0;

} // namespace trtf

static trtf::PixArtPlugin g_PixArtPlugin_instance;
static trtf::PluginRegistrar g_PixArtPlugin_reg("diffusion_pixart", &g_PixArtPlugin_instance);

#endif // TRTF_HAS_TRT
