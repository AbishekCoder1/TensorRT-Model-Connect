// FluxPlugin: handles "diffusion_flux" strategy.
// FLUX diffusion pipeline with T5 + CLIP text encoders, denoiser, and VAE.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/pipelines/diffusion_pipeline.h"

#if TRTF_HAS_TRT

namespace trtf {

class FluxPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto parts = load_diffusion_parts(ctx.bundle, ctx.config_json, ctx.hf_python);

        // Move text encoder modules into vector
        std::vector<std::unique_ptr<TrtModule>> te_modules;
        for (auto& te : parts.text_encoders)
            te_modules.push_back(std::move(te.module));

        // Try to extract CLIP tokenizer from bundle
        std::unique_ptr<ITokenizer> clip_tok;
        try {
            auto ct = extract_clip_tokenizer_from_bundle(ctx.bundle, ctx.hf_python);
            if (ct.tokenizer)
                clip_tok = std::move(ct.tokenizer);
        } catch (...) {}

        return std::make_unique<FluxPipeline>(
            std::move(te_modules),
            std::move(parts.denoiser.module),
            std::move(parts.vae.module),
            std::move(parts.config),
            std::move(parts.weights),
            std::move(parts.tokenizer),
            std::move(clip_tok),
            ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_FluxPlugin = 0;

} // namespace trtf

static trtf::FluxPlugin g_FluxPlugin_instance;
static trtf::PluginRegistrar g_FluxPlugin_reg("diffusion_flux", &g_FluxPlugin_instance);

#endif // TRTF_HAS_TRT
