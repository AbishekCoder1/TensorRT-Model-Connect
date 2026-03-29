// DecoderPlugin: handles "decoder_kv_cache" and "decoder_moe" strategies.
// Standard attention-based decoder with device-resident KV cache.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/pipelines/text_generation_pipeline.h"

#if TRTF_HAS_TRT

namespace trtf {

class DecoderPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.stream->get();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        std::unique_ptr<IInferenceState> state = std::make_unique<KvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream);
        if (!state->ok())
            throw std::runtime_error("Failed to create KvCache");

        TextGenConfig tgc;
        tgc.vocab_size = ctx.config.vocab_size;
        tgc.id_bos = ctx.config.id_bos;
        tgc.id_eos = ctx.config.id_eos;
        tgc.has_position_input = loaded.module->has_input("position_id");

        return std::make_unique<TextGenerationPipeline>(
            std::move(loaded.module), std::move(state),
            tgc, stream, std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_DecoderPlugin = 0;

} // namespace trtf

static trtf::DecoderPlugin g_DecoderPlugin_instance;
static trtf::PluginRegistrar g_DecoderPlugin_reg1("decoder_kv_cache", &g_DecoderPlugin_instance);
static trtf::PluginRegistrar g_DecoderPlugin_reg2("decoder_moe", &g_DecoderPlugin_instance);

#endif // TRTF_HAS_TRT
