// DecoderPlugin: handles "decoder_kv_cache", "decoder_moe", and "torchtrt_decoder" strategies.
// Standard attention-based decoder with device-resident KV cache.

#include "runtime/pipelines/text_generation_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded =
            load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        bool is_torchtrt = (ctx.config.runtime_strategy == "torchtrt_decoder");
        auto naming =
            is_torchtrt ? KvCache::NamingScheme::kTorchTrt : KvCache::NamingScheme::kStandard;

        cudaStream_t stream = loaded.stream->get();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<IInferenceState> state =
            std::make_unique<KvCache>(ctx.config.num_layers, ctx.config.max_cache_length, kv_dim,
                                      stream, cache_dtype, naming);
        if (!state->ok())
            throw std::runtime_error("Failed to create KvCache");

        TextGenConfig tgc;
        tgc.vocab_size = ctx.config.vocab_size;
        tgc.id_bos = ctx.config.id_bos;
        tgc.id_eos = ctx.config.id_eos;
        tgc.has_position_input = loaded.module->has_input("position_id");
        if (is_torchtrt)
            tgc.logits_output_name = "output0";

        return std::make_unique<TextGenerationPipeline>(std::move(loaded.module), std::move(state),
                                                        tgc, stream, std::move(tokenizer),
                                                        ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_DecoderPlugin = 0;

} // namespace trtf

static trtf::DecoderPlugin g_DecoderPlugin_instance;
static trtf::PluginRegistrar g_DecoderPlugin_reg1("decoder_kv_cache", &g_DecoderPlugin_instance);
static trtf::PluginRegistrar g_DecoderPlugin_reg2("decoder_moe", &g_DecoderPlugin_instance);
static trtf::PluginRegistrar g_DecoderPlugin_reg3("torchtrt_decoder", &g_DecoderPlugin_instance);

#endif // TRTF_HAS_TRT
