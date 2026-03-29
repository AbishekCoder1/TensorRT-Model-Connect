// MagpiePlugin: handles "text_to_audio_magpie" strategy.
// Magpie TTS encoder-decoder pipeline with IPA tokenizer and optional CFG.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/plugins/shared/audio_helpers.h"
#include "runtime/pipelines/magpie_pipeline.h"

#if TRTF_HAS_TRT

namespace trtf {

class MagpiePlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto shared_stream = std::make_shared<CudaStream>();

        auto enc_loaded = load_trt_module_from_plan(
            find_section(ctx.bundle, "vision_engine_plan"), "magpie encoder", shared_stream);
        auto dec_loaded = load_trt_module_from_plan(
            find_section(ctx.bundle, "engine_plan"), "magpie decoder", shared_stream);

        cudaStream_t stream = shared_stream->get();

        auto magpie_cfg = build_magpie_config(ctx.config_json, ctx.config);
        int32_t kv_dim = (ctx.config.attention_size > 0) ? ctx.config.attention_size
            : ((ctx.config.num_heads > 0 && ctx.config.head_dim > 0) ? ctx.config.num_heads * ctx.config.head_dim
               : magpie_cfg.hidden_size);

        std::unique_ptr<IInferenceState> decoder_state = std::make_unique<KvCache>(
            magpie_cfg.decoder_layers, ctx.config.max_cache_length, kv_dim, stream);
        if (!decoder_state->ok())
            throw std::runtime_error("MagpiePipeline: failed to create decoder KvCache");

        std::unique_ptr<IInferenceState> decoder_state_uncond;
        if (magpie_cfg.cfg_scale > 1.0F)
        {
            decoder_state_uncond = std::make_unique<KvCache>(
                magpie_cfg.decoder_layers, ctx.config.max_cache_length, kv_dim, stream);
        }

        const std::size_t enc_buf_size = static_cast<std::size_t>(magpie_cfg.max_source_positions) *
            static_cast<std::size_t>(magpie_cfg.hidden_size) * sizeof(float);

        std::vector<CudaBuffer> cross_k, cross_v;
        allocate_cross_kv_buffers(magpie_cfg.decoder_layers, enc_buf_size, cross_k, cross_v);

        std::vector<CudaBuffer> cross_k_uncond, cross_v_uncond;
        if (magpie_cfg.cfg_scale > 1.0F)
            allocate_cross_kv_buffers(magpie_cfg.decoder_layers, enc_buf_size,
                                       cross_k_uncond, cross_v_uncond);

        CudaBuffer encoder_output(enc_buf_size);
        CudaBuffer encoder_output_uncond(magpie_cfg.cfg_scale > 1.0F ? enc_buf_size : 0);

        auto codec_module = extract_optional_module(
            find_section(ctx.bundle, "codec_engine_plan"), "magpie codec", shared_stream);

        auto tok = make_ipa_tok(ctx.bundle);

        return std::make_unique<MagpiePipeline>(
            std::move(enc_loaded.module),
            std::move(dec_loaded.module),
            std::move(decoder_state),
            std::move(codec_module),
            std::move(decoder_state_uncond),
            std::move(cross_k),
            std::move(cross_v),
            std::move(cross_k_uncond),
            std::move(cross_v_uncond),
            std::move(encoder_output),
            std::move(encoder_output_uncond),
            section_to_floats(find_section(ctx.bundle, "magpie_audio_embed")),
            section_to_floats(find_section(ctx.bundle, "magpie_text_embed")),
            section_to_floats(find_section(ctx.bundle, "magpie_context_embed")),
            section_to_int32s(find_section(ctx.bundle, "magpie_context_lengths")),
            std::move(magpie_cfg),
            stream,
            std::move(tok),
            ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_MagpiePlugin = 0;

} // namespace trtf

static trtf::MagpiePlugin g_MagpiePlugin_instance;
static trtf::PluginRegistrar g_MagpiePlugin_reg("text_to_audio_magpie", &g_MagpiePlugin_instance);

#endif // TRTF_HAS_TRT
