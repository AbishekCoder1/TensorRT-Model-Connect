// SpeechPlugin: handles "speech_to_speech" strategy.
// Speech pipeline with temporal engine, depth engines, and mimi encoder/decoder.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/plugins/shared/audio_helpers.h"
#include "runtime/pipelines/speech_pipeline.h"

#if TRTF_HAS_TRT

namespace trtf {

class SpeechPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto shared_stream = std::make_shared<CudaStream>();
        cudaStream_t stream = shared_stream->get();

        auto speech_cfg = build_speech_config_from_bundle(ctx.bundle, ctx.config_json, ctx.config, ctx.hf_python);
        infer_speech_vocab_sizes(speech_cfg, ctx.config_json, ctx.config);

        auto temporal_loaded = load_trt_module_from_plan(
            find_section(ctx.bundle, "engine_plan"), "speech temporal", shared_stream);

        int32_t temporal_kv_dim = compute_kv_dim_kv_heads(ctx.config, ctx.config.hidden_size);

        std::unique_ptr<IInferenceState> temporal_state = std::make_unique<KvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, temporal_kv_dim, stream);
        if (!temporal_state->ok())
            throw std::runtime_error("SpeechPipeline: failed to create temporal KvCache");

        auto depth_engines = load_depth_engines(ctx.bundle, shared_stream);

        const auto depth_cfg = make_depth_engine_config(ctx.config_json, ctx.config);
        int32_t depth_kv_dim = compute_kv_dim_kv_heads(depth_cfg, depth_cfg.hidden_size);

        std::unique_ptr<IInferenceState> depth_state = std::make_unique<KvCache>(
            depth_cfg.num_layers, depth_cfg.max_cache_length, depth_kv_dim, stream);
        if (!depth_state->ok())
            throw std::runtime_error("SpeechPipeline: failed to create depth KvCache");

        auto mimi_encoder = extract_optional_module(
            find_section(ctx.bundle, "mimi_encoder_plan"), "speech mimi_encoder", shared_stream);
        auto mimi_decoder = extract_optional_module(
            find_section(ctx.bundle, "mimi_decoder_plan"), "speech mimi_decoder", shared_stream);

        return std::make_unique<SpeechPipeline>(
            std::move(mimi_encoder),
            std::move(temporal_loaded.module),
            std::move(temporal_state),
            std::move(depth_engines),
            std::move(depth_state),
            std::move(mimi_decoder),
            std::move(speech_cfg),
            stream,
            nullptr,  // subprocess_runner: default
            ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_SpeechPlugin = 0;

} // namespace trtf

static trtf::SpeechPlugin g_SpeechPlugin_instance;
static trtf::PluginRegistrar g_SpeechPlugin_reg("speech_to_speech", &g_SpeechPlugin_instance);

#endif // TRTF_HAS_TRT
