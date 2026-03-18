// BarkPlugin: handles "text_to_audio_bark" strategy.
// Bark semantic + coarse pipeline with optional codec and fine engines.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/plugins/shared/audio_helpers.h"
#include "runtime/pipelines/audio_pipeline.h"
#include "utils/json_helpers.h"

#if TRTF_HAS_TRT

namespace trtf {

class BarkPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        const auto& json = ctx.config_json;

        // Load semantic engine (main plan)
        auto sem_loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "bark semantic");

        // Load coarse engine
        auto coarse_loaded = load_trt_module_from_plan(
            find_section(ctx.bundle, "coarse_engine_plan"), "bark coarse", sem_loaded.stream);

        cudaStream_t stream = sem_loaded.stream->get();

        // Build BarkConfig
        BarkConfig bark_cfg;
        bark_cfg.sample_rate = extract_json_int(json, "sample_rate", 24000);
        bark_cfg.hidden_size = ctx.config.hidden_size;
        bark_cfg.semantic_input_vocab = extract_json_int(json, "semantic_input_vocab", 129600);
        bark_cfg.semantic_output_vocab = ctx.config.vocab_size;
        bark_cfg.text_encoding_offset = extract_json_int(json, "text_encoding_offset", 10048);
        bark_cfg.text_pad_token = extract_json_int(json, "text_pad_token", 129595);
        bark_cfg.semantic_pad_token = extract_json_int(json, "semantic_pad_token", 10000);
        bark_cfg.semantic_infer_token = extract_json_int(json, "semantic_infer_token", 129599);
        bark_cfg.semantic_vocab_size = extract_json_int(json, "semantic_vocab_size", 10000);
        bark_cfg.coarse_input_vocab = extract_json_int(json, "coarse_input_vocab", 12096);
        bark_cfg.coarse_semantic_pad_token = extract_json_int(json, "coarse_semantic_pad_token", 12048);
        bark_cfg.coarse_infer_token = extract_json_int(json, "coarse_infer_token", 12050);
        bark_cfg.n_coarse_codebooks = extract_json_int(json, "n_coarse_codebooks", 2);
        bark_cfg.codebook_size = extract_json_int(json, "codebook_size", 1024);
        bark_cfg.codec_seq_length = extract_json_int(json, "codec_seq_length", 0);
        bark_cfg.codec_upsample_factor = extract_json_int(json, "codec_upsample_factor", 320);
        bark_cfg.codec_n_codebooks = extract_json_int(json, "codec_n_codebooks", 8);
        bark_cfg.fine_hidden_size = extract_json_int(json, "fine_hidden_size", ctx.config.hidden_size);
        bark_cfg.fine_n_lm_heads = extract_json_int(json, "fine_n_lm_heads", 7);
        bark_cfg.fine_codebook_size = extract_json_int(json, "fine_codebook_size", 1056);
        bark_cfg.fine_seq_length = extract_json_int(json, "fine_seq_length", 0);

        // Create KvCaches for semantic and coarse stages
        int32_t sem_kv_dim = compute_kv_dim(ctx.config);
        auto sem_cache = std::make_unique<KvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, sem_kv_dim, stream);

        // Coarse engine may have different dimensions -- resolve with semantic fallbacks
        auto coarse_cache = make_coarse_kv_cache(json, ctx.config, stream);

        // Load embeddings
        auto sem_embed = section_to_floats(find_section(ctx.bundle, "semantic_embed"));
        auto coarse_embed = section_to_floats(find_section(ctx.bundle, "coarse_embed"));

        // Bark uses BertTokenizer WITHOUT special tokens ([CLS]/[SEP]).
        // HF's BarkProcessor calls encode(text, add_special_tokens=False).
        // Always use add_special_tokens=false to match the HF Bark pipeline.
        auto bark_tokenizer = try_create_native_tokenizer(
            ctx.bundle, /*add_special_tokens=*/false);

        auto pipeline = std::make_unique<BarkPipeline>(
            std::move(sem_loaded.module), std::move(coarse_loaded.module),
            std::move(sem_cache), std::move(coarse_cache),
            std::move(sem_embed), std::move(coarse_embed),
            std::move(bark_cfg), stream,
            std::move(bark_tokenizer), ctx.bundle.info.model_id);

        // Optional codec engine
        auto codec_loaded = try_load_trt_module_from_plan(
            find_section(ctx.bundle, "codec_engine_plan"), "bark codec", sem_loaded.stream);
        if (codec_loaded.module && codec_loaded.module->ok())
            pipeline->set_codec_module(std::move(codec_loaded.module));

        // Optional fine engine
        auto fine_loaded = try_load_trt_module_from_plan(
            find_section(ctx.bundle, "fine_engine_plan"), "bark fine", sem_loaded.stream);
        if (fine_loaded.module && fine_loaded.module->ok())
        {
            pipeline->set_fine_module(std::move(fine_loaded.module));
            auto fe = section_to_floats(find_section(ctx.bundle, "fine_embed"));
            auto fp = section_to_floats(find_section(ctx.bundle, "fine_position_embed"));
            if (!fe.empty()) pipeline->set_fine_embeddings(std::move(fe), std::move(fp));
        }

        return pipeline;
    }
};

volatile int kForceLink_BarkPlugin = 0;

} // namespace trtf

static trtf::BarkPlugin g_BarkPlugin_instance;
static trtf::PluginRegistrar g_BarkPlugin_reg("text_to_audio_bark", &g_BarkPlugin_instance);

#endif // TRTF_HAS_TRT
