// VLPlugin: handles "vision_language" strategy.
// Two-engine pipeline: vision encoder + text decoder with KV cache.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/pipelines/vl_pipeline.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "utils/json_helpers.h"

#include <iostream>

#if TRTF_HAS_TRT

namespace trtf {

class VLPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");

        cudaStream_t stream = loaded.stream->get();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        auto cache = std::make_unique<KvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle, ctx.hf_python);

        VLConfig vlc;
        vlc.vocab_size = ctx.config.vocab_size;
        vlc.id_bos = ctx.config.id_bos;
        vlc.id_eos = ctx.config.id_eos;
        vlc.image_token_id = extract_json_int(ctx.config_json, "image_token_id", -1);
        vlc.vision_output_dim = extract_json_int(ctx.config_json, "vision_output_dim", 0);
        vlc.has_position_input = loaded.module->has_input("position_id");

        bool has_vision_engine = extract_json_int(ctx.config_json, "has_vision_engine", 0) != 0;

        // Try to load the vision encoder engine from the bundle.
        std::unique_ptr<TrtModule> vision_module;
        auto vision_loaded = try_load_trt_module_from_plan(
            find_section(ctx.bundle, "vision_engine_plan"), "vision_engine_plan", loaded.stream);
        if (vision_loaded.module && vision_loaded.module->ok()) {
            vision_module = std::move(vision_loaded.module);
            std::cerr << "[trtf] Vision encoder loaded" << std::endl;
        } else if (has_vision_engine) {
            std::cerr << "[trtf] WARNING: Bundle declares vision engine but "
                         "deserialization failed" << std::endl;
        }

        // Build VL preprocessing config from bundle's config.json +
        // preprocessor_config.json sections.
        std::string config_text, preproc_text;
        const auto* config_sec = find_section(ctx.bundle, "config.json");
        if (config_sec && !config_sec->empty())
            config_text.assign(config_sec->begin(), config_sec->end());
        const auto* preproc_sec = find_section(ctx.bundle, "preprocessor_config.json");
        if (preproc_sec && !preproc_sec->empty())
            preproc_text.assign(preproc_sec->begin(), preproc_sec->end());
        auto vl_preprocess = parse_vl_preprocess_config(config_text, preproc_text);

        return std::make_unique<VLPipeline>(
            std::move(loaded.module), std::move(vision_module), std::move(cache),
            vlc, vl_preprocess, stream, std::move(tokenizer),
            ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_VLPlugin = 0;

} // namespace trtf

static trtf::VLPlugin g_VLPlugin_instance;
static trtf::PluginRegistrar g_VLPlugin_reg("vision_language", &g_VLPlugin_instance);

#endif // TRTF_HAS_TRT
