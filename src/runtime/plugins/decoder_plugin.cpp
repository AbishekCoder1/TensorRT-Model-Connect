// DecoderPlugin: handles "decoder_kv_cache" and "decoder_moe" strategies.
// Standard attention-based decoder with device-resident KV cache.
//
// The engine carries two optimization profiles (when built by the
// unified decoder builder): profile 0 for batched-sequence prefill and
// profile 1 for single-token decode. We create two IExecutionContexts
// sharing the engine, one per profile. Legacy single-profile bundles
// still work — the prefill context is left null and the pipeline falls
// back to token-by-token prefill on the decode context.

#include "runtime/core/chat_template.h"
#include "runtime/core/trt_common.h"
#include "runtime/core/trt_engine_lifecycle.h"
#include "runtime/pipelines/text_generation_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <iostream>

#if TRTF_HAS_TRT

namespace trtf {

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_dual_profile_modules(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        const auto& io = ctx.config.io_map;
        KvCacheNames kv_names;
        kv_names.position_id = io.position_id;
        kv_names.attention_mask = io.attention_mask;
        for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
            kv_names.cache_k.push_back(expand_layer_name(io.cache_k_pattern, i));
            kv_names.cache_v.push_back(expand_layer_name(io.cache_v_pattern, i));
            kv_names.present_k.push_back(expand_layer_name(io.present_k_pattern, i));
            kv_names.present_v.push_back(expand_layer_name(io.present_v_pattern, i));
        }

        cudaStream_t stream = loaded.decode->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<IInferenceState> state =
            std::make_unique<KvCache>(ctx.config.num_layers, ctx.config.max_cache_length, kv_dim,
                                      stream, cache_dtype, std::move(kv_names));
        if (!state->ok())
            throw std::runtime_error("Failed to create KvCache");

        TextGenConfig tgc;
        tgc.vocab_size = ctx.config.vocab_size;
        tgc.id_bos = ctx.config.id_bos;
        tgc.id_eos = ctx.config.id_eos;
        tgc.has_position_input = loaded.decode->has_input(io.position_id);
        tgc.token_id_name = io.token_id;
        tgc.logits_output_name = io.logits;
        tgc.present_k_pattern = io.present_k_pattern;
        tgc.present_v_pattern = io.present_v_pattern;
        tgc.num_layers = ctx.config.num_layers;
        tgc.kv_dim = kv_dim;

        auto* tok_cfg_sec = find_section(ctx.bundle, "tokenizer_config.json");
        if (tok_cfg_sec && !tok_cfg_sec->empty()) {
            std::string tok_cfg_text(tok_cfg_sec->begin(), tok_cfg_sec->end());
            std::string chat_tpl = extract_json_string(tok_cfg_text, "chat_template", "");
            tgc.chat_template_format = detect_chat_template_format(chat_tpl);
        }

        if (loaded.prefill) {
            // The unified builder uses max_prefill_length == max_cache_length
            // for profile 0's MAX shape. Read from config rather than the
            // module (input_info() stores the opt shape, not max).
            tgc.prefill_max_length = ctx.config.max_cache_length;
            if (trt_log_to_stderr_enabled()) {
                std::cerr << "[trtf] Dual-profile engine: prefill max Sq = "
                          << tgc.prefill_max_length << "\n";
            }
        }

        return std::make_unique<TextGenerationPipeline>(
            std::move(loaded.decode), std::move(state), tgc, stream, std::move(tokenizer),
            ctx.bundle.info.model_id, nullptr, std::move(loaded.prefill));
    }
};

volatile int kForceLink_DecoderPlugin = 0;

} // namespace trtf

static trtf::DecoderPlugin g_DecoderPlugin_instance;
static trtf::PluginRegistrar g_DecoderPlugin_reg1("decoder_kv_cache", &g_DecoderPlugin_instance);
static trtf::PluginRegistrar g_DecoderPlugin_reg2("decoder_moe", &g_DecoderPlugin_instance);

#endif // TRTF_HAS_TRT
