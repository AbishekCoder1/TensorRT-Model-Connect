// DecoderPlugin: handles "decoder_kv_cache" and "decoder_moe" strategies.
// Standard attention-based decoder with device-resident KV cache.

#include "runtime/core/chat_template.h"
#include "runtime/core/trt_engine_lifecycle.h"
#include "runtime/pipelines/text_generation_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/config/config_bundle.h"
#include "trtf/runtime/pipeline_registry.h"
#include "trtf/runtime/triattention_kv_cache.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <limits>
#include <sstream>

#if TRTF_HAS_TRT

namespace trtf {

namespace {

struct KvCacheRuntimeSizing {
    int32_t runtime_rows{0};
    std::uint64_t row_bytes{0};
    std::uint64_t cache_bytes{0};
    bool override_applied{false};
    bool clamped_to_bundle_max{false};
};

int32_t cache_row_dim_from_engine(const nvinfer1::ICudaEngine& engine,
                                  const std::string& tensor_name) {
    auto dims = engine.getTensorShape(tensor_name.c_str());
    if (dims.nbDims >= 2 && dims.d[1] > 0)
        return dims.d[1];
    if (engine.getNbOptimizationProfiles() > 0) {
        dims = engine.getProfileShape(tensor_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        if (dims.nbDims >= 2 && dims.d[1] > 0)
            return dims.d[1];
    }
    throw std::runtime_error("Unable to infer KV row width from engine tensor '" + tensor_name +
                             "'");
}

bool cache_input_is_dynamic(const nvinfer1::ICudaEngine& engine, const std::string& tensor_name) {
    const auto dims = engine.getTensorShape(tensor_name.c_str());
    return dims.nbDims >= 1 && dims.d[0] == -1;
}

bool cache_input_supports_runtime_rows(const nvinfer1::ICudaEngine& engine,
                                       const std::string& tensor_name) {
    if (!cache_input_is_dynamic(engine, tensor_name))
        return false;
    const int32_t num_profiles = engine.getNbOptimizationProfiles();
    if (num_profiles <= 0)
        return false;
    for (int32_t profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
        const auto min_dims = engine.getProfileShape(tensor_name.c_str(), profile_idx,
                                                     nvinfer1::OptProfileSelector::kMIN);
        const auto max_dims = engine.getProfileShape(tensor_name.c_str(), profile_idx,
                                                     nvinfer1::OptProfileSelector::kMAX);
        if (min_dims.nbDims >= 1 && max_dims.nbDims >= 1 && min_dims.d[0] < max_dims.d[0])
            return true;
    }
    return false;
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream oss;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        oss << (static_cast<double>(bytes) / kGiB) << " GiB";
        return oss.str();
    }
    if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        oss << (static_cast<double>(bytes) / kMiB) << " MiB";
        return oss.str();
    }
    oss.unsetf(std::ios::floatfield);
    oss.precision(6);
    oss << bytes << " B";
    return oss.str();
}

KvCacheRuntimeSizing
resolve_kv_cache_runtime_sizing(const PipelineContext& ctx, const nvinfer1::ICudaEngine& engine,
                                const KvCacheNames& kv_names, DType cache_dtype,
                                const TriAttentionConfig& tri_cfg, int32_t kv_dim) {
    KvCacheRuntimeSizing sizing;
    const auto elem_bytes = static_cast<std::uint64_t>(dtype_size(cache_dtype));
    sizing.row_bytes = static_cast<std::uint64_t>(ctx.config.num_layers) *
                       static_cast<std::uint64_t>(kv_dim) * elem_bytes * 2ULL;
    if (sizing.row_bytes == 0)
        throw std::runtime_error("Computed zero bytes per KV row");

    const int32_t bundle_max_rows = ctx.config.max_cache_length;
    sizing.runtime_rows = bundle_max_rows;
    sizing.cache_bytes = static_cast<std::uint64_t>(bundle_max_rows) * sizing.row_bytes;

    if (ctx.kv_cache_size_bytes == 0)
        return sizing;

    if (!cache_input_supports_runtime_rows(engine, kv_names.cache_k.front())) {
        throw std::runtime_error(
            "This bundle was not built with runtime-resizable KV cache support. "
            "Rebuild with trtf-build --dynamic-kv-cache to use --kv-cache-size.");
    }

    const std::uint64_t requested_rows_u64 = ctx.kv_cache_size_bytes / sizing.row_bytes;
    if (requested_rows_u64 == 0) {
        throw std::runtime_error("--kv-cache-size is smaller than one KV row (" +
                                 format_bytes(sizing.row_bytes) + ")");
    }

    std::uint64_t runtime_rows_u64 = requested_rows_u64;
    if (runtime_rows_u64 > static_cast<std::uint64_t>(bundle_max_rows)) {
        runtime_rows_u64 = static_cast<std::uint64_t>(bundle_max_rows);
        sizing.clamped_to_bundle_max = true;
    }
    if (runtime_rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Resolved KV cache rows exceed int32 runtime limits");
    }

    sizing.runtime_rows = static_cast<int32_t>(runtime_rows_u64);
    sizing.cache_bytes = runtime_rows_u64 * sizing.row_bytes;
    sizing.override_applied = true;

    if (tri_cfg.enabled && sizing.runtime_rows < tri_cfg.kv_budget) {
        const auto minimum_bytes = static_cast<std::uint64_t>(tri_cfg.kv_budget) * sizing.row_bytes;
        throw std::runtime_error(
            "--kv-cache-size resolves to " + std::to_string(sizing.runtime_rows) +
            " rows, but this TriAttention bundle needs at least " +
            std::to_string(tri_cfg.kv_budget) + " rows (" + format_bytes(minimum_bytes) + ")");
    }

    return sizing;
}

} // namespace

class DecoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        // Master's FFI refactor: load any TVM/FFI kernel plugins bundled alongside
        // the engine before deserializing. Required for FFI-based custom kernels.
        load_ffi_kernels_from_bundle(ctx.bundle);

        // Hydrate process-wide text_trace state from the resolved registry.
        // Replaces the old TRTF_TEXT_STEP_TRACE_* env-var reads in
        // text_generation_pipeline.cpp.
        if (ctx.runtime_config != nullptr) {
            try {
                apply_text_trace_config_from_registry(
                    ctx.runtime_config->get<std::string>("text_trace", "step_trace_path"),
                    ctx.runtime_config->get<std::int32_t>("text_trace", "step_trace_start_pos"),
                    ctx.runtime_config->get<std::int32_t>("text_trace", "step_trace_end_pos"),
                    ctx.runtime_config->get<std::int32_t>("text_trace", "step_trace_topk"));
            } catch (const std::exception&) {
                // text_trace schema not registered, or type mismatch —
                // leave tracing at its disabled default.
            }
        }

        // TriAttention needs direct ICudaEngine access (multi-profile, row-dim
        // introspection, KV sizing). Keep the manual deserialization path rather
        // than the LoadedModule helper, which hides the engine behind TrtModule.
        auto* plan = find_section(ctx.bundle, "engine_plan");
        if (plan == nullptr || plan->empty()) {
            throw std::runtime_error("engine_plan section is missing");
        }
        auto trt_runtime = create_trt_runtime();
        if (!trt_runtime) {
            throw std::runtime_error("Failed to create TRT runtime for engine_plan");
        }
        auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
            trt_runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine_plan");
        }
        auto shared_stream = std::make_shared<CudaStream>();
        if (!shared_stream->ok()) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
        nvinfer1::ICudaEngine* raw_engine = engine.release();
        auto shared_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            raw_engine, [](nvinfer1::ICudaEngine* p) { delete p; });
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        // Build KvCacheNames from IoMap patterns.
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
        std::vector<std::string> external_input_names;
        external_input_names.reserve(kv_names.cache_k.size() + kv_names.cache_v.size());
        external_input_names.insert(external_input_names.end(), kv_names.cache_k.begin(),
                                    kv_names.cache_k.end());
        external_input_names.insert(external_input_names.end(), kv_names.cache_v.begin(),
                                    kv_names.cache_v.end());

        auto make_decoder = [&](int32_t profile_idx) -> std::unique_ptr<TrtModule> {
            auto module = std::make_unique<TrtModule>(shared_engine.get(), shared_stream->get(),
                                                      profile_idx, external_input_names);
            if (!module || !module->ok()) {
                throw std::runtime_error("Failed to create TrtModule for engine_plan (profile " +
                                         std::to_string(profile_idx) + ")");
            }
            module->keep_alive(shared_engine);
            module->keep_alive(shared_stream);
            return module;
        };

        cudaStream_t stream = shared_stream->get();
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        TriAttentionConfig tri_cfg = parse_triattention_bundle_config(
            ctx.config_json, ctx.config.max_cache_length, ctx.runtime_config);
        const int32_t kv_dim = cache_row_dim_from_engine(*shared_engine, kv_names.cache_k.front());
        const auto sizing = resolve_kv_cache_runtime_sizing(ctx, *shared_engine, kv_names,
                                                            cache_dtype, tri_cfg, kv_dim);

        auto profile_rows = extract_json_int_array(ctx.config_json, "dynamic_kv_profile_rows", 16);
        if (profile_rows.empty())
            profile_rows.push_back(ctx.config.max_cache_length);
        const int32_t num_profiles = shared_engine->getNbOptimizationProfiles();
        std::vector<TextGenerationPipeline::DecoderContext> decoders;
        decoders.reserve(static_cast<std::size_t>(num_profiles > 0 ? num_profiles : 1));
        for (int32_t profile_idx = 0;
             profile_idx < num_profiles && profile_idx < static_cast<int32_t>(profile_rows.size());
             ++profile_idx) {
            const int32_t profile_max_rows = profile_rows[static_cast<std::size_t>(profile_idx)];
            if (profile_idx > 0 && profile_max_rows > sizing.runtime_rows)
                break;
            decoders.push_back(TextGenerationPipeline::DecoderContext{profile_max_rows,
                                                                      make_decoder(profile_idx)});
        }
        if (decoders.empty()) {
            decoders.push_back(TextGenerationPipeline::DecoderContext{
                ctx.config.max_cache_length,
                make_decoder(0),
            });
        }

        std::unique_ptr<IInferenceState> state;
        if (tri_cfg.enabled) {
            auto* stats_sec = find_section(ctx.bundle, tri_cfg.stats_section);
            if (stats_sec == nullptr || stats_sec->empty()) {
                throw std::runtime_error("TriAttention stats section is missing: " +
                                         tri_cfg.stats_section);
            }
            std::string stats_json(stats_sec->begin(), stats_sec->end());
            TriAttentionStats tri_stats = parse_triattention_stats_json(
                stats_json, ctx.config.num_heads, ctx.config.num_kv_heads, ctx.config.num_layers);
            state = std::make_unique<TriAttentionKvCache>(
                ctx.config.num_layers, ctx.config.num_kv_heads, sizing.runtime_rows, kv_dim, stream,
                std::move(tri_cfg), std::move(tri_stats), cache_dtype, std::move(kv_names));
        } else {
            state = std::make_unique<KvCache>(ctx.config.num_layers, sizing.runtime_rows, kv_dim,
                                              stream, cache_dtype, std::move(kv_names));
        }
        if (!state->ok())
            throw std::runtime_error("Failed to create KvCache");

        std::cerr << "[trtf] KV cache rows=" << sizing.runtime_rows
                  << " (bundle max=" << ctx.config.max_cache_length
                  << ", row=" << format_bytes(sizing.row_bytes)
                  << ", cache=" << format_bytes(sizing.cache_bytes) << ", state="
                  << format_bytes(static_cast<std::uint64_t>(state->device_memory_bytes())) << ")";
        if (sizing.override_applied) {
            std::cerr << " [requested=" << format_bytes(ctx.kv_cache_size_bytes) << "]";
            if (sizing.clamped_to_bundle_max)
                std::cerr << " [clamped-to-bundle-max]";
        }
        std::cerr << '\n';

        TextGenConfig tgc;
        tgc.vocab_size = ctx.config.vocab_size;
        tgc.id_bos = ctx.config.id_bos;
        tgc.id_eos = ctx.config.id_eos;
        tgc.has_position_input = decoders.front().module->has_input(io.position_id);
        tgc.token_id_name = io.token_id;
        tgc.logits_output_name = io.logits;

        // runtime.* namespace (replaces TRTF_DISABLE_CUDA_GRAPH, TRTF_GPU_ARGMAX).
        if (ctx.runtime_config != nullptr) {
            try {
                tgc.disable_cuda_graph =
                    ctx.runtime_config->get<bool>("runtime", "disable_cuda_graph");
                tgc.prefer_gpu_greedy =
                    ctx.runtime_config->get<bool>("runtime", "prefer_gpu_greedy");
            } catch (const std::exception&) {
                // Schema not registered or type mismatch — stay at defaults.
            }
        }

        // Detect chat template format from tokenizer_config.json
        auto* tok_cfg_sec = find_section(ctx.bundle, "tokenizer_config.json");
        if (tok_cfg_sec && !tok_cfg_sec->empty()) {
            std::string tok_cfg_text(tok_cfg_sec->begin(), tok_cfg_sec->end());
            std::string chat_tpl = extract_json_string(tok_cfg_text, "chat_template", "");
            tgc.chat_template_format = detect_chat_template_format(chat_tpl);
        }

        return std::make_unique<TextGenerationPipeline>(std::move(decoders), std::move(state), tgc,
                                                        stream, std::move(tokenizer),
                                                        ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_DecoderPlugin = 0;

} // namespace trtf

static trtf::DecoderPlugin g_DecoderPlugin_instance;
static trtf::PluginRegistrar g_DecoderPlugin_reg1("decoder_kv_cache", &g_DecoderPlugin_instance);
static trtf::PluginRegistrar g_DecoderPlugin_reg2("decoder_moe", &g_DecoderPlugin_instance);

#endif // TRTF_HAS_TRT
