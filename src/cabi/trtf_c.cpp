#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/pipeline_legacy.h"
#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"
#include "bundle/bundle_format.h"
#include "model/trt_model_definition.h"
#include "utils/data_dir.h"
#include "utils/json_helpers.h"
#include "utils/text_parsers.h"
#include "utils/trt/engine_cache.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_graph_ops.h"
#include "runtime/trt/trt_backend_shared.h"
#include "runtime/trt/trt_common.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#ifndef TRTF_VERSION_STRING
#define TRTF_VERSION_STRING "0.1.0"
#endif

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string& msg)
{
    g_last_error = msg;
}

void clear_last_error()
{
    g_last_error.clear();
}

class PipelineImpl final : public trtf::IPipeline {
public:
    PipelineImpl(std::string model_id, std::unique_ptr<trtf::ITokenizer> tokenizer,
        std::unique_ptr<trtf::IGenerationBackend> backend, std::string backend_name,
        trtf::GenerationConfig gen_config)
        : mModelId(std::move(model_id))
        , mTokenizer(std::move(tokenizer))
        , mBackend(std::move(backend))
        , mBackendName(std::move(backend_name))
        , mGenConfig(gen_config)
    {
    }

    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        if (prompt == nullptr)
        {
            mLastOutput = "";
            return mLastOutput.c_str();
        }

        trtf::GenerationConfig config = mGenConfig;
        if (max_new_tokens > 0)
        {
            config.max_new_tokens = max_new_tokens;
        }

        if (mBackend->supports_text_generation())
        {
            mLastOutput = mBackend->generate_text(prompt, config);
            return mLastOutput.c_str();
        }

        auto input_ids = mTokenizer->encode(prompt);
        auto output_ids = mBackend->generate(input_ids, config);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    const char* model_id() const override
    {
        return mModelId.c_str();
    }

    const char* backend_name() const override
    {
        return mBackendName.c_str();
    }

    bool save_bundle(const char* output_path) override
    {
        // Bundle save is implemented in Phase 3 when TRT engine serialization is ready.
        // For now, return false.
        (void) output_path;
        return false;
    }

private:
    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;
};

std::size_t resolve_max_new_tokens(std::size_t fallback)
{
    const char* env = std::getenv("TRTF_MAX_NEW_TOKENS");
    if (env == nullptr || env[0] == '\0')
    {
        return fallback;
    }
    try
    {
        const long long parsed = std::stoll(env);
        if (parsed > 0)
        {
            return static_cast<std::size_t>(parsed);
        }
    }
    catch (const std::exception&)
    {
    }
    return fallback;
}

// Resolve model alias to directory path (cheap — no weight loading).
std::string resolve_model_dir_lightweight(const std::string& input)
{
    if (trtf::iequals_ascii(input, "QWEN3") || trtf::iequals_ascii(input, "qwen3"))
    {
        const std::string real = trtf::model_path("hf/Qwen__Qwen3-0.6B");
        if (std::filesystem::exists(std::filesystem::path(real) / "config.json")
            && (std::filesystem::exists(std::filesystem::path(real) / "model.safetensors")
                || std::filesystem::exists(std::filesystem::path(real) / "model.safetensors.index.json")))
        {
            return real;
        }
        return trtf::model_path("hf/qwen3");
    }
    return input;
}

// Fast path: try to create pipeline from cached TRT engine without loading weights.
// Returns nullptr if cache miss (caller should fall back to full pipeline).
#if TRTF_HAS_TRT
PipelineImpl* try_create_from_cached_engine(const std::string& input, const std::string& model_dir)
{
    if (!std::filesystem::exists(std::filesystem::path(model_dir) / "config.json"))
    {
        return nullptr;
    }

    const std::string config_text = trtf::read_file(std::filesystem::path(model_dir) / "config.json");
    int32_t max_cache_length = trtf::parse_positive_env_int("TRTF_MAX_CACHE_LENGTH", -1);
    if (max_cache_length <= 0)
    {
        max_cache_length = trtf::extract_json_int(config_text, "max_position_embeddings", 32);
        // Apply same caps as model loader
        if (max_cache_length > 4096)
        {
            max_cache_length = 4096;
        }
    }

    const std::string index_key = trtf::BuildModelDirIndexKey(model_dir, max_cache_length);
    const auto cache_key = trtf::LookupModelDirIndex(index_key);
    if (!cache_key)
    {
        return nullptr;
    }

    const auto cached_plan = trtf::LoadTrtEnginePlanFromCache(*cache_key);
    if (!cached_plan)
    {
        return nullptr;
    }

    std::cerr << "[trtf] Fast path: loading cached TRT engine (" << cached_plan->size() / (1024 * 1024)
              << " MB) — skipping weight loading ..." << std::endl;

    trtf::TrtLogger logger;
    auto runtime_ptr = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_ptr)
    {
        return nullptr;
    }

    auto trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(cached_plan->data(), cached_plan->size()));
    if (!trt_engine)
    {
        return nullptr;
    }

    auto exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!exec_ctx)
    {
        return nullptr;
    }

    // Read minimal metadata from config for engine dimensions
    const int32_t vocab_size = trtf::extract_json_int(config_text, "vocab_size", 0);
    const int32_t hidden_size = trtf::extract_json_int(config_text, "hidden_size", 0);
    const int32_t num_layers = std::max(trtf::extract_json_int(config_text, "num_hidden_layers", 1), 1);
    const int32_t num_heads = std::max(trtf::extract_json_int(config_text, "num_attention_heads", 1), 1);
    const int32_t num_kv_heads = std::max(trtf::extract_json_int(config_text, "num_key_value_heads", 1), 1);
    // head_dim may be explicit in config (e.g., Qwen3 has head_dim=128 != hidden/heads=64)
    const int32_t head_dim = trtf::extract_json_int(config_text, "head_dim", hidden_size / num_heads);
    const int32_t attention_size = num_heads * head_dim;

    auto engine_struct = std::make_unique<trtf::DecoderStepEngine>();
    engine_struct->engine = std::move(trt_engine);
    engine_struct->context = std::move(exec_ctx);
    engine_struct->vocab_size = vocab_size;
    engine_struct->hidden_size = hidden_size;
    engine_struct->cache_state_size = attention_size;
    engine_struct->attention_mask_size = max_cache_length + 1;
    engine_struct->max_cache_length = max_cache_length;
    engine_struct->num_layers = num_layers;
    engine_struct->requires_position_input = true;

    for (int32_t i = 0; i < num_layers; ++i)
    {
        engine_struct->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        engine_struct->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        engine_struct->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        engine_struct->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }

    engine_struct->id_bos = trtf::extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    engine_struct->id_eos = trtf::extract_json_int_or_first_array(config_text, "eos_token_id", -1);

    if (!trtf::has_all_required_tensors(*engine_struct))
    {
        return nullptr;
    }

    // Create tokenizer (still need HF Python tokenizer for real models)
    std::unique_ptr<trtf::ITokenizer> tokenizer;
    if (std::filesystem::exists(std::filesystem::path(model_dir) / "tokenizer.json"))
    {
        std::cerr << "[trtf] Initializing HF tokenizer ..." << std::endl;
        auto ttok0 = std::chrono::steady_clock::now();
        tokenizer = trtf::CreateHfPythonTokenizer(model_dir);
        auto ttok1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Tokenizer ready ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
                  << " ms]" << std::endl;
    }
    else
    {
        return nullptr; // Can't run without tokenizer
    }

    // Create lightweight backend directly from pre-built engine — zero weight allocation.
    auto backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    if (!backend || !backend->is_available())
    {
        return nullptr;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, fast-path)" << std::endl;

    trtf::GenerationConfig gen_config{};
    gen_config.max_new_tokens = resolve_max_new_tokens(gen_config.max_new_tokens);

    return new PipelineImpl(input, std::move(tokenizer), std::move(backend), "trt", gen_config);
}
#endif

} // namespace

extern "C" {

trtf::IPipeline* trtf_create_pipeline(const char* model_or_bundle, int flags)
{
    clear_last_error();

    if (model_or_bundle == nullptr || model_or_bundle[0] == '\0')
    {
        set_last_error("model_or_bundle must not be null or empty");
        return nullptr;
    }

    try
    {
        const std::string input(model_or_bundle);

        // Check if input is a .trtfb bundle
        if (trtf::IsBundle(input))
        {
            set_last_error("Bundle loading not yet implemented");
            return nullptr;
        }

        // Normal model loading path
        bool prefer_trt = true;
        bool force_trt = false;
        switch (flags)
        {
        case TRTF_FORCE_TRT:
            force_trt = true;
            break;
        case TRTF_CPU_ONLY:
            prefer_trt = false;
            break;
        case TRTF_PREFER_TRT:
        default:
            break;
        }

#if TRTF_HAS_TRT
        // Fast path: if TRT is preferred and we have a cached engine, skip weight loading entirely.
        if (prefer_trt)
        {
            const std::string model_dir = resolve_model_dir_lightweight(input);
            auto t0 = std::chrono::steady_clock::now();
            PipelineImpl* fast = try_create_from_cached_engine(input, model_dir);
            if (fast != nullptr)
            {
                auto t1 = std::chrono::steady_clock::now();
                std::cerr << "[trtf] Total startup ["
                          << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                          << " ms]" << std::endl;
                return fast;
            }
        }
#endif

        // Slow path: full model resolution (loads weights from safetensors)
        auto t0 = std::chrono::steady_clock::now();

        std::cerr << "[trtf] Resolving model: " << input << " ..." << std::endl;
        const trtf::ResolvedModelSpec model_spec = trtf::ResolveTextGenerationModel(input);
        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Model resolved (kind="
                  << (model_spec.kind == trtf::ResolvedModelKind::kDecoderDefinition ? "decoder-definition"
                      : model_spec.kind == trtf::ResolvedModelKind::kHuggingFaceLocal ? "hf-local" : "custom")
                  << ") ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms]"
                  << std::endl;

        trtf::BackendSelection selection;
        selection.prefer_trt = prefer_trt;
        selection.force_trt = force_trt;

        std::cerr << "[trtf] Building runtime (backend="
                  << (force_trt ? "force-trt" : prefer_trt ? "prefer-trt" : "cpu-only")
                  << ") ..." << std::endl;
        trtf::RuntimeAssembly runtime = trtf::BuildRuntimeForTextGeneration(model_spec, selection);
        auto t2 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Runtime ready (backend=" << runtime.backend_name
                  << ") [" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms]"
                  << std::endl;

        // If TRT backend was created via slow path, save the index for next time
#if TRTF_HAS_TRT
        if (runtime.backend_name == "trt"
            && model_spec.kind == trtf::ResolvedModelKind::kDecoderDefinition)
        {
            const std::string model_dir = resolve_model_dir_lightweight(input);
            const int32_t max_cache = model_spec.decoder_model.max_cache_length;
            const std::string index_key = trtf::BuildModelDirIndexKey(model_dir, max_cache);
            trtf::TrtEngineCacheKeyParams params;
            params.requires_position_input = true;
            params.num_layers = model_spec.decoder_model.architecture.num_layers;
            const auto& ckpt = model_spec.decoder_model.checkpoint;
            // Only save index if we have the weights to compute the full cache key
            if (ckpt.has_decoder_layers)
            {
                trtf::TrtDecoderDefinition def = trtf::BuildTrtDecoderWeights(
                    *runtime.tokenizer, model_spec.decoder_model);
                const std::string cache_key = trtf::BuildTrtEngineCacheKey(def, params);
                trtf::SaveModelDirIndex(index_key, cache_key);
                std::cerr << "[trtf] Saved engine cache index for fast startup next time" << std::endl;
            }
        }
#endif

        trtf::GenerationConfig gen_config{};
        gen_config.max_new_tokens = resolve_max_new_tokens(gen_config.max_new_tokens);

        return new PipelineImpl(
            input,
            std::move(runtime.tokenizer),
            std::move(runtime.backend),
            std::move(runtime.backend_name),
            gen_config);
    }
    catch (const std::exception& e)
    {
        set_last_error(e.what());
        return nullptr;
    }
    catch (...)
    {
        set_last_error("Unknown error creating pipeline");
        return nullptr;
    }
}

const char* trtf_last_error(void)
{
    return g_last_error.c_str();
}

const char* trtf_version(void)
{
    return TRTF_VERSION_STRING;
}

int trtf_has_trt(void)
{
#if TRTF_HAS_TRT
    return 1;
#else
    return 0;
#endif
}

} // extern "C"
