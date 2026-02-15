#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"
#include "bundle/bundle_format.h"
#include "cabi/fast_path_config.h"
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
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
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

// RAII guard for thread-local engine cache config.
struct EngineCacheConfigGuard {
    EngineCacheConfigGuard(const std::string& cache_dir, bool no_cache)
    {
        trtf::EngineCacheConfig cfg;
        cfg.cache_dir = cache_dir;
        cfg.no_cache = no_cache;
        trtf::SetThreadEngineCacheConfig(cfg);
    }
    ~EngineCacheConfigGuard()
    {
        trtf::ClearThreadEngineCacheConfig();
    }
};

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

    ~PipelineImpl() override
    {
        if (!mBundleTempDir.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(mBundleTempDir, ec);
        }
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
        if (output_path == nullptr || output_path[0] == '\0')
        {
            return false;
        }

        if (mModelDir.empty())
        {
            return false;
        }

        const auto plan = mBackend->serialize_engine_plan();
        if (plan.empty())
        {
            return false;
        }

        trtf::BundleFile bundle;
        bundle.info.model_id = mModelId;
        bundle.info.model_type = mModelType;
        bundle.info.family = mFamily;
        bundle.info.vocab_size = mVocabSize;
        bundle.info.hidden_size = mHiddenSize;
        bundle.info.num_layers = mNumLayers;
        bundle.info.num_attention_heads = mNumAttentionHeads;
        bundle.info.num_key_value_heads = mNumKeyValueHeads;
        bundle.info.max_cache_length = mMaxCacheLength;

#if TRTF_HAS_TRT
        bundle.info.trt_version = std::to_string(NV_TENSORRT_MAJOR) + "."
            + std::to_string(NV_TENSORRT_MINOR) + "."
            + std::to_string(NV_TENSORRT_PATCH);
#endif

        // GPU name
#if TRTF_HAS_TRT
        {
            cudaDeviceProp props{};
            if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
            {
                bundle.info.gpu_name = props.name;
            }
        }
#endif

        // Timestamp
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t t = std::chrono::system_clock::to_time_t(now);
            char time_buf[64];
            if (std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t)) > 0)
            {
                bundle.info.created_at = time_buf;
            }
        }

        // Engine plan section
        bundle.sections.push_back({"engine_plan", plan});

        // Embed tokenizer + config files from model dir
        for (const char* filename : {"config.json", "tokenizer.json", "tokenizer_config.json"})
        {
            const std::filesystem::path file_path = std::filesystem::path(mModelDir) / filename;
            if (std::filesystem::exists(file_path))
            {
                std::ifstream in(file_path, std::ios::binary);
                if (in)
                {
                    const std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
                    bundle.sections.push_back({filename, std::vector<char>(content.begin(), content.end())});
                }
            }
        }

        trtf::WriteBundleFile(output_path, bundle);
        return true;
    }

    // Setters for metadata needed by save_bundle
    void set_model_dir(std::string dir) { mModelDir = std::move(dir); }
    void set_model_type(std::string type) { mModelType = std::move(type); }
    void set_family(std::string family) { mFamily = std::move(family); }
    void set_bundle_temp_dir(std::string dir) { mBundleTempDir = std::move(dir); }
    void set_architecture_info(int32_t vocab_size, int32_t hidden_size, int32_t num_layers,
        int32_t num_attention_heads, int32_t num_key_value_heads, int32_t max_cache_length)
    {
        mVocabSize = vocab_size;
        mHiddenSize = hidden_size;
        mNumLayers = num_layers;
        mNumAttentionHeads = num_attention_heads;
        mNumKeyValueHeads = num_key_value_heads;
        mMaxCacheLength = max_cache_length;
    }

private:
    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;

    // Metadata for save_bundle
    std::string mModelDir;
    std::string mModelType;
    std::string mFamily;
    int32_t mVocabSize{0};
    int32_t mHiddenSize{0};
    int32_t mNumLayers{0};
    int32_t mNumAttentionHeads{1};
    int32_t mNumKeyValueHeads{1};
    int32_t mMaxCacheLength{32};

    // Temp directory for extracted tokenizer files (bundle load path).
    // Cleaned up in destructor.
    std::string mBundleTempDir;
};

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
PipelineImpl* try_create_from_cached_engine(const std::string& input, const std::string& model_dir,
    int32_t max_cache_override, const std::string& hf_python)
{
    if (!std::filesystem::exists(std::filesystem::path(model_dir) / "config.json"))
    {
        return nullptr;
    }

    const std::string config_text = trtf::read_file(std::filesystem::path(model_dir) / "config.json");
    const trtf::FastPathModelConfig fp_cfg = trtf::parse_fast_path_config(config_text, max_cache_override);

    const std::string index_key = trtf::BuildModelDirIndexKey(model_dir, fp_cfg.max_cache_length);
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

    // Populate engine metadata from parsed config
    auto engine_struct = std::make_unique<trtf::DecoderStepEngine>();
    engine_struct->engine = std::move(trt_engine);
    engine_struct->context = std::move(exec_ctx);
    engine_struct->vocab_size = fp_cfg.vocab_size;
    engine_struct->hidden_size = fp_cfg.hidden_size;
    engine_struct->cache_state_size = fp_cfg.attention_size;
    engine_struct->attention_mask_size = fp_cfg.max_cache_length + 1;
    engine_struct->max_cache_length = fp_cfg.max_cache_length;
    engine_struct->num_layers = fp_cfg.num_layers;
    engine_struct->requires_position_input = true;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        engine_struct->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        engine_struct->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        engine_struct->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        engine_struct->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }

    engine_struct->id_bos = fp_cfg.id_bos;
    engine_struct->id_eos = fp_cfg.id_eos;

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
        tokenizer = trtf::CreateHfPythonTokenizer(model_dir, hf_python);
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
    // max_new_tokens set by caller via options

    auto* pipeline = new PipelineImpl(input, std::move(tokenizer), std::move(backend), "trt", gen_config);
    pipeline->set_model_dir(model_dir);
    pipeline->set_architecture_info(fp_cfg.vocab_size, fp_cfg.hidden_size, fp_cfg.num_layers,
        fp_cfg.num_heads, fp_cfg.num_kv_heads, fp_cfg.max_cache_length);

    // Read model_type and family from config.json for bundle metadata
    {
        const std::string cfg_text = trtf::read_file(std::filesystem::path(model_dir) / "config.json");
        pipeline->set_model_type(trtf::extract_json_string(cfg_text, "model_type", ""));
        std::string family = trtf::extract_json_string(cfg_text, "architecture_family", "");
        if (family.empty())
        {
            family = trtf::to_lower_ascii(trtf::extract_json_string(cfg_text, "model_type", ""));
        }
        pipeline->set_family(family);
    }

    return pipeline;
}

// Load pipeline from a .trtfb bundle file.
PipelineImpl* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python)
{
    trtf::BundleFile bundle = trtf::ReadBundleFile(bundle_path);

    // Find engine plan section
    const std::vector<char>* plan_data = nullptr;
    const std::vector<char>* config_json_data = nullptr;
    const std::vector<char>* tokenizer_json_data = nullptr;
    const std::vector<char>* tokenizer_config_data = nullptr;

    for (const auto& section : bundle.sections)
    {
        if (section.name == "engine_plan") plan_data = &section.data;
        else if (section.name == "config.json") config_json_data = &section.data;
        else if (section.name == "tokenizer.json") tokenizer_json_data = &section.data;
        else if (section.name == "tokenizer_config.json") tokenizer_config_data = &section.data;
    }

    if (plan_data == nullptr || plan_data->empty())
    {
        throw std::runtime_error("Bundle has no engine_plan section: " + bundle_path);
    }

    // Deserialize the TRT engine
    std::cerr << "[trtf] Deserializing TRT engine from bundle (" << plan_data->size() / (1024 * 1024)
              << " MB) ..." << std::endl;

    trtf::TrtLogger logger;
    auto runtime_ptr = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_ptr)
    {
        throw std::runtime_error("Failed to create TRT runtime");
    }

    auto tdeser0 = std::chrono::steady_clock::now();
    auto trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(plan_data->data(), plan_data->size()));
    if (!trt_engine)
    {
        throw std::runtime_error("Failed to deserialize engine from bundle: " + bundle_path);
    }
    auto tdeser1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Engine deserialized ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(tdeser1 - tdeser0).count()
              << " ms]" << std::endl;

    auto exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!exec_ctx)
    {
        throw std::runtime_error("Failed to create execution context from bundle engine");
    }

    // Parse config.json for model metadata
    trtf::FastPathModelConfig fp_cfg;
    if (config_json_data != nullptr && !config_json_data->empty())
    {
        const std::string config_text(config_json_data->begin(), config_json_data->end());
        fp_cfg = trtf::parse_fast_path_config(config_text, bundle.info.max_cache_length);
    }
    else
    {
        // Use bundle header metadata
        fp_cfg.vocab_size = bundle.info.vocab_size;
        fp_cfg.hidden_size = bundle.info.hidden_size;
        fp_cfg.num_layers = bundle.info.num_layers;
        fp_cfg.num_heads = bundle.info.num_attention_heads;
        fp_cfg.num_kv_heads = bundle.info.num_key_value_heads;
        fp_cfg.head_dim = fp_cfg.hidden_size / std::max(fp_cfg.num_heads, 1);
        fp_cfg.attention_size = fp_cfg.num_heads * fp_cfg.head_dim;
        fp_cfg.max_cache_length = bundle.info.max_cache_length;
    }

    // Build DecoderStepEngine
    auto engine_struct = std::make_unique<trtf::DecoderStepEngine>();
    engine_struct->engine = std::move(trt_engine);
    engine_struct->context = std::move(exec_ctx);
    engine_struct->vocab_size = fp_cfg.vocab_size;
    engine_struct->hidden_size = fp_cfg.hidden_size;
    engine_struct->cache_state_size = fp_cfg.attention_size;
    engine_struct->attention_mask_size = fp_cfg.max_cache_length + 1;
    engine_struct->max_cache_length = fp_cfg.max_cache_length;
    engine_struct->num_layers = fp_cfg.num_layers;
    engine_struct->requires_position_input = true;
    engine_struct->id_bos = fp_cfg.id_bos;
    engine_struct->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        engine_struct->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        engine_struct->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        engine_struct->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        engine_struct->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }

    if (!trtf::has_all_required_tensors(*engine_struct))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    // Extract tokenizer files to temp dir.
    // Use an owned string that auto-cleans on exception; transferred to PipelineImpl on success.
    std::string temp_dir_str;
    bool temp_dir_transferred = false;
    auto temp_dir_cleanup = [&]() {
        if (!temp_dir_transferred && !temp_dir_str.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir_str, ec);
        }
    };
    // Ensure cleanup runs on any exit path.
    struct TempDirGuard {
        std::function<void()> cleanup;
        ~TempDirGuard() { cleanup(); }
    } temp_guard{temp_dir_cleanup};

    std::unique_ptr<trtf::ITokenizer> tokenizer;
    if (tokenizer_json_data != nullptr && !tokenizer_json_data->empty())
    {
        char temp_pattern[] = "/tmp/trtfb_tok_XXXXXX";
        char* created = mkdtemp(temp_pattern);
        if (created == nullptr)
        {
            throw std::runtime_error("Failed to create temp dir for bundle tokenizer");
        }
        temp_dir_str = created;
        const std::filesystem::path temp_dir(temp_dir_str);

        // Write tokenizer.json
        {
            std::ofstream out(temp_dir / "tokenizer.json", std::ios::binary | std::ios::trunc);
            if (out)
            {
                out.write(tokenizer_json_data->data(),
                    static_cast<std::streamsize>(tokenizer_json_data->size()));
            }
        }

        // Write tokenizer_config.json if present
        if (tokenizer_config_data != nullptr && !tokenizer_config_data->empty())
        {
            std::ofstream out(temp_dir / "tokenizer_config.json", std::ios::binary | std::ios::trunc);
            if (out)
            {
                out.write(tokenizer_config_data->data(),
                    static_cast<std::streamsize>(tokenizer_config_data->size()));
            }
        }

        std::cerr << "[trtf] Initializing HF tokenizer from bundle ..." << std::endl;
        auto ttok0 = std::chrono::steady_clock::now();
        tokenizer = trtf::CreateHfPythonTokenizer(temp_dir_str, hf_python);
        auto ttok1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Tokenizer ready ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
                  << " ms]" << std::endl;
    }
    else
    {
        throw std::runtime_error("Bundle has no tokenizer.json section: " + bundle_path);
    }

    // Create backend
    auto backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, bundle)" << std::endl;

    trtf::GenerationConfig gen_config{};
    auto* pipeline = new PipelineImpl(
        bundle.info.model_id, std::move(tokenizer), std::move(backend), "trt", gen_config);
    pipeline->set_bundle_temp_dir(temp_dir_str);
    temp_dir_transferred = true; // ownership transferred to PipelineImpl
    return pipeline;
}
#endif

} // namespace

extern "C" {

trtf::IPipeline* trtf_create_pipeline(const char* model_or_bundle, int flags)
{
    TrtfPipelineOptions opts{};
    opts.flags = flags;
    opts.max_new_tokens = 0;
    opts.max_cache_length = 0;
    opts.hf_python = nullptr;
    opts.engine_cache_dir = nullptr;
    opts.no_engine_cache = 0;
    return trtf_create_pipeline_ex(model_or_bundle, &opts);
}

trtf::IPipeline* trtf_create_pipeline_ex(const char* model_or_bundle, const TrtfPipelineOptions* options)
{
    clear_last_error();

    if (model_or_bundle == nullptr || model_or_bundle[0] == '\0')
    {
        set_last_error("model_or_bundle must not be null or empty");
        return nullptr;
    }

    const int flags = options ? options->flags : TRTF_PREFER_TRT;
    const int opt_max_new_tokens = options ? options->max_new_tokens : 0;
    const int opt_max_cache_length = options ? options->max_cache_length : 0;
    const std::string hf_python = (options && options->hf_python) ? options->hf_python : "";
    const std::string engine_cache_dir = (options && options->engine_cache_dir) ? options->engine_cache_dir : "";
    const bool no_engine_cache = options ? (options->no_engine_cache != 0) : false;

    // Set thread-local engine cache config for the duration of this call.
    EngineCacheConfigGuard cache_guard(engine_cache_dir, no_engine_cache);

    try
    {
        const std::string input(model_or_bundle);

        // Check if input is a .trtfb bundle
        if (trtf::IsBundle(input))
        {
#if TRTF_HAS_TRT
            auto t0 = std::chrono::steady_clock::now();
            auto* pipeline = try_create_from_bundle(input, hf_python);
            if (pipeline != nullptr)
            {
                auto t1 = std::chrono::steady_clock::now();
                std::cerr << "[trtf] Total startup ["
                          << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                          << " ms]" << std::endl;
                return pipeline;
            }
            set_last_error("Failed to load bundle: " + input);
            return nullptr;
#else
            set_last_error("Bundle loading requires TRT support (compile with TRT)");
            return nullptr;
#endif
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
            const int32_t max_cache_override = opt_max_cache_length > 0 ? static_cast<int32_t>(opt_max_cache_length) : -1;
            auto t0 = std::chrono::steady_clock::now();
            PipelineImpl* fast = try_create_from_cached_engine(input, model_dir, max_cache_override, hf_python);
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
        trtf::ResolvedModelSpec model_spec = trtf::ResolveTextGenerationModel(input);
        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Model resolved (kind="
                  << (model_spec.kind == trtf::ResolvedModelKind::kDecoderDefinition ? "decoder-definition"
                      : "hf-local")
                  << ") ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms]"
                  << std::endl;

        // Apply max_cache_length override from API options
        if (opt_max_cache_length > 0
            && model_spec.kind == trtf::ResolvedModelKind::kDecoderDefinition)
        {
            model_spec.decoder_model.max_cache_length = opt_max_cache_length;
        }

        trtf::BackendSelection selection;
        selection.prefer_trt = prefer_trt;
        selection.force_trt = force_trt;
        selection.hf_python = hf_python;

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
        if (opt_max_new_tokens > 0)
        {
            gen_config.max_new_tokens = static_cast<std::size_t>(opt_max_new_tokens);
        }

        auto* pipeline = new PipelineImpl(
            input,
            std::move(runtime.tokenizer),
            std::move(runtime.backend),
            std::move(runtime.backend_name),
            gen_config);

        // Populate metadata for save_bundle
        if (model_spec.kind == trtf::ResolvedModelKind::kDecoderDefinition)
        {
            const std::string model_dir = resolve_model_dir_lightweight(input);
            pipeline->set_model_dir(model_dir);
            pipeline->set_family(model_spec.decoder_model.architecture.family);

            // Read model_type from config.json
            const std::filesystem::path config_path = std::filesystem::path(model_dir) / "config.json";
            if (std::filesystem::exists(config_path))
            {
                const std::string config_text = trtf::read_file(config_path);
                pipeline->set_model_type(trtf::extract_json_string(config_text, "model_type", ""));
            }

            const auto& arch = model_spec.decoder_model.architecture;
            pipeline->set_architecture_info(
                static_cast<int32_t>(model_spec.decoder_model.vocab.size()),
                model_spec.decoder_model.checkpoint.hidden_size,
                arch.num_layers,
                arch.num_attention_heads,
                arch.num_key_value_heads,
                model_spec.decoder_model.max_cache_length);
        }

        return pipeline;
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
