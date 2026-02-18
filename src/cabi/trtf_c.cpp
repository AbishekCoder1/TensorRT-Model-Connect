#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/tokenizer.h"
#include "bundle/bundle_format.h"
#include "cabi/fast_path_config.h"
#include "cabi/bundle_helpers.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_backend_shared.h"
#include "runtime/trt/mamba_backend.h"
#include "runtime/trt/mamba_decode_runtime.h"
#include "runtime/trt/vl_backend.h"
#include "runtime/trt/vision_engine.h"
#include "runtime/trt/image_preprocessor.h"
#include "runtime/trt/trt_common.h"

#include <chrono>
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

        auto input_ids = mTokenizer->encode(prompt);
        auto output_ids = mBackend->generate(input_ids, config);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    const char* generate(const char* prompt, const char* image_path,
                         std::size_t max_new_tokens) override
    {
        if (!mBackend->supports_vision() || image_path == nullptr)
        {
            return generate(prompt, max_new_tokens);
        }

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

#if TRTF_HAS_TRT
        // Native VL pipeline: preprocess image in C++, run vision TRT, tokenize, generate
        auto* vl = dynamic_cast<trtf::VLBackendFastPath*>(mBackend.get());
        if (vl != nullptr)
        {
            // Step 1-2: Load image + run vision encoder (C++ native)
            std::vector<float> image_features;
            int32_t num_features = 0;
            int32_t feature_dim = 0;
            std::string prep_error;
            if (!vl->prepare_image(std::string(image_path),
                    image_features, num_features, feature_dim, prep_error))
            {
                std::cerr << "[trtf] VL image preparation failed: " << prep_error
                          << ", falling back to text-only" << std::endl;
                auto input_ids = mTokenizer->encode(prompt);
                auto output_ids = mBackend->generate(input_ids, config);
                mLastOutput = mTokenizer->decode(output_ids);
                return mLastOutput.c_str();
            }

            // Step 3: Format VL prompt with image pad tokens (C++ string)
            const std::string formatted = trtf::format_vl_prompt(
                std::string(prompt), vl->vl_config());

            // Step 4: Tokenize via Python tokenizer subprocess
            auto input_ids = mTokenizer->encode(formatted.c_str());

            // Step 5-6: Generate with image features + decode
            auto output_ids = vl->generate_vl(
                input_ids,
                image_features.data(), num_features, feature_dim,
                {}, config);
            mLastOutput = mTokenizer->decode(output_ids);
        }
        else
#endif // TRTF_HAS_TRT
        {
            auto input_ids = mTokenizer->encode(prompt);
            auto output_ids = mBackend->generate(input_ids, config);
            mLastOutput = mTokenizer->decode(output_ids);
        }
        return mLastOutput.c_str();
    }

    bool supports_vision() const override
    {
        return mBackend->supports_vision();
    }

    const char* model_id() const override
    {
        return mModelId.c_str();
    }

    const char* backend_name() const override
    {
        return mBackendName.c_str();
    }

    void set_bundle_temp_dir(std::string dir) { mBundleTempDir = std::move(dir); }

private:
    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;
    std::string mBundleTempDir;
};

#if TRTF_HAS_TRT

// Helper: assemble a PipelineImpl from backend components + tokenizer.
PipelineImpl* make_pipeline(
    const std::string& model_id,
    trtf::TokenizerResult tok,
    std::unique_ptr<trtf::IGenerationBackend> backend,
    const std::string& backend_name)
{
    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), std::move(backend), backend_name, trtf::GenerationConfig{});
    pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    return pipeline;
}

// --- Per-strategy creation functions ---

PipelineImpl* create_mamba_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto mamba_engine = std::make_unique<trtf::MambaStepEngine>();
    mamba_engine->engine = std::move(trt_engine);
    mamba_engine->context = std::move(exec_ctx);
    mamba_engine->vocab_size = fp_cfg.vocab_size;
    mamba_engine->hidden_size = fp_cfg.hidden_size;
    mamba_engine->d_inner = fp_cfg.d_inner;
    mamba_engine->state_size = fp_cfg.state_size;
    mamba_engine->conv_kernel = fp_cfg.conv_kernel;
    mamba_engine->num_layers = fp_cfg.num_layers;
    mamba_engine->id_bos = fp_cfg.id_bos;
    mamba_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        mamba_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        mamba_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        mamba_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        mamba_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    if (!trtf::has_all_required_mamba_tensors(*mamba_engine))
    {
        throw std::runtime_error("Bundle engine missing required Mamba tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateMambaBackendFromEngine(std::move(mamba_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Mamba TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_mamba, strategy=ssm_recurrent)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_mamba");
}

PipelineImpl* create_vl_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build text decoder engine
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    // Deserialize vision engine (if present)
    std::unique_ptr<trtf::VisionStepEngine> vision_step_engine;
    if (sections.vision_plan_data != nullptr && !sections.vision_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing vision TRT engine from bundle ("
                  << sections.vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

        auto vision_trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.vision_plan_data->data(), sections.vision_plan_data->size()));
        if (!vision_trt_engine)
        {
            throw std::runtime_error("Failed to deserialize vision engine from bundle: " + bundle_path);
        }

        auto vision_exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            vision_trt_engine->createExecutionContext());
        if (!vision_exec_ctx)
        {
            throw std::runtime_error("Failed to create vision execution context");
        }

        vision_step_engine = std::make_unique<trtf::VisionStepEngine>();
        vision_step_engine->engine = std::move(vision_trt_engine);
        vision_step_engine->context = std::move(vision_exec_ctx);
        vision_step_engine->num_output_features = fp_cfg.num_image_pad_tokens;
        vision_step_engine->feature_dim = (fp_cfg.vision_output_dim > 0)
            ? fp_cfg.vision_output_dim : fp_cfg.hidden_size;

        std::cerr << "[trtf] Vision engine deserialized (features="
                  << vision_step_engine->num_output_features
                  << ", dim=" << vision_step_engine->feature_dim << ")" << std::endl;
    }

    // Parse VL preprocessing config
    std::string config_text_vl;
    if (sections.config_json_data != nullptr && !sections.config_json_data->empty())
    {
        config_text_vl.assign(sections.config_json_data->begin(), sections.config_json_data->end());
    }
    std::string preproc_text;
    if (sections.preprocessor_config_data != nullptr && !sections.preprocessor_config_data->empty())
    {
        preproc_text.assign(
            sections.preprocessor_config_data->begin(), sections.preprocessor_config_data->end());
    }
    trtf::VLPreprocessConfig vl_preproc = trtf::parse_vl_preprocess_config(config_text_vl, preproc_text);

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateVLBackendFromEngines(
        std::move(decoder_engine), std::move(vision_step_engine), fp_cfg, std::move(vl_preproc));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create VL TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_vl, strategy=vision_language)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_vl");
}

PipelineImpl* create_decoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto engine_struct = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*engine_struct))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);

    const auto& strategy = fp_cfg.runtime_strategy;
    if (strategy != "decoder_kv_cache" && strategy != "decoder_moe")
    {
        throw std::runtime_error("Unsupported runtime_strategy: " + strategy
            + " (supported: decoder_kv_cache, decoder_moe, ssm_recurrent, vision_language)");
    }

    auto backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, strategy=" << strategy << ")" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt");
}

// --- Main dispatch ---

PipelineImpl* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python)
{
    trtf::BundleFile bundle = trtf::ReadBundleFile(bundle_path);
    auto sections = trtf::find_bundle_sections(bundle);

    if (sections.plan_data == nullptr || sections.plan_data->empty())
    {
        throw std::runtime_error("Bundle has no engine_plan section: " + bundle_path);
    }

    // Deserialize TRT engine
    std::cerr << "[trtf] Deserializing TRT engine from bundle ("
              << sections.plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    trtf::TrtLogger logger;
    auto runtime_ptr = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_ptr)
    {
        throw std::runtime_error("Failed to create TRT runtime");
    }

    auto tdeser0 = std::chrono::steady_clock::now();
    auto trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(sections.plan_data->data(), sections.plan_data->size()));
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
    if (sections.config_json_data != nullptr && !sections.config_json_data->empty())
    {
        const std::string config_text(sections.config_json_data->begin(), sections.config_json_data->end());
        fp_cfg = trtf::parse_fast_path_config(config_text, bundle.info.max_cache_length);
    }
    else
    {
        fp_cfg.vocab_size = bundle.info.vocab_size;
        fp_cfg.hidden_size = bundle.info.hidden_size;
        fp_cfg.num_layers = bundle.info.num_layers;
        fp_cfg.num_heads = bundle.info.num_attention_heads;
        fp_cfg.num_kv_heads = bundle.info.num_key_value_heads;
        fp_cfg.head_dim = fp_cfg.hidden_size / std::max(fp_cfg.num_heads, 1);
        fp_cfg.attention_size = fp_cfg.num_heads * fp_cfg.head_dim;
        fp_cfg.max_cache_length = bundle.info.max_cache_length;
    }

    // Dispatch to per-strategy factory
    const auto& strategy = fp_cfg.runtime_strategy;

    if (strategy == "ssm_recurrent")
    {
        return create_mamba_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "vision_language")
    {
        return create_vl_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    return create_decoder_pipeline(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg,
        sections, bundle.info.model_id, hf_python, bundle_path);
}
#endif

} // namespace

extern "C" {

trtf::IPipeline* trtf_create_pipeline(const char* bundle_path, int flags)
{
    (void) flags;
    TrtfPipelineOptions opts{};
    opts.max_new_tokens = 0;
    opts.hf_python = nullptr;
    return trtf_create_pipeline_ex(bundle_path, &opts);
}

trtf::IPipeline* trtf_create_pipeline_ex(const char* bundle_path, const TrtfPipelineOptions* options)
{
    clear_last_error();

    if (bundle_path == nullptr || bundle_path[0] == '\0')
    {
        set_last_error("bundle_path must not be null or empty");
        return nullptr;
    }

    (void)(options ? options->max_new_tokens : 0); // reserved for future use
    const std::string hf_python = (options && options->hf_python) ? options->hf_python : "";

    try
    {
        const std::string input(bundle_path);

        if (!trtf::IsBundle(input))
        {
            set_last_error("Not a valid .trtfb bundle: " + input);
            return nullptr;
        }

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
