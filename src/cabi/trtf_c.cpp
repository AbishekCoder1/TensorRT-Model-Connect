#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/tokenizer.h"
#include "bundle/bundle_format.h"
#include "cabi/fast_path_config.h"
#include "utils/json_helpers.h"
#include "utils/text_parsers.h"
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
// Load pipeline from a .trtfb bundle file.
PipelineImpl* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python)
{
    trtf::BundleFile bundle = trtf::ReadBundleFile(bundle_path);

    // Find engine plan section
    const std::vector<char>* plan_data = nullptr;
    const std::vector<char>* vision_plan_data = nullptr;
    const std::vector<char>* config_json_data = nullptr;
    const std::vector<char>* preprocessor_config_data = nullptr;
    const std::vector<char>* tokenizer_json_data = nullptr;
    const std::vector<char>* tokenizer_config_data = nullptr;
    const std::vector<char>* vocab_json_data = nullptr;
    const std::vector<char>* merges_txt_data = nullptr;
    const std::vector<char>* special_tokens_data = nullptr;
    const std::vector<char>* tokenizer_model_data = nullptr;

    for (const auto& section : bundle.sections)
    {
        if (section.name == "engine_plan") plan_data = &section.data;
        else if (section.name == "vision_engine_plan") vision_plan_data = &section.data;
        else if (section.name == "config.json") config_json_data = &section.data;
        else if (section.name == "preprocessor_config.json") preprocessor_config_data = &section.data;
        else if (section.name == "tokenizer.json") tokenizer_json_data = &section.data;
        else if (section.name == "tokenizer_config.json") tokenizer_config_data = &section.data;
        else if (section.name == "vocab.json") vocab_json_data = &section.data;
        else if (section.name == "merges.txt") merges_txt_data = &section.data;
        else if (section.name == "special_tokens_map.json") special_tokens_data = &section.data;
        else if (section.name == "tokenizer.model") tokenizer_model_data = &section.data;
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

    // Build engine struct based on runtime strategy.
    const auto& strategy = fp_cfg.runtime_strategy;

    // Mamba/SSM path: use MambaStepEngine
    if (strategy == "ssm_recurrent")
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

        // Skip tokenizer/backend creation below, handle inline here
        // Extract tokenizer files to temp dir
        std::string mamba_temp_dir_str;
        bool mamba_temp_transferred = false;
        auto mamba_temp_cleanup = [&]() {
            if (!mamba_temp_transferred && !mamba_temp_dir_str.empty())
            {
                std::error_code ec;
                std::filesystem::remove_all(mamba_temp_dir_str, ec);
            }
        };
        struct MambaTempGuard {
            std::function<void()> cleanup;
            ~MambaTempGuard() { cleanup(); }
        } mamba_guard{mamba_temp_cleanup};

        std::unique_ptr<trtf::ITokenizer> mamba_tokenizer;
        bool mamba_has_tok = (tokenizer_json_data != nullptr && !tokenizer_json_data->empty())
            || (vocab_json_data != nullptr && !vocab_json_data->empty())
            || (tokenizer_model_data != nullptr && !tokenizer_model_data->empty());
        if (mamba_has_tok)
        {
            char mamba_temp_pattern[] = "/tmp/trtfb_tok_XXXXXX";
            char* mamba_created = mkdtemp(mamba_temp_pattern);
            if (mamba_created == nullptr)
            {
                throw std::runtime_error("Failed to create temp dir for bundle tokenizer");
            }
            mamba_temp_dir_str = mamba_created;
            const std::filesystem::path mamba_temp_dir(mamba_temp_dir_str);

            auto write_section_m = [&](const char* filename, const std::vector<char>* data) {
                if (data != nullptr && !data->empty())
                {
                    std::ofstream out(mamba_temp_dir / filename, std::ios::binary | std::ios::trunc);
                    if (out)
                    {
                        out.write(data->data(), static_cast<std::streamsize>(data->size()));
                    }
                }
            };

            write_section_m("tokenizer.json", tokenizer_json_data);
            write_section_m("tokenizer_config.json", tokenizer_config_data);
            write_section_m("vocab.json", vocab_json_data);
            write_section_m("merges.txt", merges_txt_data);
            write_section_m("special_tokens_map.json", special_tokens_data);
            write_section_m("tokenizer.model", tokenizer_model_data);

            std::cerr << "[trtf] Initializing HF tokenizer from bundle ..." << std::endl;
            auto ttok0 = std::chrono::steady_clock::now();
            mamba_tokenizer = trtf::CreateHfPythonTokenizer(mamba_temp_dir_str, hf_python);
            auto ttok1 = std::chrono::steady_clock::now();
            std::cerr << "[trtf] Tokenizer ready ["
                      << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
                      << " ms]" << std::endl;
        }
        else
        {
            throw std::runtime_error("Bundle has no tokenizer files: " + bundle_path);
        }

        auto mamba_backend = trtf::CreateMambaBackendFromEngine(std::move(mamba_engine));
        if (!mamba_backend || !mamba_backend->is_available())
        {
            throw std::runtime_error("Failed to create Mamba TRT backend from bundle engine");
        }

        std::cerr << "[trtf] Runtime ready (backend=trt_mamba, strategy=" << strategy << ")" << std::endl;

        trtf::GenerationConfig gen_config{};
        auto* pipeline = new PipelineImpl(
            bundle.info.model_id, std::move(mamba_tokenizer), std::move(mamba_backend), "trt_mamba", gen_config);
        pipeline->set_bundle_temp_dir(mamba_temp_dir_str);
        mamba_temp_transferred = true;
        return pipeline;
    }

    // Vision-Language path: two-engine (vision encoder + text decoder)
    if (strategy == "vision_language")
    {
        // Build text decoder engine struct (same as standard decoder)
        auto decoder_engine = std::make_unique<trtf::DecoderStepEngine>();
        decoder_engine->engine = std::move(trt_engine);
        decoder_engine->context = std::move(exec_ctx);
        decoder_engine->vocab_size = fp_cfg.vocab_size;
        decoder_engine->hidden_size = fp_cfg.hidden_size;
        decoder_engine->cache_state_size = fp_cfg.attention_size;
        decoder_engine->attention_mask_size = fp_cfg.max_cache_length + 1;
        decoder_engine->max_cache_length = fp_cfg.max_cache_length;
        decoder_engine->num_layers = fp_cfg.num_layers;
        decoder_engine->requires_position_input = true;
        decoder_engine->id_bos = fp_cfg.id_bos;
        decoder_engine->id_eos = fp_cfg.id_eos;

        for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
        {
            decoder_engine->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
            decoder_engine->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
            decoder_engine->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
            decoder_engine->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
        }

        if (!trtf::has_all_required_tensors(*decoder_engine))
        {
            throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
        }

        // Deserialize vision engine (if present in the bundle)
        std::unique_ptr<trtf::VisionStepEngine> vision_step_engine;
        if (vision_plan_data != nullptr && !vision_plan_data->empty())
        {
            std::cerr << "[trtf] Deserializing vision TRT engine from bundle ("
                      << vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

            auto vision_trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
                runtime_ptr->deserializeCudaEngine(vision_plan_data->data(), vision_plan_data->size()));
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

        // Parse VL preprocessing config from bundle sections
        std::string config_text_vl;
        if (config_json_data != nullptr && !config_json_data->empty())
        {
            config_text_vl.assign(config_json_data->begin(), config_json_data->end());
        }
        std::string preproc_text;
        if (preprocessor_config_data != nullptr && !preprocessor_config_data->empty())
        {
            preproc_text.assign(preprocessor_config_data->begin(), preprocessor_config_data->end());
        }
        trtf::VLPreprocessConfig vl_preproc = trtf::parse_vl_preprocess_config(
            config_text_vl, preproc_text);

        // Extract tokenizer files to temp dir
        std::string vl_temp_dir_str;
        bool vl_temp_transferred = false;
        auto vl_temp_cleanup = [&]() {
            if (!vl_temp_transferred && !vl_temp_dir_str.empty())
            {
                std::error_code ec;
                std::filesystem::remove_all(vl_temp_dir_str, ec);
            }
        };
        struct VlTempGuard {
            std::function<void()> cleanup;
            ~VlTempGuard() { cleanup(); }
        } vl_guard{vl_temp_cleanup};

        std::unique_ptr<trtf::ITokenizer> vl_tokenizer;
        bool vl_has_tok = (tokenizer_json_data != nullptr && !tokenizer_json_data->empty())
            || (vocab_json_data != nullptr && !vocab_json_data->empty())
            || (tokenizer_model_data != nullptr && !tokenizer_model_data->empty());
        if (vl_has_tok)
        {
            char vl_temp_pattern[] = "/tmp/trtfb_tok_XXXXXX";
            char* vl_created = mkdtemp(vl_temp_pattern);
            if (vl_created == nullptr)
            {
                throw std::runtime_error("Failed to create temp dir for bundle tokenizer");
            }
            vl_temp_dir_str = vl_created;
            const std::filesystem::path vl_temp_dir(vl_temp_dir_str);

            auto write_section_vl = [&](const char* filename, const std::vector<char>* data) {
                if (data != nullptr && !data->empty())
                {
                    std::ofstream out(vl_temp_dir / filename, std::ios::binary | std::ios::trunc);
                    if (out)
                    {
                        out.write(data->data(), static_cast<std::streamsize>(data->size()));
                    }
                }
            };

            write_section_vl("tokenizer.json", tokenizer_json_data);
            write_section_vl("tokenizer_config.json", tokenizer_config_data);
            write_section_vl("vocab.json", vocab_json_data);
            write_section_vl("merges.txt", merges_txt_data);
            write_section_vl("special_tokens_map.json", special_tokens_data);
            write_section_vl("tokenizer.model", tokenizer_model_data);

            // Also write preprocessor_config.json for image preprocessing
            write_section_vl("preprocessor_config.json", preprocessor_config_data);

            std::cerr << "[trtf] Initializing HF tokenizer from bundle ..." << std::endl;
            auto ttok0 = std::chrono::steady_clock::now();
            vl_tokenizer = trtf::CreateHfPythonTokenizer(vl_temp_dir_str, hf_python);
            auto ttok1 = std::chrono::steady_clock::now();
            std::cerr << "[trtf] Tokenizer ready ["
                      << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
                      << " ms]" << std::endl;
        }
        else
        {
            throw std::runtime_error("Bundle has no tokenizer files: " + bundle_path);
        }

        auto vl_backend_raw = trtf::CreateVLBackendFromEngines(
            std::move(decoder_engine),
            std::move(vision_step_engine),
            fp_cfg,
            std::move(vl_preproc));
        if (!vl_backend_raw || !vl_backend_raw->is_available())
        {
            throw std::runtime_error("Failed to create VL TRT backend from bundle engine");
        }

        std::cerr << "[trtf] Runtime ready (backend=trt_vl, strategy=" << strategy << ")" << std::endl;

        trtf::GenerationConfig gen_config{};
        auto* pipeline = new PipelineImpl(
            bundle.info.model_id, std::move(vl_tokenizer), std::move(vl_backend_raw), "trt_vl", gen_config);
        pipeline->set_bundle_temp_dir(vl_temp_dir_str);
        vl_temp_transferred = true;
        return pipeline;
    }

    // Standard decoder path: build DecoderStepEngine
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
    std::string temp_dir_str;
    bool temp_dir_transferred = false;
    auto temp_dir_cleanup = [&]() {
        if (!temp_dir_transferred && !temp_dir_str.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir_str, ec);
        }
    };
    struct TempDirGuard {
        std::function<void()> cleanup;
        ~TempDirGuard() { cleanup(); }
    } temp_guard{temp_dir_cleanup};

    std::unique_ptr<trtf::ITokenizer> tokenizer;

    // Need at least tokenizer.json, vocab.json, or tokenizer.model to initialize the tokenizer.
    bool has_tokenizer_data = (tokenizer_json_data != nullptr && !tokenizer_json_data->empty())
        || (vocab_json_data != nullptr && !vocab_json_data->empty())
        || (tokenizer_model_data != nullptr && !tokenizer_model_data->empty());
    if (has_tokenizer_data)
    {
        char temp_pattern[] = "/tmp/trtfb_tok_XXXXXX";
        char* created = mkdtemp(temp_pattern);
        if (created == nullptr)
        {
            throw std::runtime_error("Failed to create temp dir for bundle tokenizer");
        }
        temp_dir_str = created;
        const std::filesystem::path temp_dir(temp_dir_str);

        auto write_section = [&](const char* filename, const std::vector<char>* data) {
            if (data != nullptr && !data->empty())
            {
                std::ofstream out(temp_dir / filename, std::ios::binary | std::ios::trunc);
                if (out)
                {
                    out.write(data->data(), static_cast<std::streamsize>(data->size()));
                }
            }
        };

        write_section("tokenizer.json", tokenizer_json_data);
        write_section("tokenizer_config.json", tokenizer_config_data);
        write_section("vocab.json", vocab_json_data);
        write_section("merges.txt", merges_txt_data);
        write_section("special_tokens_map.json", special_tokens_data);
        write_section("tokenizer.model", tokenizer_model_data);

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
        throw std::runtime_error("Bundle has no tokenizer files: " + bundle_path);
    }

    // Create backend. At this point, only decoder_kv_cache and decoder_moe reach here
    // (ssm_recurrent already returned above).
    std::unique_ptr<trtf::IGenerationBackend> backend;

    if (strategy == "decoder_kv_cache" || strategy == "decoder_moe")
    {
        backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    }
    else
    {
        throw std::runtime_error("Unsupported runtime_strategy: " + strategy
            + " (supported: decoder_kv_cache, decoder_moe, ssm_recurrent, vision_language)");
    }

    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, strategy=" << strategy << ")" << std::endl;

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
