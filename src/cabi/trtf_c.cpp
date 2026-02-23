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
#include "runtime/trt/rwkv_backend.h"
#include "runtime/trt/rwkv_decode_runtime.h"
#include "runtime/trt/whisper_backend.h"
#include "runtime/trt/vl_backend.h"
#include "runtime/trt/vision_engine.h"
#include "runtime/trt/image_preprocessor.h"
#include "runtime/trt/diffusion_backend.h"
#include "runtime/trt/wan_diffusion_backend.h"
#include "runtime/trt/segmentation_backend.h"
#include "runtime/trt/bark_backend.h"
#include "runtime/trt/trt_common.h"

#include "stb_image_write.h"

#include <chrono>
#include <cstdio>
#include <cstring>
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

    bool supports_video() const override
    {
#if TRTF_HAS_TRT
        auto* diff = dynamic_cast<trtf::IDiffusionBackend*>(mBackend.get());
        return diff != nullptr;
#else
        return false;
#endif
    }

    int32_t generate_video(const char* prompt, const char* output_dir,
                           int32_t num_steps, float guidance_scale) override
    {
#if TRTF_HAS_TRT
        auto* diff = dynamic_cast<trtf::IDiffusionBackend*>(mBackend.get());
        if (diff == nullptr)
        {
            return -1;
        }

        // Tokenize prompt
        std::vector<int32_t> input_ids;
        if (mTokenizer && prompt != nullptr)
        {
            input_ids = mTokenizer->encode(prompt);
        }

        // Generate video
        auto video = diff->generate_video(input_ids, num_steps, guidance_scale);
        if (video.frames.empty() || video.num_frames <= 0)
        {
            return -1;
        }

        // Write PNG frames
        const std::string dir(output_dir != nullptr ? output_dir : "/tmp/trtf_frames");
        std::filesystem::create_directories(dir);

        for (int32_t t = 0; t < video.num_frames; ++t)
        {
            const auto frame_size = static_cast<std::size_t>(video.height) *
                                    static_cast<std::size_t>(video.width) * 3;
            const float* frame_f = video.frames.data() +
                static_cast<std::size_t>(t) * frame_size;

            // Convert float32 [0,1] -> uint8 [0,255]
            std::vector<uint8_t> pixels(frame_size);
            for (std::size_t i = 0; i < frame_size; ++i)
            {
                pixels[i] = static_cast<uint8_t>(
                    std::max(0.0F, std::min(255.0F, frame_f[i] * 255.0F)));
            }

            char fname[64];
            std::snprintf(fname, sizeof(fname), "/frame_%04d.png", t);
            const std::string path = dir + fname;

            stbi_write_png(path.c_str(), video.width, video.height, 3,
                           pixels.data(), video.width * 3);
        }

        std::cerr << "[trtf] Wrote " << video.num_frames << " PNG frames to "
                  << dir << std::endl;
        return video.num_frames;
#else
        (void) prompt; (void) output_dir; (void) num_steps; (void) guidance_scale;
        return -1;
#endif
    }

    const char* model_id() const override
    {
        return mModelId.c_str();
    }

    const char* backend_name() const override
    {
        return mBackendName.c_str();
    }

    bool supports_segmentation() const override
    {
#if TRTF_HAS_TRT
        return mSegBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t segment(const char* image_path, const char* output_path) override
    {
#if TRTF_HAS_TRT
        if (mSegBackend == nullptr || image_path == nullptr || output_path == nullptr)
            return -1;

        try
        {
            auto result = mSegBackend->segment_image(std::string(image_path));

            // Write PNG with raw class indices (NOT scaled to 255).
            // Each pixel stores the class index as a grayscale value.
            const int32_t w = result.width;
            const int32_t h = result.height;
            std::vector<uint8_t> pixels(static_cast<std::size_t>(w) * h);
            for (int32_t i = 0; i < w * h; ++i)
            {
                // Store raw class index (0-149 for ADE20K)
                pixels[i] = static_cast<uint8_t>(
                    std::min(result.class_map[i], 255));
            }

            stbi_write_png(output_path, w, h, 1, pixels.data(), w);
            std::cerr << "[trtf] Segmentation saved: " << output_path
                      << " (" << w << "x" << h << ", "
                      << result.num_classes << " classes)" << std::endl;
            return 0;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Segmentation error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) image_path; (void) output_path;
        return -1;
#endif
    }

    bool supports_audio() const override
    {
#if TRTF_HAS_TRT
        return mBarkBackend != nullptr;
#else
        return false;
#endif
    }

    int32_t generate_audio(const char* prompt, const char* output_path,
                           int32_t max_tokens) override
    {
#if TRTF_HAS_TRT
        if (mBarkBackend == nullptr || prompt == nullptr || output_path == nullptr)
            return -1;

        try
        {
            std::vector<int32_t> input_ids;
            if (mTokenizer)
            {
                input_ids = mTokenizer->encode(prompt);
            }

            auto result = mBarkBackend->generate_audio(
                input_ids, max_tokens > 0 ? max_tokens : 768);

            if (result.num_samples <= 0)
                return -1;

            trtf::write_wav(std::string(output_path),
                result.waveform.data(), result.num_samples, result.sample_rate);

            std::cerr << "[trtf] Audio saved: " << output_path
                      << " (" << result.num_samples << " samples @ "
                      << result.sample_rate << " Hz)" << std::endl;
            return result.num_samples;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[trtf] Audio generation error: " << e.what() << std::endl;
            return -1;
        }
#else
        (void) prompt; (void) output_path; (void) max_tokens;
        return -1;
#endif
    }

    void set_bundle_temp_dir(std::string dir) { mBundleTempDir = std::move(dir); }

#if TRTF_HAS_TRT
    void set_seg_backend(std::unique_ptr<trtf::SegmentationBackend> backend)
    {
        mSegBackend = std::move(backend);
    }
    void set_bark_backend(std::unique_ptr<trtf::BarkBackend> backend)
    {
        mBarkBackend = std::move(backend);
    }
#endif

private:
    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;
    std::string mBundleTempDir;
#if TRTF_HAS_TRT
    std::unique_ptr<trtf::SegmentationBackend> mSegBackend;
    std::unique_ptr<trtf::BarkBackend> mBarkBackend;
#endif
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

PipelineImpl* create_rwkv_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto rwkv_engine = std::make_unique<trtf::RwkvStepEngine>();
    rwkv_engine->engine = std::move(trt_engine);
    rwkv_engine->context = std::move(exec_ctx);
    rwkv_engine->vocab_size = fp_cfg.vocab_size;
    rwkv_engine->hidden_size = fp_cfg.hidden_size;
    rwkv_engine->num_layers = fp_cfg.num_layers;
    rwkv_engine->id_bos = fp_cfg.id_bos;
    rwkv_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        rwkv_engine->attn_state_input_names.push_back(trtf::layer_tensor_name("attn_state", i));
        rwkv_engine->ff_state_input_names.push_back(trtf::layer_tensor_name("ff_state", i));
        rwkv_engine->num_state_input_names.push_back(trtf::layer_tensor_name("num_state", i));
        rwkv_engine->den_state_input_names.push_back(trtf::layer_tensor_name("den_state", i));
        rwkv_engine->max_state_input_names.push_back(trtf::layer_tensor_name("max_state", i));
        rwkv_engine->present_attn_output_names.push_back(trtf::layer_tensor_name("present_attn", i));
        rwkv_engine->present_ff_output_names.push_back(trtf::layer_tensor_name("present_ff", i));
        rwkv_engine->present_num_output_names.push_back(trtf::layer_tensor_name("present_num", i));
        rwkv_engine->present_den_output_names.push_back(trtf::layer_tensor_name("present_den", i));
        rwkv_engine->present_max_output_names.push_back(trtf::layer_tensor_name("present_max", i));
    }

    if (!trtf::has_all_required_rwkv_tensors(*rwkv_engine))
    {
        throw std::runtime_error("Bundle engine missing required RWKV tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateRwkvBackendFromEngine(std::move(rwkv_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create RWKV TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_rwkv, strategy=rwkv_recurrent)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_rwkv");
}

PipelineImpl* create_whisper_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build text decoder engine (standard KV cache decoder)
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required decoder tensors: " + bundle_path);
    }

    // Deserialize encoder engine from vision_engine_plan section
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> encoder_engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> encoder_ctx;
    if (sections.vision_plan_data != nullptr && !sections.vision_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing Whisper encoder TRT engine ("
                  << sections.vision_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

        encoder_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.vision_plan_data->data(), sections.vision_plan_data->size()));
        if (!encoder_engine)
        {
            throw std::runtime_error("Failed to deserialize Whisper encoder engine: " + bundle_path);
        }
        encoder_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            encoder_engine->createExecutionContext());
        if (!encoder_ctx)
        {
            throw std::runtime_error("Failed to create Whisper encoder execution context");
        }
    }

    trtf::WhisperConfig whisper_cfg;
    whisper_cfg.num_mel_bins = fp_cfg.num_mel_bins;
    whisper_cfg.max_source_positions = fp_cfg.max_source_positions;
    whisper_cfg.max_target_positions = fp_cfg.max_target_positions;
    whisper_cfg.encoder_layers = fp_cfg.encoder_layers;
    whisper_cfg.decoder_layers = fp_cfg.decoder_layers;

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateWhisperBackend(
        std::move(decoder_engine), std::move(encoder_engine), std::move(encoder_ctx),
        whisper_cfg, fp_cfg);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Whisper TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_whisper, strategy=speech_to_text)" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_whisper");
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

PipelineImpl* create_segmentation_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& model_id,
    const std::string& bundle_path)
{
    auto seg_backend = trtf::CreateSegmentationBackend(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!seg_backend || !seg_backend->is_available())
    {
        throw std::runtime_error("Failed to create segmentation backend from bundle engine");
    }

    // Segmentation pipelines don't need a tokenizer or generation backend
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, nullptr, "trt_segmentation", trtf::GenerationConfig{});
    pipeline->set_seg_backend(std::move(seg_backend));

    std::cerr << "[trtf] Runtime ready (backend=trt_segmentation, strategy=segmentation)" << std::endl;
    return pipeline;
}

PipelineImpl* create_bark_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    // Build semantic decoder engine (primary engine)
    auto semantic_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*semantic_engine))
    {
        throw std::runtime_error("Bundle engine missing required semantic tensors: " + bundle_path);
    }

    // Override vocab_size from engine output shape.
    // Bark semantic: input_vocab=129600 but lm_head outputs only 10048.
    // The config.json vocab_size is the input embedding size, not the output.
    {
        auto logits_shape = semantic_engine->engine->getTensorShape("logits");
        if (logits_shape.nbDims >= 2)
        {
            int32_t actual_vocab = logits_shape.d[logits_shape.nbDims - 1];
            if (actual_vocab > 0 && actual_vocab != semantic_engine->vocab_size)
            {
                std::cerr << "[trtf] Semantic: output vocab " << actual_vocab
                          << " (config says " << semantic_engine->vocab_size << ")" << std::endl;
                semantic_engine->vocab_size = actual_vocab;
            }
        }
    }

    // Load embedding tables from bundle sections
    std::vector<float> semantic_embed;
    if (sections.semantic_embed_data != nullptr && !sections.semantic_embed_data->empty())
    {
        const auto n_floats = sections.semantic_embed_data->size() / sizeof(float);
        semantic_embed.resize(n_floats);
        std::memcpy(semantic_embed.data(), sections.semantic_embed_data->data(),
                     sections.semantic_embed_data->size());
        std::cerr << "[trtf] Loaded semantic embedding table ("
                  << n_floats / std::max(fp_cfg.hidden_size, 1) << " x "
                  << fp_cfg.hidden_size << ")" << std::endl;
    }
    else
    {
        throw std::runtime_error("Bundle missing semantic_embed section: " + bundle_path);
    }

    std::vector<float> coarse_embed;
    if (sections.coarse_embed_data != nullptr && !sections.coarse_embed_data->empty())
    {
        const auto n_floats = sections.coarse_embed_data->size() / sizeof(float);
        coarse_embed.resize(n_floats);
        std::memcpy(coarse_embed.data(), sections.coarse_embed_data->data(),
                     sections.coarse_embed_data->size());
        std::cerr << "[trtf] Loaded coarse embedding table ("
                  << n_floats / std::max(fp_cfg.hidden_size, 1) << " x "
                  << fp_cfg.hidden_size << ")" << std::endl;
    }
    else
    {
        throw std::runtime_error("Bundle missing coarse_embed section: " + bundle_path);
    }

    // Deserialize coarse engine
    std::unique_ptr<trtf::DecoderStepEngine> coarse_engine;
    if (sections.coarse_engine_plan_data != nullptr && !sections.coarse_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing coarse TRT engine ("
                  << sections.coarse_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto coarse_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.coarse_engine_plan_data->data(),
                sections.coarse_engine_plan_data->size()));
        if (!coarse_trt)
        {
            throw std::runtime_error("Failed to deserialize coarse engine: " + bundle_path);
        }
        auto coarse_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            coarse_trt->createExecutionContext());
        if (!coarse_ctx)
        {
            throw std::runtime_error("Failed to create coarse execution context");
        }

        // Build coarse-specific config (may differ from semantic)
        trtf::FastPathModelConfig coarse_cfg = fp_cfg;
        if (fp_cfg.coarse_hidden_size > 0)
            coarse_cfg.hidden_size = fp_cfg.coarse_hidden_size;
        if (fp_cfg.coarse_num_layers > 0)
            coarse_cfg.num_layers = fp_cfg.coarse_num_layers;
        if (fp_cfg.coarse_num_heads > 0)
        {
            coarse_cfg.num_heads = fp_cfg.coarse_num_heads;
            coarse_cfg.num_kv_heads = fp_cfg.coarse_num_heads;
        }
        coarse_cfg.vocab_size = fp_cfg.coarse_input_vocab;
        coarse_cfg.head_dim = coarse_cfg.hidden_size / std::max(coarse_cfg.num_heads, 1);
        coarse_cfg.attention_size = coarse_cfg.num_heads * coarse_cfg.head_dim;
        coarse_cfg.max_cache_length = fp_cfg.coarse_max_cache_length;

        coarse_engine = trtf::make_decoder_engine(
            std::move(coarse_trt), std::move(coarse_ctx), coarse_cfg);
        if (!trtf::has_all_required_tensors(*coarse_engine))
        {
            throw std::runtime_error("Bundle coarse engine missing required tensors: " + bundle_path);
        }
    }
    else
    {
        throw std::runtime_error("Bundle missing coarse_engine section: " + bundle_path);
    }

    // Create BarkBackend with both engines + embedding tables
    auto bark_backend = trtf::CreateBarkBackend(
        std::move(semantic_engine), std::move(coarse_engine),
        std::move(semantic_embed), std::move(coarse_embed), fp_cfg);
    if (!bark_backend || !bark_backend->is_available())
    {
        throw std::runtime_error("Failed to create Bark backend from bundle engines");
    }

    // Deserialize optional codec engine
    if (sections.codec_engine_plan_data != nullptr && !sections.codec_engine_plan_data->empty())
    {
        auto codec_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.codec_engine_plan_data->data(),
                sections.codec_engine_plan_data->size()));
        if (codec_trt)
        {
            auto codec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                codec_trt->createExecutionContext());
            bark_backend->set_codec_engine(std::move(codec_trt), std::move(codec_ctx));
        }
    }

    // Deserialize optional fine engine
    if (sections.fine_engine_plan_data != nullptr && !sections.fine_engine_plan_data->empty())
    {
        std::cerr << "[trtf] Deserializing fine TRT engine ("
                  << sections.fine_engine_plan_data->size() / (1024 * 1024)
                  << " MB) ..." << std::endl;

        auto fine_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(
                sections.fine_engine_plan_data->data(),
                sections.fine_engine_plan_data->size()));
        if (fine_trt)
        {
            auto fine_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                fine_trt->createExecutionContext());
            bark_backend->set_fine_engine(std::move(fine_trt), std::move(fine_ctx));
        }
    }

    // Load fine embedding tables
    if (sections.fine_embed_data != nullptr && !sections.fine_embed_data->empty())
    {
        const auto n_floats = sections.fine_embed_data->size() / sizeof(float);
        std::vector<float> fine_embed(n_floats);
        std::memcpy(fine_embed.data(), sections.fine_embed_data->data(),
                     sections.fine_embed_data->size());

        std::vector<float> fine_pos_embed;
        if (sections.fine_position_embed_data != nullptr &&
            !sections.fine_position_embed_data->empty())
        {
            const auto pos_n = sections.fine_position_embed_data->size() / sizeof(float);
            fine_pos_embed.resize(pos_n);
            std::memcpy(fine_pos_embed.data(),
                         sections.fine_position_embed_data->data(),
                         sections.fine_position_embed_data->size());
        }

        bark_backend->set_fine_embeddings(std::move(fine_embed),
                                           std::move(fine_pos_embed));
        std::cerr << "[trtf] Loaded fine embedding tables ("
                  << n_floats << " floats)" << std::endl;
    }

    // Tokenizer (optional for Bark)
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for Bark (" << e.what() << ")" << std::endl;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_bark, strategy=text_to_audio)" << std::endl;

    auto* pipeline = new PipelineImpl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_bark", trtf::GenerationConfig{});
    if (!tok.temp_dir.empty())
        pipeline->set_bundle_temp_dir(std::move(tok.temp_dir));
    pipeline->set_bark_backend(std::move(bark_backend));
    return pipeline;
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
            + " (supported: decoder_kv_cache, decoder_moe, ssm_recurrent, rwkv_recurrent, speech_to_text, vision_language)");
    }

    auto backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, strategy=" << strategy << ")" << std::endl;
    return make_pipeline(model_id, std::move(tok), std::move(backend), "trt");
}

PipelineImpl* create_diffusion_pipeline(
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    std::cerr << "[trtf] Creating diffusion pipeline ..." << std::endl;

    // Deserialize text encoder engines
    std::vector<trtf::DiffusionEngine> text_encoders;
    for (std::size_t i = 0; i < sections.text_encoder_plans.size(); ++i)
    {
        const auto* plan = sections.text_encoder_plans[i];
        if (plan == nullptr || plan->empty()) continue;

        std::cerr << "[trtf] Deserializing text encoder " << i
                  << " (" << plan->size() / (1024 * 1024) << " MB) ..." << std::endl;

        trtf::DiffusionEngine te;
        te.name = "text_encoder_" + std::to_string(i);
        te.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_ptr->deserializeCudaEngine(plan->data(), plan->size()));
        if (!te.engine)
            throw std::runtime_error("Failed to deserialize text encoder " + std::to_string(i));
        te.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
            te.engine->createExecutionContext());
        if (!te.context)
            throw std::runtime_error("Failed to create text encoder context " + std::to_string(i));
        text_encoders.push_back(std::move(te));
    }

    // Deserialize denoiser engine
    if (sections.denoiser_plan_data == nullptr || sections.denoiser_plan_data->empty())
        throw std::runtime_error("Bundle has no denoiser_plan section: " + bundle_path);

    std::cerr << "[trtf] Deserializing denoiser ("
              << sections.denoiser_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    trtf::DiffusionEngine denoiser;
    denoiser.name = "denoiser";
    denoiser.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.denoiser_plan_data->data(), sections.denoiser_plan_data->size()));
    if (!denoiser.engine)
        throw std::runtime_error("Failed to deserialize denoiser engine");
    denoiser.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        denoiser.engine->createExecutionContext());

    // Deserialize VAE decoder engine
    if (sections.vae_decoder_plan_data == nullptr || sections.vae_decoder_plan_data->empty())
        throw std::runtime_error("Bundle has no vae_decoder_plan section: " + bundle_path);

    std::cerr << "[trtf] Deserializing VAE decoder ("
              << sections.vae_decoder_plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    trtf::DiffusionEngine vae_decoder;
    vae_decoder.name = "vae_decoder";
    vae_decoder.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            sections.vae_decoder_plan_data->data(), sections.vae_decoder_plan_data->size()));
    if (!vae_decoder.engine)
        throw std::runtime_error("Failed to deserialize VAE decoder engine");
    vae_decoder.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        vae_decoder.engine->createExecutionContext());

    // Create backend
    auto backend = trtf::CreateDiffusionBackend(
        std::move(text_encoders), std::move(denoiser), std::move(vae_decoder), fp_cfg);
    if (!backend || !backend->is_available())
        throw std::runtime_error("Failed to create diffusion backend");

    // Load preprocessor weights (patch embedding, timestep MLP, text projection)
    if (sections.preprocessor_weights_data != nullptr &&
        !sections.preprocessor_weights_data->empty())
    {
        auto pp_weights = trtf::parse_preprocessor_weights(*sections.preprocessor_weights_data);
        backend->set_preprocessor_weights(std::move(pp_weights));
    }

    // Set paths for VAE subprocess and bundle info
    backend->set_hf_python(hf_python);
    backend->set_bundle_path(bundle_path);

    // Tokenizer (optional for diffusion — some models use sentencepiece)
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer in bundle (" << e.what() << ")" << std::endl;
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_diffusion, strategy=diffusion)" << std::endl;

    if (tok.tokenizer)
    {
        return make_pipeline(model_id, std::move(tok), std::move(backend), "trt_diffusion");
    }

    // No tokenizer — create pipeline with a null tokenizer
    auto* pipeline = new PipelineImpl(
        model_id, nullptr, std::move(backend), "trt_diffusion", trtf::GenerationConfig{});
    return pipeline;
}

// --- Main dispatch ---

PipelineImpl* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python)
{
    trtf::BundleFile bundle = trtf::ReadBundleFile(bundle_path);
    auto sections = trtf::find_bundle_sections(bundle);

    // Check for diffusion bundle early (no engine_plan needed)
    // Parse config to detect strategy before engine deserialization
    trtf::FastPathModelConfig fp_cfg_early;
    if (sections.config_json_data != nullptr && !sections.config_json_data->empty())
    {
        const std::string config_text_early(
            sections.config_json_data->begin(), sections.config_json_data->end());
        fp_cfg_early = trtf::parse_fast_path_config(config_text_early, bundle.info.max_cache_length);
    }

    if (fp_cfg_early.runtime_strategy == "diffusion")
    {
        trtf::TrtLogger logger;
        auto runtime_ptr = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime_ptr) throw std::runtime_error("Failed to create TRT runtime");
        return create_diffusion_pipeline(
            fp_cfg_early, sections, runtime_ptr,
            bundle.info.model_id, hf_python, bundle_path);
    }

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

    if (strategy == "rwkv_recurrent")
    {
        return create_rwkv_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "speech_to_text")
    {
        return create_whisper_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "vision_language")
    {
        return create_vl_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "segmentation")
    {
        return create_segmentation_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            bundle.info.model_id, bundle_path);
    }

    if (strategy == "text_to_audio")
    {
        return create_bark_pipeline(
            std::move(trt_engine), std::move(exec_ctx), fp_cfg,
            sections, runtime_ptr, bundle.info.model_id, hf_python, bundle_path);
    }

    if (strategy == "diffusion")
    {
        return create_diffusion_pipeline(
            fp_cfg, sections, runtime_ptr,
            bundle.info.model_id, hf_python, bundle_path);
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
