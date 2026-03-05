#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/tokenizer.h"
#include "bundle/bundle_format.h"
#include "cabi/config/fast_path_config.h"
#include "cabi/bundle/bundle_helpers.h"
#include "cabi/registry/backend_registry.h"
#include "cabi/registry/backend_registry_dispatch.h"
#include "cabi/factories/factory_decls.h"
#include "cabi/pipeline/pipeline_impl.h"
#include "cabi/factories/factories_audio.h"
#include "cabi/factories/factories_diffusion.h"
#include "cabi/factories/factories_vision.h"
#include "cabi/factories/factories_text.h"
#include "cabi/factories/factories_encoder.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"
#include "runtime/trt/core/trt_backend_shared.h"
#include "runtime/trt/recurrent/mamba_backend.h"
#include "runtime/trt/recurrent/mamba_decode_runtime.h"
#include "runtime/trt/recurrent/rwkv_backend.h"
#include "runtime/trt/recurrent/rwkv_decode_runtime.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/multimodal/vl_backend.h"
#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/diffusion/diffusion_backend.h"
#include "runtime/trt/diffusion/wan_diffusion_backend.h"
#include "runtime/trt/diffusion/z_image_diffusion_backend.h"
#include "runtime/trt/encoder/encoder_backend.h"
#include "runtime/trt/encoder/embedding_backend.h"
#include "runtime/trt/encoder/reranking_backend.h"
#include "runtime/trt/perception/segmentation_backend.h"
#include "runtime/trt/perception/detection_backend.h"
#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/neural_operator_backend.h"
#include "runtime/trt/recurrent/hybrid_backend.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/audio/mel_spectrogram.h"

#include "stb_image_write.h"

#include "utils/data_dir.h"
#include "utils/wav_reader.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
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

bool validate_bundle_path_or_set_error(const char* bundle_path)
{
    if (bundle_path == nullptr || bundle_path[0] == '\0')
    {
        set_last_error("bundle_path must not be null or empty");
        return false;
    }
    return true;
}

std::string resolve_hf_python_path(const TrtfPipelineOptions* options)
{
    return (options != nullptr && options->hf_python != nullptr)
        ? options->hf_python
        : "";
}

bool validate_bundle_file_or_set_error(const std::string& input)
{
    if (!trtf::IsBundle(input))
    {
        set_last_error("Not a valid .trtfb bundle: " + input);
        return false;
    }
    return true;
}

#if TRTF_HAS_TRT
trtf::IPipeline* try_create_pipeline_with_options(
    const std::string& input,
    const std::string& hf_python,
    const TrtfPipelineOptions* options);
#endif

trtf::IPipeline* create_pipeline_for_bundle(
    const std::string& input,
    const std::string& hf_python,
    const TrtfPipelineOptions* options)
{
#if TRTF_HAS_TRT
    auto* pipeline = try_create_pipeline_with_options(input, hf_python, options);
    if (pipeline != nullptr)
    {
        return pipeline;
    }
    set_last_error("Failed to load bundle: " + input);
    return nullptr;
#else
    (void) input;
    (void) hf_python;
    (void) options;
    set_last_error("Bundle loading requires TRT support (compile with TRT)");
    return nullptr;
#endif
}

#if TRTF_HAS_TRT
// --- Main dispatch ---

trtf::IPipeline* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python);

bool has_config_json_data(const trtf::BundleSections& sections)
{
    return sections.config_json_data != nullptr && !sections.config_json_data->empty();
}

trtf::FastPathModelConfig parse_fast_path_config_or_empty(
    const trtf::BundleSections& sections,
    int32_t max_cache_length)
{
    trtf::FastPathModelConfig fp_cfg;
    if (has_config_json_data(sections))
    {
        const std::string config_text(
            sections.config_json_data->begin(), sections.config_json_data->end());
        fp_cfg = trtf::parse_fast_path_config(config_text, max_cache_length);
    }
    return fp_cfg;
}

void fill_fast_path_config_from_bundle_info(
    trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleInfo& info)
{
    fp_cfg.vocab_size = info.vocab_size;
    fp_cfg.hidden_size = info.hidden_size;
    fp_cfg.num_layers = info.num_layers;
    fp_cfg.num_heads = info.num_attention_heads;
    fp_cfg.num_kv_heads = info.num_key_value_heads;
    fp_cfg.head_dim = fp_cfg.hidden_size / std::max(fp_cfg.num_heads, 1);
    fp_cfg.attention_size = fp_cfg.num_heads * fp_cfg.head_dim;
    fp_cfg.max_cache_length = info.max_cache_length;
}

trtf::FastPathModelConfig parse_or_build_fast_path_config(
    const trtf::BundleSections& sections,
    const trtf::BundleInfo& info)
{
    auto fp_cfg = parse_fast_path_config_or_empty(sections, info.max_cache_length);
    if (!has_config_json_data(sections))
    {
        fill_fast_path_config_from_bundle_info(fp_cfg, info);
    }
    return fp_cfg;
}

void ensure_bundle_has_plan_data(const trtf::BundleSections& sections, const std::string& bundle_path)
{
    if (sections.plan_data == nullptr || sections.plan_data->empty())
    {
        throw std::runtime_error("Bundle has no engine_plan section: " + bundle_path);
    }
}

trtf::TrtUniquePtr<nvinfer1::IRuntime> create_runtime_or_throw()
{
    auto runtime_ptr = trtf::create_trt_runtime();
    if (!runtime_ptr)
    {
        throw std::runtime_error("Failed to create TRT runtime");
    }
    return runtime_ptr;
}

trtf::TrtUniquePtr<nvinfer1::ICudaEngine> deserialize_engine_or_throw(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const trtf::BundleSections& sections,
    const std::string& bundle_path)
{
    auto trt_engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(sections.plan_data->data(), sections.plan_data->size()));
    if (!trt_engine)
    {
        throw std::runtime_error("Failed to deserialize engine from bundle: " + bundle_path);
    }
    return trt_engine;
}

trtf::TrtUniquePtr<nvinfer1::IExecutionContext> create_execution_context_or_throw(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine>& trt_engine)
{
    auto exec_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!exec_ctx)
    {
        throw std::runtime_error("Failed to create execution context from bundle engine");
    }
    return exec_ctx;
}

trtf::cabi::BackendRegistryDispatchContext build_registry_dispatch_context(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>* runtime_ptr,
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine>* trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext>* exec_ctx,
    trtf::FastPathModelConfig* fp_cfg,
    trtf::BundleSections* sections,
    const std::string* model_id,
    const std::string* hf_python,
    const std::string* bundle_path)
{
    trtf::cabi::BackendRegistryDispatchContext registry_ctx;
    registry_ctx.runtime_ptr = runtime_ptr;
    registry_ctx.trt_engine = trt_engine;
    registry_ctx.exec_ctx = exec_ctx;
    registry_ctx.fp_cfg = fp_cfg;
    registry_ctx.sections = sections;
    registry_ctx.model_id = model_id;
    registry_ctx.hf_python = hf_python;
    registry_ctx.bundle_path = bundle_path;
    return registry_ctx;
}

std::size_t resolve_default_max_new_tokens(const TrtfPipelineOptions* options)
{
    if (options == nullptr || options->max_new_tokens <= 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(options->max_new_tokens);
}

trtf::IPipeline* try_create_pipeline_with_options(
    const std::string& input,
    const std::string& hf_python,
    const TrtfPipelineOptions* options)
{
    const auto default_max_new_tokens = resolve_default_max_new_tokens(options);
    auto t0 = std::chrono::steady_clock::now();
    auto* pipeline = try_create_from_bundle(input, hf_python);
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    if (default_max_new_tokens > 0)
    {
        trtf::cabi::detail::set_default_max_new_tokens(pipeline, default_max_new_tokens);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Total startup ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms]" << std::endl;
    return pipeline;
}

trtf::IPipeline* try_create_from_bundle(const std::string& bundle_path, const std::string& hf_python)
{
    trtf::BundleFile bundle = trtf::ReadBundleFile(bundle_path);
    auto sections = trtf::find_bundle_sections(bundle);

    // Check for diffusion bundle early (no engine_plan needed)
    // Parse config to detect strategy before engine deserialization
    auto fp_cfg_early = parse_fast_path_config_or_empty(sections, bundle.info.max_cache_length);

    if (fp_cfg_early.runtime_strategy == "diffusion")
    {
        auto runtime_ptr = create_runtime_or_throw();
        return trtf::cabi::create_diffusion_pipeline(
            fp_cfg_early, sections, runtime_ptr,
            bundle.info.model_id, hf_python, bundle_path);
    }

    ensure_bundle_has_plan_data(sections, bundle_path);

    // Deserialize TRT engine
    std::cerr << "[trtf] Deserializing TRT engine from bundle ("
              << sections.plan_data->size() / (1024 * 1024) << " MB) ..." << std::endl;

    auto runtime_ptr = create_runtime_or_throw();

    auto tdeser0 = std::chrono::steady_clock::now();
    auto trt_engine = deserialize_engine_or_throw(runtime_ptr, sections, bundle_path);
    auto tdeser1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Engine deserialized ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(tdeser1 - tdeser0).count()
              << " ms]" << std::endl;

    auto exec_ctx = create_execution_context_or_throw(trt_engine);

    // Parse config.json for model metadata
    auto fp_cfg = parse_or_build_fast_path_config(sections, bundle.info);

    // Dispatch to per-strategy factory
    const auto& strategy = fp_cfg.runtime_strategy;
    trtf::cabi::register_builtin_backend_factories_once();

    auto registry_ctx = build_registry_dispatch_context(
        &runtime_ptr, &trt_engine, &exec_ctx, &fp_cfg, &sections,
        &bundle.info.model_id, &hf_python, &bundle_path);

    if (auto* pipeline = trtf::cabi::try_create_pipeline_from_registry(strategy, &registry_ctx))
    {
        return pipeline;
    }

    return trtf::cabi::create_decoder_pipeline(
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
    opts.image_path = nullptr;
    return trtf_create_pipeline_ex(bundle_path, &opts);
}

trtf::IPipeline* trtf_create_pipeline_ex(const char* bundle_path, const TrtfPipelineOptions* options)
{
    clear_last_error();

    if (!validate_bundle_path_or_set_error(bundle_path))
    {
        return nullptr;
    }

    const std::string hf_python = resolve_hf_python_path(options);

    try
    {
        const std::string input(bundle_path);
        if (!validate_bundle_file_or_set_error(input))
        {
            return nullptr;
        }
        return create_pipeline_for_bundle(input, hf_python, options);
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
