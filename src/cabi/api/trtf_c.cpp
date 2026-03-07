#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "bundle/bundle_format.h"
#include "cabi/config/fast_path_config.h"
#include "cabi/bundle/bundle_helpers.h"
#include "runtime/adapters/bundle/bundle_port_adapter.h"
#include "runtime/adapters/trt/trt_port_adapter.h"
#include "trtf/runtime/builders/audio/audio_strategy_builder.h"
#include "trtf/runtime/builders/diffusion/diffusion_strategy_builder.h"
#include "trtf/runtime/builders/encoder/encoder_strategy_builder.h"
#include "trtf/runtime/builders/text/text_strategy_builder.h"
#include "trtf/runtime/builders/vision/vision_strategy_builder.h"
#include "trtf/runtime/pipeline/router.h"
#include "runtime/trt/core/trt_engine_lifecycle.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

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

trtf::TrtUniquePtr<nvinfer1::IRuntime> create_runtime_or_throw()
{
    auto runtime_ptr = trtf::create_trt_runtime();
    if (!runtime_ptr)
    {
        throw std::runtime_error("Failed to create TRT runtime");
    }
    return runtime_ptr;
}

std::size_t resolve_default_max_new_tokens(const TrtfPipelineOptions* options)
{
    if (options == nullptr || options->max_new_tokens <= 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(options->max_new_tokens);
}

enum class StrategyFamily {
    kUnknown = 0,
    kText,
    kEncoder,
    kVision,
    kAudio,
    kDiffusion,
};

StrategyFamily resolve_strategy_family(const std::string& strategy)
{
    using Family = StrategyFamily;
    static const std::unordered_map<std::string, Family> kStrategyFamilies = {
        {"decoder_kv_cache", Family::kText},
        {"decoder_moe", Family::kText},
        {"ssm_recurrent", Family::kText},
        {"rwkv_recurrent", Family::kText},
        {"hybrid_mamba_attention", Family::kText},
        {"encoder_only", Family::kEncoder},
        {"embedding", Family::kEncoder},
        {"reranking", Family::kEncoder},
        {"vision_language", Family::kVision},
        {"segmentation", Family::kVision},
        {"prompted_segmentation", Family::kVision},
        {"object_detection", Family::kVision},
        {"neural_operator", Family::kVision},
        {"text_to_audio", Family::kAudio},
        {"speech_to_text", Family::kAudio},
        {"speech_to_speech", Family::kAudio},
        {"omni_multimodal", Family::kAudio},
        {"diffusion", Family::kDiffusion},
    };

    const auto it = kStrategyFamilies.find(strategy);
    return (it == kStrategyFamilies.end()) ? Family::kUnknown : it->second;
}

trtf::runtime::BuildResult build_services_with_new_runtime(
    const trtf::runtime::BuildContext& context,
    const trtf::runtime::IBundlePort& bundle_port,
    const trtf::runtime::ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime)
{
    using trtf::runtime::BuildResult;
    using trtf::runtime::BuildStatus;

    switch (resolve_strategy_family(context.strategy))
    {
    case StrategyFamily::kText:
    {
        trtf::runtime::builders::text::TextStrategyBuilder builder(bundle_port, trt_port, runtime);
        return builder.build(context);
    }
    case StrategyFamily::kEncoder:
    {
        trtf::runtime::builders::encoder::EncoderStrategyBuilder builder(bundle_port, trt_port, runtime);
        return builder.build(context);
    }
    case StrategyFamily::kVision:
    {
        trtf::runtime::builders::vision::VisionStrategyBuilder builder(bundle_port, trt_port, runtime);
        return builder.build(context);
    }
    case StrategyFamily::kAudio:
    {
        trtf::runtime::builders::audio::AudioStrategyBuilder builder(bundle_port, trt_port, runtime);
        return builder.build(context);
    }
    case StrategyFamily::kDiffusion:
    {
        trtf::runtime::builders::diffusion::DiffusionStrategyBuilder builder(bundle_port, trt_port, runtime);
        return builder.build(context);
    }
    case StrategyFamily::kUnknown:
    default:
        return BuildResult::Failure(
            BuildStatus::kUnsupportedStrategy,
            "Unsupported runtime_strategy for new runtime path: " + context.strategy);
    }
}

trtf::IPipeline* try_create_with_new_runtime(
    const trtf::BundleFile& bundle,
    const trtf::BundleSections& sections,
    const std::string& bundle_path,
    const std::string& hf_python)
{
    auto fp_cfg = parse_or_build_fast_path_config(sections, bundle.info);
    if (fp_cfg.runtime_strategy.empty())
    {
        fp_cfg.runtime_strategy = "decoder_kv_cache";
    }

    auto runtime_ptr = create_runtime_or_throw();

    trtf::runtime::BundlePortAdapter bundle_port(bundle);
    trtf::runtime::adapters::trt::TrtPortAdapter trt_port;

    trtf::runtime::BuildContext build_context;
    build_context.model_id = bundle.info.model_id;
    build_context.strategy = fp_cfg.runtime_strategy;
    build_context.hf_python = hf_python;
    build_context.bundle_path = bundle_path;
    build_context.config = &fp_cfg;
    build_context.sections = &sections;

    auto result = build_services_with_new_runtime(
        build_context, bundle_port, trt_port, runtime_ptr.get());
    if (!result.ok())
    {
        throw std::runtime_error(
            "New runtime build failed for strategy '" + build_context.strategy
            + "' from bundle '" + bundle_path + "': " + result.message);
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_new_runtime, strategy="
              << build_context.strategy << ")" << std::endl;
    return new trtf::runtime::PipelineRouter(
        std::move(result.services), bundle.info.model_id, "trt_new_runtime");
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
        auto* router = dynamic_cast<trtf::runtime::PipelineRouter*>(pipeline);
        if (router != nullptr)
        {
            router->set_default_max_new_tokens(default_max_new_tokens);
        }
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
    return try_create_with_new_runtime(bundle, sections, bundle_path, hf_python);
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
