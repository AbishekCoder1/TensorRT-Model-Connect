#include "trtf/runtime/builders/text/text_strategy_builder.h"

#include "cabi/bundle/bundle_helpers.h"
#include "runtime/services/text/generation_text_service.h"

#if TRTF_HAS_TRT
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_backend_shared.h"
#include "runtime/trt/recurrent/hybrid_backend.h"
#include "runtime/trt/recurrent/mamba_backend.h"
#include "runtime/trt/recurrent/rwkv_backend.h"
#endif

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtf::runtime::builders::text {

namespace {

struct PortCheckResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

bool is_supported_strategy(std::string_view strategy)
{
    static constexpr std::array<std::string_view, 5> kStrategies = {
        "decoder_kv_cache",
        "decoder_moe",
        "ssm_recurrent",
        "rwkv_recurrent",
        "hybrid_mamba_attention",
    };

    for (const auto candidate : kStrategies)
    {
        if (candidate == strategy)
        {
            return true;
        }
    }
    return false;
}

BuildStatus bundle_status_to_build_status(BundlePortStatus status)
{
    return status == BundlePortStatus::kMissingSection
        ? BuildStatus::kMissingDependency
        : BuildStatus::kInvalidArgument;
}

BuildStatus trt_status_to_build_status(TrtPortStatus status)
{
    return status == TrtPortStatus::kInvalidArgument
        ? BuildStatus::kInvalidArgument
        : BuildStatus::kRuntimeError;
}

BundlePortResult<trtf::FastPathModelConfig> resolve_build_config(
    const BuildContext& context,
    const IBundlePort& bundle_port)
{
    if (context.config != nullptr)
    {
        return BundlePortResult<trtf::FastPathModelConfig>::success(
            *static_cast<const trtf::FastPathModelConfig*>(context.config));
    }
    return bundle_port.parse_fast_path_config(-1);
}

PortCheckResult validate_primary_engine(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime)
{
    const auto section = bundle_port.fetch_section_bytes("engine_plan");
    if (!section.ok())
    {
        const BuildStatus status = (section.status == BundlePortStatus::kMissingSection)
            ? BuildStatus::kMissingDependency
            : BuildStatus::kInvalidArgument;
        return {status, section.message};
    }

    const auto deserialize = trt_port.deserialize_engine(
        runtime, section.value.data(), section.value.size());
    if (!deserialize.ok())
    {
        return {BuildStatus::kRuntimeError, deserialize.message};
    }

    const auto create_ctx = trt_port.create_execution_context(deserialize.value);
    if (!create_ctx.ok())
    {
        return {BuildStatus::kRuntimeError, create_ctx.message};
    }

    const auto has_logits = trt_port.has_io_tensor_named(deserialize.value, "logits");
    if (!has_logits.ok())
    {
        return {BuildStatus::kRuntimeError, has_logits.message};
    }
    if (!has_logits.value)
    {
        return {BuildStatus::kMissingDependency,
            "Engine is missing required IO tensor: logits"};
    }

    return {};
}

BuildStatus map_composition_exception_status(std::string_view message)
{
    if (message.find("missing required") != std::string_view::npos
        || message.find("has no tokenizer files") != std::string_view::npos)
    {
        return BuildStatus::kMissingDependency;
    }
    return BuildStatus::kRuntimeError;
}

#if TRTF_HAS_TRT

std::unique_ptr<trtf::IGenerationBackend> create_decoder_backend(
    const trtf::FastPathModelConfig& config,
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const std::string& bundle_label)
{
    auto decoder_engine = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), config);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_label);
    }

    const auto& strategy = config.runtime_strategy;
    if (strategy != "decoder_kv_cache" && strategy != "decoder_moe")
    {
        throw std::runtime_error("Unsupported runtime_strategy: " + strategy);
    }

    auto backend = trtf::CreateTrtBackendFromEngine(std::move(decoder_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }
    return backend;
}

std::unique_ptr<trtf::IGenerationBackend> create_mamba_backend(
    const trtf::FastPathModelConfig& config,
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const std::string& bundle_label)
{
    auto mamba_engine = std::make_unique<trtf::MambaStepEngine>();
    mamba_engine->engine = std::move(trt_engine);
    mamba_engine->context = std::move(exec_ctx);
    mamba_engine->vocab_size = config.vocab_size;
    mamba_engine->hidden_size = config.hidden_size;
    mamba_engine->d_inner = config.d_inner;
    mamba_engine->state_size = config.state_size;
    mamba_engine->conv_kernel = config.conv_kernel;
    mamba_engine->num_layers = config.num_layers;
    mamba_engine->id_bos = config.id_bos;
    mamba_engine->id_eos = config.id_eos;

    for (int32_t i = 0; i < config.num_layers; ++i)
    {
        mamba_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        mamba_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        mamba_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        mamba_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    if (!trtf::has_all_required_mamba_tensors(*mamba_engine))
    {
        throw std::runtime_error("Bundle engine missing required Mamba tensors: " + bundle_label);
    }

    auto backend = trtf::CreateMambaBackendFromEngine(std::move(mamba_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Mamba TRT backend from bundle engine");
    }
    return backend;
}

std::unique_ptr<trtf::IGenerationBackend> create_rwkv_backend(
    const trtf::FastPathModelConfig& config,
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const std::string& bundle_label)
{
    auto rwkv_engine = std::make_unique<trtf::RwkvStepEngine>();
    rwkv_engine->engine = std::move(trt_engine);
    rwkv_engine->context = std::move(exec_ctx);
    rwkv_engine->vocab_size = config.vocab_size;
    rwkv_engine->hidden_size = config.hidden_size;
    rwkv_engine->num_layers = config.num_layers;
    rwkv_engine->id_bos = config.id_bos;
    rwkv_engine->id_eos = config.id_eos;

    for (int32_t i = 0; i < config.num_layers; ++i)
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
        throw std::runtime_error("Bundle engine missing required RWKV tensors: " + bundle_label);
    }

    auto backend = trtf::CreateRwkvBackendFromEngine(std::move(rwkv_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create RWKV TRT backend from bundle engine");
    }
    return backend;
}

std::unique_ptr<trtf::IGenerationBackend> create_hybrid_backend(
    const trtf::FastPathModelConfig& config,
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const std::string& bundle_label)
{
    auto hybrid_engine = std::make_unique<trtf::HybridStepEngine>();
    hybrid_engine->engine = std::move(trt_engine);
    hybrid_engine->context = std::move(exec_ctx);
    hybrid_engine->vocab_size = config.vocab_size;
    hybrid_engine->hidden_size = config.hidden_size;
    hybrid_engine->attention_size = config.attention_size;
    hybrid_engine->max_cache_length = config.max_cache_length;
    hybrid_engine->d_inner = config.d_inner;
    hybrid_engine->d_state = config.mamba_d_state;
    hybrid_engine->d_conv = config.mamba_d_conv;
    hybrid_engine->nheads = config.mamba_nheads;
    hybrid_engine->head_dim = config.mamba_head_dim;
    hybrid_engine->conv_dim = config.conv_dim;
    hybrid_engine->num_mamba_layers = config.num_mamba_layers;
    hybrid_engine->num_attention_layers = config.num_attention_layers;
    hybrid_engine->layer_types = config.layer_types;
    hybrid_engine->id_bos = config.id_bos;
    hybrid_engine->id_eos = config.id_eos;

    for (int32_t i = 0; i < config.num_mamba_layers; ++i)
    {
        hybrid_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        hybrid_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        hybrid_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        hybrid_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    for (int32_t i = 0; i < config.num_attention_layers; ++i)
    {
        hybrid_engine->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        hybrid_engine->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        hybrid_engine->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        hybrid_engine->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }

    if (!trtf::has_all_required_hybrid_tensors(*hybrid_engine))
    {
        throw std::runtime_error("Bundle engine missing required hybrid tensors: " + bundle_label);
    }

    auto backend = trtf::CreateHybridBackendFromEngine(std::move(hybrid_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create hybrid TRT backend from bundle engine");
    }
    return backend;
}

std::unique_ptr<trtf::IGenerationBackend> create_text_backend(
    std::string_view strategy,
    const trtf::FastPathModelConfig& config,
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const std::string& bundle_label)
{
    if (strategy == "decoder_kv_cache" || strategy == "decoder_moe")
    {
        return create_decoder_backend(
            config, std::move(trt_engine), std::move(exec_ctx), bundle_label);
    }

    if (strategy == "ssm_recurrent")
    {
        return create_mamba_backend(
            config, std::move(trt_engine), std::move(exec_ctx), bundle_label);
    }

    if (strategy == "rwkv_recurrent")
    {
        return create_rwkv_backend(
            config, std::move(trt_engine), std::move(exec_ctx), bundle_label);
    }

    return create_hybrid_backend(
        config, std::move(trt_engine), std::move(exec_ctx), bundle_label);
}

BuildResult compose_native_text_services(
    const BuildContext& context,
    const trtf::FastPathModelConfig& config,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime)
{
    if (context.sections == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "TextStrategyBuilder requires BundleSections in BuildContext for text service composition");
    }

    const auto section = bundle_port.fetch_section_bytes("engine_plan");
    if (!section.ok())
    {
        return BuildResult::Failure(bundle_status_to_build_status(section.status), section.message);
    }

    const auto deserialize = trt_port.deserialize_engine(runtime, section.value.data(), section.value.size());
    if (!deserialize.ok())
    {
        return BuildResult::Failure(trt_status_to_build_status(deserialize.status), deserialize.message);
    }

    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine(deserialize.value);
    const auto create_ctx = trt_port.create_execution_context(trt_engine.get());
    if (!create_ctx.ok())
    {
        return BuildResult::Failure(trt_status_to_build_status(create_ctx.status), create_ctx.message);
    }

    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx(create_ctx.value);
    const auto& sections = *static_cast<const trtf::BundleSections*>(context.sections);
    const std::string bundle_label = !context.bundle_path.empty()
        ? context.bundle_path
        : (context.model_id.empty() ? "<text_bundle>" : context.model_id);

    try
    {
        const bool add_special = config.tokenizer_add_special_tokens_present
            ? config.tokenizer_add_special_tokens
            : true;
        auto tokenizer = trtf::extract_tokenizer_from_bundle(sections, context.hf_python, add_special);
        auto backend = create_text_backend(
            context.strategy, config, std::move(trt_engine), std::move(exec_ctx), bundle_label);
        auto tokenizer_temp_dir = std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(
            std::move(tokenizer.temp_dir));
        auto shared_tokenizer = std::shared_ptr<trtf::ITokenizer>(std::move(tokenizer.tokenizer));

        PipelineServices services;
        services.text = std::make_unique<trtf::runtime::services::text::GenerationTextService>(
            std::move(shared_tokenizer),
            std::make_unique<trtf::runtime::services::common::GenerationBackendPort>(std::move(backend)),
            std::move(tokenizer_temp_dir));
        return BuildResult::Success(std::move(services));
    }
    catch (const std::exception& ex)
    {
        return BuildResult::Failure(map_composition_exception_status(ex.what()), ex.what());
    }
}

#endif

} // namespace

TextStrategyBuilder::TextStrategyBuilder(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    ComposeTextServiceFn compose_text_service)
    : mBundlePort(bundle_port)
    , mTrtPort(trt_port)
    , mRuntime(runtime)
    , mComposeTextService(std::move(compose_text_service))
{
}

BuildResult TextStrategyBuilder::build(const BuildContext& context)
{
    if (!is_supported_strategy(context.strategy))
    {
        return BuildResult::Failure(
            BuildStatus::kUnsupportedStrategy,
            "Unsupported text strategy: " + context.strategy);
    }

    if (mRuntime == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "TextStrategyBuilder requires a non-null TensorRT runtime");
    }

    auto parsed = resolve_build_config(context, mBundlePort);
    if (!parsed.ok())
    {
        return BuildResult::Failure(
            bundle_status_to_build_status(parsed.status),
            parsed.message);
    }

    if (parsed.value.runtime_strategy.empty())
    {
        parsed.value.runtime_strategy = context.strategy;
    }

    if (!parsed.value.runtime_strategy.empty() && parsed.value.runtime_strategy != context.strategy)
    {
        return BuildResult::Failure(
            BuildStatus::kInvalidArgument,
            "Strategy mismatch: context strategy is '" + context.strategy
                + "' but bundle config declares '" + parsed.value.runtime_strategy + "'");
    }

    const auto check = validate_primary_engine(mBundlePort, mTrtPort, mRuntime);
    if (!check.ok())
    {
        return BuildResult::Failure(check.status, check.message);
    }

    return compose_text_service(context, parsed.value);
}

BuildResult TextStrategyBuilder::compose_text_service(
    const BuildContext& context,
    const trtf::FastPathModelConfig& config) const
{
    if (mComposeTextService)
    {
        return mComposeTextService(
            context, config, mBundlePort, mTrtPort, mRuntime);
    }
#if TRTF_HAS_TRT
    return compose_native_text_services(context, config, mBundlePort, mTrtPort, mRuntime);
#else
    (void) context;
    (void) config;
    return BuildResult::Failure(
        BuildStatus::kMissingDependency,
        "TextStrategyBuilder requires TRT support for text service composition");
#endif
}

} // namespace trtf::runtime::builders::text
