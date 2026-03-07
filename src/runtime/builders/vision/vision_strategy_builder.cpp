#include "trtf/runtime/builders/vision/vision_strategy_builder.h"

#include "cabi/bundle/bundle_helpers.h"
#include "runtime/services/text/generation_text_service.h"
#include "runtime/services/vision/vision_runtime_services.h"

#if TRTF_HAS_TRT
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/core/trt_backend_shared.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/multimodal/vl_backend.h"
#include "runtime/trt/multimodal/vision_engine.h"
#include "runtime/trt/perception/detection_backend.h"
#include "runtime/trt/perception/neural_operator_backend.h"
#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/segmentation_backend.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtf::runtime::builders::vision {

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
        "vision_language",
        "segmentation",
        "prompted_segmentation",
        "object_detection",
        "neural_operator",
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

PortCheckResult validate_engine_for_section(
    const IBundlePort& bundle_port,
    std::string_view section_name)
{
    const auto section = bundle_port.fetch_section_bytes(section_name);
    if (!section.ok())
    {
        const BuildStatus status = (section.status == BundlePortStatus::kMissingSection)
            ? BuildStatus::kMissingDependency
            : BuildStatus::kInvalidArgument;
        return {status, section.message};
    }
    return {};
}

PortCheckResult validate_optional_vision_section(
    const IBundlePort& bundle_port,
    std::string_view strategy)
{
    if (strategy == "prompted_segmentation")
    {
        return validate_engine_for_section(bundle_port, "vision_engine_plan");
    }

    if (strategy != "vision_language")
    {
        return {};
    }

    if (!bundle_port.has_section("vision_engine_plan"))
    {
        return {};
    }

    return validate_engine_for_section(bundle_port, "vision_engine_plan");
}

BuildStatus map_composition_exception_status(std::string_view message)
{
    if (message.find("has no tokenizer files") != std::string_view::npos
        || message.find("missing vision_engine_plan") != std::string_view::npos
        || message.find("missing required") != std::string_view::npos)
    {
        return BuildStatus::kMissingDependency;
    }
    return BuildStatus::kRuntimeError;
}

#if TRTF_HAS_TRT

struct OwnedEngineContext {
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> context;
};

struct EngineLoadResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;
    OwnedEngineContext value;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

BuildResult failure_from_port(const PortCheckResult& check)
{
    return BuildResult::Failure(check.status, check.message);
}

EngineLoadResult load_engine_context_for_section(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view section_name)
{
    const auto section = bundle_port.fetch_section_bytes(section_name);
    if (!section.ok())
    {
        return {bundle_status_to_build_status(section.status), section.message, {}};
    }

    const auto deserialize = trt_port.deserialize_engine(runtime, section.value.data(), section.value.size());
    if (!deserialize.ok())
    {
        return {trt_status_to_build_status(deserialize.status), deserialize.message, {}};
    }

    OwnedEngineContext loaded;
    loaded.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(deserialize.value);
    const auto create_ctx = trt_port.create_execution_context(loaded.engine.get());
    if (!create_ctx.ok())
    {
        return {trt_status_to_build_status(create_ctx.status), create_ctx.message, {}};
    }

    loaded.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(create_ctx.value);
    return {BuildStatus::kOk, {}, std::move(loaded)};
}

BundlePortResult<std::vector<char>> fetch_optional_section(
    const IBundlePort& bundle_port,
    std::string_view section_name)
{
    if (!bundle_port.has_section(section_name))
    {
        return BundlePortResult<std::vector<char>>::success({});
    }
    return bundle_port.fetch_section_bytes(section_name);
}

trtf::VLPreprocessConfig parse_vl_preprocess_from_bundle(const IBundlePort& bundle_port)
{
    const auto config_json = fetch_optional_section(bundle_port, "config.json");
    const auto preprocessor_json = fetch_optional_section(bundle_port, "preprocessor_config.json");
    const std::string config_text = config_json.ok()
        ? std::string(config_json.value.begin(), config_json.value.end())
        : std::string();
    const std::string preproc_text = preprocessor_json.ok()
        ? std::string(preprocessor_json.value.begin(), preprocessor_json.value.end())
        : std::string();
    return trtf::parse_vl_preprocess_config(config_text, preproc_text);
}

BuildResult build_vl_services(
    const BuildContext& context,
    const trtf::FastPathModelConfig& config,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::BundleSections& sections,
    OwnedEngineContext primary,
    const std::string& bundle_label)
{
    auto decoder_engine = trtf::make_decoder_engine(
        std::move(primary.engine), std::move(primary.context), config);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_label);
    }

    std::unique_ptr<trtf::VisionStepEngine> vision_engine;
    if (bundle_port.has_section("vision_engine_plan"))
    {
        auto vision = load_engine_context_for_section(
            bundle_port, trt_port, runtime, "vision_engine_plan");
        if (!vision.ok())
        {
            return BuildResult::Failure(vision.status, vision.message);
        }

        vision_engine = std::make_unique<trtf::VisionStepEngine>();
        vision_engine->engine = std::move(vision.value.engine);
        vision_engine->context = std::move(vision.value.context);
        vision_engine->num_output_features = config.num_image_pad_tokens;
        vision_engine->feature_dim = (config.vision_output_dim > 0)
            ? config.vision_output_dim
            : config.hidden_size;
    }

    auto tokenizer = trtf::extract_tokenizer_from_bundle(
        sections, context.hf_python, config.tokenizer_add_special_tokens);
    auto backend = trtf::CreateVLBackendFromEngines(
        std::move(decoder_engine),
        std::move(vision_engine),
        config,
        parse_vl_preprocess_from_bundle(bundle_port));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create VL TRT backend from bundle engine");
    }

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

BuildResult build_segmentation_services(
    OwnedEngineContext primary,
    const trtf::FastPathModelConfig& config)
{
    auto backend = trtf::CreateSegmentationBackend(
        std::move(primary.engine), std::move(primary.context), config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create segmentation backend from bundle engine");
    }

    PipelineServices services;
    services.segmentation = std::make_unique<trtf::runtime::services::vision::SegmentationService>(
        std::make_unique<trtf::runtime::services::common::SegmentationBackendPort>(std::move(backend)));
    return BuildResult::Success(std::move(services));
}

BuildResult build_prompted_segmentation_services(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary)
{
    auto decoder = load_engine_context_for_section(
        bundle_port, trt_port, runtime, "vision_engine_plan");
    if (!decoder.ok())
    {
        return BuildResult::Failure(decoder.status, decoder.message);
    }

    auto backend = trtf::CreateSamBackend(
        std::move(primary.engine),
        std::move(primary.context),
        std::move(decoder.value.engine),
        std::move(decoder.value.context),
        config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create SAM backend from bundle engines");
    }

    PipelineServices services;
    services.segmentation = std::make_unique<trtf::runtime::services::vision::PromptedSegmentationService>(
        std::make_unique<trtf::runtime::services::common::SamBackendPort>(std::move(backend)));
    return BuildResult::Success(std::move(services));
}

BuildResult build_detection_services(
    OwnedEngineContext primary,
    const trtf::FastPathModelConfig& config)
{
    auto backend = trtf::CreateDetectionBackend(
        std::move(primary.engine), std::move(primary.context), config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create detection backend from bundle engine");
    }

    PipelineServices services;
    services.detection = std::make_unique<trtf::runtime::services::vision::DetectionService>(
        std::make_unique<trtf::runtime::services::common::DetectionBackendPort>(std::move(backend)));
    return BuildResult::Success(std::move(services));
}

BuildResult build_neural_operator_services(
    OwnedEngineContext primary,
    const trtf::FastPathModelConfig& config)
{
    auto backend = trtf::CreateNeuralOperatorBackend(
        std::move(primary.engine), std::move(primary.context), config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create neural operator backend from bundle engine");
    }

    PipelineServices services;
    services.solve = std::make_unique<trtf::runtime::services::vision::NeuralOperatorService>(
        std::make_unique<trtf::runtime::services::common::NeuralOperatorBackendPort>(std::move(backend)));
    return BuildResult::Success(std::move(services));
}

BuildResult compose_native_vision_services(
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
            "VisionStrategyBuilder requires BundleSections in BuildContext for vision service composition");
    }

    auto primary = load_engine_context_for_section(
        bundle_port, trt_port, runtime, "engine_plan");
    if (!primary.ok())
    {
        return BuildResult::Failure(primary.status, primary.message);
    }

    const auto& sections = *static_cast<const trtf::BundleSections*>(context.sections);
    const std::string bundle_label = !context.bundle_path.empty()
        ? context.bundle_path
        : (context.model_id.empty() ? "<vision_bundle>" : context.model_id);

    try
    {
        if (context.strategy == "vision_language")
        {
            return build_vl_services(
                context,
                config,
                bundle_port,
                trt_port,
                runtime,
                sections,
                std::move(primary.value),
                bundle_label);
        }

        if (context.strategy == "segmentation")
        {
            return build_segmentation_services(std::move(primary.value), config);
        }

        if (context.strategy == "prompted_segmentation")
        {
            return build_prompted_segmentation_services(
                bundle_port, trt_port, runtime, config, std::move(primary.value));
        }

        if (context.strategy == "object_detection")
        {
            return build_detection_services(std::move(primary.value), config);
        }

        return build_neural_operator_services(std::move(primary.value), config);
    }
    catch (const std::exception& ex)
    {
        return BuildResult::Failure(map_composition_exception_status(ex.what()), ex.what());
    }
}

#endif

} // namespace

VisionStrategyBuilder::VisionStrategyBuilder(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    ComposeVisionServiceFn compose_vision_service)
    : mBundlePort(bundle_port)
    , mTrtPort(trt_port)
    , mRuntime(runtime)
    , mComposeVisionService(std::move(compose_vision_service))
{
}

BuildResult VisionStrategyBuilder::build(const BuildContext& context)
{
    if (!is_supported_strategy(context.strategy))
    {
        return BuildResult::Failure(
            BuildStatus::kUnsupportedStrategy,
            "Unsupported vision strategy: " + context.strategy);
    }

    if (mRuntime == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "VisionStrategyBuilder requires a non-null TensorRT runtime");
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

    const auto primary = validate_engine_for_section(mBundlePort, "engine_plan");
    if (!primary.ok())
    {
        return failure_from_port(primary);
    }

    const auto optional = validate_optional_vision_section(mBundlePort, context.strategy);
    if (!optional.ok())
    {
        return failure_from_port(optional);
    }

    return compose_vision_service(context, parsed.value);
}

BuildResult VisionStrategyBuilder::compose_vision_service(
    const BuildContext& context,
    const trtf::FastPathModelConfig& config) const
{
    if (mComposeVisionService)
    {
        return mComposeVisionService(
            context, config, mBundlePort, mTrtPort, mRuntime);
    }
#if TRTF_HAS_TRT
    return compose_native_vision_services(context, config, mBundlePort, mTrtPort, mRuntime);
#else
    (void) context;
    (void) config;
    return BuildResult::Failure(
        BuildStatus::kMissingDependency,
        "VisionStrategyBuilder requires TRT support for vision service composition");
#endif
}

} // namespace trtf::runtime::builders::vision
