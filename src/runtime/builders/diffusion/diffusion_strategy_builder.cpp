#include "trtf/runtime/builders/diffusion/diffusion_strategy_builder.h"

#if TRTF_HAS_TRT
#include "cabi/bundle/bundle_helpers.h"
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/diffusion/diffusion_backend.h"
#include "runtime/trt/diffusion/z_image_diffusion_backend.h"
#endif

#if TRTF_HAS_TRT
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#endif
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtf::runtime::builders::diffusion {

namespace {

struct PortCheckResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

#if TRTF_HAS_TRT

class DiffusionVideoService final : public IVideoService {
public:
    DiffusionVideoService(
        std::unique_ptr<trtf::IDiffusionBackend> backend,
        std::unique_ptr<trtf::ITokenizer> tokenizer,
        std::string tokenizer_temp_dir,
        std::string clip_tokenizer_temp_dir)
        : mBackend(std::move(backend))
        , mTokenizer(std::move(tokenizer))
        , mTokenizerTempDir(std::move(tokenizer_temp_dir))
        , mClipTokenizerTempDir(std::move(clip_tokenizer_temp_dir))
    {
    }

    ~DiffusionVideoService() override
    {
        remove_temp_dir(mTokenizerTempDir);
        remove_temp_dir(mClipTokenizerTempDir);
    }

    VideoGenerationResult generate_video(const VideoGenerationRequest& request) override
    {
        if (mBackend == nullptr)
        {
            return VideoGenerationResult::Failure(
                RuntimeServiceStatus::kInvalidArgument, "diffusion backend missing");
        }

        try
        {
            mBackend->set_prompt(request.prompt);
            std::vector<int32_t> input_ids;
            if (mTokenizer != nullptr)
            {
                const std::string prepared = mBackend->prepare_prompt(request.prompt);
                input_ids = mTokenizer->encode(prepared);
            }

            auto video = mBackend->generate_video(input_ids, request.num_steps, request.guidance_scale);
            if (video.frames.empty() || video.num_frames <= 0)
            {
                return VideoGenerationResult::Failure(
                    RuntimeServiceStatus::kRuntimeError, "diffusion backend produced no frames");
            }
            return VideoGenerationResult::Success(
                trtf::runtime::adapters::io::VideoFrameArtifact{
                    std::move(video.frames), video.num_frames, video.width, video.height});
        }
        catch (const std::exception& e)
        {
            return VideoGenerationResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
        }
    }

private:
    static void remove_temp_dir(const std::string& path)
    {
        if (path.empty())
        {
            return;
        }
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    std::unique_ptr<trtf::IDiffusionBackend> mBackend;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::string mTokenizerTempDir;
    std::string mClipTokenizerTempDir;
};

#endif // TRTF_HAS_TRT

bool has_data(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

const trtf::FastPathModelConfig* resolve_fast_path_config(const BuildContext& context)
{
    return static_cast<const trtf::FastPathModelConfig*>(context.config);
}

bool has_bundle_sections(const BuildContext& context)
{
#if TRTF_HAS_TRT
    return static_cast<const trtf::BundleSections*>(context.sections);
#else
    return context.sections != nullptr;
#endif
}

#if TRTF_HAS_TRT

const trtf::BundleSections* resolve_bundle_sections(const BuildContext& context)
{
    return static_cast<const trtf::BundleSections*>(context.sections);
}

trtf::DiffusionEngine load_diffusion_engine(
    nvinfer1::IRuntime* runtime,
    const std::vector<char>* plan_data,
    const std::string& missing_error,
    const std::string& deserialize_error,
    const std::string& context_error,
    std::string name)
{
    if (!has_data(plan_data))
    {
        throw std::runtime_error(missing_error);
    }

    auto* raw_engine = runtime->deserializeCudaEngine(plan_data->data(), plan_data->size());
    if (raw_engine == nullptr)
    {
        throw std::runtime_error(deserialize_error);
    }

    trtf::DiffusionEngine engine;
    engine.name = std::move(name);
    engine.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(raw_engine);
    auto* raw_context = engine.engine->createExecutionContext();
    if (raw_context == nullptr)
    {
        throw std::runtime_error(context_error);
    }
    engine.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(raw_context);
    return engine;
}

std::vector<trtf::DiffusionEngine> load_text_encoder_engines(
    const trtf::BundleSections& sections,
    nvinfer1::IRuntime* runtime)
{
    std::vector<trtf::DiffusionEngine> text_encoders;
    for (std::size_t i = 0; i < sections.text_encoder_plans.size(); ++i)
    {
        const auto* plan = sections.text_encoder_plans[i];
        if (!has_data(plan))
        {
            continue;
        }

        text_encoders.push_back(load_diffusion_engine(
            runtime,
            plan,
            "Bundle has no text encoder plan section",
            "Failed to deserialize text encoder " + std::to_string(i),
            "Failed to create text encoder context " + std::to_string(i),
            "text_encoder_" + std::to_string(i)));
    }
    return text_encoders;
}

template <typename BackendT>
void maybe_load_preprocessor_weights(
    BackendT& backend,
    const trtf::BundleSections& sections)
{
    if (!has_data(sections.preprocessor_weights_data))
    {
        return;
    }

    auto pp_weights = trtf::parse_preprocessor_weights(*sections.preprocessor_weights_data);
    backend.set_preprocessor_weights(std::move(pp_weights));

    auto* z_image = dynamic_cast<trtf::ZImageDiffusionBackend*>(&backend);
    if (z_image != nullptr)
    {
        z_image->load_z_image_preprocessor_weights(*sections.preprocessor_weights_data);
    }
}

std::string maybe_load_clip_tokenizer(
    trtf::IDiffusionBackend& backend,
    const trtf::BundleSections& sections,
    const std::string& hf_python)
{
    if (!has_data(sections.clip_vocab_json_data))
    {
        return {};
    }

    auto clip_tok = trtf::extract_clip_tokenizer_from_bundle(sections, hf_python);
    const std::string temp_dir = clip_tok.temp_dir;
    backend.set_clip_tokenizer(std::move(clip_tok.tokenizer));
    return temp_dir;
}

trtf::TokenizerResult try_extract_diffusion_tokenizer(
    const trtf::BundleSections& sections,
    const std::string& hf_python)
{
    trtf::TokenizerResult tok{nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, /*add_special_tokens=*/true);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer in bundle (" << e.what() << ")" << std::endl;
    }
    return tok;
}

std::unique_ptr<IVideoService> create_default_video_service(const BuildContext& context, nvinfer1::IRuntime* runtime)
{
    const auto* fp_cfg = resolve_fast_path_config(context);
    const auto* sections = resolve_bundle_sections(context);
    if (fp_cfg == nullptr || sections == nullptr)
    {
        return nullptr;
    }

    auto denoiser = load_diffusion_engine(
        runtime,
        sections->denoiser_plan_data,
        "Bundle has no denoiser_plan section",
        "Failed to deserialize denoiser engine",
        "Failed to create denoiser context",
        "denoiser");
    auto vae_decoder = load_diffusion_engine(
        runtime,
        sections->vae_decoder_plan_data,
        "Bundle has no vae_decoder_plan section",
        "Failed to deserialize VAE decoder engine",
        "Failed to create VAE decoder context",
        "vae_decoder");
    auto text_encoders = load_text_encoder_engines(*sections, runtime);

    auto backend = trtf::CreateDiffusionBackend(
        std::move(text_encoders), std::move(denoiser), std::move(vae_decoder), *fp_cfg);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create diffusion backend");
    }

    const std::string bundle_label = !context.bundle_path.empty()
        ? context.bundle_path
        : (context.model_id.empty() ? "<diffusion_bundle>" : context.model_id);
    maybe_load_preprocessor_weights(*backend, *sections);
    backend->set_hf_python(context.hf_python);
    backend->set_bundle_path(bundle_label);
    const std::string clip_temp_dir = maybe_load_clip_tokenizer(*backend, *sections, context.hf_python);
    auto tok = try_extract_diffusion_tokenizer(*sections, context.hf_python);

    return std::make_unique<DiffusionVideoService>(
        std::move(backend), std::move(tok.tokenizer), std::move(tok.temp_dir), clip_temp_dir);
}

#else

std::unique_ptr<IVideoService> create_default_video_service(
    const BuildContext& /*context*/, nvinfer1::IRuntime* /*runtime*/)
{
    throw std::runtime_error(
        "DiffusionStrategyBuilder requires TRT support for default video service composition");
}

#endif

DiffusionStrategyBuilder::VideoServiceFactory& video_service_factory()
{
    static DiffusionStrategyBuilder::VideoServiceFactory factory = &create_default_video_service;
    return factory;
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

PortCheckResult maybe_validate_optional_text_encoder(
    const IBundlePort& bundle_port,
    std::string_view section_name)
{
    if (!bundle_port.has_section(section_name))
    {
        return {};
    }

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

BuildResult validate_diffusion_build_request(
    const BuildContext& context,
    nvinfer1::IRuntime* runtime)
{
    if (context.strategy != "diffusion")
    {
        return BuildResult::Failure(
            BuildStatus::kUnsupportedStrategy,
            "Unsupported diffusion strategy: " + context.strategy);
    }
    if (runtime == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "DiffusionStrategyBuilder requires a non-null TensorRT runtime");
    }
    BuildResult result;
    result.status = BuildStatus::kOk;
    return result;
}

BuildResult validate_diffusion_bundle_sections(const IBundlePort& bundle_port)
{
    const auto denoiser = validate_engine_for_section(bundle_port, "denoiser_plan");
    if (!denoiser.ok())
    {
        return BuildResult::Failure(denoiser.status, denoiser.message);
    }

    const auto vae = validate_engine_for_section(bundle_port, "vae_decoder_plan");
    if (!vae.ok())
    {
        return BuildResult::Failure(vae.status, vae.message);
    }

    for (int idx = 0; idx < 3; ++idx)
    {
        const auto text_encoder =
            maybe_validate_optional_text_encoder(bundle_port, "text_encoder_" + std::to_string(idx) + "_plan");
        if (!text_encoder.ok())
        {
            return BuildResult::Failure(text_encoder.status, text_encoder.message);
        }
    }
    BuildResult result;
    result.status = BuildStatus::kOk;
    return result;
}

BuildResult validate_diffusion_compose_context(const BuildContext& context)
{
    if (resolve_fast_path_config(context) == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kInvalidArgument,
            "DiffusionStrategyBuilder requires BuildContext.config");
    }
    if (!has_bundle_sections(context))
    {
        return BuildResult::Failure(
            BuildStatus::kInvalidArgument,
            "DiffusionStrategyBuilder requires BuildContext.sections");
    }
    BuildResult result;
    result.status = BuildStatus::kOk;
    return result;
}

BuildResult compose_diffusion_video_service(
    const BuildContext& context,
    nvinfer1::IRuntime* runtime)
{
    PipelineServices services;
    try
    {
        services.video = video_service_factory()(context, runtime);
    }
    catch (const std::exception& e)
    {
        return BuildResult::Failure(BuildStatus::kRuntimeError, e.what());
    }
    if (services.video == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kRuntimeError,
            "DiffusionStrategyBuilder failed to compose a video service");
    }
    return BuildResult::Success(std::move(services));
}

} // namespace

DiffusionStrategyBuilder::DiffusionStrategyBuilder(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime)
    : mBundlePort(bundle_port)
    , mTrtPort(trt_port)
    , mRuntime(runtime)
{
}

void DiffusionStrategyBuilder::set_video_service_factory_for_tests(VideoServiceFactory factory)
{
    video_service_factory() = factory != nullptr ? factory : &create_default_video_service;
}

BuildResult DiffusionStrategyBuilder::build(const BuildContext& context)
{
    auto request = validate_diffusion_build_request(context, mRuntime);
    if (!request.ok())
    {
        return BuildResult::Failure(request.status, std::move(request.message));
    }

    auto sections = validate_diffusion_bundle_sections(mBundlePort);
    if (!sections.ok())
    {
        return BuildResult::Failure(sections.status, std::move(sections.message));
    }

    auto compose_context = validate_diffusion_compose_context(context);
    if (!compose_context.ok())
    {
        return BuildResult::Failure(compose_context.status, std::move(compose_context.message));
    }

    return compose_diffusion_video_service(context, mRuntime);
}

} // namespace trtf::runtime::builders::diffusion
