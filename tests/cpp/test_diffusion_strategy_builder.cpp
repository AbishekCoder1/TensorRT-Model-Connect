#include "trtf/runtime/builders/diffusion/diffusion_strategy_builder.h"
#include "test_strategy_builder_helpers.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using trtf::runtime::BuildContext;
using trtf::runtime::BuildStatus;
using trtf::runtime::builders::diffusion::DiffusionStrategyBuilder;
using trtf::runtime::IVideoService;
using trtf::runtime::testhelpers::FakeBundlePort;
using trtf::runtime::testhelpers::FakeTrtPort;
using trtf::runtime::testhelpers::check;
using trtf::runtime::testhelpers::fake_runtime_ptr;
using trtf::runtime::testhelpers::print_summary_and_exit_if_failures;

namespace {

std::vector<char> bytes_from_text(const std::string& text)
{
    return std::vector<char>(text.begin(), text.end());
}

class FakeVideoService final : public IVideoService {
public:
    trtf::runtime::VideoGenerationResult generate_video(
        const trtf::runtime::VideoGenerationRequest& /*request*/) override
    {
        return trtf::runtime::VideoGenerationResult::Success({{0.1F, 0.2F, 0.3F}, 7, 1, 1});
    }
};

std::unique_ptr<IVideoService> make_fake_video_service(
    const BuildContext& /*context*/, nvinfer1::IRuntime* /*runtime*/)
{
    return std::make_unique<FakeVideoService>();
}

std::unique_ptr<IVideoService> fail_video_service_factory(
    const BuildContext& /*context*/, nvinfer1::IRuntime* /*runtime*/)
{
    throw std::runtime_error("video service factory failed");
}

BuildContext make_context(const std::string& strategy)
{
    BuildContext context;
    context.model_id = "diffusion-model";
    context.strategy = strategy;
    return context;
}

trtf::FastPathModelConfig make_config()
{
    trtf::FastPathModelConfig config;
    config.runtime_strategy = "diffusion";
    config.diffusion_backend_type = "wan_3d";
    return config;
}

const void* make_sections_handle()
{
    static const int sentinel = 1;
    return &sentinel;
}

void test_unsupported_strategy_rejected()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "unsupported: result not ok");
    check(result.status == BuildStatus::kUnsupportedStrategy,
        "unsupported: status");
}

void test_missing_denoiser_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["vae_decoder_plan"] = bytes_from_text("vae");
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("diffusion"));
    check(!result.ok(), "missing denoiser: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing denoiser: status");
}

void test_missing_vae_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["denoiser_plan"] = bytes_from_text("denoiser");
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("diffusion"));
    check(!result.ok(), "missing vae: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing vae: status");
}

void test_optional_text_encoder_empty_section_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["denoiser_plan"] = bytes_from_text("denoiser");
    bundle.sections["vae_decoder_plan"] = bytes_from_text("vae");
    bundle.sections["text_encoder_0_plan"] = {};
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("diffusion"));
    check(!result.ok(), "optional text encoder empty section: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "optional text encoder empty section: status");
}

void test_missing_context_config_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["denoiser_plan"] = bytes_from_text("denoiser");
    bundle.sections["vae_decoder_plan"] = bytes_from_text("vae");
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    BuildContext context = make_context("diffusion");
    context.sections = make_sections_handle();

    const auto result = builder.build(context);
    check(!result.ok(), "missing context config: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "missing context config: status");
}

void test_missing_context_sections_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["denoiser_plan"] = bytes_from_text("denoiser");
    bundle.sections["vae_decoder_plan"] = bytes_from_text("vae");
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto config = make_config();
    BuildContext context = make_context("diffusion");
    context.config = &config;

    const auto result = builder.build(context);
    check(!result.ok(), "missing context sections: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "missing context sections: status");
}

void test_factory_failure_propagates()
{
    FakeBundlePort bundle;
    bundle.sections["denoiser_plan"] = bytes_from_text("denoiser");
    bundle.sections["vae_decoder_plan"] = bytes_from_text("vae");
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto config = make_config();
    BuildContext context = make_context("diffusion");
    context.config = &config;
    context.sections = make_sections_handle();

    DiffusionStrategyBuilder::set_video_service_factory_for_tests(&fail_video_service_factory);
    const auto result = builder.build(context);
    DiffusionStrategyBuilder::set_video_service_factory_for_tests(nullptr);

    check(!result.ok(), "factory failure: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "factory failure: status");
    check(result.message == "video service factory failed",
        "factory failure: message");
}

void test_success_sets_video_service()
{
    FakeBundlePort bundle;
    bundle.sections["denoiser_plan"] = bytes_from_text("denoiser");
    bundle.sections["vae_decoder_plan"] = bytes_from_text("vae");
    bundle.sections["text_encoder_0_plan"] = bytes_from_text("te0");
    FakeTrtPort trt;
    DiffusionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto config = make_config();
    BuildContext context = make_context("diffusion");
    context.config = &config;
    context.sections = make_sections_handle();

    DiffusionStrategyBuilder::set_video_service_factory_for_tests(&make_fake_video_service);
    const auto result = builder.build(context);
    DiffusionStrategyBuilder::set_video_service_factory_for_tests(nullptr);

    check(result.ok(), "success: result ok");
    check(result.services.video != nullptr, "success: video service set");
    check(trt.deserialize_calls == 0, "success: build avoids eager TRT deserialize");
    check(trt.create_context_calls == 0, "success: build avoids eager TRT context creation");
}

} // namespace

int main()
{
    test_unsupported_strategy_rejected();
    test_missing_denoiser_rejected();
    test_missing_vae_rejected();
    test_missing_context_config_rejected();
    test_missing_context_sections_rejected();
    test_optional_text_encoder_empty_section_rejected();
    test_factory_failure_propagates();
    test_success_sets_video_service();
    print_summary_and_exit_if_failures("test_diffusion_strategy_builder");
    return 0;
}
