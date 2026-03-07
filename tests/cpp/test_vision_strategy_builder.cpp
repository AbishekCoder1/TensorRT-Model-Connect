#include "trtf/runtime/builders/vision/vision_strategy_builder.h"

#include "test_strategy_builder_helpers.h"

#include <cstring>
#include <string>
#include <vector>

using trtf::runtime::BuildContext;
using trtf::runtime::BuildStatus;
using trtf::runtime::builders::vision::VisionStrategyBuilder;
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

trtf::runtime::adapters::io::DecodedImage make_decoded_image()
{
    trtf::runtime::adapters::io::DecodedImage image;
    image.pixels = {255, 0, 0};
    image.width = 1;
    image.height = 1;
    image.channels = 3;
    return image;
}

class ComposedVisionTextService final : public trtf::runtime::ITextService {
public:
    explicit ComposedVisionTextService(bool supports_vision)
        : mSupportsVision(supports_vision)
    {
    }

    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        mLast = std::string("text|")
            + (prompt == nullptr ? "" : prompt)
            + "|"
            + std::to_string(max_new_tokens);
        return mLast.c_str();
    }

    const char* generate(
        const char* prompt, const trtf::runtime::adapters::io::DecodedImage& image, std::size_t max_new_tokens) override
    {
        mLast = std::string("vl|")
            + (prompt == nullptr ? "" : prompt)
            + "|"
            + std::to_string(image.width)
            + "|"
            + std::to_string(max_new_tokens);
        return mLast.c_str();
    }

    bool supports_vision() const override
    {
        return mSupportsVision;
    }

private:
    bool mSupportsVision{false};
    std::string mLast;
};

class ComposedSegmentationService final : public trtf::runtime::ISegmentationService {
public:
    explicit ComposedSegmentationService(bool prompted)
        : mPrompted(prompted)
    {
    }

    trtf::runtime::SegmentationResult segment(const trtf::runtime::SegmentationRequest& /*request*/) override
    {
        return trtf::runtime::SegmentationResult::Success({{1, 2, 3, 4}, 2, 2});
    }

    bool supports_prompted() const override
    {
        return mPrompted;
    }

    trtf::runtime::PromptedSegmentationResult segment_prompt(
        const trtf::runtime::PromptedSegmentationRequest& /*request*/) override
    {
        if (!mPrompted)
        {
            return trtf::runtime::PromptedSegmentationResult::Failure(
                trtf::runtime::RuntimeServiceStatus::kUnsupported, "prompted segmentation disabled");
        }
        return trtf::runtime::PromptedSegmentationResult::Success({{1.0F, 0.0F, 0.0F, 1.0F}, {0.9F}, 13, 2, 2});
    }

private:
    bool mPrompted{false};
};

class ComposedDetectionService final : public trtf::runtime::IDetectionService {
public:
    trtf::runtime::DetectionResult detect(const trtf::runtime::DetectionRequest& /*request*/) override
    {
        return trtf::runtime::DetectionResult::Success({{{1, 0.95F, 0.0F, 1.0F, 2.0F, 3.0F}}});
    }
};

class ComposedSolveService final : public trtf::runtime::ISolveService {
public:
    const float* solve(const float* /*branch_input*/, int32_t /*branch_len*/, const float* /*trunk_input*/,
        int32_t /*trunk_len*/, int32_t* out_dim) override
    {
        if (out_dim != nullptr)
        {
            *out_dim = 4;
        }
        return mSolveValues;
    }

    const float* solve_field(const float* /*field_input*/, int32_t /*input_size*/, int32_t* out_channels,
        int32_t* out_h, int32_t* out_w) override
    {
        if (out_channels != nullptr)
        {
            *out_channels = 1;
        }
        if (out_h != nullptr)
        {
            *out_h = 2;
        }
        if (out_w != nullptr)
        {
            *out_w = 2;
        }
        return mFieldValues;
    }

private:
    float mSolveValues[4]{1.0F, 2.0F, 3.0F, 4.0F};
    float mFieldValues[4]{5.0F, 6.0F, 7.0F, 8.0F};
};

BuildContext make_context(const std::string& strategy)
{
    BuildContext context;
    context.model_id = "vision-model";
    context.strategy = strategy;
    return context;
}

void test_unsupported_strategy_rejected()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    VisionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("embedding"));
    check(!result.ok(), "unsupported: result not ok");
    check(result.status == BuildStatus::kUnsupportedStrategy,
        "unsupported: status");
}

void test_missing_primary_engine_rejected()
{
    FakeBundlePort bundle;
    bundle.config.runtime_strategy = "vision_language";
    FakeTrtPort trt;
    VisionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("vision_language"));
    check(!result.ok(), "missing engine: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing engine: status");
}

void test_prompted_segmentation_requires_vision_plan()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "prompted_segmentation";
    FakeTrtPort trt;
    VisionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("prompted_segmentation"));
    check(!result.ok(), "prompted requires vision: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "prompted requires vision: status");
}

void test_strategy_mismatch_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "object_detection";
    FakeTrtPort trt;
    VisionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("segmentation"));
    check(!result.ok(), "strategy mismatch: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "strategy mismatch: status");
}

void test_vision_language_optional_vision_plan_empty_section_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.sections["vision_engine_plan"] = {};
    bundle.config.runtime_strategy = "vision_language";
    FakeTrtPort trt;
    VisionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("vision_language"));
    check(!result.ok(), "vision optional empty section: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "vision optional empty section: status");
}

void test_missing_composition_context_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "vision_language";
    FakeTrtPort trt;
    VisionStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("vision_language"));
    check(!result.ok(), "missing composition context: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing composition context: status");
#if TRTF_HAS_TRT
    check(result.message.find("BundleSections") != std::string::npos,
        "missing composition context: mentions BundleSections");
#else
    check(result.message.find("TRT support") != std::string::npos,
        "missing composition context: mentions TRT support");
#endif
}

void test_success_uses_composer_services()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.sections["vision_engine_plan"] = bytes_from_text("vision-plan");
    bundle.config.runtime_strategy.clear();
    bundle.config.hidden_size = 128;
    FakeTrtPort trt;

    int compose_calls = 0;
    std::string last_strategy;
    int32_t last_hidden_size = 0;
    VisionStrategyBuilder builder(
        bundle,
        trt,
        fake_runtime_ptr(),
        [&](const BuildContext& context, const trtf::FastPathModelConfig& config,
            const trtf::runtime::IBundlePort&, const trtf::runtime::ITrtPort&,
            nvinfer1::IRuntime*) {
            ++compose_calls;
            last_strategy = context.strategy;
            last_hidden_size = config.hidden_size;

            trtf::runtime::PipelineServices services;
            if (context.strategy == "vision_language")
            {
                services.text = std::make_unique<ComposedVisionTextService>(true);
            }
            else if (context.strategy == "segmentation")
            {
                services.segmentation = std::make_unique<ComposedSegmentationService>(false);
            }
            else if (context.strategy == "prompted_segmentation")
            {
                services.segmentation = std::make_unique<ComposedSegmentationService>(true);
            }
            else if (context.strategy == "object_detection")
            {
                services.detection = std::make_unique<ComposedDetectionService>();
            }
            else
            {
                services.solve = std::make_unique<ComposedSolveService>();
            }

            return trtf::runtime::BuildResult::Success(std::move(services));
        });

    const auto vision = builder.build(make_context("vision_language"));
    check(vision.ok(), "vision composer: result ok");
    check(vision.services.text != nullptr, "vision composer: text service");
    if (vision.services.text != nullptr)
    {
        check(vision.services.text->supports_vision(), "vision composer: vision support");
        const char* generated = vision.services.text->generate("hello", make_decoded_image(), 4);
        check(generated != nullptr, "vision composer: generate non-null");
        check(std::strcmp(generated, "vl|hello|1|4") == 0,
            "vision composer: generated text");
    }

    const auto segmentation = builder.build(make_context("segmentation"));
    check(segmentation.ok(), "segmentation composer: result ok");
    check(segmentation.services.segmentation != nullptr,
        "segmentation composer: segmentation service");
    if (segmentation.services.segmentation != nullptr)
    {
        check(!segmentation.services.segmentation->supports_prompted(),
            "segmentation composer: prompted false");
        const auto result = segmentation.services.segmentation->segment({make_decoded_image()});
        check(result.ok() && result.value.width == 2 && result.value.height == 2,
            "segmentation composer: segment delegates");
    }

    const auto prompted = builder.build(make_context("prompted_segmentation"));
    check(prompted.ok(), "prompted composer: result ok");
    check(prompted.services.segmentation != nullptr,
        "prompted composer: segmentation service");
    if (prompted.services.segmentation != nullptr)
    {
        check(prompted.services.segmentation->supports_prompted(),
            "prompted composer: prompted true");
        const auto result = prompted.services.segmentation->segment_prompt({make_decoded_image(), 0.1F, 0.2F, true});
        check(result.ok() && result.value.num_masks == 13,
            "prompted composer: segment prompt delegates");
    }

    const auto detection = builder.build(make_context("object_detection"));
    check(detection.ok(), "detection composer: result ok");
    check(detection.services.detection != nullptr,
        "detection composer: detection service");
    if (detection.services.detection != nullptr)
    {
        const auto result = detection.services.detection->detect({make_decoded_image(), 0.5F});
        check(result.ok() && result.value.detections.size() == 1,
            "detection composer: detect delegates");
    }

    const auto neural = builder.build(make_context("neural_operator"));
    check(neural.ok(), "neural composer: result ok");
    check(neural.services.solve != nullptr,
        "neural composer: solve service");
    if (neural.services.solve != nullptr)
    {
        int32_t out_dim = 0;
        const float* solved = neural.services.solve->solve(nullptr, 0, nullptr, 0, &out_dim);
        check(solved != nullptr, "neural composer: solve returns data");
        check(out_dim == 4, "neural composer: solve dim");
        int32_t out_channels = 0;
        int32_t out_h = 0;
        int32_t out_w = 0;
        const float* solved_field = neural.services.solve->solve_field(nullptr, 0, &out_channels, &out_h, &out_w);
        check(solved_field != nullptr, "neural composer: solve field returns data");
        check(out_channels == 1 && out_h == 2 && out_w == 2,
            "neural composer: solve field shape");
    }

    check(compose_calls == 5, "composer: called once per strategy");
    check(last_strategy == "neural_operator", "composer: last strategy forwarded");
    check(last_hidden_size == 128, "composer: config forwarded");
    check(trt.deserialize_calls == 0, "composer: build avoids eager TRT deserialize");
    check(trt.create_context_calls == 0, "composer: build avoids eager TRT context creation");
}

void test_context_config_used_when_available()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.parse_status = trtf::runtime::BundlePortStatus::kMissingSection;
    bundle.parse_message = "config missing";
    FakeTrtPort trt;

    trtf::FastPathModelConfig config;
    config.runtime_strategy = "object_detection";
    config.hidden_size = 256;

    int compose_calls = 0;
    VisionStrategyBuilder builder(
        bundle,
        trt,
        fake_runtime_ptr(),
        [&](const BuildContext&, const trtf::FastPathModelConfig& composed_config,
            const trtf::runtime::IBundlePort&, const trtf::runtime::ITrtPort&,
            nvinfer1::IRuntime*) {
            ++compose_calls;
            check(composed_config.hidden_size == 256,
                "context config: forwarded hidden size");

            trtf::runtime::PipelineServices services;
            services.detection = std::make_unique<ComposedDetectionService>();
            return trtf::runtime::BuildResult::Success(std::move(services));
        });

    auto context = make_context("object_detection");
    context.config = &config;
    const auto result = builder.build(context);
    check(result.ok(), "context config: result ok");
    check(compose_calls == 1, "context config: composer called");
}

} // namespace

int main()
{
    test_unsupported_strategy_rejected();
    test_missing_primary_engine_rejected();
    test_prompted_segmentation_requires_vision_plan();
    test_strategy_mismatch_rejected();
    test_vision_language_optional_vision_plan_empty_section_rejected();
    test_missing_composition_context_rejected();
    test_success_uses_composer_services();
    test_context_config_used_when_available();
    print_summary_and_exit_if_failures("test_vision_strategy_builder");
    return 0;
}
