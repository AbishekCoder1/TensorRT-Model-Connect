#include "trtf/runtime/contracts/contracts.h"

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

int failures = 0;

void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class DummyTextService final : public trtf::runtime::ITextService {
public:
    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        (void) prompt;
        (void) max_new_tokens;
        return "ok";
    }
};

class DummyBuilder final : public trtf::runtime::IStrategyBuilder {
public:
    trtf::runtime::BuildResult build(const trtf::runtime::BuildContext& context) override
    {
        (void) context;
        trtf::runtime::PipelineServices services;
        services.text = std::make_unique<DummyTextService>();
        return trtf::runtime::BuildResult::Success(std::move(services));
    }
};

void test_compile_shape()
{
    static_assert(std::is_abstract_v<trtf::runtime::ITextService>, "ITextService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::IVideoService>, "IVideoService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::IAudioService>, "IAudioService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::ITranscriptionService>,
        "ITranscriptionService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::IEmbeddingService>, "IEmbeddingService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::IRerankService>, "IRerankService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::ISegmentationService>,
        "ISegmentationService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::IDetectionService>, "IDetectionService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::ISolveService>, "ISolveService must be abstract");
    static_assert(std::is_abstract_v<trtf::runtime::IStrategyBuilder>, "IStrategyBuilder must be abstract");

    static_assert(std::is_same_v<decltype(trtf::runtime::PipelineServices{}.text),
                      std::unique_ptr<trtf::runtime::ITextService>>,
        "PipelineServices::text must be unique_ptr<ITextService>");
    static_assert(std::is_same_v<decltype(trtf::runtime::PipelineServices{}.video),
                      std::unique_ptr<trtf::runtime::IVideoService>>,
        "PipelineServices::video must be unique_ptr<IVideoService>");
    static_assert(std::is_same_v<decltype(trtf::runtime::PipelineServices{}.audio),
                      std::unique_ptr<trtf::runtime::IAudioService>>,
        "PipelineServices::audio must be unique_ptr<IAudioService>");
}

void test_pipeline_services_default_state()
{
    trtf::runtime::PipelineServices services;
    check(services.text == nullptr, "default text service is null");
    check(services.video == nullptr, "default video service is null");
    check(services.audio == nullptr, "default audio service is null");
    check(services.transcription == nullptr, "default transcription service is null");
    check(services.embedding == nullptr, "default embedding service is null");
    check(services.rerank == nullptr, "default rerank service is null");
    check(services.segmentation == nullptr, "default segmentation service is null");
    check(services.detection == nullptr, "default detection service is null");
    check(services.solve == nullptr, "default solve service is null");
    check(services.empty(), "default PipelineServices is empty");
}

void test_build_context_default_state()
{
    trtf::runtime::BuildContext context;
    check(context.model_id.empty(), "default BuildContext model_id is empty");
    check(context.strategy.empty(), "default BuildContext strategy is empty");
    check(context.hf_python.empty(), "default BuildContext hf_python is empty");
    check(context.bundle_path.empty(), "default BuildContext bundle_path is empty");
    check(context.config == nullptr, "default BuildContext config is null");
    check(context.sections == nullptr, "default BuildContext sections is null");
    check(context.runtime_handles.empty(), "default BuildContext runtime_handles is empty");
}

void test_build_result_default_state()
{
    trtf::runtime::BuildResult result;
    check(result.status == trtf::runtime::BuildStatus::kUnspecified,
        "default BuildResult status is unspecified");
    check(!result.ok(), "default BuildResult is not ok");
    check(result.message.empty(), "default BuildResult message is empty");
    check(result.services.empty(), "default BuildResult services are empty");
}

void test_build_result_helpers()
{
    trtf::runtime::BuildResult failure
        = trtf::runtime::BuildResult::Failure(trtf::runtime::BuildStatus::kUnsupportedStrategy, "unsupported");
    check(!failure.ok(), "failure BuildResult is not ok");
    check(failure.status == trtf::runtime::BuildStatus::kUnsupportedStrategy, "failure status preserved");
    check(failure.message == "unsupported", "failure message preserved");
    check(failure.services.empty(), "failure services stay empty");

    trtf::runtime::PipelineServices services;
    services.text = std::make_unique<DummyTextService>();
    trtf::runtime::BuildResult success = trtf::runtime::BuildResult::Success(std::move(services));
    check(success.ok(), "success BuildResult is ok");
    check(success.status == trtf::runtime::BuildStatus::kOk, "success status is ok");
    check(success.message.empty(), "success message is empty");
    check(success.services.text != nullptr, "success carries service ownership");
    check(!success.services.empty(), "success services not empty");
}

void test_strategy_builder_contract()
{
    DummyBuilder builder;
    trtf::runtime::BuildContext context;
    context.model_id = "model";
    context.strategy = "strategy";
    trtf::runtime::BuildResult result = builder.build(context);
    check(result.ok(), "builder returns ok result");
    check(result.services.text != nullptr, "builder returns text service");
}

} // namespace

int main()
{
    test_compile_shape();
    test_pipeline_services_default_state();
    test_build_context_default_state();
    test_build_result_default_state();
    test_build_result_helpers();
    test_strategy_builder_contract();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
