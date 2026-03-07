#include "trtf/runtime/builders/audio/audio_strategy_builder.h"

#include "test_strategy_builder_helpers.h"

#include <string>
#include <vector>

using trtf::runtime::BuildContext;
using trtf::runtime::BuildStatus;
using trtf::runtime::builders::audio::AudioStrategyBuilder;
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

BuildContext make_context(const std::string& strategy)
{
    BuildContext context;
    context.model_id = "audio-model";
    context.strategy = strategy;
    return context;
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
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("encoder_only"));
    check(!result.ok(), "unsupported: result not ok");
    check(result.status == BuildStatus::kUnsupportedStrategy,
        "unsupported: status");
}

void test_missing_primary_engine_rejected()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("text_to_audio"));
    check(!result.ok(), "missing engine: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing engine: status");
}

void test_optional_text_to_audio_engine_failure_propagates()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.sections["vision_engine_plan"] = bytes_from_text("vision");
    FakeTrtPort trt;
    trt.fail_deserialize_on_call = 2;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("text_to_audio"));
    check(!result.ok(), "optional text_to_audio failure: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "optional text_to_audio failure: status");
}

void test_speech_to_text_success_sets_transcription_service()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    FakeTrtPort trt;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("speech_to_text"));
    check(result.ok(), "speech_to_text success: result ok");
    check(result.services.transcription != nullptr,
        "speech_to_text success: transcription service");
}

void test_speech_to_speech_success_and_optional_engines()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.sections["depth_engine_plan"] = bytes_from_text("depth");
    bundle.sections["mimi_encoder_plan"] = bytes_from_text("mimi-enc");
    bundle.sections["mimi_decoder_plan"] = bytes_from_text("mimi-dec");
    FakeTrtPort trt;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("speech_to_speech"));
    check(result.ok(), "speech_to_speech success: result ok");
    check(result.services.audio != nullptr, "speech_to_speech success: audio service");
    check(result.services.audio->supports_speech(),
        "speech_to_speech success: supports speech");
}

void test_omni_success_sets_audio_and_text_services()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.sections["audio_encoder_plan"] = bytes_from_text("audio-encoder");
    bundle.sections["talker_engine_plan"] = bytes_from_text("talker");
    bundle.sections["code2wav_engine_plan"] = bytes_from_text("code2wav");
    FakeTrtPort trt;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("omni_multimodal"));
    check(result.ok(), "omni success: result ok");
    check(result.services.audio != nullptr, "omni success: audio service");
    check(result.services.text != nullptr, "omni success: text service");
}

void test_production_context_strategy_mismatch_rejected_before_trt_load()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    trtf::FastPathModelConfig config;
    config.runtime_strategy = "omni_multimodal";

    BuildContext context = make_context("speech_to_text");
    context.config = &config;
    context.sections = make_sections_handle();

    const auto result = builder.build(context);
    check(!result.ok(), "production mismatch: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "production mismatch: status");
    check(trt.deserialize_calls == 0,
        "production mismatch: no TRT deserialize");
}

void test_production_context_config_parse_failure_propagates()
{
    FakeBundlePort bundle;
    bundle.parse_status = trtf::runtime::BundlePortStatus::kMissingSection;
    bundle.parse_message = "missing config";
    FakeTrtPort trt;
    AudioStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    BuildContext context = make_context("speech_to_text");
    context.sections = make_sections_handle();

    const auto result = builder.build(context);
    check(!result.ok(), "production parse failure: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "production parse failure: status");
    check(trt.deserialize_calls == 0,
        "production parse failure: no TRT deserialize");
}

} // namespace

int main()
{
    test_unsupported_strategy_rejected();
    test_missing_primary_engine_rejected();
    test_optional_text_to_audio_engine_failure_propagates();
    test_speech_to_text_success_sets_transcription_service();
    test_speech_to_speech_success_and_optional_engines();
    test_omni_success_sets_audio_and_text_services();
    test_production_context_strategy_mismatch_rejected_before_trt_load();
    test_production_context_config_parse_failure_propagates();
    print_summary_and_exit_if_failures("test_audio_strategy_builder");
    return 0;
}
