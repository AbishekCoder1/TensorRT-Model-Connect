#include "trtf/runtime/builders/text/text_strategy_builder.h"

#include "test_strategy_builder_helpers.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using trtf::runtime::BuildContext;
using trtf::runtime::BuildStatus;
using trtf::runtime::builders::text::TextStrategyBuilder;
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

class ComposedTextService final : public trtf::runtime::ITextService {
public:
    explicit ComposedTextService(std::string prefix)
        : mPrefix(std::move(prefix))
    {
    }

    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        mLast = mPrefix + "|" + (prompt == nullptr ? "" : prompt) + "|" + std::to_string(max_new_tokens);
        return mLast.c_str();
    }

private:
    std::string mPrefix;
    std::string mLast;
};

BuildContext make_context(const std::string& strategy)
{
    BuildContext context;
    context.model_id = "test-model";
    context.strategy = strategy;
    return context;
}

void test_unsupported_strategy_rejected()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("encoder_only"));
    check(!result.ok(), "unsupported: result not ok");
    check(result.status == BuildStatus::kUnsupportedStrategy,
        "unsupported: status");
}

void test_missing_engine_plan_rejected()
{
    FakeBundlePort bundle;
    bundle.config.runtime_strategy = "decoder_kv_cache";
    FakeTrtPort trt;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "missing engine: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing engine: status");
}

void test_deserialize_failure_mapped_to_runtime_error()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "decoder_kv_cache";
    FakeTrtPort trt;
    trt.fail_deserialize_on_call = 1;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "deserialize failure: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "deserialize failure: status");
}

void test_context_failure_mapped_to_runtime_error()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "decoder_kv_cache";
    FakeTrtPort trt;
    trt.fail_context_on_call = 1;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "context failure: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "context failure: status");
}

void test_missing_logits_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "decoder_kv_cache";
    FakeTrtPort trt;
    trt.has_requested_tensor = false;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "missing logits: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing logits: status");
}

void test_missing_composition_context_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "decoder_kv_cache";
    FakeTrtPort trt;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
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

void test_success_uses_composer_service()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "decoder_kv_cache";
    bundle.config.hidden_size = 128;
    FakeTrtPort trt;

    int compose_calls = 0;
    std::string composed_strategy;
    int32_t composed_hidden_size = 0;
    TextStrategyBuilder builder(
        bundle,
        trt,
        fake_runtime_ptr(),
        [&](const BuildContext& context, const trtf::FastPathModelConfig& config,
            const trtf::runtime::IBundlePort&, const trtf::runtime::ITrtPort&,
            nvinfer1::IRuntime*) {
            ++compose_calls;
            composed_strategy = context.strategy;
            composed_hidden_size = config.hidden_size;

            trtf::runtime::PipelineServices services;
            services.text = std::make_unique<ComposedTextService>("composed");
            return trtf::runtime::BuildResult::Success(std::move(services));
        });

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(result.ok(), "success composer: result ok");
    check(result.status == BuildStatus::kOk, "success composer: status");
    check(result.services.text != nullptr, "success composer: text service created");
    check(compose_calls == 1, "success composer: composer called once");
    check(composed_strategy == "decoder_kv_cache",
        "success composer: strategy forwarded");
    check(composed_hidden_size == 128,
        "success composer: config forwarded");

    const char* generated = result.services.text->generate("hello", 4);
    check(generated != nullptr, "success composer: generate returns non-null");
    check(std::strcmp(generated, "composed|hello|4") == 0,
        "success composer: generated text comes from composed service");
}

void test_context_config_used_when_available()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.parse_status = trtf::runtime::BundlePortStatus::kMissingSection;
    bundle.parse_message = "config missing";
    FakeTrtPort trt;

    trtf::FastPathModelConfig config;
    config.runtime_strategy = "rwkv_recurrent";
    config.hidden_size = 256;

    int compose_calls = 0;
    TextStrategyBuilder builder(
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
            services.text = std::make_unique<ComposedTextService>("context");
            return trtf::runtime::BuildResult::Success(std::move(services));
        });

    auto context = make_context("rwkv_recurrent");
    context.config = &config;
    const auto result = builder.build(context);
    check(result.ok(), "context config: result ok");
    check(compose_calls == 1, "context config: composer called");
}

void test_config_strategy_mismatch_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "rwkv_recurrent";
    FakeTrtPort trt;
    TextStrategyBuilder builder(bundle, trt, fake_runtime_ptr());

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "config mismatch: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "config mismatch: status");
}

} // namespace

int main()
{
    test_unsupported_strategy_rejected();
    test_missing_engine_plan_rejected();
    test_deserialize_failure_mapped_to_runtime_error();
    test_context_failure_mapped_to_runtime_error();
    test_missing_logits_rejected();
    test_missing_composition_context_rejected();
    test_success_uses_composer_service();
    test_context_config_used_when_available();
    test_config_strategy_mismatch_rejected();
    print_summary_and_exit_if_failures("test_text_strategy_builder");
    return 0;
}
