#include "trtf/runtime/builders/encoder/encoder_strategy_builder.h"

#include "test_strategy_builder_helpers.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using trtf::runtime::BuildContext;
using trtf::runtime::BuildStatus;
using trtf::runtime::IEmbeddingService;
using trtf::runtime::IRerankService;
using trtf::runtime::ITextService;
using trtf::runtime::PipelineServices;
using trtf::runtime::builders::encoder::EncoderStrategyBuilder;
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

class FakeEncodingTextService final : public ITextService {
public:
    const char* generate(const char* /*prompt*/, std::size_t /*max_new_tokens*/) override
    {
        mLast.clear();
        return mLast.c_str();
    }

    bool supports_encoding() const override
    {
        return true;
    }

    const float* encode(const char* /*text*/, int32_t* out_seq_len, int32_t* out_hidden_size) override
    {
        if (out_seq_len != nullptr)
        {
            *out_seq_len = 1;
        }
        if (out_hidden_size != nullptr)
        {
            *out_hidden_size = 4;
        }
        return mValues;
    }

private:
    float mValues[4]{0.1F, 0.2F, 0.3F, 0.4F};
    std::string mLast;
};

class FakeEmbeddingService final : public IEmbeddingService {
public:
    const float* embed(const char* /*text*/, int32_t* out_dim) override
    {
        if (out_dim != nullptr)
        {
            *out_dim = 3;
        }
        return mValues;
    }

private:
    float mValues[3]{1.0F, 2.0F, 3.0F};
};

class FakeRerankService final : public IRerankService {
public:
    float rerank(const char* /*query*/, const char* /*document*/) override
    {
        return 0.5F;
    }
};

BuildContext make_context(const std::string& strategy)
{
    BuildContext context;
    context.model_id = "encoder-model";
    context.strategy = strategy;
    return context;
}

EncoderStrategyBuilder make_builder(FakeBundlePort& bundle, FakeTrtPort& trt)
{
    return EncoderStrategyBuilder(
        bundle,
        trt,
        fake_runtime_ptr(),
        [](const BuildContext& context, const trtf::FastPathModelConfig&,
            const trtf::runtime::IBundlePort&, const trtf::runtime::ITrtPort&,
            nvinfer1::IRuntime*) {
            PipelineServices services;
            if (context.strategy == "encoder_only")
            {
                services.text = std::make_unique<FakeEncodingTextService>();
            }
            else if (context.strategy == "embedding")
            {
                services.embedding = std::make_unique<FakeEmbeddingService>();
            }
            else if (context.strategy == "reranking")
            {
                services.rerank = std::make_unique<FakeRerankService>();
            }
            return trtf::runtime::BuildResult::Success(std::move(services));
        });
}

void test_unsupported_strategy_rejected()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("decoder_kv_cache"));
    check(!result.ok(), "unsupported: result not ok");
    check(result.status == BuildStatus::kUnsupportedStrategy,
        "unsupported: status");
}

void test_missing_runtime_rejected()
{
    FakeBundlePort bundle;
    FakeTrtPort trt;
    EncoderStrategyBuilder builder(bundle, trt, nullptr);

    const auto result = builder.build(make_context("encoder_only"));
    check(!result.ok(), "missing runtime: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing runtime: status");
}

void test_strategy_mismatch_rejected()
{
    FakeBundlePort bundle;
    bundle.config.runtime_strategy = "embedding";
    FakeTrtPort trt;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("encoder_only"));
    check(!result.ok(), "strategy mismatch: result not ok");
    check(result.status == BuildStatus::kInvalidArgument,
        "strategy mismatch: status");
}

void test_missing_primary_engine_rejected()
{
    FakeBundlePort bundle;
    bundle.config.runtime_strategy = "encoder_only";
    FakeTrtPort trt;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("encoder_only"));
    check(!result.ok(), "missing primary: result not ok");
    check(result.status == BuildStatus::kMissingDependency,
        "missing primary: status");
}

void test_deserialize_failure_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "encoder_only";
    FakeTrtPort trt;
    trt.fail_deserialize_on_call = 1;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("encoder_only"));
    check(!result.ok(), "deserialize: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "deserialize: status");
}

void test_embedding_metadata_failure_rejected()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "embedding";
    FakeTrtPort trt;
    trt.metadata_failure = true;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("embedding"));
    check(!result.ok(), "metadata failure: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "metadata failure: status");
    check(trt.has_io_calls == 1, "metadata failure: queried engine metadata");
    check(trt.last_tensor_name == "input_embed", "metadata failure: queried input_embed");
}

void test_optional_vision_failure_rejected_for_embedding()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.sections["vision_engine_plan"] = bytes_from_text("vision-plan");
    bundle.config.runtime_strategy = "embedding";
    FakeTrtPort trt;
    trt.fail_deserialize_on_call = 2;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("embedding"));
    check(!result.ok(), "optional vision failure: result not ok");
    check(result.status == BuildStatus::kRuntimeError,
        "optional vision failure: status");
    check(trt.deserialize_calls == 2, "optional vision failure: primary and vision plans checked");
}

void test_encoder_only_success_sets_encoding_capability()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "encoder_only";
    FakeTrtPort trt;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("encoder_only"));
    check(result.ok(), "encoder_only success: result ok");
    check(result.services.text != nullptr, "encoder_only success: text service set");
    if (result.services.text != nullptr)
    {
        check(result.services.text->supports_encoding(), "encoder_only success: supports encoding");
    }
    check(trt.has_io_calls == 0, "encoder_only success: no embedding metadata query");
}

void test_embedding_success_sets_embedding_service()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "embedding";
    FakeTrtPort trt;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("embedding"));
    check(result.ok(), "embedding success: result ok");
    check(result.services.embedding != nullptr, "embedding success: embedding service set");
    check(trt.has_io_calls == 1, "embedding success: input_embed metadata queried");

    if (result.services.embedding != nullptr)
    {
        int32_t dim = 0;
        const float* values = result.services.embedding->embed("hello", &dim);
        check(values != nullptr, "embedding success: embed returns values");
        check(dim == 3, "embedding success: expected dimension");
    }
}

void test_reranking_success_sets_rerank_service()
{
    FakeBundlePort bundle;
    bundle.sections["engine_plan"] = bytes_from_text("plan");
    bundle.config.runtime_strategy = "reranking";
    FakeTrtPort trt;
    auto builder = make_builder(bundle, trt);

    const auto result = builder.build(make_context("reranking"));
    check(result.ok(), "reranking success: result ok");
    check(result.services.rerank != nullptr, "reranking success: rerank service set");
    if (result.services.rerank != nullptr)
    {
        check(result.services.rerank->rerank("q", "d") == 0.5F,
            "reranking success: score");
    }
}

} // namespace

int main()
{
    test_unsupported_strategy_rejected();
    test_missing_runtime_rejected();
    test_strategy_mismatch_rejected();
    test_missing_primary_engine_rejected();
    test_deserialize_failure_rejected();
    test_embedding_metadata_failure_rejected();
    test_optional_vision_failure_rejected_for_embedding();
    test_encoder_only_success_sets_encoding_capability();
    test_embedding_success_sets_embedding_service();
    test_reranking_success_sets_rerank_service();
    print_summary_and_exit_if_failures("test_encoder_strategy_builder");
    return 0;
}
