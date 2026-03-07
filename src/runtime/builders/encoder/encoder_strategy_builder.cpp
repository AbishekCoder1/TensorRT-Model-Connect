#include "trtf/runtime/builders/encoder/encoder_strategy_builder.h"

#include "trtf/tokenizer.h"

#if TRTF_HAS_TRT
#include "runtime/trt/core/trt_common.h"
#include "runtime/trt/encoder/embedding_backend.h"
#include "runtime/trt/encoder/encoder_backend.h"
#include "runtime/trt/encoder/reranking_backend.h"
#include "runtime/trt/multimodal/image_preprocessor.h"
#include "runtime/trt/multimodal/vision_engine.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace trtf::runtime::builders::encoder {

namespace {

struct PortCheckResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

struct ConfigCheckResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;
    trtf::FastPathModelConfig config{};

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

struct TokenizerAssets {
    std::unique_ptr<trtf::ITokenizer> tokenizer;
    std::string temp_dir;

    TokenizerAssets() = default;

    TokenizerAssets(TokenizerAssets&& other) noexcept
        : tokenizer(std::move(other.tokenizer))
        , temp_dir(std::move(other.temp_dir))
    {
        other.temp_dir.clear();
    }

    TokenizerAssets& operator=(TokenizerAssets&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            tokenizer = std::move(other.tokenizer);
            temp_dir = std::move(other.temp_dir);
            other.temp_dir.clear();
        }
        return *this;
    }

    TokenizerAssets(const TokenizerAssets&) = delete;
    TokenizerAssets& operator=(const TokenizerAssets&) = delete;

    ~TokenizerAssets()
    {
        cleanup();
    }

private:
    void cleanup() noexcept
    {
        tokenizer.reset();
        if (!temp_dir.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir, ec);
            temp_dir.clear();
        }
    }
};

bool is_supported_strategy(std::string_view strategy)
{
    static constexpr std::array<std::string_view, 3> kStrategies = {
        "encoder_only",
        "embedding",
        "reranking",
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

ConfigCheckResult resolve_fast_path_config(const BuildContext& context, const IBundlePort& bundle_port)
{
    ConfigCheckResult result;

    if (context.config != nullptr)
    {
        result.config = *static_cast<const trtf::FastPathModelConfig*>(context.config);
    }
    else
    {
        const auto parsed = bundle_port.parse_fast_path_config(-1);
        if (!parsed.ok())
        {
            const BuildStatus status = (parsed.status == BundlePortStatus::kMissingSection)
                ? BuildStatus::kMissingDependency
                : BuildStatus::kInvalidArgument;
            result.status = status;
            result.message = parsed.message;
            return result;
        }
        result.config = parsed.value;
    }

    if (!result.config.runtime_strategy.empty() && result.config.runtime_strategy != context.strategy)
    {
        result.status = BuildStatus::kInvalidArgument;
        result.message = "Strategy mismatch: context strategy is '" + context.strategy
            + "' but bundle config declares '" + result.config.runtime_strategy + "'";
    }

    return result;
}

PortCheckResult validate_engine_for_section(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
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

    const auto deserialize = trt_port.deserialize_engine(runtime, section.value.data(), section.value.size());
    if (!deserialize.ok())
    {
        return {BuildStatus::kRuntimeError, deserialize.message};
    }

    const auto create_ctx = trt_port.create_execution_context(deserialize.value);
    if (!create_ctx.ok())
    {
        return {BuildStatus::kRuntimeError, create_ctx.message};
    }

    return {};
}

PortCheckResult validate_embedding_primary_engine(
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

    const auto deserialize = trt_port.deserialize_engine(runtime, section.value.data(), section.value.size());
    if (!deserialize.ok())
    {
        return {BuildStatus::kRuntimeError, deserialize.message};
    }

    const auto has_input_embed = trt_port.has_io_tensor_named(deserialize.value, "input_embed");
    if (!has_input_embed.ok())
    {
        return {BuildStatus::kRuntimeError, has_input_embed.message};
    }

    const auto create_ctx = trt_port.create_execution_context(deserialize.value);
    if (!create_ctx.ok())
    {
        return {BuildStatus::kRuntimeError, create_ctx.message};
    }

    return {};
}

PortCheckResult validate_optional_vision_engine(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view strategy)
{
    if (strategy != "embedding" || !bundle_port.has_section("vision_engine_plan"))
    {
        return {};
    }

    return validate_engine_for_section(bundle_port, trt_port, runtime, "vision_engine_plan");
}

PortCheckResult validate_strategy_dependencies(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view strategy)
{
    const auto primary = (strategy == "embedding")
        ? validate_embedding_primary_engine(bundle_port, trt_port, runtime)
        : validate_engine_for_section(bundle_port, trt_port, runtime, "engine_plan");
    if (!primary.ok())
    {
        return primary;
    }

    return validate_optional_vision_engine(bundle_port, trt_port, runtime, strategy);
}

std::vector<char> fetch_optional_section(const IBundlePort& bundle_port, std::string_view section_name)
{
    if (!bundle_port.has_section(section_name))
    {
        return {};
    }

    const auto section = bundle_port.fetch_section_bytes(section_name);
    if (!section.ok())
    {
        return {};
    }
    return section.value;
}

bool has_tokenizer_payload(
    const std::vector<char>& tokenizer_json,
    const std::vector<char>& vocab_json,
    const std::vector<char>& tokenizer_model)
{
    return !tokenizer_json.empty() || !vocab_json.empty() || !tokenizer_model.empty();
}

std::string create_tokenizer_temp_dir()
{
    char temp_pattern[] = "/tmp/trtf_runtime_tok_XXXXXX";
    char* created = mkdtemp(temp_pattern);
    if (created == nullptr)
    {
        return {};
    }
    return created;
}

void write_optional_section_file(
    const std::string& dir,
    const char* filename,
    const std::vector<char>& data)
{
    if (dir.empty() || data.empty())
    {
        return;
    }

    std::ofstream out(std::filesystem::path(dir) / filename, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        return;
    }

    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

TokenizerAssets try_extract_tokenizer_from_bundle(
    const IBundlePort& bundle_port,
    const std::string& hf_python)
{
    TokenizerAssets assets;

    const auto tokenizer_json = fetch_optional_section(bundle_port, "tokenizer.json");
    const auto tokenizer_config = fetch_optional_section(bundle_port, "tokenizer_config.json");
    const auto vocab_json = fetch_optional_section(bundle_port, "vocab.json");
    const auto merges_txt = fetch_optional_section(bundle_port, "merges.txt");
    const auto special_tokens = fetch_optional_section(bundle_port, "special_tokens_map.json");
    const auto tokenizer_model = fetch_optional_section(bundle_port, "tokenizer.model");
    const auto preprocessor_config = fetch_optional_section(bundle_port, "preprocessor_config.json");

    if (!has_tokenizer_payload(tokenizer_json, vocab_json, tokenizer_model))
    {
        return assets;
    }

    assets.temp_dir = create_tokenizer_temp_dir();
    if (assets.temp_dir.empty())
    {
        return assets;
    }

    write_optional_section_file(assets.temp_dir, "tokenizer.json", tokenizer_json);
    write_optional_section_file(assets.temp_dir, "tokenizer_config.json", tokenizer_config);
    write_optional_section_file(assets.temp_dir, "vocab.json", vocab_json);
    write_optional_section_file(assets.temp_dir, "merges.txt", merges_txt);
    write_optional_section_file(assets.temp_dir, "special_tokens_map.json", special_tokens);
    write_optional_section_file(assets.temp_dir, "tokenizer.model", tokenizer_model);
    write_optional_section_file(assets.temp_dir, "preprocessor_config.json", preprocessor_config);

    try
    {
        assets.tokenizer = trtf::CreateHfPythonTokenizer(
            assets.temp_dir,
            hf_python,
            /*add_special_tokens=*/true);
    }
    catch (...)
    {
        assets = TokenizerAssets{};
    }

    return assets;
}

std::string bytes_to_string(const std::vector<char>& bytes)
{
    return std::string(bytes.begin(), bytes.end());
}

#if TRTF_HAS_TRT

struct OwnedEngine {
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> context;
};

struct EngineLoadResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;
    OwnedEngine value;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

struct MetadataResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;
    bool value{false};

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

class EncoderTextService final : public ITextService {
public:
    EncoderTextService(std::unique_ptr<trtf::EncoderBackend> backend, TokenizerAssets tokenizer_assets)
        : mTokenizerAssets(std::move(tokenizer_assets))
        , mBackend(std::move(backend))
    {
    }

    const char* generate(const char* /*prompt*/, std::size_t /*max_new_tokens*/) override
    {
        mLastOutput.clear();
        return mLastOutput.c_str();
    }

    bool supports_encoding() const override
    {
        return mBackend != nullptr && mBackend->is_available();
    }

    const float* encode(const char* text, int32_t* out_seq_len, int32_t* out_hidden_size) override
    {
        if (mBackend == nullptr || text == nullptr)
        {
            return nullptr;
        }

        try
        {
            mLastResult = mBackend->encode(tokenize_or_empty(text));
            if (out_seq_len != nullptr)
            {
                *out_seq_len = mLastResult.seq_length;
            }
            if (out_hidden_size != nullptr)
            {
                *out_hidden_size = mLastResult.hidden_size;
            }
            return mLastResult.hidden_states.data();
        }
        catch (...)
        {
            return nullptr;
        }
    }

private:
    std::vector<int32_t> tokenize_or_empty(std::string_view text) const
    {
        if (mTokenizerAssets.tokenizer == nullptr)
        {
            return {};
        }
        return mTokenizerAssets.tokenizer->encode(std::string(text));
    }

    TokenizerAssets mTokenizerAssets;
    std::unique_ptr<trtf::EncoderBackend> mBackend;
    trtf::EncoderResult mLastResult;
    std::string mLastOutput;
};

class EmbeddingService final : public IEmbeddingService {
public:
    EmbeddingService(std::unique_ptr<trtf::EmbeddingBackend> backend, TokenizerAssets tokenizer_assets)
        : mTokenizerAssets(std::move(tokenizer_assets))
        , mBackend(std::move(backend))
    {
    }

    const float* embed(const char* text, int32_t* out_dim) override
    {
        if (mBackend == nullptr || text == nullptr)
        {
            return nullptr;
        }

        try
        {
            mLastResult = mBackend->embed(tokenize_or_empty(text));
            if (out_dim != nullptr)
            {
                *out_dim = mLastResult.embedding_dim;
            }
            return mLastResult.embedding.data();
        }
        catch (...)
        {
            return nullptr;
        }
    }

    const float* embed_image(const adapters::io::DecodedImage& image, int32_t* out_dim) override
    {
        if (mBackend == nullptr || !mBackend->has_vision() || image.empty())
        {
            return nullptr;
        }

        try
        {
            const auto prompt = trtf::format_vl_prompt("", mBackend->vl_config());
            mLastResult = mBackend->embed_with_image(tokenize_or_empty(prompt), image);
            if (out_dim != nullptr)
            {
                *out_dim = mLastResult.embedding_dim;
            }
            return mLastResult.embedding.data();
        }
        catch (...)
        {
            return nullptr;
        }
    }

    const float* embed_image_text(
        const char* text, const adapters::io::DecodedImage& image, int32_t* out_dim) override
    {
        if (mBackend == nullptr || !mBackend->has_vision() || text == nullptr || image.empty())
        {
            return nullptr;
        }

        try
        {
            const auto prompt = trtf::format_vl_prompt(std::string(text), mBackend->vl_config());
            mLastResult = mBackend->embed_with_image(tokenize_or_empty(prompt), image);
            if (out_dim != nullptr)
            {
                *out_dim = mLastResult.embedding_dim;
            }
            return mLastResult.embedding.data();
        }
        catch (...)
        {
            return nullptr;
        }
    }

private:
    std::vector<int32_t> tokenize_or_empty(std::string_view text) const
    {
        if (mTokenizerAssets.tokenizer == nullptr)
        {
            return {};
        }
        return mTokenizerAssets.tokenizer->encode(std::string(text));
    }

    TokenizerAssets mTokenizerAssets;
    std::unique_ptr<trtf::EmbeddingBackend> mBackend;
    trtf::EmbeddingResult mLastResult;
};

class RerankService final : public IRerankService {
public:
    RerankService(std::unique_ptr<trtf::RerankingBackend> backend, TokenizerAssets tokenizer_assets)
        : mTokenizerAssets(std::move(tokenizer_assets))
        , mBackend(std::move(backend))
    {
    }

    float rerank(const char* query, const char* document) override
    {
        if (mBackend == nullptr || query == nullptr || document == nullptr)
        {
            return 0.0F;
        }

        try
        {
            const std::string combined = std::string(query) + " " + std::string(document);
            return mBackend->rerank(tokenize_or_empty(combined)).score;
        }
        catch (...)
        {
            return 0.0F;
        }
    }

private:
    std::vector<int32_t> tokenize_or_empty(std::string_view text) const
    {
        if (mTokenizerAssets.tokenizer == nullptr)
        {
            return {};
        }
        return mTokenizerAssets.tokenizer->encode(std::string(text));
    }

    TokenizerAssets mTokenizerAssets;
    std::unique_ptr<trtf::RerankingBackend> mBackend;
};

EngineLoadResult load_engine_for_section(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view section_name)
{
    const auto section = bundle_port.fetch_section_bytes(section_name);
    if (!section.ok())
    {
        const BuildStatus status = (section.status == BundlePortStatus::kMissingSection)
            ? BuildStatus::kMissingDependency
            : BuildStatus::kInvalidArgument;
        return {status, section.message, {}};
    }

    const auto deserialize = trt_port.deserialize_engine(runtime, section.value.data(), section.value.size());
    if (!deserialize.ok())
    {
        return {BuildStatus::kRuntimeError, deserialize.message, {}};
    }

    OwnedEngine owned;
    owned.engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(deserialize.value);

    const auto create_ctx = trt_port.create_execution_context(owned.engine.get());
    if (!create_ctx.ok())
    {
        return {BuildStatus::kRuntimeError, create_ctx.message, {}};
    }

    owned.context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(create_ctx.value);
    return {BuildStatus::kOk, {}, std::move(owned)};
}

MetadataResult inspect_input_embed_support(const ITrtPort& trt_port, const OwnedEngine& engine)
{
    const auto has_input_embed = trt_port.has_io_tensor_named(engine.engine.get(), "input_embed");
    if (!has_input_embed.ok())
    {
        return {BuildStatus::kRuntimeError, has_input_embed.message, false};
    }
    return {BuildStatus::kOk, {}, has_input_embed.value};
}

trtf::VLPreprocessConfig parse_embedding_vl_config(const IBundlePort& bundle_port)
{
    const auto config_json = fetch_optional_section(bundle_port, "config.json");
    const auto preprocessor_json = fetch_optional_section(bundle_port, "preprocessor_config.json");
    return trtf::parse_vl_preprocess_config(
        bytes_to_string(config_json),
        bytes_to_string(preprocessor_json));
}

trtf::VisionStepEngine make_vision_step_engine(
    OwnedEngine vision_engine,
    const trtf::FastPathModelConfig& config)
{
    trtf::VisionStepEngine step_engine;
    step_engine.engine = std::move(vision_engine.engine);
    step_engine.context = std::move(vision_engine.context);
    step_engine.pixel_input_name = "pixel_values";
    step_engine.features_output_name = "vision_features";
    step_engine.num_output_features = config.num_image_pad_tokens;
    step_engine.feature_dim = (config.vision_output_dim > 0)
        ? config.vision_output_dim
        : config.hidden_size;
    return step_engine;
}

trtf::EncoderConfig make_encoder_backend_config(const trtf::FastPathModelConfig& config)
{
    trtf::EncoderConfig backend_config;
    backend_config.max_seq_length = config.max_cache_length;
    backend_config.hidden_size = config.hidden_size;
    backend_config.type_vocab_size = config.type_vocab_size;
    return backend_config;
}

trtf::EmbeddingConfig make_embedding_backend_config(
    const trtf::FastPathModelConfig& config,
    bool has_input_embed)
{
    trtf::EmbeddingConfig backend_config;
    backend_config.max_seq_length = config.max_cache_length;
    backend_config.hidden_size = config.hidden_size;
    backend_config.embedding_dim = (config.embedding_dim > 0) ? config.embedding_dim : config.hidden_size;
    backend_config.has_input_embed = has_input_embed;
    if (config.image_token_id >= 0)
    {
        backend_config.img_context_token_id = config.image_token_id;
    }
    return backend_config;
}

trtf::RerankingConfig make_reranking_backend_config(const trtf::FastPathModelConfig& config)
{
    trtf::RerankingConfig backend_config;
    backend_config.max_seq_length = config.max_cache_length;
    backend_config.hidden_size = config.hidden_size;
    return backend_config;
}

BuildResult build_encoder_only_services(
    const trtf::FastPathModelConfig& config,
    OwnedEngine primary,
    TokenizerAssets tokenizer_assets)
{
    PipelineServices services;
    auto backend = std::make_unique<trtf::EncoderBackend>(
        std::move(primary.engine),
        std::move(primary.context),
        make_encoder_backend_config(config));
    services.text = std::make_unique<EncoderTextService>(
        std::move(backend),
        std::move(tokenizer_assets));
    return BuildResult::Success(std::move(services));
}

BuildResult build_embedding_services(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngine primary,
    TokenizerAssets tokenizer_assets)
{
    const auto has_input_embed = inspect_input_embed_support(trt_port, primary);
    if (!has_input_embed.ok())
    {
        return BuildResult::Failure(has_input_embed.status, has_input_embed.message);
    }

    auto backend = std::make_unique<trtf::EmbeddingBackend>(
        std::move(primary.engine),
        std::move(primary.context),
        make_embedding_backend_config(config, has_input_embed.value));

    if (bundle_port.has_section("vision_engine_plan"))
    {
        auto vision = load_engine_for_section(
            bundle_port, trt_port, runtime, "vision_engine_plan");
        if (!vision.ok())
        {
            return BuildResult::Failure(vision.status, vision.message);
        }

        if (has_input_embed.value)
        {
            auto step_engine = std::make_unique<trtf::VisionStepEngine>(
                make_vision_step_engine(std::move(vision.value), config));
            backend->set_vision_engine(
                std::move(step_engine),
                parse_embedding_vl_config(bundle_port));
        }
    }

    PipelineServices services;
    services.embedding = std::make_unique<EmbeddingService>(
        std::move(backend),
        std::move(tokenizer_assets));
    return BuildResult::Success(std::move(services));
}

BuildResult build_reranking_services(
    const trtf::FastPathModelConfig& config,
    OwnedEngine primary,
    TokenizerAssets tokenizer_assets)
{
    auto backend = std::make_unique<trtf::RerankingBackend>(
        std::move(primary.engine),
        std::move(primary.context),
        make_reranking_backend_config(config));

    PipelineServices services;
    services.rerank = std::make_unique<RerankService>(
        std::move(backend),
        std::move(tokenizer_assets));
    return BuildResult::Success(std::move(services));
}

BuildResult build_production_encoder_services(
    const BuildContext& context,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config)
{
    auto primary = load_engine_for_section(bundle_port, trt_port, runtime, "engine_plan");
    if (!primary.ok())
    {
        return BuildResult::Failure(primary.status, primary.message);
    }

    TokenizerAssets tokenizer_assets = try_extract_tokenizer_from_bundle(
        bundle_port, context.hf_python);

    if (context.strategy == "encoder_only")
    {
        return build_encoder_only_services(
            config, std::move(primary.value), std::move(tokenizer_assets));
    }
    if (context.strategy == "embedding")
    {
        return build_embedding_services(
            bundle_port,
            trt_port,
            runtime,
            config,
            std::move(primary.value),
            std::move(tokenizer_assets));
    }
    return build_reranking_services(
        config, std::move(primary.value), std::move(tokenizer_assets));
}

#endif // TRTF_HAS_TRT

} // namespace

EncoderStrategyBuilder::EncoderStrategyBuilder(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    ComposeEncoderServicesFn compose_encoder_services)
    : mBundlePort(bundle_port)
    , mTrtPort(trt_port)
    , mRuntime(runtime)
    , mComposeEncoderServices(std::move(compose_encoder_services))
{
}

BuildResult EncoderStrategyBuilder::build(const BuildContext& context)
{
    if (!is_supported_strategy(context.strategy))
    {
        return BuildResult::Failure(
            BuildStatus::kUnsupportedStrategy,
            "Unsupported encoder strategy: " + context.strategy);
    }

    if (mRuntime == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "EncoderStrategyBuilder requires a non-null TensorRT runtime");
    }

    const auto config = resolve_fast_path_config(context, mBundlePort);
    if (!config.ok())
    {
        return BuildResult::Failure(config.status, config.message);
    }

    if (mComposeEncoderServices)
    {
        const auto validation = validate_strategy_dependencies(
            mBundlePort, mTrtPort, mRuntime, context.strategy);
        if (!validation.ok())
        {
            return BuildResult::Failure(validation.status, validation.message);
        }
        return compose_encoder_services(context, config.config);
    }

#if !TRTF_HAS_TRT
    (void) config;
    return BuildResult::Failure(
        BuildStatus::kMissingDependency,
        "EncoderStrategyBuilder requires TRTF_HAS_TRT for production service composition");
#else
    return build_production_encoder_services(
        context, mBundlePort, mTrtPort, mRuntime, config.config);
#endif
}

BuildResult EncoderStrategyBuilder::compose_encoder_services(
    const BuildContext& context,
    const trtf::FastPathModelConfig& config) const
{
    if (!mComposeEncoderServices)
    {
        return BuildResult::Failure(
            BuildStatus::kRuntimeError,
            "EncoderStrategyBuilder has no injected compose function");
    }

    auto result = mComposeEncoderServices(
        context, config, mBundlePort, mTrtPort, mRuntime);
    if (!result.ok() || !result.services.empty())
    {
        return result;
    }

    return BuildResult::Failure(
        BuildStatus::kRuntimeError,
        "EncoderStrategyBuilder compose override returned no services");
}

} // namespace trtf::runtime::builders::encoder
