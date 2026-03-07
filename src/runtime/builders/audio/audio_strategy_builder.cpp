#include "trtf/runtime/builders/audio/audio_strategy_builder.h"

#include "cabi/bundle/bundle_helpers.h"
#include "runtime/builders/audio/audio_bundle_validation.h"
#include "runtime/services/audio/audio_runtime_services.h"
#include "runtime/services/text/generation_text_service.h"
#include "trtf/tokenizer.h"

#if TRTF_HAS_TRT
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/core/trt_backend_shared.h"
#endif

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtf::runtime::builders::audio {

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

struct ConfigLoadResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;
    trtf::FastPathModelConfig value{};

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

struct OwnedEngineContext {
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> context;
};

struct OwnedEngineLoadResult {
    BuildStatus status{BuildStatus::kOk};
    std::string message;
    OwnedEngineContext value;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }
};

BuildResult failure_from_engine_load(const OwnedEngineLoadResult& result)
{
    return BuildResult::Failure(result.status, result.message);
}

bool has_production_bundle_context(const BuildContext& context)
{
    return context.sections != nullptr;
}

const trtf::BundleSections* bundle_sections_from_context(const BuildContext& context)
{
    return static_cast<const trtf::BundleSections*>(context.sections);
}

bool has_data(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

std::string lowercase_ascii(std::string_view text)
{
    std::string lower(text);
    for (char& ch : lower)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return lower;
}

BuildStatus map_factory_error_status(std::string_view message)
{
    const std::string lower = lowercase_ascii(message);
    if (lower.find("missing") != std::string::npos
        || lower.find("no tokenizer files") != std::string::npos)
    {
        return BuildStatus::kMissingDependency;
    }
    if (lower.find("invalid") != std::string::npos
        || lower.find("empty") != std::string::npos
        || lower.find("parse") != std::string::npos
        || lower.find("mismatch") != std::string::npos)
    {
        return BuildStatus::kInvalidArgument;
    }
    return BuildStatus::kRuntimeError;
}

ConfigLoadResult resolve_fast_path_config(const BuildContext& context, const IBundlePort& bundle_port)
{
    if (context.config != nullptr)
    {
        return {BuildStatus::kOk, {}, *static_cast<const trtf::FastPathModelConfig*>(context.config)};
    }

    const auto parsed = bundle_port.parse_fast_path_config(-1);
    if (!parsed.ok())
    {
        const BuildStatus status = (parsed.status == BundlePortStatus::kMissingSection)
            ? BuildStatus::kMissingDependency
            : BuildStatus::kInvalidArgument;
        return {status, parsed.message, {}};
    }

    return {BuildStatus::kOk, {}, parsed.value};
}

OwnedEngineLoadResult load_engine_context_for_section(
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

    const auto create_ctx = trt_port.create_execution_context(deserialize.value);
    if (!create_ctx.ok())
    {
        trtf::TrtDeleter<nvinfer1::ICudaEngine>{}(deserialize.value);
        return {BuildStatus::kRuntimeError, create_ctx.message, {}};
    }

    OwnedEngineContext engine_context;
    engine_context.engine.reset(deserialize.value);
    engine_context.context.reset(create_ctx.value);
    return {BuildStatus::kOk, {}, std::move(engine_context)};
}

OwnedEngineLoadResult maybe_load_engine_context_for_section(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view section_name)
{
    if (!bundle_port.has_section(section_name))
    {
        return {BuildStatus::kOk, {}, {}};
    }
    return load_engine_context_for_section(bundle_port, trt_port, runtime, section_name);
}

std::vector<float> load_required_float_section(
    const std::vector<char>* data,
    const char* section_name,
    const std::string& bundle_path,
    bool log_load = true)
{
    if (!has_data(data))
    {
        throw std::runtime_error(
            std::string("Bundle missing ") + section_name + " section: " + bundle_path);
    }

    const auto n_floats = data->size() / sizeof(float);
    std::vector<float> values(n_floats);
    std::memcpy(values.data(), data->data(), data->size());
    if (log_load)
    {
        std::cerr << "[trtf] Loaded " << section_name << " (" << n_floats << " floats)" << std::endl;
    }
    return values;
}

std::vector<int32_t> load_optional_int32_section(const std::vector<char>* data)
{
    if (!has_data(data))
    {
        return {};
    }

    const auto count = data->size() / sizeof(int32_t);
    std::vector<int32_t> values(count);
    std::memcpy(values.data(), data->data(), data->size());
    return values;
}

trtf::TokenizerResult try_extract_tokenizer(
    const trtf::BundleSections& sections,
    const std::string& hf_python,
    bool add_special_tokens,
    const char* warning_context)
{
    trtf::TokenizerResult tok = {nullptr, ""};
    try
    {
        tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, add_special_tokens);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for " << warning_context
                  << " (" << e.what() << ")" << std::endl;
    }
    return tok;
}

trtf::TokenizerResult create_magpie_tokenizer_or_throw(const trtf::BundleSections& sections)
{
    trtf::TokenizerResult tok = {nullptr, ""};
    if (!has_data(sections.magpie_ipa_phoneme_dict_data) || !has_data(sections.magpie_ipa_vocab_data))
    {
        throw std::runtime_error(
            "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, magpie_ipa_vocab). Rebuild the bundle with the latest trtf-build.");
    }

    tok.tokenizer = trtf::CreateIpaTokenizer(
        sections.magpie_ipa_phoneme_dict_data->data(), sections.magpie_ipa_phoneme_dict_data->size(),
        has_data(sections.magpie_ipa_heteronyms_data) ? sections.magpie_ipa_heteronyms_data->data() : nullptr,
        has_data(sections.magpie_ipa_heteronyms_data) ? sections.magpie_ipa_heteronyms_data->size() : 0,
        sections.magpie_ipa_vocab_data->data(), sections.magpie_ipa_vocab_data->size(),
        has_data(sections.magpie_ipa_config_data) ? sections.magpie_ipa_config_data->data() : nullptr,
        has_data(sections.magpie_ipa_config_data) ? sections.magpie_ipa_config_data->size() : 0);
    return tok;
}

std::vector<float> load_required_bark_embed(
    const std::vector<char>* data,
    const char* missing_section_name,
    const char* log_name,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& bundle_path)
{
    auto values = load_required_float_section(data, missing_section_name, bundle_path, false);
    const auto n_floats = values.size();
    std::cerr << "[trtf] Loaded " << log_name << " embedding table ("
              << n_floats / std::max(fp_cfg.hidden_size, 1)
              << " x " << fp_cfg.hidden_size << ")" << std::endl;
    return values;
}

trtf::FastPathModelConfig make_bark_coarse_config(const trtf::FastPathModelConfig& fp_cfg)
{
    trtf::FastPathModelConfig coarse_cfg = fp_cfg;
    if (fp_cfg.coarse_hidden_size > 0)
    {
        coarse_cfg.hidden_size = fp_cfg.coarse_hidden_size;
    }
    if (fp_cfg.coarse_num_layers > 0)
    {
        coarse_cfg.num_layers = fp_cfg.coarse_num_layers;
    }
    if (fp_cfg.coarse_num_heads > 0)
    {
        coarse_cfg.num_heads = fp_cfg.coarse_num_heads;
        coarse_cfg.num_kv_heads = fp_cfg.coarse_num_heads;
    }
    coarse_cfg.vocab_size = fp_cfg.coarse_input_vocab;
    coarse_cfg.head_dim = coarse_cfg.hidden_size / std::max(coarse_cfg.num_heads, 1);
    coarse_cfg.attention_size = coarse_cfg.num_heads * coarse_cfg.head_dim;
    coarse_cfg.max_cache_length = fp_cfg.coarse_max_cache_length;
    return coarse_cfg;
}

std::unique_ptr<trtf::DecoderStepEngine> create_required_bark_coarse_engine(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& bundle_path)
{
    auto coarse = load_engine_context_for_section(bundle_port, trt_port, runtime, "coarse_engine_plan");
    if (!coarse.ok())
    {
        throw std::runtime_error(coarse.message.empty()
                ? std::string("Bundle missing coarse_engine section: ") + bundle_path
                : coarse.message);
    }

    auto coarse_engine = trtf::make_decoder_engine(
        std::move(coarse.value.engine), std::move(coarse.value.context), make_bark_coarse_config(fp_cfg));
    if (!trtf::has_all_required_tensors(*coarse_engine))
    {
        throw std::runtime_error("Bundle coarse engine missing required tensors: " + bundle_path);
    }
    return coarse_engine;
}

template <typename BarkBackendT>
void maybe_load_bark_fine_embeddings(BarkBackendT& bark_backend, const trtf::BundleSections& sections)
{
    if (!has_data(sections.fine_embed_data))
    {
        return;
    }

    const auto n_floats = sections.fine_embed_data->size() / sizeof(float);
    std::vector<float> fine_embed(n_floats);
    std::memcpy(fine_embed.data(), sections.fine_embed_data->data(), sections.fine_embed_data->size());

    std::vector<float> fine_pos_embed;
    if (has_data(sections.fine_position_embed_data))
    {
        const auto pos_floats = sections.fine_position_embed_data->size() / sizeof(float);
        fine_pos_embed.resize(pos_floats);
        std::memcpy(
            fine_pos_embed.data(),
            sections.fine_position_embed_data->data(),
            sections.fine_position_embed_data->size());
    }

    bark_backend.set_fine_embeddings(std::move(fine_embed), std::move(fine_pos_embed));
}

trtf::FastPathModelConfig make_omni_talker_config(const trtf::FastPathModelConfig& fp_cfg)
{
    trtf::FastPathModelConfig talker_cfg;
    talker_cfg.hidden_size = fp_cfg.omni_talker_hidden_size;
    talker_cfg.num_layers = fp_cfg.omni_talker_num_layers;
    talker_cfg.num_heads = std::max(fp_cfg.omni_talker_hidden_size / 64, 1);
    talker_cfg.num_kv_heads = talker_cfg.num_heads;
    talker_cfg.head_dim = talker_cfg.hidden_size / std::max(talker_cfg.num_heads, 1);
    talker_cfg.attention_size = talker_cfg.num_heads * talker_cfg.head_dim;
    talker_cfg.max_cache_length = fp_cfg.omni_talker_max_cache_length;
    talker_cfg.vocab_size = fp_cfg.omni_codebook_size * fp_cfg.omni_n_codebooks;
    return talker_cfg;
}

trtf::WhisperConfig make_whisper_config(const trtf::FastPathModelConfig& fp_cfg)
{
    trtf::WhisperConfig whisper_cfg;
    whisper_cfg.num_mel_bins = fp_cfg.num_mel_bins;
    whisper_cfg.max_source_positions = fp_cfg.max_source_positions;
    whisper_cfg.max_target_positions = fp_cfg.max_target_positions;
    whisper_cfg.encoder_layers = fp_cfg.encoder_layers;
    whisper_cfg.decoder_layers = fp_cfg.decoder_layers;
    whisper_cfg.mel_length = fp_cfg.mel_length;
    whisper_cfg.decoder_start_token_ids = fp_cfg.decoder_start_token_ids;
    if (fp_cfg.eot_token_id >= 0)
    {
        whisper_cfg.eot_token_id = fp_cfg.eot_token_id;
    }
    return whisper_cfg;
}

template <typename SpeechBackendT>
trtf::SpeechConfig& mutable_speech_config(SpeechBackendT& speech_backend)
{
    return const_cast<trtf::SpeechConfig&>(speech_backend.config());
}

void maybe_load_speech_depth_projection(
    trtf::SpeechConfig& cfg_ref,
    const std::vector<char>* proj_data,
    const trtf::FastPathModelConfig& fp_cfg)
{
    if (!has_data(proj_data))
    {
        return;
    }

    const auto num_floats = proj_data->size() / sizeof(float);
    cfg_ref.depth_projection.resize(num_floats);
    std::memcpy(cfg_ref.depth_projection.data(), proj_data->data(), proj_data->size());
    cfg_ref.temporal_hidden_for_proj = fp_cfg.hidden_size;
}

void maybe_load_speech_audio_embeddings(
    trtf::SpeechConfig& cfg_ref,
    const std::vector<char>* emb_data,
    const trtf::FastPathModelConfig& fp_cfg)
{
    if (!has_data(emb_data))
    {
        return;
    }

    const auto num_floats = emb_data->size() / sizeof(float);
    cfg_ref.audio_embeddings.resize(num_floats);
    std::memcpy(cfg_ref.audio_embeddings.data(), emb_data->data(), emb_data->size());
    if (fp_cfg.speech_num_codebooks > 0 && fp_cfg.hidden_size > 0)
    {
        cfg_ref.audio_vocab_size = static_cast<int32_t>(
            num_floats / (static_cast<std::size_t>(fp_cfg.speech_num_codebooks) * fp_cfg.hidden_size));
    }
}

void maybe_load_temporal_text_embedding(
    trtf::SpeechConfig& cfg_ref,
    const std::vector<char>* tte_data,
    const trtf::FastPathModelConfig& fp_cfg)
{
    if (!has_data(tte_data))
    {
        return;
    }

    const auto num_floats = tte_data->size() / sizeof(float);
    cfg_ref.temporal_text_embedding.resize(num_floats);
    std::memcpy(cfg_ref.temporal_text_embedding.data(), tte_data->data(), tte_data->size());
    if (fp_cfg.hidden_size > 0)
    {
        cfg_ref.temporal_text_vocab = static_cast<int32_t>(num_floats / static_cast<std::size_t>(fp_cfg.hidden_size));
    }
}

void maybe_load_depth_text_embedding(
    trtf::SpeechConfig& cfg_ref,
    const std::vector<char>* depth_data,
    const trtf::FastPathModelConfig& fp_cfg)
{
    if (!has_data(depth_data))
    {
        return;
    }

    const auto num_floats = depth_data->size() / sizeof(float);
    cfg_ref.depth_text_embedding.resize(num_floats);
    std::memcpy(cfg_ref.depth_text_embedding.data(), depth_data->data(), depth_data->size());
    if (fp_cfg.speech_depth_hidden_size > 0)
    {
        cfg_ref.depth_text_vocab = static_cast<int32_t>(num_floats / static_cast<std::size_t>(fp_cfg.speech_depth_hidden_size));
    }
}

void maybe_load_depth_audio_embeddings(
    trtf::SpeechConfig& cfg_ref,
    const std::vector<char>* dae_data,
    const trtf::FastPathModelConfig& fp_cfg)
{
    if (!has_data(dae_data))
    {
        return;
    }

    const auto num_floats = dae_data->size() / sizeof(float);
    cfg_ref.depth_audio_embeddings.resize(num_floats);
    std::memcpy(cfg_ref.depth_audio_embeddings.data(), dae_data->data(), dae_data->size());
    const int32_t audio_vocab = cfg_ref.audio_vocab_size;
    const int32_t depth_hidden = fp_cfg.speech_depth_hidden_size;
    if (audio_vocab > 0 && depth_hidden > 0)
    {
        cfg_ref.num_depformer_emb = static_cast<int32_t>(
            num_floats / (static_cast<std::size_t>(audio_vocab) * depth_hidden));
    }
}

void load_optional_speech_tables(
    trtf::SpeechConfig& cfg_ref,
    const trtf::BundleSections& sections,
    const trtf::FastPathModelConfig& fp_cfg)
{
    maybe_load_speech_depth_projection(cfg_ref, sections.depth_projection_data, fp_cfg);
    maybe_load_speech_audio_embeddings(cfg_ref, sections.audio_embeddings_data, fp_cfg);
    maybe_load_temporal_text_embedding(cfg_ref, sections.temporal_text_embedding_data, fp_cfg);
    maybe_load_depth_text_embedding(cfg_ref, sections.depth_text_embedding_data, fp_cfg);
    maybe_load_depth_audio_embeddings(cfg_ref, sections.depth_audio_embeddings_data, fp_cfg);
}

trtf::FastPathModelConfig make_speech_depth_config(const trtf::FastPathModelConfig& fp_cfg)
{
    trtf::FastPathModelConfig depth_cfg = fp_cfg;
    if (fp_cfg.speech_depth_hidden_size > 0)
    {
        depth_cfg.hidden_size = fp_cfg.speech_depth_hidden_size;
    }
    if (fp_cfg.speech_depth_num_layers > 0)
    {
        depth_cfg.num_layers = fp_cfg.speech_depth_num_layers;
    }
    if (fp_cfg.speech_depth_num_heads > 0)
    {
        depth_cfg.num_heads = fp_cfg.speech_depth_num_heads;
        depth_cfg.num_kv_heads = fp_cfg.speech_depth_num_kv_heads > 0
            ? fp_cfg.speech_depth_num_kv_heads
            : fp_cfg.speech_depth_num_heads;
    }
    depth_cfg.vocab_size = fp_cfg.speech_codebook_size;
    depth_cfg.head_dim = depth_cfg.hidden_size / std::max(depth_cfg.num_heads, 1);
    depth_cfg.attention_size = depth_cfg.num_heads * depth_cfg.head_dim;
    depth_cfg.max_cache_length = fp_cfg.speech_num_codebooks + 2;
    return depth_cfg;
}

template <typename SpeechBackendT>
void attach_speech_depth_engines(
    SpeechBackendT& speech_backend,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& depth_cfg)
{
    if (!sections.depth_engine_plans.empty())
    {
        for (std::size_t cb = 0; cb < sections.depth_engine_plans.size(); ++cb)
        {
            const std::string section_name = "depth_engine_plan_" + std::to_string(cb);
            if (!bundle_port.has_section(section_name))
            {
                continue;
            }
            auto depth = load_engine_context_for_section(bundle_port, trt_port, runtime, section_name);
            if (!depth.ok())
            {
                continue;
            }
            auto depth_engine = trtf::make_decoder_engine(
                std::move(depth.value.engine), std::move(depth.value.context), depth_cfg);
            speech_backend.set_depth_engine(static_cast<int32_t>(cb), std::move(depth_engine));
        }
        return;
    }

    if (!bundle_port.has_section("depth_engine_plan"))
    {
        return;
    }
    auto depth = load_engine_context_for_section(bundle_port, trt_port, runtime, "depth_engine_plan");
    if (!depth.ok())
    {
        return;
    }
    auto depth_engine = trtf::make_decoder_engine(
        std::move(depth.value.engine), std::move(depth.value.context), depth_cfg);
    speech_backend.set_depth_engine(std::move(depth_engine));
}

BuildResult build_magpie_audio_services(
    const BuildContext& /*context*/,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    auto decoder_engine = trtf::make_decoder_engine(
        std::move(primary.engine), std::move(primary.context), config);
    auto encoder = load_engine_context_for_section(bundle_port, trt_port, runtime, "vision_engine_plan");
    if (!encoder.ok())
    {
        return failure_from_engine_load(encoder);
    }

    auto audio_embed = load_required_float_section(
        sections.magpie_audio_embed_data, "magpie_audio_embed", bundle_source);
    auto text_embed = load_required_float_section(
        sections.magpie_text_embed_data, "magpie_text_embed", bundle_source);
    auto context_embed = load_required_float_section(
        sections.magpie_context_embed_data, "magpie_context_embed", bundle_source);
    auto context_lengths = load_optional_int32_section(sections.magpie_context_lengths_data);

    auto backend = trtf::CreateMagpieTTSBackend(
        std::move(decoder_engine),
        std::move(encoder.value.engine),
        std::move(encoder.value.context),
        std::move(audio_embed),
        std::move(text_embed),
        std::move(context_embed),
        std::move(context_lengths),
        config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create MagpieTTS backend from bundle engines");
    }

    auto codec = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "codec_engine_plan");
    if (codec.ok() && codec.value.engine != nullptr)
    {
        backend->set_codec_engine(std::move(codec.value.engine), std::move(codec.value.context));
    }

    auto tokenizer = create_magpie_tokenizer_or_throw(sections);
    auto tokenizer_temp_dir = std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(
        std::move(tokenizer.temp_dir));
    auto shared_tokenizer = std::shared_ptr<trtf::ITokenizer>(std::move(tokenizer.tokenizer));
    PipelineServices services;
    services.audio = std::make_unique<trtf::runtime::services::audio::MagpieAudioService>(
        std::make_unique<trtf::runtime::services::common::MagpieAudioGenerationPort>(std::move(backend)),
        std::move(shared_tokenizer),
        std::move(tokenizer_temp_dir),
        500);
    return BuildResult::Success(std::move(services));
}

BuildResult build_bark_audio_services(
    const BuildContext& context,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    auto semantic_engine = trtf::make_decoder_engine(
        std::move(primary.engine), std::move(primary.context), config);
    if (!trtf::has_all_required_tensors(*semantic_engine))
    {
        throw std::runtime_error("Bundle engine missing required semantic tensors: " + bundle_source);
    }

    auto semantic_embed = load_required_bark_embed(
        sections.semantic_embed_data, "semantic_embed", "semantic", config, bundle_source);
    auto coarse_embed = load_required_bark_embed(
        sections.coarse_embed_data, "coarse_embed", "coarse", config, bundle_source);
    auto coarse_engine = create_required_bark_coarse_engine(
        bundle_port, trt_port, runtime, config, bundle_source);

    auto backend = trtf::CreateBarkBackend(
        std::move(semantic_engine),
        std::move(coarse_engine),
        std::move(semantic_embed),
        std::move(coarse_embed),
        config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Bark backend from bundle engines");
    }

    auto codec = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "codec_engine_plan");
    if (codec.ok() && codec.value.engine != nullptr)
    {
        backend->set_codec_engine(std::move(codec.value.engine), std::move(codec.value.context));
    }
    auto fine = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "fine_engine_plan");
    if (fine.ok() && fine.value.engine != nullptr)
    {
        backend->set_fine_engine(std::move(fine.value.engine), std::move(fine.value.context));
    }
    maybe_load_bark_fine_embeddings(*backend, sections);

    auto tokenizer = try_extract_tokenizer(sections, context.hf_python, false, "Bark");
    auto tokenizer_temp_dir = std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(
        std::move(tokenizer.temp_dir));
    auto shared_tokenizer = std::shared_ptr<trtf::ITokenizer>(std::move(tokenizer.tokenizer));
    PipelineServices services;
    services.audio = std::make_unique<trtf::runtime::services::audio::BarkAudioService>(
        std::make_unique<trtf::runtime::services::common::BarkAudioGenerationPort>(std::move(backend)),
        std::move(shared_tokenizer),
        std::move(tokenizer_temp_dir),
        768);
    return BuildResult::Success(std::move(services));
}

BuildResult build_text_to_audio_services(
    const BuildContext& context,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    if (config.is_magpie_tts)
    {
        validate_text_to_audio_bundle_sections(
            TextToAudioBundleKind::kMagpieTts, sections, bundle_source);
        return build_magpie_audio_services(
            context,
            sections,
            bundle_port,
            trt_port,
            runtime,
            config,
            std::move(primary),
            bundle_source);
    }

    validate_text_to_audio_bundle_sections(
        TextToAudioBundleKind::kBark, sections, bundle_source);
    return build_bark_audio_services(
        context,
        sections,
        bundle_port,
        trt_port,
        runtime,
        config,
        std::move(primary),
        bundle_source);
}

BuildResult build_whisper_transcription_services(
    const BuildContext& context,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    auto decoder_engine = trtf::make_decoder_engine(
        std::move(primary.engine), std::move(primary.context), config);
    if (!trtf::has_all_required_tensors(*decoder_engine))
    {
        throw std::runtime_error("Bundle engine missing required decoder tensors: " + bundle_source);
    }

    OwnedEngineContext encoder_context;
    if (bundle_port.has_section("vision_engine_plan"))
    {
        auto encoder = load_engine_context_for_section(
            bundle_port, trt_port, runtime, "vision_engine_plan");
        if (!encoder.ok())
        {
            return failure_from_engine_load(encoder);
        }
        encoder_context = std::move(encoder.value);
    }

    auto tokenizer = trtf::extract_tokenizer_from_bundle(sections, context.hf_python);
    auto backend = trtf::CreateWhisperBackend(
        std::move(decoder_engine),
        std::move(encoder_context.engine),
        std::move(encoder_context.context),
        make_whisper_config(config),
        config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Whisper TRT backend from bundle engine");
    }

    auto mel_fb = trtf::load_mel_filterbank(sections);
    auto tokenizer_temp_dir = std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(
        std::move(tokenizer.temp_dir));
    auto shared_tokenizer = std::shared_ptr<trtf::ITokenizer>(std::move(tokenizer.tokenizer));
    PipelineServices services;
    services.transcription = std::make_unique<trtf::runtime::services::audio::WhisperTranscriptionService>(
        std::make_unique<trtf::runtime::services::common::WhisperTranscriptionPort>(std::move(backend)),
        std::move(shared_tokenizer),
        std::move(tokenizer_temp_dir),
        trtf::runtime::services::common::MelSpectrogramConfig{
            std::move(mel_fb.data),
            mel_fb.n_freq_bins,
            mel_fb.n_mel_bins,
            config.mel_n_fft,
            config.mel_hop_length,
            config.mel_chunk_length,
            config.mel_sampling_rate,
        },
        224);
    return BuildResult::Success(std::move(services));
}

BuildResult build_speech_to_speech_services(
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    const std::string& hf_python,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    auto temporal_engine = trtf::make_decoder_engine(
        std::move(primary.engine), std::move(primary.context), config);
    if (!trtf::has_all_required_tensors(*temporal_engine))
    {
        throw std::runtime_error("Bundle engine missing required temporal tensors: " + bundle_source);
    }

    auto backend = trtf::CreateSpeechBackend(std::move(temporal_engine), config, hf_python);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create speech backend from bundle engine");
    }

    auto& cfg_ref = mutable_speech_config(*backend);
    load_optional_speech_tables(cfg_ref, sections, config);
    attach_speech_depth_engines(
        *backend,
        sections,
        bundle_port,
        trt_port,
        runtime,
        make_speech_depth_config(config));

    auto mimi_encoder = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "mimi_encoder_plan");
    if (mimi_encoder.ok() && mimi_encoder.value.engine != nullptr)
    {
        backend->set_mimi_encoder(std::move(mimi_encoder.value.engine), std::move(mimi_encoder.value.context));
    }
    auto mimi_decoder = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "mimi_decoder_plan");
    if (mimi_decoder.ok() && mimi_decoder.value.engine != nullptr)
    {
        backend->set_mimi_decoder(std::move(mimi_decoder.value.engine), std::move(mimi_decoder.value.context));
    }

    PipelineServices services;
    services.audio = std::make_unique<trtf::runtime::services::audio::SpeechToSpeechAudioService>(
        std::make_unique<trtf::runtime::services::common::SpeechSynthesisPort>(std::move(backend)));
    return BuildResult::Success(std::move(services));
}

BuildResult build_omni_services(
    const BuildContext& context,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    auto thinker_engine = trtf::make_decoder_engine(
        std::move(primary.engine), std::move(primary.context), config);
    if (!trtf::has_all_required_tensors(*thinker_engine))
    {
        throw std::runtime_error("Bundle engine missing required Thinker tensors: " + bundle_source);
    }

    auto backend = trtf::CreateOmniBackend(std::move(thinker_engine), config);
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Omni backend from bundle engine");
    }

    auto audio_encoder = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "audio_encoder_plan");
    if (audio_encoder.ok() && audio_encoder.value.engine != nullptr)
    {
        backend->set_audio_encoder(std::move(audio_encoder.value.engine), std::move(audio_encoder.value.context));
    }
    auto talker = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "talker_engine_plan");
    if (talker.ok() && talker.value.engine != nullptr)
    {
        auto talker_engine = trtf::make_decoder_engine(
            std::move(talker.value.engine), std::move(talker.value.context), make_omni_talker_config(config));
        backend->set_talker_engine(std::move(talker_engine));
    }
    auto code2wav = maybe_load_engine_context_for_section(bundle_port, trt_port, runtime, "code2wav_engine_plan");
    if (code2wav.ok() && code2wav.value.engine != nullptr)
    {
        backend->set_code2wav_engine(std::move(code2wav.value.engine), std::move(code2wav.value.context));
    }

    auto tokenizer = try_extract_tokenizer(sections, context.hf_python, false, "Omni");
    auto tokenizer_temp_dir = std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(
        std::move(tokenizer.temp_dir));
    auto shared_backend = std::shared_ptr<trtf::OmniBackend>(backend.release());
    auto shared_tokenizer = std::shared_ptr<trtf::ITokenizer>(std::move(tokenizer.tokenizer));
    trtf::GenerationConfig text_generation_config;
    text_generation_config.max_new_tokens = 128;

    PipelineServices services;
    services.audio = std::make_unique<trtf::runtime::services::audio::OmniAudioService>(
        std::make_unique<trtf::runtime::services::common::OmniAudioGenerationPort>(shared_backend),
        shared_tokenizer,
        tokenizer_temp_dir,
        768);
    services.text = std::make_unique<trtf::runtime::services::text::GenerationTextService>(
        shared_tokenizer,
        std::make_unique<trtf::runtime::services::common::OmniTextGenerationPort>(shared_backend),
        tokenizer_temp_dir,
        text_generation_config);
    return BuildResult::Success(std::move(services));
}

BuildResult build_audio_strategy_services(
    const BuildContext& context,
    const trtf::BundleSections& sections,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    const trtf::FastPathModelConfig& config,
    OwnedEngineContext primary,
    const std::string& bundle_source)
{
    if (context.strategy == "text_to_audio")
    {
        return build_text_to_audio_services(
            context,
            sections,
            bundle_port,
            trt_port,
            runtime,
            config,
            std::move(primary),
            bundle_source);
    }

    if (context.strategy == "speech_to_text")
    {
        return build_whisper_transcription_services(
            context,
            sections,
            bundle_port,
            trt_port,
            runtime,
            config,
            std::move(primary),
            bundle_source);
    }

    if (context.strategy == "speech_to_speech")
    {
        return build_speech_to_speech_services(
            sections,
            bundle_port,
            trt_port,
            runtime,
            config,
            context.hf_python,
            std::move(primary),
            bundle_source);
    }

    return build_omni_services(
        context,
        sections,
        bundle_port,
        trt_port,
        runtime,
        config,
        std::move(primary),
        bundle_source);
}

BuildResult build_real_services(
    const BuildContext& context,
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime)
{
    const auto config_result = resolve_fast_path_config(context, bundle_port);
    if (!config_result.ok())
    {
        return BuildResult::Failure(config_result.status, config_result.message);
    }

    const auto* sections = bundle_sections_from_context(context);
    if (sections == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "AudioStrategyBuilder requires bundle section metadata for production composition");
    }

    if (!config_result.value.runtime_strategy.empty() && config_result.value.runtime_strategy != context.strategy)
    {
        return BuildResult::Failure(
            BuildStatus::kInvalidArgument,
            "Strategy mismatch: context strategy is '" + context.strategy
                + "' but bundle config declares '" + config_result.value.runtime_strategy + "'");
    }

    auto primary = load_engine_context_for_section(bundle_port, trt_port, runtime, "engine_plan");
    if (!primary.ok())
    {
        return BuildResult::Failure(primary.status, primary.message);
    }

    const std::string bundle_source = !context.bundle_path.empty()
        ? context.bundle_path
        : (context.model_id.empty() ? "<audio-builder-bundle>" : context.model_id);

    try
    {
        return build_audio_strategy_services(
            context,
            *sections,
            bundle_port,
            trt_port,
            runtime,
            config_result.value,
            std::move(primary.value),
            bundle_source);
    }
    catch (const std::exception& ex)
    {
        return BuildResult::Failure(map_factory_error_status(ex.what()), ex.what());
    }
}

#endif // TRTF_HAS_TRT

class StubAudioService final : public IAudioService {
public:
    explicit StubAudioService(bool speech)
        : mSpeech(speech)
    {
    }

    AudioGenerationResult generate_audio(const AudioGenerationRequest& /*request*/) override
    {
        return AudioGenerationResult::Success(trtf::runtime::adapters::io::AudioArtifact{});
    }

    bool supports_speech() const override
    {
        return mSpeech;
    }

    SpeechSynthesisResult speak(const SpeechSynthesisRequest& /*request*/) override
    {
        if (!mSpeech)
        {
            return SpeechSynthesisResult::Failure(RuntimeServiceStatus::kUnsupported, "speech not supported");
        }
        return SpeechSynthesisResult::Success(trtf::runtime::adapters::io::AudioArtifact{});
    }

private:
    bool mSpeech{false};
};

class StubTranscriptionService final : public ITranscriptionService {
public:
    TranscriptionResult transcribe(const TranscriptionRequest& /*request*/) override
    {
        return TranscriptionResult::Success("transcribed");
    }
};

class StubOmniTextService final : public ITextService {
public:
    const char* generate(const char* prompt, std::size_t /*max_new_tokens*/) override
    {
        mLast = prompt == nullptr ? "" : prompt;
        return mLast.c_str();
    }

private:
    std::string mLast;
};

bool is_supported_strategy(std::string_view strategy)
{
    static constexpr std::array<std::string_view, 4> kStrategies = {
        "text_to_audio",
        "speech_to_text",
        "speech_to_speech",
        "omni_multimodal",
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

PortCheckResult maybe_validate_optional_engine(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view section_name)
{
    if (!bundle_port.has_section(section_name))
    {
        return {};
    }
    return validate_engine_for_section(bundle_port, trt_port, runtime, section_name);
}

PortCheckResult validate_optional_engine_sequence(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::initializer_list<std::string_view> section_names)
{
    for (const auto section_name : section_names)
    {
        const auto check = maybe_validate_optional_engine(
            bundle_port, trt_port, runtime, section_name);
        if (!check.ok())
        {
            return check;
        }
    }
    return {};
}

PortCheckResult validate_optional_audio_sections(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime,
    std::string_view strategy)
{
    if (strategy == "text_to_audio")
    {
        return validate_optional_engine_sequence(
            bundle_port,
            trt_port,
            runtime,
            {"vision_engine_plan", "codec_engine_plan", "coarse_engine_plan"});
    }

    if (strategy == "speech_to_text")
    {
        return validate_optional_engine_sequence(
            bundle_port, trt_port, runtime, {"vision_engine_plan"});
    }

    if (strategy == "speech_to_speech")
    {
        return validate_optional_engine_sequence(
            bundle_port,
            trt_port,
            runtime,
            {"depth_engine_plan", "mimi_encoder_plan", "mimi_decoder_plan"});
    }

    if (strategy == "omni_multimodal")
    {
        return validate_optional_engine_sequence(
            bundle_port,
            trt_port,
            runtime,
            {"audio_encoder_plan", "talker_engine_plan", "code2wav_engine_plan"});
    }

    return {};
}

} // namespace

AudioStrategyBuilder::AudioStrategyBuilder(
    const IBundlePort& bundle_port,
    const ITrtPort& trt_port,
    nvinfer1::IRuntime* runtime)
    : mBundlePort(bundle_port)
    , mTrtPort(trt_port)
    , mRuntime(runtime)
{
}

BuildResult AudioStrategyBuilder::build(const BuildContext& context)
{
    if (!is_supported_strategy(context.strategy))
    {
        return BuildResult::Failure(
            BuildStatus::kUnsupportedStrategy,
            "Unsupported audio strategy: " + context.strategy);
    }

    if (mRuntime == nullptr)
    {
        return BuildResult::Failure(
            BuildStatus::kMissingDependency,
            "AudioStrategyBuilder requires a non-null TensorRT runtime");
    }

#if TRTF_HAS_TRT
    if (has_production_bundle_context(context))
    {
        return build_real_services(context, mBundlePort, mTrtPort, mRuntime);
    }
#endif

    const auto primary = validate_engine_for_section(mBundlePort, mTrtPort, mRuntime, "engine_plan");
    if (!primary.ok())
    {
        return BuildResult::Failure(primary.status, primary.message);
    }

    const auto optional = validate_optional_audio_sections(mBundlePort, mTrtPort, mRuntime, context.strategy);
    if (!optional.ok())
    {
        return BuildResult::Failure(optional.status, optional.message);
    }

    PipelineServices services;
    if (context.strategy == "text_to_audio")
    {
        services.audio = std::make_unique<StubAudioService>(false);
    }
    else if (context.strategy == "speech_to_text")
    {
        services.transcription = std::make_unique<StubTranscriptionService>();
    }
    else if (context.strategy == "speech_to_speech")
    {
        services.audio = std::make_unique<StubAudioService>(true);
    }
    else
    {
        services.audio = std::make_unique<StubAudioService>(true);
        services.text = std::make_unique<StubOmniTextService>();
    }

    return BuildResult::Success(std::move(services));
}

} // namespace trtf::runtime::builders::audio
