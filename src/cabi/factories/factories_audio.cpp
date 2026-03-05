#include "cabi/factories/factories_audio.h"

#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/core/trt_backend_shared.h"
#include "trtf/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

namespace {

constexpr std::size_t kBytesPerMb = 1024 * 1024;

bool has_data(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

void set_decoder_vocab_from_logits(
    trtf::DecoderStepEngine& decoder_engine, bool log_mismatch = false)
{
    auto logits_shape = decoder_engine.engine->getTensorShape("logits");
    int32_t actual_vocab = -1;
    if (logits_shape.nbDims >= 2)
    {
        actual_vocab = logits_shape.d[logits_shape.nbDims - 1];
    }
    else if (logits_shape.nbDims == 1)
    {
        actual_vocab = logits_shape.d[0];
    }

    if (actual_vocab <= 0)
    {
        return;
    }
    if (log_mismatch && actual_vocab != decoder_engine.vocab_size)
    {
        std::cerr << "[trtf] Semantic: output vocab " << actual_vocab
                  << " (config says " << decoder_engine.vocab_size << ")"
                  << std::endl;
    }
    decoder_engine.vocab_size = actual_vocab;
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
            std::string("Bundle missing ") + section_name + " section: "
            + bundle_path);
    }

    const auto n_floats = data->size() / sizeof(float);
    std::vector<float> values(n_floats);
    std::memcpy(values.data(), data->data(), data->size());
    if (log_load)
    {
        std::cerr << "[trtf] Loaded " << section_name << " (" << n_floats
                  << " floats)" << std::endl;
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
        tok = trtf::extract_tokenizer_from_bundle(
            sections, hf_python, add_special_tokens);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: no tokenizer for " << warning_context
                  << " (" << e.what() << ")" << std::endl;
    }
    return tok;
}

trtf::TokenizerResult create_magpie_tokenizer_or_throw(
    const trtf::BundleSections& sections)
{
    trtf::TokenizerResult tok = {nullptr, ""};
    if (!has_data(sections.magpie_ipa_phoneme_dict_data)
        || !has_data(sections.magpie_ipa_vocab_data))
    {
        throw std::runtime_error(
            "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, "
            "magpie_ipa_vocab). Rebuild the bundle with the latest trtf-build.");
    }

    try
    {
        const auto* dict = sections.magpie_ipa_phoneme_dict_data;
        const auto* het = sections.magpie_ipa_heteronyms_data;
        const auto* voc = sections.magpie_ipa_vocab_data;
        const auto* cfg = sections.magpie_ipa_config_data;
        tok.tokenizer = trtf::CreateIpaTokenizer(
            dict->data(), dict->size(),
            has_data(het) ? het->data() : nullptr,
            has_data(het) ? het->size() : 0,
            voc->data(), voc->size(),
            has_data(cfg) ? cfg->data() : nullptr,
            has_data(cfg) ? cfg->size() : 0);
        std::cerr << "[trtf] Using native IPA tokenizer from bundle"
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf] Warning: native IPA tokenizer failed: "
                  << e.what() << std::endl;
    }

    if (!tok.tokenizer)
    {
        throw std::runtime_error(
            "Bundle missing IPA tokenizer sections (magpie_ipa_phoneme_dict, "
            "magpie_ipa_vocab). Rebuild the bundle with the latest trtf-build.");
    }
    return tok;
}

std::pair<trtf::TrtUniquePtr<nvinfer1::ICudaEngine>,
          trtf::TrtUniquePtr<nvinfer1::IExecutionContext>>
load_required_engine_context(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::vector<char>* plan_data,
    const std::string& missing_error,
    const std::string& deserialize_error,
    const std::string& context_error,
    const char* log_label)
{
    if (!has_data(plan_data))
    {
        throw std::runtime_error(missing_error);
    }

    std::cerr << "[trtf] Deserializing " << log_label << " ("
              << plan_data->size() / kBytesPerMb << " MB) ..."
              << std::endl;
    auto engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            plan_data->data(), plan_data->size()));
    if (!engine)
    {
        throw std::runtime_error(deserialize_error);
    }

    auto context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext());
    if (!context)
    {
        throw std::runtime_error(context_error);
    }
    return {std::move(engine), std::move(context)};
}

template <typename Setter>
void set_optional_engine(
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::vector<char>* plan_data,
    const char* log_label,
    Setter&& setter)
{
    if (!has_data(plan_data))
    {
        return;
    }
    if (log_label != nullptr && log_label[0] != '\0')
    {
        std::cerr << "[trtf] Deserializing " << log_label << " ("
                  << plan_data->size() / kBytesPerMb << " MB) ..."
                  << std::endl;
    }

    auto engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            plan_data->data(), plan_data->size()));
    if (!engine)
    {
        return;
    }

    auto context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext());
    setter(std::move(engine), std::move(context));
}

std::vector<float> load_required_bark_embed(
    const std::vector<char>* data,
    const char* missing_section_name,
    const char* log_name,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& bundle_path)
{
    auto values = load_required_float_section(
        data, missing_section_name, bundle_path, /*log_load=*/false);
    const auto n_floats = values.size();
    std::cerr << "[trtf] Loaded " << log_name << " embedding table ("
              << n_floats / std::max(fp_cfg.hidden_size, 1)
              << " x " << fp_cfg.hidden_size << ")" << std::endl;
    return values;
}

trtf::FastPathModelConfig make_bark_coarse_config(
    const trtf::FastPathModelConfig& fp_cfg)
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
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const trtf::BundleSections& sections,
    const trtf::FastPathModelConfig& fp_cfg,
    const std::string& bundle_path)
{
    auto [coarse_trt, coarse_ctx] = load_required_engine_context(
        runtime_ptr,
        sections.coarse_engine_plan_data,
        "Bundle missing coarse_engine section: " + bundle_path,
        "Failed to deserialize coarse engine: " + bundle_path,
        "Failed to create coarse execution context",
        "coarse TRT engine");

    auto coarse_engine = trtf::make_decoder_engine(
        std::move(coarse_trt), std::move(coarse_ctx),
        make_bark_coarse_config(fp_cfg));
    if (!trtf::has_all_required_tensors(*coarse_engine))
    {
        throw std::runtime_error(
            "Bundle coarse engine missing required tensors: " + bundle_path);
    }
    return coarse_engine;
}

template <typename BarkBackendT>
void maybe_load_bark_fine_embeddings(
    BarkBackendT& bark_backend, const trtf::BundleSections& sections)
{
    if (!has_data(sections.fine_embed_data))
    {
        return;
    }

    const auto n_floats = sections.fine_embed_data->size() / sizeof(float);
    std::vector<float> fine_embed(n_floats);
    std::memcpy(
        fine_embed.data(),
        sections.fine_embed_data->data(),
        sections.fine_embed_data->size());

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

    bark_backend.set_fine_embeddings(
        std::move(fine_embed), std::move(fine_pos_embed));
    std::cerr << "[trtf] Loaded fine embedding tables (" << n_floats
              << " floats)" << std::endl;
}

trtf::FastPathModelConfig make_omni_talker_config(
    const trtf::FastPathModelConfig& fp_cfg)
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

template <typename OmniBackendT>
void set_optional_omni_talker_engine(
    OmniBackendT& omni_backend,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::vector<char>* plan_data,
    const trtf::FastPathModelConfig& fp_cfg)
{
    set_optional_engine(
        runtime_ptr, plan_data, "Omni Talker TRT engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> talker_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> talker_ctx)
        {
            auto talker_engine = trtf::make_decoder_engine(
                std::move(talker_trt), std::move(talker_ctx),
                make_omni_talker_config(fp_cfg));
            omni_backend.set_talker_engine(std::move(talker_engine));
        });
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
    std::memcpy(
        cfg_ref.depth_projection.data(), proj_data->data(), proj_data->size());
    cfg_ref.temporal_hidden_for_proj = fp_cfg.hidden_size;
    std::cerr << "[trtf] Loaded depth_projection: " << num_floats
              << " floats" << std::endl;
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
    std::memcpy(
        cfg_ref.audio_embeddings.data(), emb_data->data(), emb_data->size());
    if (fp_cfg.speech_num_codebooks > 0 && fp_cfg.hidden_size > 0)
    {
        cfg_ref.audio_vocab_size = static_cast<int32_t>(
            num_floats
            / (static_cast<std::size_t>(fp_cfg.speech_num_codebooks)
               * fp_cfg.hidden_size));
    }
    std::cerr << "[trtf] Loaded audio_embeddings: " << num_floats
              << " floats (" << fp_cfg.speech_num_codebooks
              << " codebooks x " << cfg_ref.audio_vocab_size
              << " vocab x " << fp_cfg.hidden_size << " hidden)"
              << std::endl;
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
    std::memcpy(
        cfg_ref.temporal_text_embedding.data(),
        tte_data->data(),
        tte_data->size());
    if (fp_cfg.hidden_size > 0)
    {
        cfg_ref.temporal_text_vocab = static_cast<int32_t>(
            num_floats / static_cast<std::size_t>(fp_cfg.hidden_size));
    }
    std::cerr << "[trtf] Loaded temporal_text_embedding: " << num_floats
              << " floats (" << cfg_ref.temporal_text_vocab
              << " vocab x " << fp_cfg.hidden_size << " hidden)"
              << std::endl;
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
    std::memcpy(
        cfg_ref.depth_text_embedding.data(),
        depth_data->data(),
        depth_data->size());
    if (fp_cfg.speech_depth_hidden_size > 0)
    {
        cfg_ref.depth_text_vocab = static_cast<int32_t>(
            num_floats / static_cast<std::size_t>(fp_cfg.speech_depth_hidden_size));
    }
    std::cerr << "[trtf] Loaded depth_text_embedding: " << num_floats
              << " floats (" << cfg_ref.depth_text_vocab
              << " vocab x " << fp_cfg.speech_depth_hidden_size << " hidden)"
              << std::endl;
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
    std::memcpy(
        cfg_ref.depth_audio_embeddings.data(),
        dae_data->data(),
        dae_data->size());
    const int32_t audio_vocab = cfg_ref.audio_vocab_size;
    const int32_t depth_hidden = fp_cfg.speech_depth_hidden_size;
    if (audio_vocab > 0 && depth_hidden > 0)
    {
        cfg_ref.num_depformer_emb = static_cast<int32_t>(
            num_floats / (static_cast<std::size_t>(audio_vocab) * depth_hidden));
    }
    std::cerr << "[trtf] Loaded depth_audio_embeddings: " << num_floats
              << " floats (" << cfg_ref.num_depformer_emb
              << " codebooks x " << audio_vocab
              << " vocab x " << depth_hidden << " hidden)"
              << std::endl;
}

void load_optional_speech_tables(
    trtf::SpeechConfig& cfg_ref,
    const trtf::BundleSections& sections,
    const trtf::FastPathModelConfig& fp_cfg)
{
    maybe_load_speech_depth_projection(
        cfg_ref, sections.depth_projection_data, fp_cfg);
    maybe_load_speech_audio_embeddings(
        cfg_ref, sections.audio_embeddings_data, fp_cfg);
    maybe_load_temporal_text_embedding(
        cfg_ref, sections.temporal_text_embedding_data, fp_cfg);
    maybe_load_depth_text_embedding(
        cfg_ref, sections.depth_text_embedding_data, fp_cfg);
    maybe_load_depth_audio_embeddings(
        cfg_ref, sections.depth_audio_embeddings_data, fp_cfg);
}

trtf::FastPathModelConfig make_speech_depth_config(
    const trtf::FastPathModelConfig& fp_cfg)
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
    const std::vector<const std::vector<char>*>& depth_engine_plans,
    const std::vector<char>* depth_engine_plan_data,
    const trtf::FastPathModelConfig& depth_cfg,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr)
{
    if (!depth_engine_plans.empty())
    {
        std::cerr << "[trtf] Deserializing " << depth_engine_plans.size()
                  << " per-codebook depth engines ..." << std::endl;
        for (std::size_t cb = 0; cb < depth_engine_plans.size(); ++cb)
        {
            const auto* plan = depth_engine_plans[cb];
            if (!has_data(plan))
            {
                continue;
            }
            std::cerr << "[trtf] Deserializing depth engine cb=" << cb
                      << " (" << plan->size() / kBytesPerMb << " MB) ..."
                      << std::endl;

            auto depth_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
                runtime_ptr->deserializeCudaEngine(plan->data(), plan->size()));
            if (!depth_trt)
            {
                continue;
            }

            auto depth_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
                depth_trt->createExecutionContext());
            auto depth_engine = trtf::make_decoder_engine(
                std::move(depth_trt), std::move(depth_ctx), depth_cfg);
            speech_backend.set_depth_engine(
                static_cast<int32_t>(cb), std::move(depth_engine));
        }
        return;
    }

    if (!has_data(depth_engine_plan_data))
    {
        return;
    }
    std::cerr << "[trtf] Deserializing depth TRT engine ("
              << depth_engine_plan_data->size() / kBytesPerMb << " MB) ..."
              << std::endl;
    auto depth_trt = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime_ptr->deserializeCudaEngine(
            depth_engine_plan_data->data(), depth_engine_plan_data->size()));
    if (!depth_trt)
    {
        return;
    }

    auto depth_ctx = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(
        depth_trt->createExecutionContext());
    auto depth_engine = trtf::make_decoder_engine(
        std::move(depth_trt), std::move(depth_ctx), depth_cfg);
    speech_backend.set_depth_engine(std::move(depth_engine));
}

} // namespace

trtf::IPipeline* create_magpie_tts_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    (void) hf_python; // Kept for factory signature consistency; Magpie uses native bundle tokenizer data.

    auto decoder_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    set_decoder_vocab_from_logits(*decoder_engine);

    auto [encoder_engine, encoder_ctx] = load_required_engine_context(
        runtime_ptr,
        sections.vision_plan_data,
        "Bundle missing encoder engine (vision_engine_plan) for MagpieTTS: "
            + bundle_path,
        "Failed to deserialize MagpieTTS encoder engine: " + bundle_path,
        "Failed to create MagpieTTS encoder execution context",
        "MagpieTTS encoder TRT engine");

    auto audio_embed = load_required_float_section(
        sections.magpie_audio_embed_data, "magpie_audio_embed", bundle_path);
    auto text_embed = load_required_float_section(
        sections.magpie_text_embed_data, "magpie_text_embed", bundle_path);
    auto context_embed = load_required_float_section(
        sections.magpie_context_embed_data, "magpie_context_embed", bundle_path);
    auto context_lengths = load_optional_int32_section(
        sections.magpie_context_lengths_data);

    auto magpie_backend = trtf::CreateMagpieTTSBackend(
        std::move(decoder_engine), std::move(encoder_engine), std::move(encoder_ctx),
        std::move(audio_embed), std::move(text_embed),
        std::move(context_embed), std::move(context_lengths), fp_cfg);
    if (!magpie_backend || !magpie_backend->is_available())
    {
        throw std::runtime_error("Failed to create MagpieTTS backend from bundle engines");
    }

    set_optional_engine(
        runtime_ptr,
        sections.codec_engine_plan_data,
        "MagpieTTS codec engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> codec_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> codec_ctx)
        {
            magpie_backend->set_codec_engine(
                std::move(codec_trt), std::move(codec_ctx));
        });

    auto tok = create_magpie_tokenizer_or_throw(sections);

    std::cerr << "[trtf] Runtime ready (backend=trt_magpie_tts, strategy=text_to_audio)" << std::endl;

    auto* pipeline = detail::create_pipeline_impl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_magpie_tts");
    if (!tok.temp_dir.empty())
    {
        detail::set_bundle_temp_dir(pipeline, std::move(tok.temp_dir));
    }
    detail::set_magpie_tts_backend(pipeline, std::move(magpie_backend));
    return pipeline;
}

trtf::IPipeline* create_bark_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto semantic_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*semantic_engine))
    {
        throw std::runtime_error("Bundle engine missing required semantic tensors: " + bundle_path);
    }
    set_decoder_vocab_from_logits(*semantic_engine, /*log_mismatch=*/true);

    auto semantic_embed = load_required_bark_embed(
        sections.semantic_embed_data,
        "semantic_embed",
        "semantic",
        fp_cfg,
        bundle_path);
    auto coarse_embed = load_required_bark_embed(
        sections.coarse_embed_data,
        "coarse_embed",
        "coarse",
        fp_cfg,
        bundle_path);
    auto coarse_engine = create_required_bark_coarse_engine(
        runtime_ptr, sections, fp_cfg, bundle_path);

    auto bark_backend = trtf::CreateBarkBackend(
        std::move(semantic_engine), std::move(coarse_engine),
        std::move(semantic_embed), std::move(coarse_embed), fp_cfg);
    if (!bark_backend || !bark_backend->is_available())
    {
        throw std::runtime_error("Failed to create Bark backend from bundle engines");
    }

    set_optional_engine(
        runtime_ptr,
        sections.codec_engine_plan_data,
        "",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> codec_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> codec_ctx)
        {
            bark_backend->set_codec_engine(
                std::move(codec_trt), std::move(codec_ctx));
        });
    set_optional_engine(
        runtime_ptr,
        sections.fine_engine_plan_data,
        "fine TRT engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> fine_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> fine_ctx)
        {
            bark_backend->set_fine_engine(
                std::move(fine_trt), std::move(fine_ctx));
        });
    maybe_load_bark_fine_embeddings(*bark_backend, sections);

    auto tok = try_extract_tokenizer(
        sections, hf_python, /*add_special_tokens=*/false, "Bark");

    std::cerr << "[trtf] Runtime ready (backend=trt_bark, strategy=text_to_audio)" << std::endl;

    auto* pipeline = detail::create_pipeline_impl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_bark");
    if (!tok.temp_dir.empty())
    {
        detail::set_bundle_temp_dir(pipeline, std::move(tok.temp_dir));
    }
    detail::set_bark_backend(pipeline, std::move(bark_backend));
    return pipeline;
}

trtf::IPipeline* create_omni_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto thinker_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*thinker_engine))
    {
        throw std::runtime_error("Bundle engine missing required Thinker tensors: " + bundle_path);
    }

    auto omni_backend = trtf::CreateOmniBackend(std::move(thinker_engine), fp_cfg);
    if (!omni_backend || !omni_backend->is_available())
    {
        throw std::runtime_error("Failed to create Omni backend from bundle engine");
    }

    set_optional_engine(
        runtime_ptr,
        sections.audio_encoder_plan_data,
        "Omni audio encoder TRT engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> audio_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> audio_ctx)
        {
            omni_backend->set_audio_encoder(
                std::move(audio_trt), std::move(audio_ctx));
        });
    set_optional_omni_talker_engine(
        *omni_backend,
        runtime_ptr,
        sections.talker_engine_plan_data,
        fp_cfg);
    set_optional_engine(
        runtime_ptr,
        sections.code2wav_engine_plan_data,
        "Omni Code2Wav TRT engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> code2wav_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> code2wav_ctx)
        {
            omni_backend->set_code2wav_engine(
                std::move(code2wav_trt), std::move(code2wav_ctx));
        });

    auto tok = try_extract_tokenizer(
        sections, hf_python, /*add_special_tokens=*/false, "Omni");

    std::cerr << "[trtf] Runtime ready (backend=trt_omni, strategy=omni_multimodal)" << std::endl;

    auto* pipeline = detail::create_pipeline_impl(
        model_id, std::move(tok.tokenizer), nullptr, "trt_omni");
    if (!tok.temp_dir.empty())
    {
        detail::set_bundle_temp_dir(pipeline, std::move(tok.temp_dir));
    }
    detail::set_omni_backend(pipeline, std::move(omni_backend));
    return pipeline;
}

trtf::IPipeline* create_speech_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    trtf::TrtUniquePtr<nvinfer1::IRuntime>& runtime_ptr,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto temporal_engine = trtf::make_decoder_engine(
        std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*temporal_engine))
    {
        throw std::runtime_error(
            "Bundle engine missing required temporal tensors: " + bundle_path);
    }

    auto speech_backend = trtf::CreateSpeechBackend(
        std::move(temporal_engine), fp_cfg, hf_python);
    if (!speech_backend || !speech_backend->is_available())
    {
        throw std::runtime_error(
            "Failed to create speech backend from bundle engine");
    }

    auto& cfg_ref = mutable_speech_config(*speech_backend);
    load_optional_speech_tables(cfg_ref, sections, fp_cfg);
    attach_speech_depth_engines(
        *speech_backend,
        sections.depth_engine_plans,
        sections.depth_engine_plan_data,
        make_speech_depth_config(fp_cfg),
        runtime_ptr);
    set_optional_engine(
        runtime_ptr,
        sections.mimi_encoder_plan_data,
        "Mimi encoder TRT engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> enc_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> enc_ctx)
        {
            speech_backend->set_mimi_encoder(
                std::move(enc_trt), std::move(enc_ctx));
        });
    set_optional_engine(
        runtime_ptr,
        sections.mimi_decoder_plan_data,
        "Mimi decoder TRT engine",
        [&](trtf::TrtUniquePtr<nvinfer1::ICudaEngine> dec_trt,
            trtf::TrtUniquePtr<nvinfer1::IExecutionContext> dec_ctx)
        {
            speech_backend->set_mimi_decoder(
                std::move(dec_trt), std::move(dec_ctx));
        });

    std::cerr << "[trtf] Runtime ready (backend=trt_speech, strategy=speech_to_speech)"
              << std::endl;

    // Speech pipelines don't need a tokenizer or generation backend
    auto* pipeline = detail::create_pipeline_impl(
        model_id, nullptr, nullptr, "trt_speech");
    detail::set_speech_backend(pipeline, std::move(speech_backend));
    return pipeline;
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
